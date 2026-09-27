"""
Face detection using MTCNN (Multi-task Cascaded Convolutional Networks).

Uses facenet-pytorch's MTCNN implementation which provides:
- Face bounding boxes
- Detection confidence scores
- 5 facial landmarks (left eye, right eye, nose, left mouth, right mouth)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from facenet_pytorch import MTCNN
from PIL import Image


@dataclass
class FaceDetection:
    """Result of face detection on a single image."""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2 in image coordinates
    confidence: float
    landmarks: Optional[np.ndarray] = None  # Shape (5, 2) - 5 landmarks (x, y)
    image_size: Tuple[int, int] = (0, 0)  # width, height


@dataclass
class FaceDetectorConfig:
    """Configuration for MTCNN face detector."""
    min_face_size: int = 20
    thresholds: Tuple[float, float, float] = (0.6, 0.7, 0.7)
    factor: float = 0.709
    keep_all: bool = True
    device: str = "cpu"  # or "cuda"
    # Post-processing
    min_confidence: float = 0.9  # Minimum confidence to accept detection
    select_largest: bool = True  # If multiple faces, select largest by area


class FaceDetector:
    """
    MTCNN-based face detector.
    
    Detects faces and 5 facial landmarks in images.
    """
    
    def __init__(self, config: Optional[FaceDetectorConfig] = None):
        self.config = config or FaceDetectorConfig()
        self.detector = MTCNN(
            min_face_size=self.config.min_face_size,
            thresholds=self.config.thresholds,
            factor=self.config.factor,
            keep_all=self.config.keep_all,
            device=self.config.device,
            post_process=False,  # We handle post-processing ourselves
        )
    
    def detect(self, image: Image.Image) -> List[FaceDetection]:
        """
        Detect faces in an image.
        
        Args:
            image: PIL Image in RGB format
            
        Returns:
            List of FaceDetection objects, sorted by confidence (highest first)
        """
        width, height = image.size
        
        # MTCNN expects RGB image
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Detect faces
        # Returns: boxes (N, 4), probs (N,), landmarks (N, 5, 2)
        boxes, probs, landmarks = self.detector.detect(image, landmarks=True)
        
        detections = []
        
        if boxes is not None:
            for i, (box, prob, lm) in enumerate(zip(boxes, probs, landmarks)):
                if prob is None or prob < self.config.min_confidence:
                    continue
                
                # Convert box to integers (x1, y1, x2, y2)
                x1, y1, x2, y2 = box.astype(int)
                
                # Clip to image boundaries
                x1 = max(0, min(x1, width - 1))
                y1 = max(0, min(y1, height - 1))
                x2 = max(0, min(x2, width))
                y2 = max(0, min(y2, height))
                
                if x2 <= x1 or y2 <= y1:
                    continue
                
                detection = FaceDetection(
                    bbox=(x1, y1, x2, y2),
                    confidence=float(prob),
                    landmarks=lm if lm is not None else None,
                    image_size=(width, height),
                )
                detections.append(detection)
        
        # Sort by confidence (highest first)
        detections.sort(key=lambda d: d.confidence, reverse=True)
        
        return detections
    
    def detect_primary(self, image: Image.Image) -> Optional[FaceDetection]:
        """
        Detect the primary face in an image.
        
        Selection rule (deterministic):
        1. Filter by minimum confidence
        2. If multiple faces remain, select the largest by area
        3. If still tied, select the one closest to image center
        
        Args:
            image: PIL Image in RGB format
            
        Returns:
            Primary FaceDetection or None if no face detected
        """
        detections = self.detect(image)
        
        if not detections:
            return None
        
        if len(detections) == 1:
            return detections[0]
        
        # Multiple faces - apply selection rule
        if self.config.select_largest:
            # Select largest by area
            primary = max(detections, key=lambda d: (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]))
        else:
            # Select highest confidence (already sorted)
            primary = detections[0]
        
        return primary


def expand_bbox(
    bbox: Tuple[int, int, int, int],
    image_size: Tuple[int, int],
    margin: float = 0.2,
) -> Tuple[int, int, int, int]:
    """
    Expand bounding box by a percentage margin around its center.
    
    The expansion is symmetric: margin * width added to EACH SIDE (left and right),
    margin * height added to EACH SIDE (top and bottom).
    
    For margin=0.2 (20%): a 100x100 box becomes 140x140 (20% on each side = 40% total).
    
    Args:
        bbox: (x1, y1, x2, y2) in image coordinates
        image_size: (width, height) of the image
        margin: Expansion factor per side (0.2 = 20% expansion on each side)
        
    Returns:
        Expanded bbox (x1, y1, x2, y2) clipped to image boundaries
    """
    x1, y1, x2, y2 = bbox
    width, height = image_size
    
    box_w = x2 - x1
    box_h = y2 - y1
    
    # Center of the box
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    
    # New half-width and half-height after expansion
    # margin is per side, so total expansion is 2 * margin
    new_half_w = box_w * (1 + 2 * margin) / 2
    new_half_h = box_h * (1 + 2 * margin) / 2
    
    # New coordinates
    new_x1 = cx - new_half_w
    new_y1 = cy - new_half_h
    new_x2 = cx + new_half_w
    new_y2 = cy + new_half_h
    
    # Clip to image boundaries
    new_x1 = max(0, int(np.floor(new_x1)))
    new_y1 = max(0, int(np.floor(new_y1)))
    new_x2 = min(width, int(np.ceil(new_x2)))
    new_y2 = min(height, int(np.ceil(new_y2)))
    
    # Ensure valid box
    if new_x2 <= new_x1:
        new_x2 = min(width, new_x1 + 1)
    if new_y2 <= new_y1:
        new_y2 = min(height, new_y1 + 1)
    
    return (new_x1, new_y1, new_x2, new_y2)


# Canonical 5-point landmark template for 224x224 aligned face
# Based on standard facial landmark positions (normalized to 224x224)
# Order: left_eye, right_eye, nose, left_mouth, right_mouth
CANONICAL_LANDMARKS_224 = np.array([
    [84.0, 84.0],    # Left eye
    [140.0, 84.0],   # Right eye
    [112.0, 120.0],  # Nose
    [84.0, 156.0],   # Left mouth
    [140.0, 156.0],  # Right mouth
], dtype=np.float32)


def estimate_similarity_transform(
    src_landmarks: np.ndarray,
    dst_landmarks: np.ndarray = CANONICAL_LANDMARKS_224,
) -> Optional[np.ndarray]:
    """
    Estimate similarity transform (rotation + translation + uniform scaling)
    from source landmarks to destination landmarks.
    
    Uses Umeyama's method for least-squares similarity transform estimation.
    
    Args:
        src_landmarks: Source landmarks (N, 2)
        dst_landmarks: Destination landmarks (N, 2)
        
    Returns:
        2x3 affine transformation matrix, or None if estimation fails
    """
    if src_landmarks.shape != dst_landmarks.shape:
        return None
    if src_landmarks.shape[0] < 2:
        return None
    
    # Umeyama's algorithm for similarity transform
    src = src_landmarks.astype(np.float64)
    dst = dst_landmarks.astype(np.float64)
    
    # Centroids
    src_mean = np.mean(src, axis=0)
    dst_mean = np.mean(dst, axis=0)
    
    # Centered coordinates
    src_centered = src - src_mean
    dst_centered = dst - dst_mean
    
    # Compute covariance matrix
    cov = src_centered.T @ dst_centered
    
    # SVD
    U, _, Vt = np.linalg.svd(cov)
    
    # Rotation matrix
    R = Vt.T @ U.T
    
    # Ensure proper rotation (det = 1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # Scale
    src_var = np.sum(src_centered ** 2)
    if src_var < 1e-10:
        return None
    scale = np.trace(R @ cov) / src_var
    
    # Translation
    t = dst_mean - scale * (R @ src_mean)
    
    # Construct 2x3 affine matrix
    M = np.zeros((2, 3), dtype=np.float32)
    M[:2, :2] = (scale * R).astype(np.float32)
    M[:2, 2] = t.astype(np.float32)
    
    return M


def apply_similarity_transform(
    image: Image.Image,
    transform: np.ndarray,
    output_size: Tuple[int, int] = (224, 224),
) -> Image.Image:
    """
    Apply similarity transform to image using PIL.
    
    Args:
        image: Input PIL Image
        transform: 2x3 affine transformation matrix
        output_size: Output image size (width, height)
        
    Returns:
        Transformed PIL Image
    """
    # Convert to numpy for OpenCV warp
    img_array = np.array(image)
    
    # Use OpenCV for affine warp
    import cv2
    warped = cv2.warpAffine(
        img_array,
        transform,
        output_size,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    
    return Image.fromarray(warped)


def compute_quality_metrics(
    image: Image.Image,
    face_bbox: Optional[Tuple[int, int, int, int]] = None,
) -> dict:
    """
    Compute quality metrics for a face crop.
    
    Args:
        image: PIL Image (face crop)
        face_bbox: Original face bbox in source image (optional)
        
    Returns:
        Dictionary of quality metrics
    """
    img_array = np.array(image)
    
    # Convert to grayscale for blur detection
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    
    # Blur score (Laplacian variance) - higher = sharper
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    
    # Brightness (mean pixel value) - 0-255
    brightness = float(np.mean(img_array))
    
    # Face size
    face_size = None
    if face_bbox:
        w = face_bbox[2] - face_bbox[0]
        h = face_bbox[3] - face_bbox[1]
        face_size = float(w * h)
    
    return {
        "blur_score": blur_score,
        "brightness": brightness,
        "face_size": face_size,
        "resolution": f"{image.width}x{image.height}",
    }


import cv2