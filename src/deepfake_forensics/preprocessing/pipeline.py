"""
Preprocessing pipeline for EXP-S01.

Orchestrates face detection, bounding box expansion, landmark alignment,
and standardized 224x224 face crop generation.
"""

import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image

from deepfake_forensics.preprocessing.face_detector import (
    CANONICAL_LANDMARKS_224,
    FaceDetection,
    FaceDetector,
    FaceDetectorConfig,
    apply_similarity_transform,
    compute_quality_metrics,
    estimate_similarity_transform,
    expand_bbox,
    transform_landmarks_to_crop,
)


def _default_canonical_landmarks() -> np.ndarray:
    """Factory function for canonical landmarks."""
    return CANONICAL_LANDMARKS_224.copy()


class ProcessingStatus:
    """Explicit processing status codes."""
    SUCCESS = "SUCCESS"
    NO_FACE = "NO_FACE"
    MULTIPLE_FACE_AMBIGUOUS = "MULTIPLE_FACE_AMBIGUOUS"
    LANDMARK_FAILURE = "LANDMARK_FAILURE"
    ALIGNMENT_FAILURE = "ALIGNMENT_FAILURE"
    DECODE_FAILURE = "DECODE_FAILURE"
    CORRUPT_MEDIA = "CORRUPT_MEDIA"
    LOW_QUALITY = "LOW_QUALITY"
    VIDEO_DECODE_FAILURE = "VIDEO_DECODE_FAILURE"
    NO_FRAMES = "NO_FRAMES"


@dataclass
class PreprocessingConfig:
    """Configuration for preprocessing pipeline."""
    # Output
    target_size: Tuple[int, int] = (224, 224)
    output_format: str = "jpg"
    output_quality: int = 95
    
    # Face detection
    face_detector: str = "mtcnn"
    min_face_size: int = 20
    min_confidence: float = 0.9
    select_largest: bool = True
    
    # Bounding box expansion
    bbox_margin: float = 0.2  # 20% expansion
    
    # Alignment
    align_method: str = "similarity"
    canonical_landmarks: np.ndarray = field(default_factory=_default_canonical_landmarks)
    
    # Quality thresholds
    min_blur_score: float = 50.0  # Minimum Laplacian variance
    min_face_size_pixels: int = 1000  # Minimum face area in pixels
    
    # Video processing
    frame_interval: int = 1
    max_frames_per_video: Optional[int] = None
    
    # Versioning
    version: str = "1.0"
    detector_version: str = "mtcnn-facenet-pytorch-2.5.3"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization."""
        return {
            "target_size": self.target_size,
            "output_format": self.output_format,
            "output_quality": self.output_quality,
            "face_detector": self.face_detector,
            "min_face_size": self.min_face_size,
            "min_confidence": self.min_confidence,
            "select_largest": self.select_largest,
            "bbox_margin": self.bbox_margin,
            "align_method": self.align_method,
            "canonical_landmarks": self.canonical_landmarks.tolist(),
            "min_blur_score": self.min_blur_score,
            "min_face_size_pixels": self.min_face_size_pixels,
            "frame_interval": self.frame_interval,
            "max_frames_per_video": self.max_frames_per_video,
            "version": self.version,
            "detector_version": self.detector_version,
        }
    
    def config_hash(self) -> str:
        """Compute hash of configuration for versioning."""
        config_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]


@dataclass
class PreprocessedSample:
    """Result of preprocessing a single media file/frame."""
    # Identifiers
    sample_id: str
    media_id: str
    dataset: str
    label: int
    split: str
    identity_id: Optional[str] = None
    source_id: Optional[str] = None
    manipulation_type: Optional[str] = None
    generator: Optional[str] = None
    
    # Source
    source_path: str = ""
    source_sha256: Optional[str] = None
    
    # Derived artifact
    derived_path: str = ""
    derived_sha256: Optional[str] = None
    
    # Face detection
    face_detected: bool = False
    face_bbox: Optional[Tuple[int, int, int, int]] = None  # x1, y1, x2, y2 in source
    expanded_bbox: Optional[Tuple[int, int, int, int]] = None
    detector_confidence: Optional[float] = None
    
    # Landmarks
    landmarks: Optional[List[Tuple[float, float]]] = None
    landmarks_crop: Optional[List[Tuple[float, float]]] = None
    landmark_status: str = ProcessingStatus.NO_FACE
    
    # Alignment
    alignment_status: str = ProcessingStatus.NO_FACE
    alignment_transform: Optional[List[List[float]]] = None
    
    # Quality
    blur_score: Optional[float] = None
    brightness: Optional[float] = None
    face_size: Optional[float] = None
    
    # Video-specific
    frame_index: Optional[int] = None
    timestamp: Optional[float] = None
    tracking_id: Optional[str] = None
    
    # Status
    processing_status: str = ProcessingStatus.NO_FACE
    
    # Metadata
    preprocessing_version: str = "1.0"
    preprocessing_config_hash: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for manifest generation."""
        d = {
            "sample_id": self.sample_id,
            "media_id": self.media_id,
            "dataset": self.dataset,
            "label": self.label,
            "split": self.split,
            "identity_id": self.identity_id,
            "source_id": self.source_id,
            "manipulation_type": self.manipulation_type,
            "generator": self.generator,
            "source_path": self.source_path,
            "source_sha256": self.source_sha256,
            "file_path": self.derived_path,
            "derived_sha256": self.derived_sha256,
            "face_detected": self.face_detected,
            "face_bbox_x": self.face_bbox[0] if self.face_bbox else None,
            "face_bbox_y": self.face_bbox[1] if self.face_bbox else None,
            "face_bbox_w": (self.face_bbox[2] - self.face_bbox[0]) if self.face_bbox else None,
            "face_bbox_h": (self.face_bbox[3] - self.face_bbox[1]) if self.face_bbox else None,
            "expanded_bbox_x": self.expanded_bbox[0] if self.expanded_bbox else None,
            "expanded_bbox_y": self.expanded_bbox[1] if self.expanded_bbox else None,
            "expanded_bbox_w": (self.expanded_bbox[2] - self.expanded_bbox[0]) if self.expanded_bbox else None,
            "expanded_bbox_h": (self.expanded_bbox[3] - self.expanded_bbox[1]) if self.expanded_bbox else None,
            "detector_confidence": self.detector_confidence,
            "landmarks": self.landmarks,
            "landmarks_crop": self.landmarks_crop,
            "landmark_status": self.landmark_status,
            "alignment_status": self.alignment_status,
            "alignment_transform": self.alignment_transform,
            "blur_score": self.blur_score,
            "brightness": self.brightness,
            "face_size": self.face_size,
            "frame_index": self.frame_index,
            "timestamp": self.timestamp,
            "tracking_id": self.tracking_id,
            "processing_status": self.processing_status,
            "preprocessing_version": self.preprocessing_version,
            "preprocessing_config_hash": self.preprocessing_config_hash,
            "created_at": self.created_at,
        }
        return d


@dataclass
class PreprocessingResult:
    """Result of preprocessing a single media file."""
    samples: List[PreprocessedSample]
    config: PreprocessingConfig
    media_id: str
    source_path: str
    source_sha256: str
    total_frames: int = 1
    processed_frames: int = 0
    failed_frames: int = 0


class ImagePreprocessor:
    """
    Preprocesses a single image file.
    """
    
    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self.detector = FaceDetector(FaceDetectorConfig(
            min_face_size=config.min_face_size,
            min_confidence=config.min_confidence,
            keep_all=True,
            select_largest=config.select_largest,
            device="cpu",
        ))
    
    def compute_sha256(self, file_path: Path) -> str:
        """Compute SHA256 hash of a file."""
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    
    def process(
        self,
        source_path: Path,
        output_dir: Path,
        sample_id: str,
        media_id: str,
        dataset: str,
        label: int,
        split: str,
        identity_id: Optional[str] = None,
        source_id: Optional[str] = None,
        manipulation_type: Optional[str] = None,
        generator: Optional[str] = None,
    ) -> PreprocessedSample:
        """
        Process a single image.
        
        Args:
            source_path: Path to source image
            output_dir: Directory to save derived face crop
            sample_id: Unique sample identifier
            media_id: Media identifier
            dataset: Dataset name
            label: 0=authentic, 1=manipulated
            split: train/val/test
            identity_id: Identity identifier (optional)
            source_id: Source identifier (optional)
            manipulation_type: Manipulation type (optional)
            generator: Generator name (optional)
            
        Returns:
            PreprocessedSample with all metadata
        """
        # Initialize sample
        sample = PreprocessedSample(
            sample_id=sample_id,
            media_id=media_id,
            dataset=dataset,
            label=label,
            split=split,
            identity_id=identity_id,
            source_id=source_id,
            manipulation_type=manipulation_type,
            generator=generator,
            source_path=str(source_path),
            preprocessing_version=self.config.version,
            preprocessing_config_hash=self.config.config_hash(),
        )
        
        # Compute source hash
        try:
            sample.source_sha256 = self.compute_sha256(source_path)
        except Exception as e:
            sample.processing_status = ProcessingStatus.CORRUPT_MEDIA
            return sample
        
        # Load image
        try:
            image = Image.open(source_path)
            if image.mode != "RGB":
                image = image.convert("RGB")
        except Exception as e:
            sample.processing_status = ProcessingStatus.DECODE_FAILURE
            return sample
        
        # Detect face
        primary_face = self.detector.detect_primary(image)
        
        if primary_face is None:
            sample.processing_status = ProcessingStatus.NO_FACE
            return sample
        
        # Store detection info
        sample.face_detected = True
        sample.face_bbox = primary_face.bbox
        sample.detector_confidence = primary_face.confidence
        
        # Expand bbox by 20%
        expanded = expand_bbox(
            primary_face.bbox,
            primary_face.image_size,
            margin=self.config.bbox_margin,
        )
        sample.expanded_bbox = expanded
        
        # Check landmarks
        if primary_face.landmarks is None:
            sample.landmark_status = ProcessingStatus.LANDMARK_FAILURE
            sample.processing_status = ProcessingStatus.LANDMARK_FAILURE
            return sample
        
        # Convert landmarks to list of tuples (source image coordinates)
        landmarks_list = [(float(x), float(y)) for x, y in primary_face.landmarks]
        sample.landmarks = landmarks_list
        sample.landmark_status = ProcessingStatus.SUCCESS
        
        # Crop expanded region from source image
        x1, y1, x2, y2 = expanded
        expanded_crop = image.crop((x1, y1, x2, y2))
        
        # Transform landmarks to expanded-crop coordinates
        src_lm_crop = transform_landmarks_to_crop(
            primary_face.landmarks.astype(np.float32),
            expanded,
        )
        
        # Store crop-relative landmarks as well
        sample.landmarks_crop = [(float(x), float(y)) for x, y in src_lm_crop]
        
        # Estimate similarity transform from crop-relative landmarks
        transform = estimate_similarity_transform(src_lm_crop, self.config.canonical_landmarks)
        
        if transform is None:
            sample.alignment_status = ProcessingStatus.ALIGNMENT_FAILURE
            sample.processing_status = ProcessingStatus.ALIGNMENT_FAILURE
            return sample
        
        sample.alignment_transform = transform.tolist()
        
        # Apply alignment to expanded crop (not full image)
        aligned = apply_similarity_transform(
            expanded_crop,
            transform,
            output_size=self.config.target_size,
        )
        
        sample.alignment_status = ProcessingStatus.SUCCESS
        
        # Compute quality metrics on aligned face
        quality = compute_quality_metrics(aligned, face_bbox=primary_face.bbox)
        sample.blur_score = quality["blur_score"]
        sample.brightness = quality["brightness"]
        sample.face_size = quality["face_size"]
        
        # Check quality thresholds
        if sample.blur_score is not None and sample.blur_score < self.config.min_blur_score:
            sample.processing_status = ProcessingStatus.LOW_QUALITY
            # Still save but mark as low quality
        elif sample.face_size is not None and sample.face_size < self.config.min_face_size_pixels:
            sample.processing_status = ProcessingStatus.LOW_QUALITY
        else:
            sample.processing_status = ProcessingStatus.SUCCESS
        
        # Save derived artifact
        output_dir.mkdir(parents=True, exist_ok=True)
        derived_filename = f"{sample_id}.{self.config.output_format}"
        derived_path = output_dir / derived_filename
        
        # Save as JPEG with quality
        if self.config.output_format.lower() in ("jpg", "jpeg"):
            aligned.save(derived_path, "JPEG", quality=self.config.output_quality)
        else:
            aligned.save(derived_path)
        
        sample.derived_path = str(derived_path)
        
        # Compute derived artifact hash
        try:
            sample.derived_sha256 = self.compute_sha256(derived_path)
        except Exception:
            pass
        
        return sample


class VideoPreprocessor:
    """
    Preprocesses a video file frame by frame.
    """
    
    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self.detector = FaceDetector(FaceDetectorConfig(
            min_face_size=config.min_face_size,
            min_confidence=config.min_confidence,
            keep_all=True,
            select_largest=config.select_largest,
            device="cpu",
        ))
    
    def compute_sha256(self, file_path: Path) -> str:
        """Compute SHA256 hash of a file."""
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    
    def process(
        self,
        source_path: Path,
        output_dir: Path,
        media_id: str,
        dataset: str,
        label: int,
        split: str,
        identity_id: Optional[str] = None,
        source_id: Optional[str] = None,
        manipulation_type: Optional[str] = None,
        generator: Optional[str] = None,
    ) -> PreprocessingResult:
        """
        Process a video file.
        
        Args:
            source_path: Path to source video
            output_dir: Directory to save derived face crops
            media_id: Media identifier
            dataset: Dataset name
            label: 0=authentic, 1=manipulated
            split: train/val/test
            identity_id: Identity identifier (optional)
            source_id: Source identifier (optional)
            manipulation_type: Manipulation type (optional)
            generator: Generator name (optional)
            
        Returns:
            PreprocessingResult with all samples
        """
        import cv2
        
        # Compute source hash
        source_sha256 = self.compute_sha256(source_path)
        
        # Open video
        cap = cv2.VideoCapture(str(source_path))
        if not cap.isOpened():
            return PreprocessingResult(
                samples=[],
                config=self.config,
                media_id=media_id,
                source_path=str(source_path),
                source_sha256=source_sha256,
                total_frames=0,
                processed_frames=0,
                failed_frames=1,
            )
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        samples = []
        processed = 0
        failed = 0
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % self.config.frame_interval != 0:
                frame_idx += 1
                continue
            
            if self.config.max_frames_per_video and processed >= self.config.max_frames_per_video:
                break
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)
            
            # Generate sample_id
            sample_id = f"{media_id}_frame_{frame_idx:06d}"
            
            # Process frame
            sample = PreprocessedSample(
                sample_id=sample_id,
                media_id=media_id,
                dataset=dataset,
                label=label,
                split=split,
                identity_id=identity_id,
                source_id=source_id,
                manipulation_type=manipulation_type,
                generator=generator,
                source_path=str(source_path),
                source_sha256=source_sha256,
                frame_index=frame_idx,
                timestamp=frame_idx / fps if fps > 0 else None,
                preprocessing_version=self.config.version,
                preprocessing_config_hash=self.config.config_hash(),
            )
            
            # Detect face
            primary_face = self.detector.detect_primary(image)
            
            if primary_face is None:
                sample.processing_status = ProcessingStatus.NO_FACE
                failed += 1
            else:
                sample.face_detected = True
                sample.face_bbox = primary_face.bbox
                sample.detector_confidence = primary_face.confidence
                
                # Expand bbox
                expanded = expand_bbox(
                    primary_face.bbox,
                    primary_face.image_size,
                    margin=self.config.bbox_margin,
                )
                sample.expanded_bbox = expanded
                
                if primary_face.landmarks is None:
                    sample.landmark_status = ProcessingStatus.LANDMARK_FAILURE
                    sample.processing_status = ProcessingStatus.LANDMARK_FAILURE
                    failed += 1
                else:
                    landmarks_list = [(float(x), float(y)) for x, y in primary_face.landmarks]
                    sample.landmarks = landmarks_list
                    sample.landmark_status = ProcessingStatus.SUCCESS
                    
                    # Expand bbox
                    expanded = expand_bbox(
                        primary_face.bbox,
                        primary_face.image_size,
                        margin=self.config.bbox_margin,
                    )
                    sample.expanded_bbox = expanded
                    
                    # Crop expanded region from source image
                    x1, y1, x2, y2 = expanded
                    expanded_crop = image.crop((x1, y1, x2, y2))
                    
                    # Transform landmarks to expanded-crop coordinates
                    src_lm_crop = transform_landmarks_to_crop(
                        primary_face.landmarks.astype(np.float32),
                        expanded,
                    )
                    sample.landmarks_crop = [(float(x), float(y)) for x, y in src_lm_crop]
                    
                    # Estimate similarity transform from crop-relative landmarks
                    transform = estimate_similarity_transform(src_lm_crop, self.config.canonical_landmarks)
                    
                    if transform is None:
                        sample.alignment_status = ProcessingStatus.ALIGNMENT_FAILURE
                        sample.processing_status = ProcessingStatus.ALIGNMENT_FAILURE
                        failed += 1
                    else:
                        sample.alignment_transform = transform.tolist()
                        
                        # Apply alignment to expanded crop (not full image)
                        aligned = apply_similarity_transform(
                            expanded_crop,
                            transform,
                            output_size=self.config.target_size,
                        )
                        
                        sample.alignment_status = ProcessingStatus.SUCCESS
                        
                        # Quality metrics
                        quality = compute_quality_metrics(aligned, face_bbox=primary_face.bbox)
                        sample.blur_score = quality["blur_score"]
                        sample.brightness = quality["brightness"]
                        sample.face_size = quality["face_size"]
                        
                        if (sample.blur_score is not None and sample.blur_score < self.config.min_blur_score) or \
                           (sample.face_size is not None and sample.face_size < self.config.min_face_size_pixels):
                            sample.processing_status = ProcessingStatus.LOW_QUALITY
                        else:
                            sample.processing_status = ProcessingStatus.SUCCESS
                        
                        # Save
                        output_dir.mkdir(parents=True, exist_ok=True)
                        derived_filename = f"{sample_id}.{self.config.output_format}"
                        derived_path = output_dir / derived_filename
                        
                        if self.config.output_format.lower() in ("jpg", "jpeg"):
                            aligned.save(derived_path, "JPEG", quality=self.config.output_quality)
                        else:
                            aligned.save(derived_path)
                        
                        sample.derived_path = str(derived_path)
                        sample.derived_sha256 = self.compute_sha256(derived_path)
                        
                        processed += 1
            
            samples.append(sample)
            frame_idx += 1
        
        cap.release()
        
        return PreprocessingResult(
            samples=samples,
            config=self.config,
            media_id=media_id,
            source_path=str(source_path),
            source_sha256=source_sha256,
            total_frames=total_frames,
            processed_frames=processed,
            failed_frames=failed,
        )


def create_manifest(samples: List[PreprocessedSample], output_path: Path) -> None:
    """
    Create a training-ready manifest CSV from preprocessed samples.
    
    Only includes samples with SUCCESS status.
    LOW_QUALITY and failed samples are excluded from training manifest
    but can be written to an audit manifest via create_audit_manifest.
    """
    # Filter for SUCCESS only (training-ready)
    training_samples = [s for s in samples if s.processing_status == ProcessingStatus.SUCCESS]
    
    if not training_samples:
        # Create empty manifest with correct columns
        df = pd.DataFrame(columns=training_samples[0].to_dict().keys() if training_samples else [])
        df.to_csv(output_path, index=False)
        return
    
    df = pd.DataFrame([s.to_dict() for s in training_samples])
    df.to_csv(output_path, index=False)


def create_audit_manifest(samples: List[PreprocessedSample], output_path: Path) -> None:
    """
    Create an audit manifest containing all samples including LOW_QUALITY and failures.
    
    This provides traceability for samples excluded from the training manifest.
    Includes: sample_id, media_id, frame_index, processing_status, quality metrics,
    source_path, source_sha256, and all preprocessing metadata.
    """
    if not samples:
        df = pd.DataFrame()
        df.to_csv(output_path, index=False)
        return
    
    df = pd.DataFrame([s.to_dict() for s in samples])
    df.to_csv(output_path, index=False)


def load_preprocessing_config(config_path: Optional[Path] = None) -> PreprocessingConfig:
    """Load preprocessing configuration from YAML file or return defaults."""
    if config_path and config_path.exists():
        import yaml
        with open(config_path, "r") as f:
            data = yaml.safe_load(f)
        # Convert nested dicts
        if "canonical_landmarks" in data:
            data["canonical_landmarks"] = np.array(data["canonical_landmarks"], dtype=np.float32)
        if "target_size" in data:
            data["target_size"] = tuple(data["target_size"])
        return PreprocessingConfig(**data)
    return PreprocessingConfig()


def save_preprocessing_config(config: PreprocessingConfig, output_path: Path) -> None:
    """Save preprocessing configuration to YAML file."""
    import yaml
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False)