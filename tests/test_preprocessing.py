"""
Tests for face detector and preprocessing utilities.
"""

import pytest
import numpy as np
from PIL import Image

from deepfake_forensics.preprocessing.face_detector import (
    CANONICAL_LANDMARKS_224,
    FaceDetection,
    FaceDetector,
    FaceDetectorConfig,
    expand_bbox,
    estimate_similarity_transform,
    apply_similarity_transform,
    compute_quality_metrics,
)


class TestExpandBBox:
    """Test 20% bounding box expansion.
    
    Convention: margin=0.2 means 20% expansion on EACH SIDE (left/right, top/bottom).
    So a 100x100 box becomes 140x140 (40% total increase).
    """
    
    def test_normal_expansion(self):
        """Test normal bbox expansion."""
        bbox = (100, 100, 200, 200)  # 100x100 box at center
        image_size = (500, 500)
        
        expanded = expand_bbox(bbox, image_size, margin=0.2)
        
        # 20% expansion on each side = 40% total increase
        # Original: 100x100, New: 140x140
        # Center: (150, 150), New bounds: (80, 80) to (220, 220)
        assert expanded == (80, 80, 220, 220)
    
    def test_expansion_clipped_at_boundary(self):
        """Test bbox expansion clipped at image boundary."""
        # Box at (10,10) to (50,50) = 40x40 box
        # 20% expansion per side = 8 pixels each side
        # New: (2, 2) to (58, 58), clipped at 0 -> (2, 2) to (58, 58)
        bbox = (10, 10, 50, 50)  # Near top-left corner
        image_size = (100, 100)
        
        expanded = expand_bbox(bbox, image_size, margin=0.2)
        
        # Should be clipped at 0 (but box is 10px from edge, 8px expansion gives 2)
        assert expanded[0] == 2
        assert expanded[1] == 2
        assert expanded[2] == 58
        assert expanded[3] == 58
    
    def test_small_face_expansion(self):
        """Test expansion of small face."""
        bbox = (100, 100, 110, 110)  # 10x10 face
        image_size = (224, 224)
        
        expanded = expand_bbox(bbox, image_size, margin=0.2)
        
        # Should expand to 14x14 (10 * 1.4)
        assert expanded[2] - expanded[0] == 14
        assert expanded[3] - expanded[1] == 14
    
    def test_non_square_face(self):
        """Test expansion of non-square face."""
        bbox = (100, 100, 150, 200)  # 50x100 face
        image_size = (300, 300)
        
        expanded = expand_bbox(bbox, image_size, margin=0.2)
        
        # 20% expansion per side = 40% total
        # width: 50 * 1.4 = 70, height: 100 * 1.4 = 140
        assert expanded[2] - expanded[0] == 70
        assert expanded[3] - expanded[1] == 140
        
        # Center should remain same
        orig_cx = (100 + 150) / 2
        orig_cy = (100 + 200) / 2
        new_cx = (expanded[0] + expanded[2]) / 2
        new_cy = (expanded[1] + expanded[3]) / 2
        assert abs(orig_cx - new_cx) < 1
        assert abs(orig_cy - new_cy) < 1
    
    def test_invalid_bbox(self):
        """Test handling of invalid bbox (x2 <= x1 or y2 <= y1)."""
        bbox = (100, 100, 100, 100)  # Zero area
        image_size = (224, 224)
        
        expanded = expand_bbox(bbox, image_size, margin=0.2)
        
        # Should produce valid bbox with at least 1 pixel
        assert expanded[2] > expanded[0]
        assert expanded[3] > expanded[1]
    
    def test_large_margin(self):
        """Test with large margin that exceeds image."""
        # Box at (200,200) to (250,250) = 50x50 box
        # 200% expansion per side = 200% * 2 = 400% total = 5x size = 250x250
        # Center at (225, 225). New: (225-125, 225-125) to (225+125, 225+125) = (100, 100) to (350, 350)
        # Clipped to (100, 100, 300, 300)
        bbox = (200, 200, 250, 250)
        image_size = (300, 300)
        
        expanded = expand_bbox(bbox, image_size, margin=2.0)  # 200% expansion per side
        
        # Should be clipped to image boundaries
        assert expanded[0] == 100
        assert expanded[1] == 100
        assert expanded[2] == 300
        assert expanded[3] == 300


class TestCanonicalLandmarks:
    """Test canonical landmark template."""
    
    def test_canonical_shape(self):
        """Test canonical landmarks have correct shape."""
        assert CANONICAL_LANDMARKS_224.shape == (5, 2)
        assert CANONICAL_LANDMARKS_224.dtype == np.float32
    
    def test_canonical_within_bounds(self):
        """Test canonical landmarks are within 224x224."""
        assert np.all(CANONICAL_LANDMARKS_224[:, 0] >= 0)
        assert np.all(CANONICAL_LANDMARKS_224[:, 0] < 224)
        assert np.all(CANONICAL_LANDMARKS_224[:, 1] >= 0)
        assert np.all(CANONICAL_LANDMARKS_224[:, 1] < 224)
    
    def test_canonical_order(self):
        """Test landmark order: left_eye, right_eye, nose, left_mouth, right_mouth."""
        # Left eye should be left of right eye
        assert CANONICAL_LANDMARKS_224[0, 0] < CANONICAL_LANDMARKS_224[1, 0]
        # Eyes should be above nose
        assert CANONICAL_LANDMARKS_224[0, 1] < CANONICAL_LANDMARKS_224[2, 1]
        assert CANONICAL_LANDMARKS_224[1, 1] < CANONICAL_LANDMARKS_224[2, 1]
        # Nose should be above mouth
        assert CANONICAL_LANDMARKS_224[2, 1] < CANONICAL_LANDMARKS_224[3, 1]
        # Left mouth should be left of right mouth
        assert CANONICAL_LANDMARKS_224[3, 0] < CANONICAL_LANDMARKS_224[4, 0]


class TestSimilarityTransform:
    """Test similarity transform estimation and application."""
    
    def test_identity_transform(self):
        """Test identity transform when src == dst."""
        transform = estimate_similarity_transform(CANONICAL_LANDMARKS_224, CANONICAL_LANDMARKS_224)
        
        assert transform is not None
        assert transform.shape == (2, 3)
        # Should be approximately identity (scale=1, rotation=0, translation=0)
        assert abs(transform[0, 0] - 1.0) < 1e-5
        assert abs(transform[0, 1] - 0.0) < 1e-5
        assert abs(transform[1, 0] - 0.0) < 1e-5
        assert abs(transform[1, 1] - 1.0) < 1e-5
        assert abs(transform[0, 2]) < 1e-5
        assert abs(transform[1, 2]) < 1e-5
    
    def test_translation_only(self):
        """Test pure translation."""
        shifted = CANONICAL_LANDMARKS_224 + np.array([10.0, 20.0], dtype=np.float32)
        transform = estimate_similarity_transform(shifted, CANONICAL_LANDMARKS_224)
        
        assert transform is not None
        # Should translate by (-10, -20)
        assert abs(transform[0, 2] - (-10.0)) < 1e-5
        assert abs(transform[1, 2] - (-20.0)) < 1e-5
    
    def test_scaling(self):
        """Test uniform scaling."""
        scaled = CANONICAL_LANDMARKS_224 * 2.0
        transform = estimate_similarity_transform(scaled, CANONICAL_LANDMARKS_224)
        
        assert transform is not None
        # Should scale by 0.5
        assert abs(transform[0, 0] - 0.5) < 1e-5
        assert abs(transform[1, 1] - 0.5) < 1e-5
    
    def test_rotation(self):
        """Test rotation."""
        # Rotate 90 degrees around center
        center = np.mean(CANONICAL_LANDMARKS_224, axis=0)
        centered = CANONICAL_LANDMARKS_224 - center
        rot = np.array([[0, -1], [1, 0]], dtype=np.float32)  # 90 deg
        rotated = (centered @ rot.T) + center
        
        transform = estimate_similarity_transform(rotated, CANONICAL_LANDMARKS_224)
        
        assert transform is not None
        # Should have rotation component
        # R = [[0, 1], [-1, 0]] for -90 degrees (inverse of 90)
        assert abs(transform[0, 0] - 0.0) < 1e-5
        assert abs(transform[0, 1] - 1.0) < 1e-5
        assert abs(transform[1, 0] - (-1.0)) < 1e-5
        assert abs(transform[1, 1] - 0.0) < 1e-5
    
    def test_insufficient_landmarks(self):
        """Test failure with insufficient landmarks."""
        transform = estimate_similarity_transform(CANONICAL_LANDMARKS_224[:1])
        assert transform is None
        
        transform = estimate_similarity_transform(np.array([]).reshape(0, 2))
        assert transform is None
    
    def test_mismatched_shapes(self):
        """Test failure with mismatched shapes."""
        transform = estimate_similarity_transform(CANONICAL_LANDMARKS_224[:3], CANONICAL_LANDMARKS_224)
        assert transform is None


class TestQualityMetrics:
    """Test quality metric computation."""
    
    def test_quality_metrics_basic(self):
        """Test basic quality metrics computation."""
        # Create a test image
        img = Image.new("RGB", (224, 224), color=(128, 128, 128))
        
        metrics = compute_quality_metrics(img)
        
        assert "blur_score" in metrics
        assert "brightness" in metrics
        assert "face_size" in metrics
        assert "resolution" in metrics
        assert metrics["resolution"] == "224x224"
        assert metrics["brightness"] == 128.0
        assert metrics["face_size"] is None  # No bbox provided
    
    def test_quality_metrics_with_bbox(self):
        """Test quality metrics with face bbox."""
        img = Image.new("RGB", (224, 224), color=(128, 128, 128))
        bbox = (50, 50, 150, 150)  # 100x100 face
        
        metrics = compute_quality_metrics(img, face_bbox=bbox)
        
        assert metrics["face_size"] == 10000.0  # 100 * 100
    
    def test_blur_score_variation(self):
        """Test blur score varies with sharpness."""
        # Sharp image (high frequency)
        sharp = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        # Blur image (low frequency)
        blur = Image.new("RGB", (224, 224), color=(128, 128, 128))
        
        sharp_metrics = compute_quality_metrics(sharp)
        blur_metrics = compute_quality_metrics(blur)
        
        # Sharp image should have higher Laplacian variance
        assert sharp_metrics["blur_score"] > blur_metrics["blur_score"]


class TestApplySimilarityTransform:
    """Test applying similarity transform to images."""
    
    def test_apply_identity(self):
        """Test applying identity transform."""
        img = Image.new("RGB", (224, 224), color=(255, 0, 0))
        transform = np.eye(2, 3, dtype=np.float32)
        
        result = apply_similarity_transform(img, transform, (224, 224))
        
        assert result.size == (224, 224)
        # Should be same color
        assert np.array_equal(np.array(result), np.array(img))
    
    def test_apply_translation(self):
        """Test applying translation."""
        img = Image.new("RGB", (224, 224), color=(0, 255, 0))
        transform = np.array([[1, 0, 10], [0, 1, 20]], dtype=np.float32)
        
        result = apply_similarity_transform(img, transform, (224, 224))
        
        assert result.size == (224, 224)
    
    def test_output_size(self):
        """Test output size is respected."""
        img = Image.new("RGB", (224, 224), color=(0, 0, 255))
        transform = np.eye(2, 3, dtype=np.float32)
        
        result = apply_similarity_transform(img, transform, (112, 112))
        
        assert result.size == (112, 112)


class TestFaceDetectorConfig:
    """Test FaceDetectorConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = FaceDetectorConfig()
        
        assert config.min_face_size == 20
        assert config.thresholds == (0.6, 0.7, 0.7)
        assert config.factor == 0.709
        assert config.keep_all is True
        assert config.device == "cpu"
        assert config.min_confidence == 0.9
        assert config.select_largest is True


class TestProcessingStatus:
    """Test processing status constants."""
    
    def test_status_values(self):
        """Test processing status values."""
        from deepfake_forensics.preprocessing.pipeline import ProcessingStatus
        
        assert ProcessingStatus.SUCCESS == "SUCCESS"
        assert ProcessingStatus.NO_FACE == "NO_FACE"
        assert ProcessingStatus.MULTIPLE_FACE_AMBIGUOUS == "MULTIPLE_FACE_AMBIGUOUS"
        assert ProcessingStatus.LANDMARK_FAILURE == "LANDMARK_FAILURE"
        assert ProcessingStatus.ALIGNMENT_FAILURE == "ALIGNMENT_FAILURE"
        assert ProcessingStatus.DECODE_FAILURE == "DECODE_FAILURE"
        assert ProcessingStatus.CORRUPT_MEDIA == "CORRUPT_MEDIA"
        assert ProcessingStatus.LOW_QUALITY == "LOW_QUALITY"
        assert ProcessingStatus.VIDEO_DECODE_FAILURE == "VIDEO_DECODE_FAILURE"
        assert ProcessingStatus.NO_FRAMES == "NO_FRAMES"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])