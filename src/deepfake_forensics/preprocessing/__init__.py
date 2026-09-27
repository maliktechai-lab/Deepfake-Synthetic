"""
Preprocessing module for deepfake forensics.

This module handles face detection, alignment, and cropping to produce
versioned derived artifacts from raw media.

================================================================================
IMPORTANT NOTE FOR EXP-S01
================================================================================

The preprocessing pipeline implements:
- Face detection using MTCNN (facenet-pytorch)
- 20% bounding box expansion around detected face center
- 5-point facial landmark detection
- Similarity alignment to canonical 224x224 template
- Quality metadata generation (blur, brightness, face size)
- Real SHA-256 hashing of source and derived artifacts
- Versioned preprocessing configuration

EXP-S01 requires pre-generated aligned face crops. Use scripts/preprocess.py
to generate them from raw media.

Directory structure:
  data/
  ├── raw/           # Immutable raw media (never modified)
  ├── derived/       # Generated face crops
  │   ├── faces/     # 224x224 aligned face images
  │   └── ...        # Other derived artifacts
  └── manifests/     # Training manifests with preprocessing metadata
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

# Face detector exports
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

# Pipeline exports
from deepfake_forensics.preprocessing.pipeline import (
    ImagePreprocessor,
    VideoPreprocessor,
    PreprocessingConfig,
    PreprocessedSample,
    PreprocessingResult,
    ProcessingStatus,
    create_manifest,
    create_audit_manifest,
    load_preprocessing_config,
    save_preprocessing_config,
)

__all__ = [
    # Face detector
    "CANONICAL_LANDMARKS_224",
    "FaceDetection",
    "FaceDetector",
    "FaceDetectorConfig",
    "apply_similarity_transform",
    "compute_quality_metrics",
    "estimate_similarity_transform",
    "expand_bbox",
    "transform_landmarks_to_crop",
    # Pipeline
    "ImagePreprocessor",
    "VideoPreprocessor",
    "PreprocessingConfig",
    "PreprocessedSample",
    "PreprocessingResult",
    "ProcessingStatus",
    "create_manifest",
    "create_audit_manifest",
    "load_preprocessing_config",
    "save_preprocessing_config",
]
