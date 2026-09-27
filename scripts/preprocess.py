#!/usr/bin/env python3
"""
Preprocessing script for EXP-S01.

Usage:
    python scripts/preprocess.py --input-dir data/raw --output-dir data/derived --manifest data/manifests/preprocessed.csv --config configs/preprocessing.yaml
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd

from deepfake_forensics.preprocessing import (
    ImagePreprocessor,
    VideoPreprocessor,
    PreprocessingConfig,
    load_preprocessing_config,
    save_preprocessing_config,
    create_manifest,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run preprocessing pipeline for EXP-S01")
    parser.add_argument("--input-dir", type=str, required=True, help="Input directory with raw media")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for derived artifacts")
    parser.add_argument("--manifest", type=str, required=True, help="Output manifest CSV path")
    parser.add_argument("--config", type=str, default=None, help="Preprocessing config YAML")
    parser.add_argument("--dataset", type=str, default="Unknown", help="Dataset name")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"], help="Split")
    parser.add_argument("--label", type=int, default=0, choices=[0, 1], help="Label (0=authentic, 1=manipulated)")
    parser.add_argument("--identity-col", type=str, default=None, help="Column in manifest for identity_id")
    parser.add_argument("--source-col", type=str, default=None, help="Column in manifest for source_id")
    parser.add_argument("--recursive", action="store_true", help="Search input directory recursively")
    parser.add_argument("--extensions", type=str, default=".jpg,.jpeg,.png,.mp4,.avi,.mov", help="File extensions to process")
    parser.add_argument("--dry-run", action="store_true", help="Don't save files, just show what would be processed")
    return parser.parse_args()


def find_media_files(input_dir: Path, extensions: list, recursive: bool) -> list:
    """Find all media files in input directory."""
    files = []
    if recursive:
        for ext in extensions:
            files.extend(input_dir.rglob(f"*{ext}"))
    else:
        for ext in extensions:
            files.extend(input_dir.glob(f"*{ext}"))
    return sorted(files)


def infer_metadata(file_path: Path, args) -> dict:
    """Infer metadata from file path and arguments."""
    # Default metadata
    metadata = {
        "dataset": args.dataset,
        "label": args.label,
        "split": args.split,
        "identity_id": None,
        "source_id": None,
        "manipulation_type": "None" if args.label == 0 else "Deepfakes",
        "generator": "None" if args.label == 0 else "faceswap",
    }
    
    # Try to infer from directory structure
    # e.g., data/raw/FF++/train/id_001/video.mp4
    parts = file_path.relative_to(Path(args.input_dir)).parts
    if len(parts) >= 2:
        metadata["dataset"] = parts[0]
    if len(parts) >= 3:
        metadata["split"] = parts[1]
    if len(parts) >= 4:
        # Try to extract identity from directory name
        metadata["identity_id"] = parts[2]
    
    return metadata


def main():
    args = parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    manifest_path = Path(args.manifest)
    
    if not input_dir.exists():
        print(f"Error: Input directory {input_dir} does not exist")
        sys.exit(1)
    
    # Load config
    config_path = Path(args.config) if args.config else None
    config = load_preprocessing_config(config_path)
    
    print(f"Preprocessing config:")
    print(f"  Face detector: {config.face_detector}")
    print(f"  Bbox margin: {config.bbox_margin} (20% expansion)")
    print(f"  Target size: {config.target_size}")
    print(f"  Min confidence: {config.min_confidence}")
    print(f"  Output format: {config.output_format}")
    
    # Save config for reproducibility
    output_dir.mkdir(parents=True, exist_ok=True)
    save_preprocessing_config(config, output_dir / "preprocessing_config.yaml")
    
    # Find media files
    extensions = [e.strip() for e in args.extensions.split(",")]
    media_files = find_media_files(input_dir, extensions, args.recursive)
    
    print(f"\nFound {len(media_files)} media files")
    
    # Initialize preprocessors
    image_preprocessor = ImagePreprocessor(config)
    video_preprocessor = VideoPreprocessor(config)
    
    # Video extensions
    video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    
    all_samples = []
    
    for media_file in media_files:
        # Generate media_id from file path
        media_id = media_file.stem
        
        # Infer metadata
        metadata = infer_metadata(media_file, args)
        
        is_video = media_file.suffix.lower() in video_exts
        
        if args.dry_run:
            print(f"  Would process: {media_file} (video={is_video})")
            continue
        
        print(f"  Processing: {media_file}")
        
        try:
            if is_video:
                result = video_preprocessor.process(
                    source_path=media_file,
                    output_dir=output_dir / "faces",
                    media_id=media_id,
                    **metadata,
                )
                all_samples.extend(result.samples)
                print(f"    Frames: {result.total_frames}, Processed: {result.processed_frames}, Failed: {result.failed_frames}")
            else:
                sample = image_preprocessor.process(
                    source_path=media_file,
                    output_dir=output_dir / "faces",
                    sample_id=media_id,
                    media_id=media_id,
                    **metadata,
                )
                all_samples.append(sample)
                print(f"    Status: {sample.processing_status}")
        except Exception as e:
            print(f"    ERROR: {e}")
    
    # Create manifest
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    create_manifest(all_samples, manifest_path)
    
    # Print summary
    successful = [s for s in all_samples if s.processing_status == "SUCCESS"]
    low_quality = [s for s in all_samples if s.processing_status == "LOW_QUALITY"]
    failed = [s for s in all_samples if s.processing_status not in ("SUCCESS", "LOW_QUALITY")]
    
    print(f"\n{'='*50}")
    print(f"Preprocessing complete!")
    print(f"  Total samples: {len(all_samples)}")
    print(f"  Successful: {len(successful)}")
    print(f"  Low quality: {len(low_quality)}")
    print(f"  Failed: {len(failed)}")
    print(f"  Manifest saved to: {manifest_path}")
    
    if failed:
        print(f"\nFailure breakdown:")
        from collections import Counter
        status_counts = Counter(s.processing_status for s in failed)
        for status, count in status_counts.items():
            print(f"  {status}: {count}")


if __name__ == "__main__":
    main()