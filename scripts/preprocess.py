#!/usr/bin/env python3
"""
Preprocessing script for EXP-S01.

Usage:
    python scripts/preprocess.py --input-dir data/raw --output-dir data/derived --manifest data/manifests/preprocessed.csv --config configs/preprocessing.yaml --metadata data/metadata.csv

The --metadata CSV must contain per-file metadata with columns:
- file_path (relative to input-dir or absolute)
- label (0=authentic, 1=manipulated) [REQUIRED]
- dataset [REQUIRED]
- split (train/val/test) [REQUIRED]
- identity_id (optional)
- source_id (optional)
- manipulation_type (optional)
- generator (optional)

This script does NOT invent metadata. All metadata must be provided explicitly.
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
    create_audit_manifest,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run preprocessing pipeline for EXP-S01",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
    # Process with explicit metadata CSV (REQUIRED)
    python scripts/preprocess.py \
        --input-dir data/raw \
        --output-dir data/derived \
        --manifest data/manifests/train.csv \
        --metadata data/metadata.csv \
        --config configs/preprocessing.yaml

    # Dry run to see what would be processed
    python scripts/preprocess.py \
        --input-dir data/raw \
        --output-dir data/derived \
        --manifest data/manifests/train.csv \
        --metadata data/metadata.csv \
        --dry-run

METADATA CSV FORMAT (required columns):
    file_path,label,dataset,split,identity_id,source_id,manipulation_type,generator
    video1.mp4,1,FF++,train,id_001,src_001,Deepfakes,faceswap
    video2.mp4,0,FF++,train,id_002,src_002,None,None
"""
    )
    parser.add_argument("--input-dir", type=str, required=True, help="Input directory with raw media")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for derived artifacts")
    parser.add_argument("--manifest", type=str, required=True, help="Output training-ready manifest CSV path")
    parser.add_argument("--audit-manifest", type=str, default=None, help="Output audit manifest CSV path (includes LOW_QUALITY/failed)")
    parser.add_argument("--config", type=str, default=None, help="Preprocessing config YAML")
    parser.add_argument("--metadata", type=str, required=True, help="Metadata CSV with per-file labels/splits (REQUIRED)")
    parser.add_argument("--recursive", action="store_true", help="Search input directory recursively")
    parser.add_argument("--extensions", type=str, default=".jpg,.jpeg,.png,.mp4,.avi,.mov,.mkv,.webm", help="File extensions to process")
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


def load_metadata(metadata_path: Path, input_dir: Path) -> dict:
    """
    Load metadata from CSV.
    
    Returns dict mapping file_path (relative to input_dir) -> metadata dict.
    Required columns: file_path, label, dataset, split
    Optional columns: identity_id, source_id, manipulation_type, generator
    """
    df = pd.read_csv(metadata_path)
    
    required_cols = {"file_path", "label", "dataset", "split"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Metadata CSV missing required columns: {missing}")
    
    metadata_map = {}
    for _, row in df.iterrows():
        # Normalize file path
        file_path = Path(row["file_path"])
        if not file_path.is_absolute():
            file_path = (input_dir / file_path).resolve()
        else:
            file_path = file_path.resolve()
        
        metadata_map[str(file_path)] = {
            "label": int(row["label"]),
            "dataset": str(row["dataset"]),
            "split": str(row["split"]),
            "identity_id": str(row["identity_id"]) if pd.notna(row.get("identity_id")) else None,
            "source_id": str(row["source_id"]) if pd.notna(row.get("source_id")) else None,
            "manipulation_type": str(row["manipulation_type"]) if pd.notna(row.get("manipulation_type")) else None,
            "generator": str(row["generator"]) if pd.notna(row.get("generator")) else None,
        }
    
    return metadata_map


def main():
    args = parse_args()
    
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    manifest_path = Path(args.manifest).resolve()
    audit_manifest_path = Path(args.audit_manifest).resolve() if args.audit_manifest else None
    metadata_path = Path(args.metadata).resolve()
    
    if not input_dir.exists():
        print(f"Error: Input directory {input_dir} does not exist")
        sys.exit(1)
    
    if not metadata_path.exists():
        print(f"Error: Metadata file {metadata_path} does not exist")
        sys.exit(1)
    
    # Load metadata
    try:
        metadata_map = load_metadata(metadata_path, input_dir)
    except Exception as e:
        print(f"Error loading metadata: {e}")
        sys.exit(1)
    
    print(f"Loaded metadata for {len(metadata_map)} files")
    
    # Load config
    config_path = Path(args.config) if args.config else None
    config = load_preprocessing_config(config_path)
    
    print(f"Preprocessing config:")
    print(f"  Face detector: {config.face_detector}")
    print(f"  Bbox margin: {config.bbox_margin} (20% expansion per side)")
    print(f"  Target size: {config.target_size}")
    print(f"  Min confidence: {config.min_confidence}")
    print(f"  Output format: {config.output_format}")
    
    # Save config for reproducibility
    output_dir.mkdir(parents=True, exist_ok=True)
    save_preprocessing_config(config, output_dir / "preprocessing_config.yaml")
    
    # Find media files
    extensions = [e.strip() for e in args.extensions.split(",")]
    media_files = find_media_files(input_dir, extensions, args.recursive)
    
    print(f"\nFound {len(media_files)} media files in input directory")
    
    # Match media files with metadata
    matched_files = []
    unmatched_files = []
    
    for media_file in media_files:
        media_file_resolved = media_file.resolve()
        if str(media_file_resolved) in metadata_map:
            matched_files.append((media_file_resolved, metadata_map[str(media_file_resolved)]))
        else:
            unmatched_files.append(str(media_file_resolved))
    
    if unmatched_files:
        print(f"WARNING: {len(unmatched_files)} files have no metadata entry (will be skipped):")
        for f in unmatched_files[:10]:
            print(f"  {f}")
        if len(unmatched_files) > 10:
            print(f"  ... and {len(unmatched_files) - 10} more")
    
    if not matched_files:
        print("ERROR: No files matched with metadata. Check your metadata CSV.")
        sys.exit(1)
    
    print(f"Processing {len(matched_files)} files with metadata...")
    
    # Initialize preprocessors
    image_preprocessor = ImagePreprocessor(config)
    video_preprocessor = VideoPreprocessor(config)
    
    # Video extensions
    video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    
    all_samples = []
    
    for media_file, metadata in matched_files:
        media_id = media_file.stem
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
            import traceback
            traceback.print_exc()
    
    # Create training-ready manifest (SUCCESS only)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    create_manifest(all_samples, manifest_path)
    
    # Create audit manifest (all samples including LOW_QUALITY/failed)
    if audit_manifest_path:
        create_audit_manifest(all_samples, audit_manifest_path)
        print(f"  Audit manifest saved to: {audit_manifest_path}")
    
    # Print summary
    successful = [s for s in all_samples if s.processing_status == "SUCCESS"]
    low_quality = [s for s in all_samples if s.processing_status == "LOW_QUALITY"]
    failed = [s for s in all_samples if s.processing_status not in ("SUCCESS", "LOW_QUALITY")]
    
    print(f"\n{'='*50}")
    print(f"Preprocessing complete!")
    print(f"  Total samples: {len(all_samples)}")
    print(f"  Successful (training-ready): {len(successful)}")
    print(f"  Low quality (excluded from training): {len(low_quality)}")
    print(f"  Failed: {len(failed)}")
    print(f"  Training manifest saved to: {manifest_path}")
    
    if failed:
        print(f"\nFailure breakdown:")
        from collections import Counter
        status_counts = Counter(s.processing_status for s in failed)
        for status, count in status_counts.items():
            print(f"  {status}: {count}")


if __name__ == "__main__":
    main()