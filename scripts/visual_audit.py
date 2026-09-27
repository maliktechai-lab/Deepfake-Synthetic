#!/usr/bin/env python3
"""
Visual audit tool for preprocessing results.

Usage:
    python scripts/visual_audit.py --manifest data/manifests/preprocessed.csv --num-samples 10 --output audit.png
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from deepfake_forensics.preprocessing.face_detector import (
    FaceDetector,
    FaceDetectorConfig,
    expand_bbox,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Visual audit of preprocessing results")
    parser.add_argument("--manifest", type=str, required=True, help="Preprocessed manifest CSV")
    parser.add_argument("--source-root", type=str, default="data/raw", help="Root directory for raw media")
    parser.add_argument("--output", type=str, default="audit.png", help="Output image path")
    parser.add_argument("--num-samples", type=int, default=8, help="Number of samples to visualize")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed for sampling")
    return parser.parse_args()


def draw_bbox(draw: ImageDraw, bbox: tuple, color: str, width: int = 3, label: str = None):
    """Draw bounding box on image."""
    x1, y1, x2, y2 = bbox
    draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
    if label:
        # Draw label background
        text_bbox = draw.textbbox((x1, y1 - 20), label)
        draw.rectangle(text_bbox, fill=color)
        draw.text((x1, y1 - 20), label, fill="white")


def draw_landmarks(draw: ImageDraw, landmarks: list, color: str = "red", radius: int = 3):
    """Draw facial landmarks on image."""
    for i, (x, y) in enumerate(landmarks):
        draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=color)
        draw.text((x + radius + 2, y - radius), str(i), fill=color)


def create_audit_grid(samples_df: pd.DataFrame, source_root: Path, num_samples: int, seed: int) -> Image.Image:
    """Create visual audit grid."""
    np.random.seed(seed)
    
    # Filter for successful samples
    success_df = samples_df[samples_df["processing_status"] == "SUCCESS"].copy()
    
    if len(success_df) == 0:
        raise ValueError("No successful samples in manifest")
    
    # Sample
    if len(success_df) > num_samples:
        success_df = success_df.sample(n=num_samples, random_state=seed)
    
    # Initialize detector for visualization
    detector = FaceDetector(FaceDetectorConfig(min_confidence=0.5, keep_all=True))
    
    # Grid layout
    cols = 4
    rows = int(np.ceil(len(success_df) / cols))
    
    cell_width = 600
    cell_height = 450
    padding = 10
    
    grid_width = cols * cell_width + (cols + 1) * padding
    grid_height = rows * cell_height + (rows + 1) * padding
    
    grid = Image.new("RGB", (grid_width, grid_height), "white")
    draw = ImageDraw.Draw(grid)
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except:
        font = ImageFont.load_default()
        font_small = ImageFont.load_default()
    
    for idx, (_, row) in enumerate(success_df.iterrows()):
        r = idx // cols
        c = idx % cols
        
        x_offset = padding + c * (cell_width + padding)
        y_offset = padding + r * (cell_height + padding)
        
        # Load source image
        source_path = source_root / row["source_path"] if not Path(row["source_path"]).is_absolute() else Path(row["source_path"])
        
        try:
            source_img = Image.open(source_path)
            if source_img.mode != "RGB":
                source_img = source_img.convert("RGB")
        except Exception as e:
            # Draw error placeholder
            error_img = Image.new("RGB", (cell_width, cell_height), "lightgray")
            error_draw = ImageDraw.Draw(error_img)
            error_draw.text((20, 20), f"Error loading:\n{row['sample_id']}\n{e}", fill="red", font=font)
            grid.paste(error_img, (x_offset, y_offset))
            continue
        
        # Resize for display (maintain aspect ratio)
        display_img = source_img.copy()
        display_img.thumbnail((cell_width, cell_height - 80), Image.Resampling.LANCZOS)
        
        # Create cell image with space for labels
        cell_img = Image.new("RGB", (cell_width, cell_height), "white")
        cell_draw = ImageDraw.Draw(cell_img)
        
        # Paste source image
        img_x = (cell_width - display_img.width) // 2
        cell_img.paste(display_img, (img_x, 10))
        
        # Draw original face bbox
        if row.get("face_bbox_x") is not None:
            bbox = (
                row["face_bbox_x"],
                row["face_bbox_y"],
                row["face_bbox_x"] + row["face_bbox_w"],
                row["face_bbox_y"] + row["face_bbox_h"],
            )
            # Scale bbox to display coordinates
            scale_x = display_img.width / source_img.width
            scale_y = display_img.height / source_img.height
            display_bbox = (
                int(bbox[0] * scale_x) + img_x,
                int(bbox[1] * scale_y) + 10,
                int(bbox[2] * scale_x) + img_x,
                int(bbox[3] * scale_y) + 10,
            )
            draw_bbox(cell_draw, display_bbox, "blue", width=2, label="Detected")
        
        # Draw expanded bbox
        if row.get("expanded_bbox_x") is not None:
            bbox = (
                row["expanded_bbox_x"],
                row["expanded_bbox_y"],
                row["expanded_bbox_x"] + row["expanded_bbox_w"],
                row["expanded_bbox_y"] + row["expanded_bbox_h"],
            )
            scale_x = display_img.width / source_img.width
            scale_y = display_img.height / source_img.height
            display_bbox = (
                int(bbox[0] * scale_x) + img_x,
                int(bbox[1] * scale_y) + 10,
                int(bbox[2] * scale_x) + img_x,
                int(bbox[3] * scale_y) + 10,
            )
            draw_bbox(cell_draw, display_bbox, "green", width=2, label="Expanded (20%)")
        
        # Draw landmarks
        if row.get("landmarks") is not None:
            import ast
            try:
                landmarks = ast.literal_eval(row["landmarks"]) if isinstance(row["landmarks"], str) else row["landmarks"]
                scale_x = display_img.width / source_img.width
                scale_y = display_img.height / source_img.height
                display_landmarks = [
                    (int(x * scale_x) + img_x, int(y * scale_y) + 10)
                    for x, y in landmarks
                ]
                draw_landmarks(cell_draw, display_landmarks, "red", radius=2)
            except:
                pass
        
        # Add title
        title = f"{row['sample_id']} | {row['dataset']} | {row['split']} | label={row['label']}"
        cell_draw.text((10, display_img.height + 20), title, fill="black", font=font_small)
        
        # Add metadata
        meta_lines = [
            f"Conf: {row.get('detector_confidence', 'N/A'):.3f}" if row.get('detector_confidence') else "Conf: N/A",
            f"Blur: {row.get('blur_score', 'N/A'):.1f}" if row.get('blur_score') else "Blur: N/A",
            f"Bright: {row.get('brightness', 'N/A'):.1f}" if row.get('brightness') else "Bright: N/A",
            f"FaceSize: {row.get('face_size', 'N/A'):.0f}" if row.get('face_size') else "FaceSize: N/A",
            f"Status: {row['processing_status']}",
        ]
        for i, line in enumerate(meta_lines):
            cell_draw.text((10, display_img.height + 40 + i * 18), line, fill="black", font=font_small)
        
        # Paste aligned face crop on the right side
        derived_path = Path(row["file_path"])
        if derived_path.exists():
            try:
                aligned_img = Image.open(derived_path)
                if aligned_img.mode != "RGB":
                    aligned_img = aligned_img.convert("RGB")
                # Place aligned crop at right side of cell
                aligned_x = cell_width - 120
                cell_img.paste(aligned_img.resize((112, 112)), (aligned_x, 10))
                cell_draw.text((aligned_x, 130), "Aligned\n224×224", fill="black", font=font_small)
            except:
                pass
        
        # Paste cell into grid
        grid.paste(cell_img, (x_offset, y_offset))
    
    return grid


def main():
    args = parse_args()
    
    manifest_path = Path(args.manifest)
    source_root = Path(args.source_root)
    output_path = Path(args.output)
    
    if not manifest_path.exists():
        print(f"Error: Manifest {manifest_path} does not exist")
        sys.exit(1)
    
    print(f"Loading manifest: {manifest_path}")
    df = pd.read_csv(manifest_path)
    
    print(f"Total samples: {len(df)}")
    print(f"Status breakdown:")
    print(df["processing_status"].value_counts())
    
    print(f"\nCreating visual audit for {args.num_samples} samples...")
    grid = create_audit_grid(df, source_root, args.num_samples, args.random_seed)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(output_path)
    print(f"Audit image saved to: {output_path}")


if __name__ == "__main__":
    main()