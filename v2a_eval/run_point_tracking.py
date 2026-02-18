#!/usr/bin/env python3
"""
Point tracking for V2A evaluation using GT masks
Skips change detection and SAM2 - uses GT masks directly
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
import numpy as np
from PIL import Image


def run_tapip3d_tracking(scene_dir: Path, max_query_points: int = 300):
    """
    Run TAPIP3D point tracking using GT masks
    
    Args:
        scene_dir: Prepared V2A scene directory
        max_query_points: Number of points to track
    """
    
    print(f"\n{'='*80}")
    print(f"Running TAPIP3D Point Tracking")
    print(f"Scene: {scene_dir.name}")
    print(f"{'='*80}\n")
    
    # Paths
    articulated_dir = scene_dir / "articulated"
    transforms_path = articulated_dir / "transforms.json"
    masks_dir = articulated_dir / "mask"
    output_dir = articulated_dir / "tapip3d"
    
    # Validate inputs
    if not transforms_path.exists():
        raise FileNotFoundError(f"Transforms not found: {transforms_path}")
    
    if not masks_dir.exists():
        raise FileNotFoundError(f"Masks directory not found: {masks_dir}")
    
    # Get first mask 
    mask_files = sorted(masks_dir.glob("*.png"))
    if not mask_files:
        raise FileNotFoundError(f"No mask files found in {masks_dir}")
    
    first_mask = mask_files[0]
    print(f"Using initial mask: {first_mask.name}")
    
    # Visualize mask coverage
    mask = np.array(Image.open(first_mask))
    mask_coverage = (mask > 127).sum() / mask.size * 100
    print(f"Mask coverage: {mask_coverage:.1f}%")
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    # Setup TAPIP3D environment
    tapip3d_dir = os.environ.get("TAPIP3D_DIR", "/local/home/pmishra/arti-splatfacto/third_party/tapip3d")
    if not Path(tapip3d_dir).exists():
        raise FileNotFoundError(f"TAPIP3D directory not found: {tapip3d_dir}")
    
    print(f"\n{'='*80}")
    print(f"TAPIP3D Configuration")
    print(f"{'='*80}")
    print(f"TAPIP3D directory: {tapip3d_dir}")
    print(f"Transforms: {transforms_path}")
    print(f"Initial mask: {first_mask}")
    print(f"Output: {output_dir}")
    print(f"Max query points: {max_query_points}")
    print(f"{'='*80}\n")
    
    # Run TAPIP3D
    cmd = [
        "python", "inference_nerf_mask.py",
        "--input_path", str(transforms_path.resolve()),
        "--mask_path", str(first_mask.resolve()),
        "--output_dir", str(output_dir.resolve()),
        "--max_query_points", str(max_query_points)
    ]
    
    print(f"Running command:")
    print(f"  cd {tapip3d_dir}")
    print(f"  {' '.join(cmd)}\n")
    
    # Change to TAPIP3D directory and run
    original_dir = os.getcwd()
    try:
        os.chdir(tapip3d_dir)
        
        # Set PYTHONPATH
        env = os.environ.copy()
        env["PYTHONPATH"] = tapip3d_dir
        
        result = subprocess.run(
            cmd,
            env=env,
            check=True,
            capture_output=False,
            text=True
        )
        
        print(f"\n✓ TAPIP3D tracking complete!")
        
    except subprocess.CalledProcessError as e:
        print(f"\n✗ TAPIP3D failed with error code {e.returncode}")
        raise
    
    finally:
        os.chdir(original_dir)
    
    # Validate output
    expected_output = output_dir / "tapip3d_trajectory.npz"
    if not expected_output.exists():
        raise FileNotFoundError(f"Expected output not found: {expected_output}")
    
    tracks = np.load(expected_output, allow_pickle=True)

    print(f"\n{'='*80}")
    print("Tracking Results Summary")
    print(f"{'='*80}")
    print(f"Output directory: {output_dir}")
    print("Available keys:", tracks.files)

    for key in tracks.files:
        arr = tracks[key]
        print(f"{key}: shape={arr.shape}, dtype={arr.dtype}")
    print(f"{'='*80}\n")



def main():
    parser = argparse.ArgumentParser(
        description="Run TAPIP3D point tracking on V2A scene using GT masks"
    )
    parser.add_argument(
        "--scene", 
        type=str, 
        required=True,
        help="Path to prepared V2A scene directory"
    )
    parser.add_argument(
        "--max_query_points",
        type=int,
        default=250,
        help="Maximum number of query points to track (default: 300)"
    )
    
    args = parser.parse_args()
    
    scene_dir = Path(args.scene)
    if not scene_dir.exists():
        raise FileNotFoundError(f"Scene directory not found: {scene_dir}")
    
    run_tapip3d_tracking(scene_dir, args.max_query_points)


if __name__ == "__main__":
    main()