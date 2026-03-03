#!/usr/bin/env python3
"""
Step 4: TAPIP3D Point Tracking.
Runs on HOST in local conda environment (tapip3d), NOT in nerfstudio container.

Reads from:
  <joint_dir>/transforms.json   <- camera metadata
  <joint_dir>/mask/             <- SAM2 masks (uses first frame as query)

Writes to:
  <joint_dir>/tapip3d/          <- trajectory output (tapip3d_trajectory.npz)

Usage:
    conda activate tapip3d
    python -m arti4d.run_point_tracking \
        --joint_dir        /local/home/pmishra/nerfstudio-data/.../articulated_joint_0 \
        --max_query_points 300
"""

import os
import argparse
import subprocess
from pathlib import Path
import numpy as np
from PIL import Image

TAPIP3D_DIR = "/local/home/pmishra/arti-splatfacto/third_party/tapip3d"


def run_point_tracking(joint_dir: Path, max_query_points: int = 300) -> Path:
    joint_dir   = Path(joint_dir)
    tapip3d_dir = Path(os.environ.get("TAPIP3D_DIR", TAPIP3D_DIR))

    transforms_path = joint_dir / "transforms.json"
    masks_dir       = joint_dir / "mask"
    output_dir      = joint_dir / "tapip3d"

    print(f"\n{'='*60}")
    print(f"STEP 4: TAPIP3D POINT TRACKING")
    print(f"{'='*60}")
    print(f"  Joint dir:   {joint_dir}")
    print(f"  Transforms:  {transforms_path}")
    print(f"  Masks:       {masks_dir}")
    print(f"  Output:      {output_dir}")
    print(f"  TAPIP3D:     {tapip3d_dir}")

    if not transforms_path.exists():
        raise FileNotFoundError(f"transforms.json not found: {transforms_path}")
    if not tapip3d_dir.exists():
        raise FileNotFoundError(f"TAPIP3D directory not found: {tapip3d_dir}")

    mask_files = sorted(masks_dir.glob("*.png"))
    if not mask_files:
        raise FileNotFoundError(
            f"No mask files found in {masks_dir}\n"
            f"Run SAM2 VOS (STEP 3) first."
        )

    first_mask = mask_files[0]
    mask       = np.array(Image.open(first_mask))
    coverage   = (mask > 127).sum() / mask.size * 100
    print(f"  Query mask:  {first_mask.name}  ({coverage:.1f}% coverage)")

    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python", "inference_nerf_mask.py",
        "--input_path",       str(transforms_path.resolve()),
        "--mask_path",        str(first_mask.resolve()),
        "--output_dir",       str(output_dir.resolve()),
        "--max_query_points", str(max_query_points),
    ]

    print(f"\nCommand:\n  cd {tapip3d_dir}\n  " + " \\\n    ".join(cmd) + "\n")

    env              = os.environ.copy()
    env["PYTHONPATH"] = str(tapip3d_dir)
    original_dir     = os.getcwd()

    try:
        os.chdir(tapip3d_dir)
        subprocess.run(cmd, env=env, check=True)
    finally:
        os.chdir(original_dir)

    expected = output_dir / "tapip3d_trajectory.npz"
    if not expected.exists():
        raise FileNotFoundError(f"Expected output not found: {expected}")

    tracks = np.load(expected, allow_pickle=True)
    print(f"\n✓ Point tracking complete → {output_dir}")
    for key in tracks.files:
        print(f"  {key}: shape={tracks[key].shape}, dtype={tracks[key].dtype}")

    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description="Step 4: TAPIP3D point tracking (run in local tapip3d conda env)"
    )
    parser.add_argument("--joint_dir",         required=True,
                        help="Path to articulated_joint_<N>/ directory (host path)")
    parser.add_argument("--max_query_points",  type=int, default=500)
    args = parser.parse_args()

    run_point_tracking(
        joint_dir        = Path(args.joint_dir),
        max_query_points = args.max_query_points,
    )


if __name__ == "__main__":
    main()