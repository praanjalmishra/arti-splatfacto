#!/usr/bin/env python3
"""
Step 3: SAM2 Video Object Segmentation.
Runs on HOST in local conda environment (sam2), NOT in nerfstudio container.

Propagates the change detection mask through the articulated video sequence
to produce per-frame part masks.

Reads from:
  <joint_dir>/frames/     <- RGB frames
  <joint_dir>/cd_mask/    <- change detection seed masks

Writes to:
  <joint_dir>/mask/       <- per-frame SAM2 masks

Usage:
    conda activate sam2
    python -m arti4d.run_sam2_vos \
        --joint_dir /local/home/pmishra/nerfstudio-data/.../articulated_joint_0 \
        --sam2_dir  /local/home/pmishra/arti-splatfacto/third_party/sam2
"""

import os
import argparse
import subprocess
from pathlib import Path


def run_sam2_vos(joint_dir: Path, sam2_dir: Path) -> Path:
    joint_dir = Path(joint_dir)
    sam2_dir  = Path(sam2_dir)

    frames_dir      = joint_dir / "frames"
    cd_mask_dir     = joint_dir / "cd_mask"
    output_mask_dir = joint_dir / "mask"
    sam2_script     = sam2_dir / "tools" / "run_sam2_vos_wrapper.py"

    print(f"\n{'='*60}")
    print(f"STEP 3: SAM2 VIDEO OBJECT SEGMENTATION")
    print(f"{'='*60}")
    print(f"  Joint dir:  {joint_dir}")
    print(f"  Frames:     {frames_dir}")
    print(f"  CD mask:    {cd_mask_dir}")
    print(f"  Output:     {output_mask_dir}")

    if not frames_dir.exists():
        raise FileNotFoundError(f"Frames not found: {frames_dir}")
    if not cd_mask_dir.exists() or not list(cd_mask_dir.glob("*")):
        raise FileNotFoundError(
            f"CD mask not found or empty: {cd_mask_dir}\n"
            f"Run change detection (STEP 2) first."
        )
    if not sam2_script.exists():
        raise FileNotFoundError(f"SAM2 script not found: {sam2_script}")

    output_mask_dir.mkdir(parents=True, exist_ok=True)

    n_frames = len(list(frames_dir.glob("*.jpg")) + list(frames_dir.glob("*.png")))
    print(f"  Frames to process: {n_frames}")

    cmd = [
        "python", str(sam2_script),
        "--frames_dir",      str(frames_dir),
        "--mask_path",       str(cd_mask_dir),
        "--output_mask_dir", str(output_mask_dir),
    ]

    print(f"\nCommand:\n  " + " \\\n    ".join(cmd) + "\n")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(sam2_dir)
    subprocess.run(cmd, env=env, check=True)

    output_masks = sorted(output_mask_dir.glob("*.png"))
    if not output_masks:
        raise RuntimeError(f"SAM2 produced no output masks in {output_mask_dir}")

    print(f"\n✓ SAM2 VOS complete")
    print(f"  {len(output_masks)} masks → {output_mask_dir}")

    return output_mask_dir


def main():
    parser = argparse.ArgumentParser(
        description="Step 3: SAM2 VOS (run in local sam2 conda env)"
    )
    parser.add_argument("--joint_dir", required=True,
                        help="Path to articulated_joint_<N>/ directory (host path)")
    parser.add_argument("--sam2_dir",  default=None,
                        help="SAM2 repo path (default: $SAM2_DIR env var)")
    args = parser.parse_args()

    sam2_dir = Path(
        args.sam2_dir or
        os.environ.get("SAM2_DIR",
                       "/local/home/pmishra/arti-splatfacto/third_party/sam2")
    )

    run_sam2_vos(
        joint_dir = Path(args.joint_dir),
        sam2_dir  = sam2_dir,
    )

    print(f"\nNext step (host conda: tapip3d):")
    print(f"  python -m arti4d.run_point_tracking --joint_dir {args.joint_dir}")


if __name__ == "__main__":
    main()