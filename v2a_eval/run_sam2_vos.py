#!/usr/bin/env python3
"""
Step 7: SAM2 Video Object Segmentation.
Runs in LOCAL conda environment (sam2 env), NOT in nerfstudio container.

Propagates the change detection mask through the articulated video sequence
to get per-frame part-level masks (moving part only).

Inputs:
  - <scene_dir>/articulated/frames/   : RGB frames
  - <scene_dir>/cd_mask/              : change detection seed mask

Outputs:
  - <scene_dir>/mask_cd/              : per-frame part masks

Usage (from local machine, sam2 conda env):
    conda activate sam2
    python -m v2a_eval.run_sam2_vos \
        --scene outputs/v2a_eval/7265_joint_0_bg_view_0 \
        --sam2_dir /local/home/pmishra/cvg/sam2
"""

import os
import json
import argparse
import subprocess
from pathlib import Path


def run_sam2_vos(
    scene_dir: Path,
    sam2_dir: Path,
):
    """
    Run SAM2 VOS to propagate CD mask through articulated video.

    Args:
        scene_dir: Prepared V2A scene directory
        sam2_dir:  Path to SAM2 repository
    """
    scene_dir = Path(scene_dir)
    sam2_dir  = Path(sam2_dir)

    frames_dir      = scene_dir / "articulated" / "frames"
    cd_mask_dir     = scene_dir / "articulated" / "cd_mask"
    output_mask_dir = scene_dir / "articulated"  / "mask"

    sam2_script = sam2_dir / "tools" / "run_sam2_vos_wrapper.py"

    print(f"\n{'='*60}")
    print(f"STEP 7: SAM2 VIDEO OBJECT SEGMENTATION")
    print(f"{'='*60}")
    print(f"Scene:   {scene_dir.name}")
    print(f"Frames:  {frames_dir}")
    print(f"CD mask: {cd_mask_dir}")
    print(f"Output:  {output_mask_dir}")

    # Validate inputs
    if not frames_dir.exists():
        raise FileNotFoundError(
            f"Frames not found: {frames_dir}\n"
            f"Run prepare_scene.py first."
        )
    if not cd_mask_dir.exists() or not list(cd_mask_dir.glob("*")):
        raise FileNotFoundError(
            f"CD mask not found or empty: {cd_mask_dir}\n"
            f"Run run_change_detection.py first (in nerfstudio container)."
        )
    if not sam2_script.exists():
        raise FileNotFoundError(f"SAM2 script not found: {sam2_script}")

    output_mask_dir.mkdir(parents=True, exist_ok=True)

    n_frames = len(list(frames_dir.glob("*.png")))
    print(f"Frames to process: {n_frames}")

    cmd = [
        "python", str(sam2_script),
        "--frames_dir",       str(frames_dir),
        "--mask_path",        str(cd_mask_dir),
        "--output_mask_dir",  str(output_mask_dir),
    ]

    print(f"\nCommand:\n  " + " \\\n    ".join(cmd) + "\n")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(sam2_dir)

    subprocess.run(cmd, env=env, check=True)

    # Validate output
    output_masks = sorted(output_mask_dir.glob("*.png"))

    if not output_masks:
        raise RuntimeError(
            f"SAM2 produced no output masks in {output_mask_dir}"
        )

    print(f"\n✓ SAM2 VOS complete")
    print(f"  Output masks: {len(output_masks)}")
    print(f"  Saved to:     {output_mask_dir}")

    # Update transforms_arti.json with CD masks
    _update_transforms_with_cd_masks(scene_dir, output_masks)

    return output_mask_dir


def _update_transforms_with_cd_masks(scene_dir: Path, output_masks: list):
    """
    Create transforms_arti_cd.json with mask_path pointing to CD masks.
    Preserves everything else from transforms_arti.json.
    """
    arti_transforms = scene_dir / "transforms_arti.json"
    output_path     = scene_dir / "transforms_arti_cd.json"

    if not arti_transforms.exists():
        print(f"⚠  transforms_arti.json not found — skipping transforms update.")
        print(f"   Run merge_arti_data_v2a.py first, then re-run this script.")
        return

    with open(arti_transforms) as f:
        transforms = json.load(f)

    # Build index → relative mask path lookup
    # SAM2 output naming: frame_00001.png, frame_00002.png ...
    cd_mask_lookup = {}
    for mask_path in output_masks:
        # Extract frame index from filename
        stem = mask_path.stem          # e.g. "frame_00001"
        try:
            idx = int(stem.split("_")[-1])
            cd_mask_lookup[idx] = f"mask_cd/{mask_path.name}"
        except ValueError:
            continue

    updated_frames = []
    swapped = 0

    for idx, frame in enumerate(transforms["frames"]):
        frame = dict(frame)
        if idx in cd_mask_lookup:
            frame["mask_path"] = cd_mask_lookup[idx]
            swapped += 1
        updated_frames.append(frame)

    transforms["frames"] = updated_frames

    with open(output_path, "w") as f:
        json.dump(transforms, f, indent=2)

    print(f"\n✓ Created: {output_path}")
    print(f"  CD masks injected: {swapped}/{len(updated_frames)} frames")
    print(f"\nTwo mask options now available:")
    print(f"  GT alpha:  transforms_arti.json     (full object)")
    print(f"  CD+SAM2:   transforms_arti_cd.json  (moving part only)")


def main():
    parser = argparse.ArgumentParser(
        description="Step 7: SAM2 VOS (run in local sam2 conda env)"
    )
    parser.add_argument("--scene",    required=True,
                        help="Prepared V2A scene directory")
    parser.add_argument("--sam2_dir", default=None,
                        help="SAM2 repo path (default: $SAM2_DIR env var)")

    args = parser.parse_args()

    sam2_dir = Path(
        args.sam2_dir or
        os.environ.get("SAM2_DIR", "/local/home/pmishra/arti-splatfacto/third_party/sam2")
    )

    run_sam2_vos(
        scene_dir = Path(args.scene),
        sam2_dir  = sam2_dir,
    )


if __name__ == "__main__":
    main()