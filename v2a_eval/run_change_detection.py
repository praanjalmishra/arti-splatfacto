#!/usr/bin/env python3
"""
Step 6: Change Detection using DINO features against canonical 3DGS.
Runs inside nerfstudio container.

Inputs:
  - Trained canonical 3DGS config.yml
  - Articulated frames + transforms.json (state=1)

Outputs:
  - <scene_dir>/cd_mask/  : change detection masks

Usage:
    python -m v2a_eval.run_change_detection \
        --scene outputs/v2a_eval/7265_joint_0_bg_view_0
"""

import os
import argparse
import subprocess
from pathlib import Path


def find_canonical_checkpoint(canonical_dir: Path):
    """Find the most recent trained 3DGS config under canonical/."""
    configs = sorted(canonical_dir.rglob("config.yml"))

    if not configs:
        raise FileNotFoundError(
            f"No config.yml found under {canonical_dir}\n"
            f"Run first: ns-train qed-splatter --data {canonical_dir}/transforms.json"
        )

    return configs[-1]   # most recent


def run_change_detection(
    scene_dir: Path,
    project_root: Path,
    change_cfg: str = "config_change.yaml",
    debug: bool = False,
):
    """
    Run DINO-based change detection against canonical 3DGS.

    Renders canonical views at articulated camera poses, computes DINO
    feature difference, outputs a change mask into <scene_dir>/cd_mask/.
    """
    scene_dir    = Path(scene_dir)
    project_root = Path(project_root)

    canonical_dir   = scene_dir / "canonical"
    transforms_path = scene_dir / "articulated" / "transforms.json"
    cd_output_dir   = scene_dir / "articulated"

    config = find_canonical_checkpoint(canonical_dir)

    print(f"\n{'='*60}")
    print(f"STEP 6: CHANGE DETECTION")
    print(f"{'='*60}")
    print(f"Scene:      {scene_dir.name}")
    print(f"GS config:  {config}")
    print(f"Transforms: {transforms_path}")
    print(f"Output:     {cd_output_dir}")

    if not transforms_path.exists():
        raise FileNotFoundError(
            f"Articulated transforms not found: {transforms_path}\n"
            f"Run prepare_scene.py first."
        )

    cmd = [
        "python", "-m", "change_det.change_detection_sam",
        "--config",    str(config),
        "--output",    str(cd_output_dir),
        "--transform", str(transforms_path),
        "--params",    change_cfg,
    ]

    if debug:
        cmd.append("--debug")

    print(f"\nCommand:\n  " + " \\\n    ".join(cmd) + "\n")

    subprocess.run(cmd, cwd=project_root, check=True)

    # Validate output
    cd_output_dir.mkdir(exist_ok=True)
    cd_files = list(cd_output_dir.glob("*"))

    if not cd_files:
        raise RuntimeError(
            f"Change detection produced no output in {cd_output_dir}"
        )

    print(f"\n✓ Change detection complete")
    print(f"  Output files: {len(cd_files)} in {cd_output_dir}")

    return cd_output_dir


def main():
    parser = argparse.ArgumentParser(
        description="Step 6: Change detection (nerfstudio container)"
    )
    parser.add_argument("--scene",        required=True,
                        help="Prepared V2A scene directory")
    parser.add_argument("--project_root", default=None,
                        help="arti-splatfacto root (default: $PROJECT_ROOT)")
    parser.add_argument("--change_cfg",   default="config_change.yaml",
                        help="Change detection config YAML")
    parser.add_argument("--debug",        action="store_true")

    args = parser.parse_args()

    project_root = Path(
        args.project_root or
        os.environ.get("PROJECT_ROOT", ".")
    )

    cd_output = run_change_detection(
        scene_dir    = Path(args.scene),
        project_root = project_root,
        change_cfg   = args.change_cfg,
        debug        = args.debug,
    )

    print(f"\nNext step (on local conda env):")
    print(f"  python -m v2a_eval.run_sam2_vos --scene {args.scene}")


if __name__ == "__main__":
    main()