#!/usr/bin/env python3
"""
Step 2: Change Detection.
Runs inside nerfstudio container.

Renders canonical 3DGS at articulated_joint_<N> camera poses, computes
DINO feature difference, writes change masks to:
  <joint_dir>/cd_mask/

Usage:
    python -m arti4d.run_change_detection \
        --canonical_model /workspace/data/.../canonical/splatfacto \
        --joint_dir       /workspace/data/.../articulated_joint_0 \
        [--joint 0]       \
        [--debug]
"""

import argparse
import os
import subprocess
from pathlib import Path


def find_canonical_checkpoint(canonical_splatfacto_dir: Path) -> Path:
    """
    Find the most recent config.yml under canonical/splatfacto/.
    ns-train writes: canonical/splatfacto/canonical/<experiment>/<timestamp>/config.yml
    """
    configs = sorted(canonical_splatfacto_dir.rglob("config.yml"))
    if not configs:
        raise FileNotFoundError(
            f"No config.yml found under {canonical_splatfacto_dir}\n"
            f"Has STEP 1 (ns-train canonical) completed?"
        )
    return configs[-1]


def run_change_detection(
    canonical_model_dir: Path,
    joint_dir: Path,
    joint_idx: int,
    project_root: Path,
    change_cfg: str = "config_change_arti4d.yaml",
    debug: bool = False,
) -> Path:
    canonical_model_dir = Path(canonical_model_dir)
    joint_dir           = Path(joint_dir)

    transforms_path = joint_dir / "transforms.json"
    config          = find_canonical_checkpoint(canonical_model_dir)
    output_dir      = joint_dir 
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"STEP 2: CHANGE DETECTION  —  joint {joint_idx}")
    print(f"{'='*60}")
    print(f"  GS config:    {config}")
    print(f"  Transforms:   {transforms_path}")
    print(f"  Output:       {output_dir}")

    if not transforms_path.exists():
        raise FileNotFoundError(f"transforms.json not found: {transforms_path}")

    cmd = [
        "python", "-m", "change_det.change_detection_sam",
        "--config",    str(config),
        "--output",    str(output_dir),
        "--transform", str(transforms_path),
        "--params",    change_cfg,
    ]
    if debug:
        cmd.append("--debug")

    print(f"\nCommand:\n  " + " \\\n    ".join(cmd) + "\n")
    subprocess.run(cmd, cwd=project_root, check=True)

    cd_files = list(output_dir.glob("*"))
    if not cd_files:
        raise RuntimeError(f"Change detection produced no output in {output_dir}")

    print(f"\n✓ Change detection complete  ({len(cd_files)} files → {output_dir})")
    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description="Step 2: Change detection — canonical 3DGS vs articulated frames"
    )
    parser.add_argument("--canonical_model", required=True,
                        help="Path to canonical/splatfacto/ directory")
    parser.add_argument("--joint_dir", required=True,
                        help="Path to articulated_joint_<N>/ directory")
    parser.add_argument("--joint",        type=int, default=0,
                        help="Joint index (for logging only, default: 0)")
    parser.add_argument("--project_root", default=None,
                        help="arti-splatfacto root (default: $PROJECT_ROOT or cwd)")
    parser.add_argument("--change_cfg",   default="config_change_arti4d.yaml")
    parser.add_argument("--debug",        action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root or os.environ.get("PROJECT_ROOT", "."))

    run_change_detection(
        canonical_model_dir = Path(args.canonical_model),
        joint_dir           = Path(args.joint_dir),
        joint_idx           = args.joint,
        project_root        = project_root,
        change_cfg          = args.change_cfg,
        debug               = args.debug,
    )

    print(f"\nNext step (host conda: sam2):")
    print(f"  python -m arti4d.run_sam2_vos --joint_dir {args.joint_dir}")


if __name__ == "__main__":
    main()