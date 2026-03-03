#!/usr/bin/env python3
"""
Step 5: RANSAC Joint Estimation.
Runs inside nerfstudio container.

Reads from:
  <joint_dir>/tapip3d/          <- TAPIP3D trajectory output
  <joint_dir>/transforms.json   <- camera metadata

Writes to:
  <joint_dir>/ransac_joints/    <- joint_schemas.json

Usage:
    python -m arti4d.run_joint_estimation \
        --joint_dir /workspace/data/.../articulated_joint_0
"""

import os
import json
import argparse
import subprocess
from pathlib import Path


def run_joint_estimation(
    joint_dir: Path,
    max_iterations: int = 200,
    error_threshold: float = 0.05,
    min_inliers: int = 15,
    visibility_threshold: float = 0.8,
    no_viz: bool = True,
) -> Path:
    joint_dir  = Path(joint_dir)

    tapip3d_dir    = joint_dir / "tapip3d"
    transforms_path = joint_dir / "transforms.json"
    output_dir     = joint_dir / "ransac_joints"

    # ── Locate TAPIP3D result file ────────────────────────────────────────────
    result_candidates = [
        tapip3d_dir / "result.npz",
        tapip3d_dir / "tapip3d_trajectory.npz",
        tapip3d_dir / "tracks_3d.npy",
        *sorted(tapip3d_dir.glob("*.npz")),
    ]
    tapip3d_result = next((p for p in result_candidates if p.exists()), None)

    # ── Validate ──────────────────────────────────────────────────────────────
    missing = []
    if not tapip3d_dir.exists():
        missing.append(f"tapip3d dir:     {tapip3d_dir}")
    if tapip3d_result is None:
        missing.append(f"result file in   {tapip3d_dir}  (run STEP 4 first)")
    if not transforms_path.exists():
        missing.append(f"transforms.json: {transforms_path}")
    if missing:
        raise FileNotFoundError(
            "Missing required inputs:\n" +
            "\n".join(f"  ✗ {m}" for m in missing)
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"STEP 5: 4D RANSAC JOINT ESTIMATION")
    print(f"{'='*60}")
    print(f"  Joint dir:    {joint_dir}")
    print(f"  Result file:  {tapip3d_result.name}")
    print(f"  Transforms:   {transforms_path}")
    print(f"  Output:       {output_dir}")

    cmd = [
        "python", "-m", "joint_estimator.main",
        "--tapip3d_result",       str(tapip3d_result),
        "--camera_metadata",      str(transforms_path),
        "--out_dir",              str(output_dir),
        "--max_iterations",       str(max_iterations),
        "--error_threshold",      str(error_threshold),
        "--min_inliers",          str(min_inliers),
        "--visibility_threshold", str(visibility_threshold),
    ]
    if no_viz:
        cmd.append("--no_viz")

    print(f"\nCommand:\n  " + " \\\n    ".join(cmd) + "\n")

    project_root = Path(os.environ.get("PROJECT_ROOT", "."))
    subprocess.run(cmd, cwd=project_root, check=True)

    schema_path = output_dir / "joint_schemas.json"
    if not schema_path.exists():
        raise RuntimeError(
            f"joint_schemas.json not found after estimation.\n"
            f"Expected: {schema_path}"
        )

    with open(schema_path) as f:
        schemas = json.load(f)

    if isinstance(schemas, list):
        schemas_list = schemas
    else:
        schemas_list = [schemas]

    jtype = schemas_list[0].get("joint_type", "unknown") if schemas_list else "unknown"
    print(f"\n✓ Joint estimation complete — type: {jtype}")
    print(f"  {len(schemas_list)} schema(s) → {schema_path}")

    return schema_path


def main():
    parser = argparse.ArgumentParser(
        description="Step 5: RANSAC joint estimation"
    )
    parser.add_argument("--joint_dir", required=True,
                        help="Path to articulated_joint_<N>/ directory (container path)")
    parser.add_argument("--max_iterations",       type=int,   default=100)
    parser.add_argument("--error_threshold",      type=float, default=0.05)
    parser.add_argument("--min_inliers",          type=int,   default=20)
    parser.add_argument("--visibility_threshold", type=float, default=0.8)
    parser.add_argument("--viz", action="store_true",
                        help="Enable 3D visualisation (disabled by default)")
    args = parser.parse_args()

    run_joint_estimation(
        joint_dir            = Path(args.joint_dir),
        max_iterations       = args.max_iterations,
        error_threshold      = args.error_threshold,
        min_inliers          = args.min_inliers,
        visibility_threshold = args.visibility_threshold,
        no_viz               = not args.viz,
    )

    print(f"\nNext step (container):")
    print(f"  python -m arti4d.run_merge_arti_data --joint_dir {args.joint_dir}")


if __name__ == "__main__":
    main()