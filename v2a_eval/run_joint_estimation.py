#!/usr/bin/env python3
"""
V2A Joint Estimation Wrapper
Runs 4D RANSAC on TAPIP3D tracks from V2A prepared scene.
Skips temporal voxels, SAM2, change detection.
"""

import os
import json
import argparse
import subprocess
from pathlib import Path


def run_joint_estimation(
    scene_dir: Path,
    output_dir: Path = None,
    max_iterations: int = 200,
    error_threshold: float = 0.05,
    min_inliers: int = 15,
    visibility_threshold: float = 0.8,
    no_viz: bool = True,
):
    """
    Run 4D RANSAC joint estimation on a prepared V2A scene.

    Args:
        scene_dir:            Prepared V2A scene (output of prepare_scene.py)
        output_dir:           Where to save results (default: scene_dir/ransac_joints)
        max_iterations:       RANSAC iterations
        error_threshold:      Inlier threshold in metres
        min_inliers:          Minimum inliers to accept a model
        visibility_threshold: TAPIP3D visibility filter
        no_viz:               Disable 3D visualisation (required for batch runs)
    """
    scene_dir = Path(scene_dir)

    # ------------------------------------------------------------------ paths
    tapip3d_dir  = scene_dir / "articulated" / "tapip3d"
    camera_meta  = scene_dir / "articulated" / "transforms.json"
    output_dir   = scene_dir / "articulated" / "ransac_joints"

    # Locate the TAPIP3D result file
    result_candidates = [
        tapip3d_dir / "result.npz",
        tapip3d_dir / "tracks_3d.npy",
        *tapip3d_dir.glob("*.result.npz"),
        *tapip3d_dir.glob("*.npz"),
    ]
    tapip3d_result = next((p for p in result_candidates if p.exists()), None)

    # ---------------------------------------------------------------- validate
    missing = []
    if not tapip3d_dir.exists():  missing.append(str(tapip3d_dir))
    if not camera_meta.exists():  missing.append(str(camera_meta))
    if tapip3d_result is None:    missing.append(f"result file in {tapip3d_dir}")
    if missing:
        raise FileNotFoundError(
            f"Missing required inputs for {scene_dir.name}:\n" +
            "\n".join(f"  - {m}" for m in missing)
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"4D RANSAC Joint Estimation")
    print(f"Scene: {scene_dir.name}")
    print(f"TAPIP3D result: {tapip3d_result.name}")
    print(f"Camera metadata: {camera_meta}")
    print(f"Output: {output_dir}")
    print(f"{'='*80}\n")

    # ---------------------------------------------------------------- build cmd
    cmd = [
        "python", "-m", "joint_estimator.main",
        "--tapip3d_result",      str(tapip3d_result),
        "--camera_metadata",     str(camera_meta),
        "--out_dir",             str(output_dir),
        "--max_iterations",      str(max_iterations),
        "--error_threshold",     str(error_threshold),
        "--min_inliers",         str(min_inliers),
        "--visibility_threshold",str(visibility_threshold),
    ]

    if no_viz:
        cmd.append("--no_viz")

    print("Command:")
    print("  " + " \\\n    ".join(cmd))
    print()

    # ---------------------------------------------------------------- run
    project_root = Path(os.environ.get("PROJECT_ROOT", "."))
    result = subprocess.run(cmd, cwd=project_root, check=True)

    # ---------------------------------------------------------------- validate output
    schema_path = output_dir / "joint_schemas.json"
    if not schema_path.exists():
        raise RuntimeError(
            f"joint_schemas.json not found after estimation.\n"
            f"Expected: {schema_path}"
        )

    # Summarise what was estimated
    with open(schema_path) as f:
        schemas = json.load(f)

    joint_type = (
        schemas[0].get("joint_type", "unknown")
        if isinstance(schemas, list) and len(schemas) > 0
        else "unknown"
    )


    return schema_path


def main():
    parser = argparse.ArgumentParser(
        description="Run 4D RANSAC joint estimation on a prepared V2A scene"
    )
    parser.add_argument("--scene",   required=True,
                        help="Path to prepared V2A scene directory")
    parser.add_argument("--output",  default=None,
                        help="Output directory (default: <scene>/ransac_joints)")
    parser.add_argument("--max_iterations",       type=int,   default=200)
    parser.add_argument("--error_threshold",      type=float, default=0.05)
    parser.add_argument("--min_inliers",          type=int,   default=15)
    parser.add_argument("--visibility_threshold", type=float, default=0.8)
    parser.add_argument("--viz", action="store_true",
                        help="Enable 3D visualisation (disabled by default for batch)")
    args = parser.parse_args()

    run_joint_estimation(
        scene_dir            = args.scene,
        output_dir           = args.output,
        max_iterations       = args.max_iterations,
        error_threshold      = args.error_threshold,
        min_inliers          = args.min_inliers,
        visibility_threshold = args.visibility_threshold,
        no_viz               = not args.viz,
    )


if __name__ == "__main__":
    main()