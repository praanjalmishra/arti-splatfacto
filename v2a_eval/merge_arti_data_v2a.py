#!/usr/bin/env python3
"""
V2A-specific merge: attach RANSAC joint parameters + interpolated angles
to GT transforms.

The articulated sequence covers a full motion cycle (open → close or similar).
RANSAC produces per_frame_angles for a subsampled set of frames.
We interpolate to get the joint angle for every frame using time.
"""

import json
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, Tuple


def load_joint_schemas(path: Path) -> Dict:
    """Load and validate joint schemas from RANSAC output."""
    with open(path) as f:
        schemas = json.load(f)

    # Handle both list and dict format
    if isinstance(schemas, list):
        schemas = schemas[0]

    required = ["joint_type", "joint_axis", "joint_pivot"]
    missing = [k for k in required if k not in schemas]
    if missing:
        raise ValueError(f"joint_schemas.json missing fields: {missing}")

    return schemas


def build_per_frame_lookup(schemas: Dict) -> Tuple[Dict[int, float], str]:
    """
    Extract per-frame angle/translation from schemas.
    Converts degrees → radians for revolute joints.

    Returns:
        per_frame_data: {frame_idx: value}
        unit: 'rad' or 'm'
    """
    joint_type = schemas["joint_type"]

    if joint_type == "revolute":
        raw = schemas.get("per_frame_angles", {})
        # Convert degrees → radians
        per_frame = {int(k): np.radians(float(v)) for k, v in raw.items()}
        unit = "rad"
    elif joint_type in ("prismatic", "slider"):
        raw = schemas.get("per_frame_translations", {})
        per_frame = {int(k): float(v) for k, v in raw.items()}
        unit = "m"
    else:
        raise ValueError(f"Unknown joint type: {joint_type}")

    if not per_frame:
        raise ValueError(
            f"No per-frame data found in schemas for joint type '{joint_type}'"
        )

    return per_frame, unit


def interpolate_angle(
    time: float,
    per_frame_data: Dict[int, float],
    n_ransac_frames: int,
) -> float:
    """
    Interpolate joint angle for a given normalized time [0, 1].

    Maps time → fractional frame index → linear interpolation
    between the two nearest RANSAC-measured frames.

    Args:
        time:            Normalized time in [0, 1]
        per_frame_data:  {ransac_frame_idx: angle_value}
        n_ransac_frames: Total RANSAC frames (max key + 1)
    """
    # Map normalized time to fractional frame index
    frame_float = time * (n_ransac_frames - 1)

    sorted_frames = sorted(per_frame_data.keys())
    min_frame = sorted_frames[0]
    max_frame = sorted_frames[-1]

    # Clamp to measured range
    if frame_float <= min_frame:
        return per_frame_data[min_frame]
    if frame_float >= max_frame:
        return per_frame_data[max_frame]

    # Find surrounding frames and interpolate
    lower = max(f for f in sorted_frames if f <= frame_float)
    upper = min(f for f in sorted_frames if f >= frame_float)

    if lower == upper:
        return per_frame_data[lower]

    alpha = (frame_float - lower) / (upper - lower)
    return per_frame_data[lower] * (1 - alpha) + per_frame_data[upper] * alpha


def merge_v2a(
    scene_dir: Path,
    output_path: Path = None,
):

    """
    Merge RANSAC joint parameters + interpolated angles into V2A transforms.

    For each articulated frame:
      - joint_angle is interpolated from per_frame_angles using frame's time
      - For canonical frames (state=0): joint_angle = 0.0
    """
    scene_dir = Path(scene_dir)
    output_path = scene_dir / "articulated" / "transforms_post.json"

    # ── Load inputs ───────────────────────────────────────────────────
    schemas       = scene_dir / "articulated" / "ransac_joints" / "joint_schemas.json"
    transforms_path = scene_dir / "articulated" / "transforms.json"

    with open(transforms_path) as f:
        transforms = json.load(f)

    schemas = load_joint_schemas(schemas)
    per_frame_data, unit = build_per_frame_lookup(schemas)


    # Number of frames RANSAC actually tracked
    n_ransac_frames = max(per_frame_data.keys()) + 1

    print(f"\n{'='*60}")
    print(f"V2A ARTICULATION MERGE")
    print(f"{'='*60}")
    print(f"Scene:          {scene_dir.name}")
    print(f"Joint type:     {schemas['joint_type']}")
    print(f"Joint axis:     {[round(x, 4) for x in schemas['joint_axis']]}")
    print(f"Joint pivot:    {[round(x, 4) for x in schemas['joint_pivot']]}")
    print(f"RANSAC frames:  {len(per_frame_data)} (indices 0–{n_ransac_frames-1})")
    print(f"Video frames:   {len(transforms['frames'])}")

    # Print angle range
    all_values = list(per_frame_data.values())
    if unit == "rad":
        print(f"Angle range:    {np.degrees(min(all_values)):.1f}° → "
              f"{np.degrees(max(all_values)):.1f}°")
    else:
        print(f"Translation range: {min(all_values):.3f}m → {max(all_values):.3f}m")

    # ── Build output transforms ───────────────────────────────────────
    output = {
        "camera_angle_x": transforms.get("camera_angle_x"),
        "camera_angle_y": transforms.get("camera_angle_y"),
        "fl_x":  transforms["fl_x"],
        "fl_y":  transforms["fl_y"],
        "cx":    transforms["cx"],
        "cy":    transforms["cy"],
        "w":     transforms["w"],
        "h":     transforms["h"],
        "articulations": [{
            "joint_type":  schemas["joint_type"],
            "joint_axis":  schemas["joint_axis"],
            "joint_pivot": schemas["joint_pivot"],
            "joint_limits": [
                float(min(all_values)),
                float(max(all_values)),
            ],
        }],
        "frames": [],
    }

    # ── Process frames ────────────────────────────────────────────────
    angle_stats = []

    for frame_data in transforms["frames"]:
        state = frame_data["state"]
        time  = frame_data["time"]

        # Interpolate from RANSAC per_frame_angles using time
        joint_angle = interpolate_angle(time, per_frame_data, n_ransac_frames)


        file_path = Path(frame_data["file_path"])
        mask_file_path = str(Path("mask") / file_path.name)


        output_frame = {
            "file_path":        str(file_path),
            "mask_file_path":   mask_file_path,
            "depth_file_path":  frame_data.get("depth_file_path"),
            "transform_matrix": frame_data["transform_matrix"],
            "time":             time,
            "state":            state,
            "joint_angle":      round(float(joint_angle), 3),
        }


        # Remove None values
        output_frame = {k: v for k, v in output_frame.items() if v is not None}
        output["frames"].append(output_frame)
        angle_stats.append(joint_angle)

    # ── Save ──────────────────────────────────────────────────────────
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"✓ Merge complete")
    print(f"{'='*60}")
    print(f"Output:         {output_path}")
    print(f"Total frames:   {len(output['frames'])}")

    if unit == "rad":
        print(f"Angle range:    {np.degrees(min(angle_stats)):.1f}° → "
              f"{np.degrees(max(angle_stats)):.1f}°")
        # Sanity check: show 5 sample frames
        print(f"\nSample interpolated angles:")
        n = len(output["frames"])
        for idx in [0, n//4, n//2, 3*n//4, n-1]:
            f = output["frames"][idx]
            print(f"  frame {idx:4d}  t={f['time']:.3f}  "
                  f"angle={np.degrees(f['joint_angle']):.1f}°")
    else:
        print(f"Translation range: {min(angle_stats):.3f}m → {max(angle_stats):.3f}m")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Merge RANSAC joint params into V2A transforms (with interpolation)"
    )
    # parser.add_argument("--joint_schemas",
    #                     help="Path to ransac_joints/joint_schemas.json")
    parser.add_argument("--scene_dir",     required=True,
                        help="Prepared V2A scene directory")
    parser.add_argument("--output",        default=None,
                        help="Output path (default: <scene_dir>/transforms_arti.json)")
    args = parser.parse_args()

    merge_v2a(
        scene_dir=Path(args.scene_dir),
        output_path=Path(args.output) if args.output else None,
    )



if __name__ == "__main__":
    main()