#!/usr/bin/env python3
"""
Step 6: Merge articulation data — writes transforms_post.json.
Runs inside nerfstudio container.

Reads from:
  <joint_dir>/transforms.json              <- camera poses + intrinsics
  <joint_dir>/ransac_joints/joint_schemas.json  <- RANSAC joint params
  <joint_dir>/depth/                       <- depth files (.png or .npy)
  <joint_dir>/mask/                        <- SAM2 masks

Writes:
  <joint_dir>/transforms_post.json         <- final merged output

Output format:
  {
    "camera_model": "OPENCV",
    "fl_x": ..., "fl_y": ..., "cx": ..., "cy": ..., "w": ..., "h": ...,
    "k1": ..., "k2": ..., "p1": ..., "p2": ...,
    "articulations": [{
      "joint_type":   "revolute" | "prismatic",
      "joint_axis":   [ax, ay, az],
      "joint_pivot":  [px, py, pz],
      "joint_limits": [min, max]
    }],
    "frames": [{
      "file_path":        "frames/frame_00001.jpg",
      "transform_matrix": [[...]],
      "depth_file_path":  "depth/frame_00001.png",   // .png or .npy
      "mask_file_path":   "mask/frame_00001.png",
      "time":             0.0,
      "joint_angle":      0.0
    }, ...]
  }

Usage:
    python -m arti4d.run_merge_arti_data \
        --joint_dir /workspace/data/.../articulated_joint_0
"""

import json
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, Tuple


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_joint_schemas(path: Path) -> Dict:
    with open(path) as f:
        schemas = json.load(f)
    if isinstance(schemas, list):
        schemas = schemas[0]
    required = ["joint_type", "joint_axis", "joint_pivot"]
    missing  = [k for k in required if k not in schemas]
    if missing:
        raise ValueError(f"joint_schemas.json missing fields: {missing}")
    return schemas


def build_per_frame_lookup(schemas: Dict) -> Tuple[Dict[int, float], str]:
    """
    Extract per-frame angle or translation from schemas.
    Revolute:  degrees → radians
    Prismatic: metres (unchanged)
    Returns (per_frame_data, unit)
    """
    joint_type = schemas["joint_type"]
    if joint_type == "revolute":
        raw = schemas.get("per_frame_angles", {})
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


def interpolate_value(
    time: float,
    per_frame_data: Dict[int, float],
    n_ransac_frames: int,
) -> float:
    frame_float   = time * (n_ransac_frames - 1)
    sorted_frames = sorted(per_frame_data.keys())
    min_frame     = sorted_frames[0]
    max_frame     = sorted_frames[-1]

    if frame_float <= min_frame:
        return per_frame_data[min_frame]
    if frame_float >= max_frame:
        return per_frame_data[max_frame]

    lower = max(f for f in sorted_frames if f <= frame_float)
    upper = min(f for f in sorted_frames if f >= frame_float)
    if lower == upper:
        return per_frame_data[lower]

    alpha = (frame_float - lower) / (upper - lower)
    return per_frame_data[lower] * (1 - alpha) + per_frame_data[upper] * alpha


def find_depth_file(depth_dir: Path, frame_stem: str) -> str | None:
    """
    Find the depth file for a given frame stem, accepting .png or .npy.
    frame_stem: e.g. 'frame_00001'
    Returns relative path string like 'depth/frame_00001.png', or None.
    """
    for ext in (".png", ".npy"):
        candidate = depth_dir / f"{frame_stem}{ext}"
        if candidate.exists():
            return f"depth/{frame_stem}{ext}"
    return None


def find_mask_file(mask_dir: Path, frame_stem: str) -> str | None:
    """Find mask file for a given frame stem."""
    candidate = mask_dir / f"{frame_stem}.png"
    if candidate.exists():
        return f"mask/{frame_stem}.png"
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Core
# ──────────────────────────────────────────────────────────────────────────────

def merge_arti_data(joint_dir: Path) -> Path:
    joint_dir = Path(joint_dir)

    transforms_path = joint_dir / "transforms.json"
    schemas_path    = joint_dir / "ransac_joints" / "joint_schemas.json"
    depth_dir       = joint_dir / "depth"
    mask_dir        = joint_dir / "mask"
    output_path     = joint_dir / "transforms_post.json"

    # ── Validate inputs ───────────────────────────────────────────────────────
    missing = []
    if not transforms_path.exists():
        missing.append(str(transforms_path))
    if not schemas_path.exists():
        missing.append(str(schemas_path))
    if missing:
        raise FileNotFoundError(
            "Missing required inputs:\n" +
            "\n".join(f"  ✗ {p}" for p in missing)
        )

    # ── Load ──────────────────────────────────────────────────────────────────
    with open(transforms_path) as f:
        transforms = json.load(f)

    schemas          = load_joint_schemas(schemas_path)
    per_frame_data, unit = build_per_frame_lookup(schemas)
    n_ransac_frames  = max(per_frame_data.keys()) + 1

    all_values = list(per_frame_data.values())

    print(f"\n{'='*60}")
    print(f"STEP 6: MERGE ARTICULATION DATA")
    print(f"{'='*60}")
    print(f"  Joint dir:      {joint_dir}")
    print(f"  Joint type:     {schemas['joint_type']}")
    print(f"  Joint axis:     {[round(x, 4) for x in schemas['joint_axis']]}")
    print(f"  Joint pivot:    {[round(x, 4) for x in schemas['joint_pivot']]}")
    print(f"  RANSAC frames:  {len(per_frame_data)}  (0–{n_ransac_frames-1})")
    print(f"  Video frames:   {len(transforms['frames'])}")
    if unit == "rad":
        print(f"  Angle range:    {np.degrees(min(all_values)):.1f}° → "
              f"{np.degrees(max(all_values)):.1f}°")
    else:
        print(f"  Transl. range:  {min(all_values):.3f}m → {max(all_values):.3f}m")

    # ── Build output ──────────────────────────────────────────────────────────
    # Carry over all intrinsic fields from transforms.json
    intrinsic_keys = ["camera_model", "fl_x", "fl_y", "cx", "cy", "w", "h",
                      "k1", "k2", "k3", "p1", "p2", "k4", "k5", "k6"]
    output = {k: transforms[k] for k in intrinsic_keys if k in transforms}

    output["articulations"] = [{
        "joint_type":   schemas["joint_type"],
        "joint_axis":   schemas["joint_axis"],
        "joint_pivot":  schemas["joint_pivot"],
        "joint_limits": [float(min(all_values)), float(max(all_values))],
    }]
    output["frames"] = []

    n_frames = len(transforms["frames"])
    value_stats = []

    for i, frame_data in enumerate(transforms["frames"]):
        # Normalised time [0, 1] across the articulated sequence
        time = i / max(n_frames - 1, 1)

        joint_value = interpolate_value(time, per_frame_data, n_ransac_frames)
        value_stats.append(joint_value)

        # Derive frame stem from file_path, e.g. "frames/frame_00001.jpg" → "frame_00001"
        file_path  = Path(frame_data["file_path"])
        frame_stem = file_path.stem   # e.g. "frame_00001"

        depth_rel = find_depth_file(depth_dir, frame_stem)
        mask_rel  = find_mask_file(mask_dir, frame_stem)

        out_frame = {
            "file_path":        str(file_path),
            "transform_matrix": frame_data["transform_matrix"],
            "time":             round(time, 6),
            "joint_angle":      round(float(joint_value), 6),
        }
        if depth_rel:
            out_frame["depth_file_path"] = depth_rel
        if mask_rel:
            out_frame["mask_file_path"] = mask_rel

        output["frames"].append(out_frame)

    # ── Save ──────────────────────────────────────────────────────────────────
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    # ── Summary ───────────────────────────────────────────────────────────────
    n_with_depth = sum(1 for fr in output["frames"] if "depth_file_path" in fr)
    n_with_mask  = sum(1 for fr in output["frames"] if "mask_file_path" in fr)

    print(f"\n✓ Merge complete → {output_path}")
    print(f"  Total frames:   {len(output['frames'])}")
    print(f"  With depth:     {n_with_depth}")
    print(f"  With mask:      {n_with_mask}")

    if unit == "rad":
        print(f"  Angle range:    {np.degrees(min(value_stats)):.1f}° → "
              f"{np.degrees(max(value_stats)):.1f}°")
        print(f"\n  Sample interpolated angles:")
        n = len(output["frames"])
        for idx in [0, n // 4, n // 2, 3 * n // 4, n - 1]:
            fr = output["frames"][idx]
            print(f"    frame {idx:4d}  t={fr['time']:.3f}  "
                  f"angle={np.degrees(fr['joint_angle']):.1f}°")
    else:
        print(f"  Transl. range:  {min(value_stats):.3f}m → {max(value_stats):.3f}m")

    return output_path


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Step 6: Merge RANSAC joint params → transforms_post.json"
    )
    parser.add_argument("--joint_dir", required=True,
                        help="Path to articulated_joint_<N>/ directory (container path)")
    args = parser.parse_args()

    merge_arti_data(joint_dir=Path(args.joint_dir))


if __name__ == "__main__":
    main()