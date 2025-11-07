"""
Merge joint_schemas.json and transforms_post.json into ArtiSplatfacto format.

Usage:
    python merge_arti_data.py \
        --joint_schemas dataset/sync_data/joint_schemas.json \
        --transforms_post dataset/sync_data/multiview/transforms_post.json \
        --output dataset/sync_data/multiview/transforms.json \
        --mask_dir dataset/sync_data/multiview/masks
"""

import argparse
import json
import numpy as np
from pathlib import Path
from typing import Dict, List


def interpolate_joint_angle(time: float, per_frame_data: Dict[int, float], 
                            n_temporal_steps: int) -> float:
    """
    Interpolate joint angle/translation for a given normalized time [0,1].
    
    Args:
        time: Normalized time in [0,1]
        per_frame_data: Dict mapping frame_idx to angle/translation
        n_temporal_steps: Total number of temporal steps in video sequence
        
    Returns:
        Interpolated joint angle/translation
    """
    if not per_frame_data:
        return 0.0
    
    # Map normalized time to frame index
    frame_float = time * (n_temporal_steps - 1)
    frame_idx = int(np.round(frame_float))
    
    # Clamp to valid range
    frame_idx = max(0, min(frame_idx, n_temporal_steps - 1))
    
    # Get value at this frame (or interpolate if needed)
    if frame_idx in per_frame_data:
        return per_frame_data[frame_idx]
    
    # Linear interpolation between nearest frames
    sorted_frames = sorted(per_frame_data.keys())
    
    # Find surrounding frames
    lower_frame = max([f for f in sorted_frames if f <= frame_idx], default=sorted_frames[0])
    upper_frame = min([f for f in sorted_frames if f >= frame_idx], default=sorted_frames[-1])
    
    if lower_frame == upper_frame:
        return per_frame_data[lower_frame]
    
    # Linear interpolation
    alpha = (frame_idx - lower_frame) / (upper_frame - lower_frame)
    return per_frame_data[lower_frame] * (1 - alpha) + per_frame_data[upper_frame] * alpha


def merge_transforms(joint_schemas_path: str, transforms_post_path: str, 
                     output_path: str, mask_dir: bool) -> None:
    """
    Merge joint schemas and multi-view transforms into ArtiSplatfacto format.
    
    Args:
        joint_schemas_path: Path to joint_schemas.json from 4D RANSAC
        transforms_post_path: Path to transforms_post.json from multi-view capture
        output_path: Path to save merged transforms.json
        mask_dir: Optional directory containing mask files
    """
    # Load input files
    with open(joint_schemas_path, 'r') as f:
        joint_schemas = json.load(f)
    
    with open(transforms_post_path, 'r') as f:
        transforms_post = json.load(f)
    
    print(f"Loaded {len(joint_schemas)} joint schemas")
    print(f"Loaded {len(transforms_post['frames'])} multi-view frames")
    
    output = {
        "camera_model": transforms_post.get("camera_model", "PINHOLE"),
        "fl_x": transforms_post["fl_x"],
        "fl_y": transforms_post["fl_y"],
        "cx": transforms_post["cx"],
        "cy": transforms_post["cy"],
        "w": transforms_post["w"],
        "h": transforms_post["h"],
        "k1": transforms_post.get("k1", 0.0),
        "k2": transforms_post.get("k2", 0.0),
        "p1": transforms_post.get("p1", 0.0),
        "p2": transforms_post.get("p2", 0.0),
    }
    
    if "ply_file_path" in transforms_post:
        output["ply_file_path"] = transforms_post["ply_file_path"]
    
    # Build articulation block from joint schemas
    articulations = []
    for joint in joint_schemas:
        joint_type = joint["joint_type"]
        
        articulation = {
            "joint_type": joint_type,
            "joint_axis": joint["joint_axis"],
            "joint_pivot": joint["joint_pivot"],
        }
        
        # Add limits if present
        if "joint_limits" in joint and joint["joint_limits"][0] is not None:
            articulation["joint_limits"] = joint["joint_limits"]
        
        # Extract per-frame data for interpolation
        if joint_type == "revolute":
            per_frame_key = "per_frame_angles"
        else:
            per_frame_key = "per_frame_translations"

        per_frame_data = joint.get(per_frame_key, {})

        # Convert string keys to int
        per_frame_data = {int(k): v for k, v in per_frame_data.items()}

        # If revolute, convert from degrees → radians and round
        if joint_type == "revolute":
            per_frame_data = {k: round(np.radians(v), 4) for k, v in per_frame_data.items()}

        articulation["_per_frame_data"] = per_frame_data  # Store for interpolation
        articulations.append(articulation)

    
    output["articulations"] = articulations
    
    # Determine number of temporal steps from per-frame data
    n_temporal_steps = 0
    for art in articulations:
        if art["_per_frame_data"]:
            n_temporal_steps = max(n_temporal_steps, max(art["_per_frame_data"].keys()) + 1)
    
    if n_temporal_steps == 0:
        print("Warning: No per-frame data found, using default temporal steps")
        n_temporal_steps = 10
    
    print(f"Detected {n_temporal_steps} temporal steps from joint schemas")
    
    # Process frames
    output_frames = []
    n_frames = len(transforms_post["frames"])

    for i, frame_data in enumerate(transforms_post["frames"]):
        # Copy base frame data
        output_frame = {
            "file_path": frame_data["file_path"],
            "transform_matrix": frame_data["transform_matrix"],
        }

        if "depth_file_path" in frame_data:
            output_frame["depth_file_path"] = frame_data["depth_file_path"]

        # Derive normalized time based on frame index (if no explicit "time")
        time = frame_data.get("time", i / (n_frames - 1) if n_frames > 1 else 0.0)
        output_frame["time"] = time

        if "joint_angle" in frame_data:
            output_frame["joint_angle_gt"] = frame_data["joint_angle"]

        # Interpolate joint angles
        joint_angles = []
        for articulation in articulations:
            per_frame_data = articulation["_per_frame_data"]
            interpolated_value = interpolate_joint_angle(time, per_frame_data, n_temporal_steps)
            interpolated_value = round(interpolated_value, 3)
            joint_angles.append(interpolated_value)

        output_frame["joint_angle"] = joint_angles[0] if len(joint_angles) == 1 else joint_angles

        if mask_dir:
            frame_name = Path(frame_data["file_path"]).name
            frame_name = Path(frame_name).with_suffix(".png")
            output_frame["mask_file_path"] = str(Path("mask") / frame_name)

        output_frames.append(output_frame)

    
    output["frames"] = output_frames
    
    # Remove temporary per-frame data from articulations
    for art in output["articulations"]:
        art.pop("_per_frame_data", None)
    
    # Save merged output
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n=== MERGE COMPLETE ===")
    print(f"Output: {output_path}")
    print(f"Frames: {len(output_frames)}")
    print(f"Articulations: {len(articulations)}")
    print(f"Time range: [{min(f['time'] for f in output_frames):.3f}, {max(f['time'] for f in output_frames):.3f}]")
    
    # Print joint angle statistics
    for i, art in enumerate(articulations):
        angles = [f["joint_angle"] if isinstance(f["joint_angle"], float) else f["joint_angle"][i] 
                  for f in output_frames]
        print(f"\nJoint {i} ({art['joint_type']}):")
        print(f"  Range: [{min(angles):.4f}, {max(angles):.4f}]")
        if art["joint_type"] == "revolute":
            print(f"  Range (degrees): [{np.degrees(min(angles)):.1f}°, {np.degrees(max(angles)):.1f}°]")
        else:
            print(f"  Range (meters): [{min(angles):.3f}m, {max(angles):.3f}m]")


def main():
    parser = argparse.ArgumentParser(
        description="Merge joint schemas and multi-view transforms for ArtiSplatfacto"
    )
    
    parser.add_argument("--joint_schemas", type=str, required=True,
                       help="Path to joint_schemas.json from 4D RANSAC")
    parser.add_argument("--transforms_post", type=str, required=True,
                       help="Path to transforms_post.json from multi-view capture")
    parser.add_argument("--output", type=str, required=True,
                       help="Path to save merged transforms.json")
    parser.add_argument(
        "--use_masks",
        action="store_true",
        help="If set, include masks alongside frames"
    )


    args = parser.parse_args()
    
    # Validate inputs
    if not Path(args.joint_schemas).exists():
        print(f"Error: Joint schemas not found: {args.joint_schemas}")
        return 1
    
    if not Path(args.transforms_post).exists():
        print(f"Error: Transforms not found: {args.transforms_post}")
        return 1
    
    # Create output directory
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    # Merge transforms
    merge_transforms(
        joint_schemas_path=args.joint_schemas,
        transforms_post_path=args.transforms_post,
        output_path=args.output,
        mask_dir=args.use_masks
    )
    
    return 0


if __name__ == "__main__":
    exit(main())