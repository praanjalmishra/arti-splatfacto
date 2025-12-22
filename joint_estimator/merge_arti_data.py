"""
Merge joint_schemas.json and transforms_post.json into ArtiSplatfacto format.

Improvements:
- Robust linear interpolation handling sparse temporal sampling
- Better edge case handling (extrapolation, single frame, etc.)
- Support for multiple interpolation methods
- Validation and warnings for data quality issues

Usage:
    python merge_arti_data.py \
        --joint_schemas dataset/sync_data/joint_schemas.json \
        --transforms_post dataset/sync_data/multiview/transforms_post.json \
        --output dataset/sync_data/multiview/transforms.json \
        --mask_dir dataset/sync_data/multiview/mask
"""

import argparse
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings


def interpolate_joint_value(
    time: float,
    per_frame_data: Dict[int, float],
    n_temporal_steps: int,
    method: str = 'linear',
    extrapolate: bool = False
) -> float:
    """
    Robust interpolation of joint angle/translation for a given normalized time.
    
    Args:
        time: Normalized time in [0, 1]
        per_frame_data: Dict mapping frame_idx to angle/translation
        n_temporal_steps: Total number of temporal steps in video sequence
        method: Interpolation method ('linear', 'nearest', 'cubic')
        extrapolate: Whether to extrapolate outside the measured range
        
    Returns:
        Interpolated joint value
        
    Edge Cases Handled:
        - Empty per_frame_data → returns 0.0
        - Single frame → returns that frame's value
        - Time outside measured range → clamps or extrapolates based on flag
        - Sparse sampling → interpolates between nearest frames
    """
    # Handle empty data
    if not per_frame_data:
        return 0.0
    
    # Handle single frame
    if len(per_frame_data) == 1:
        return list(per_frame_data.values())[0]
    
    # Map normalized time [0,1] to frame index [0, n_temporal_steps-1]
    frame_float = time * (n_temporal_steps - 1)
    
    # Get sorted frame indices and values
    sorted_frames = sorted(per_frame_data.keys())
    min_frame = sorted_frames[0]
    max_frame = sorted_frames[-1]
    
    # Handle exact match
    frame_idx_rounded = int(np.round(frame_float))
    if frame_idx_rounded in per_frame_data:
        return per_frame_data[frame_idx_rounded]
    
    # Handle extrapolation (before first frame or after last frame)
    if frame_float < min_frame:
        if extrapolate and len(sorted_frames) >= 2:
            # Linear extrapolation using first two points
            f0, f1 = sorted_frames[0], sorted_frames[1]
            v0, v1 = per_frame_data[f0], per_frame_data[f1]
            slope = (v1 - v0) / (f1 - f0)
            return v0 + slope * (frame_float - f0)
        else:
            # Clamp to first frame
            return per_frame_data[min_frame]
    
    if frame_float > max_frame:
        if extrapolate and len(sorted_frames) >= 2:
            # Linear extrapolation using last two points
            f0, f1 = sorted_frames[-2], sorted_frames[-1]
            v0, v1 = per_frame_data[f0], per_frame_data[f1]
            slope = (v1 - v0) / (f1 - f0)
            return v1 + slope * (frame_float - f1)
        else:
            # Clamp to last frame
            return per_frame_data[max_frame]
    
    # Interpolation between two frames
    if method == 'nearest':
        # Nearest neighbor
        nearest_frame = min(sorted_frames, key=lambda f: abs(f - frame_float))
        return per_frame_data[nearest_frame]
    
    elif method == 'linear':
        # Linear interpolation
        # Find surrounding frames
        lower_frame = max([f for f in sorted_frames if f <= frame_float])
        upper_frame = min([f for f in sorted_frames if f >= frame_float])
        
        if lower_frame == upper_frame:
            return per_frame_data[lower_frame]
        
        # Linear interpolation
        alpha = (frame_float - lower_frame) / (upper_frame - lower_frame)
        return per_frame_data[lower_frame] * (1 - alpha) + per_frame_data[upper_frame] * alpha
    
    elif method == 'cubic':
        # Cubic spline interpolation
        from scipy.interpolate import interp1d
        frames = np.array(sorted_frames, dtype=float)
        values = np.array([per_frame_data[f] for f in sorted_frames])
        
        # Need at least 4 points for cubic, fall back to linear otherwise
        if len(frames) >= 4:
            f = interp1d(frames, values, kind='cubic', fill_value='extrapolate' if extrapolate else (values[0], values[-1]))
        else:
            f = interp1d(frames, values, kind='linear', fill_value='extrapolate' if extrapolate else (values[0], values[-1]))
        
        return float(f(frame_float))
    
    else:
        raise ValueError(f"Unknown interpolation method: {method}")


def validate_joint_schemas(joint_schemas: List[Dict]) -> Tuple[bool, List[str]]:
    """
    Validate joint schemas for common issues.
    
    Returns:
        (is_valid, warnings_list)
    """
    warnings_list = []
    
    for i, joint in enumerate(joint_schemas):
        # Check required fields
        if "joint_type" not in joint:
            warnings_list.append(f"Joint {i}: Missing 'joint_type' field")
        
        if "joint_axis" not in joint or len(joint["joint_axis"]) != 3:
            warnings_list.append(f"Joint {i}: Invalid 'joint_axis' field")
        
        if "joint_pivot" not in joint or len(joint["joint_pivot"]) != 3:
            warnings_list.append(f"Joint {i}: Invalid 'joint_pivot' field")
        
        # Check per-frame data
        joint_type = joint.get("joint_type")
        if joint_type == "revolute":
            per_frame_key = "per_frame_angles"
        elif joint_type == "prismatic":
            per_frame_key = "per_frame_translations"
        else:
            warnings_list.append(f"Joint {i}: Unknown joint type '{joint_type}'")
            continue
        
        per_frame_data = joint.get(per_frame_key, {})
        
        if not per_frame_data:
            warnings_list.append(f"Joint {i}: No per-frame data found")
        elif len(per_frame_data) == 1:
            warnings_list.append(f"Joint {i}: Only 1 frame measured (no motion)")
        
        # Check for sparse sampling
        if per_frame_data:
            frames = sorted([int(k) for k in per_frame_data.keys()])
            frame_range = frames[-1] - frames[0] + 1
            if len(frames) < frame_range:
                sparsity = len(frames) / frame_range
                warnings_list.append(
                    f"Joint {i}: Sparse sampling detected "
                    f"({len(frames)}/{frame_range} = {sparsity:.1%} frames measured)"
                )
        
        # Check metadata if available
        metadata = joint.get("metadata", {})
        if metadata.get("is_sparse_sampling", False):
            measured_frames = metadata.get("measured_frames", [])
            if measured_frames:
                warnings_list.append(
                    f"Joint {i}: Sparse sampling confirmed by metadata "
                    f"({len(measured_frames)} frames measured)"
                )
    
    is_valid = len([w for w in warnings_list if "Missing" in w or "Invalid" in w]) == 0
    return is_valid, warnings_list


def merge_transforms(
    joint_schemas_path: str,
    transforms_post_path: str,
    output_path: str,
    mask_dir: bool,
    interpolation_method: str = 'linear',
    extrapolate: bool = False
) -> None:
    """
    Merge joint schemas and multi-view transforms into ArtiSplatfacto format.
    
    Args:
        joint_schemas_path: Path to joint_schemas.json from 4D RANSAC
        transforms_post_path: Path to transforms_post.json from multi-view capture
        output_path: Path to save merged transforms.json
        mask_dir: Optional directory containing mask files
        interpolation_method: 'linear', 'nearest', or 'cubic'
        extrapolate: Whether to extrapolate beyond measured frames
    """
    # Load input files
    with open(joint_schemas_path, 'r') as f:
        joint_schemas = json.load(f)
    
    with open(transforms_post_path, 'r') as f:
        transforms_post = json.load(f)
    
    print(f"\n{'='*60}")
    print(f"MERGING ARTICULATION DATA")
    print(f"{'='*60}")
    print(f"Loaded {len(joint_schemas)} joint schemas")
    print(f"Loaded {len(transforms_post['frames'])} multi-view frames")
    
    # Validate joint schemas
    is_valid, warnings_list = validate_joint_schemas(joint_schemas)
    
    if warnings_list:
        print(f"\n⚠️  {len(warnings_list)} validation warnings:")
        for warning in warnings_list[:10]:  # Show first 10
            print(f"   - {warning}")
        if len(warnings_list) > 10:
            print(f"   ... and {len(warnings_list) - 10} more")
    
    if not is_valid:
        print("\n❌ Critical validation errors found. Please fix joint_schemas.json")
        return
    
    # Build output structure
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
    print("\nProcessing joint schemas...")
    articulations = []
    per_frame_data_list = []  # Store for interpolation
    
    for i, joint in enumerate(joint_schemas):
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
            unit = "degrees"
        else:
            per_frame_key = "per_frame_translations"
            unit = "meters"
        
        per_frame_data_raw = joint.get(per_frame_key, {})
        
        # Convert string keys to int
        per_frame_data = {int(k): float(v) for k, v in per_frame_data_raw.items()}
        
        # For revolute joints: convert degrees → radians
        if joint_type == "revolute":
            per_frame_data = {k: np.radians(v) for k, v in per_frame_data.items()}
        
        per_frame_data_list.append(per_frame_data)
        
        # Print joint info
        if per_frame_data:
            frames = sorted(per_frame_data.keys())
            values = [per_frame_data[f] for f in frames]
            print(f"\n  Joint {i} ({joint_type}):")
            print(f"    Measured frames: {frames[0]} to {frames[-1]} ({len(frames)} frames)")
            if joint_type == "revolute":
                print(f"    Value range: {np.degrees(min(values)):.1f}° to {np.degrees(max(values)):.1f}°")
            else:
                print(f"    Value range: {min(values):.3f}m to {max(values):.3f}m")
        
        articulations.append(articulation)
    
    output["articulations"] = articulations
    
    # Determine number of temporal steps from per-frame data
    n_temporal_steps = 0
    for per_frame_data in per_frame_data_list:
        if per_frame_data:
            n_temporal_steps = max(n_temporal_steps, max(per_frame_data.keys()) + 1)
    
    if n_temporal_steps == 0:
        print("\n⚠️  Warning: No per-frame data found, using default temporal steps")
        n_temporal_steps = len(transforms_post["frames"])
    
    print(f"\nDetected {n_temporal_steps} temporal steps")
    print(f"Interpolation method: {interpolation_method}")
    print(f"Extrapolation: {'enabled' if extrapolate else 'disabled'}")
    
    # Process frames
    print("\nInterpolating joint values for all frames...")
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
        
        # Preserve ground truth if available
        if "joint_angle" in frame_data:
            output_frame["joint_angle_gt"] = frame_data["joint_angle"]
        
        # Interpolate joint values for each articulation
        joint_values = []
        for j, per_frame_data in enumerate(per_frame_data_list):
            interpolated_value = interpolate_joint_value(
                time=time,
                per_frame_data=per_frame_data,
                n_temporal_steps=n_temporal_steps,
                method=interpolation_method,
                extrapolate=extrapolate
            )
            # Round to reasonable precision
            interpolated_value = round(interpolated_value, 4)
            joint_values.append(interpolated_value)
        
        # Store joint angles (single value or array)
        if len(joint_values) == 1:
            output_frame["joint_angle"] = joint_values[0]
        else:
            output_frame["joint_angle"] = joint_values
        
        # Add mask path if requested
        if mask_dir:
            frame_name = Path(frame_data["file_path"]).name
            frame_name = Path(frame_name).with_suffix(".png")
            output_frame["mask_file_path"] = str(Path("mask") / frame_name)
        
        output_frames.append(output_frame)
    
    output["frames"] = output_frames
    
    # Save merged output
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"✅ MERGE COMPLETE")
    print(f"{'='*60}")
    print(f"Output: {output_path}")
    print(f"Frames: {len(output_frames)}")
    print(f"Articulations: {len(articulations)}")
    print(f"Time range: [{min(f['time'] for f in output_frames):.3f}, {max(f['time'] for f in output_frames):.3f}]")
    
    # Print joint value statistics
    for i, art in enumerate(articulations):
        if len(joint_values) == 1:
            values = [f["joint_angle"] for f in output_frames]
        else:
            values = [f["joint_angle"][i] for f in output_frames]
        
        print(f"\nJoint {i} ({art['joint_type']}) - Interpolated Values:")
        print(f"  Range: [{min(values):.4f}, {max(values):.4f}]")
        
        if art["joint_type"] == "revolute":
            print(f"  Range (degrees): [{np.degrees(min(values)):.1f}°, {np.degrees(max(values)):.1f}°]")
        else:
            print(f"  Range (meters): [{min(values):.3f}m, {max(values):.3f}m]")
        
        # Show some sample interpolations
        sample_indices = [0, len(output_frames)//4, len(output_frames)//2, 
                         3*len(output_frames)//4, len(output_frames)-1]
        print(f"  Sample values at t=[0, 0.25, 0.5, 0.75, 1.0]:")
        for idx in sample_indices:
            if idx < len(output_frames):
                val = values[idx]
                t = output_frames[idx]["time"]
                if art["joint_type"] == "revolute":
                    print(f"    t={t:.3f}: {val:.4f} rad ({np.degrees(val):.1f}°)")
                else:
                    print(f"    t={t:.3f}: {val:.4f} m")


def main():
    parser = argparse.ArgumentParser(
        description="Merge joint schemas and multi-view transforms for ArtiSplatfacto",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic merge with linear interpolation
  python merge_arti_data.py \\
      --joint_schemas joint_schemas.json \\
      --transforms_post transforms_post.json \\
      --output transforms.json
  
  # With cubic interpolation and masks
  python merge_arti_data.py \\
      --joint_schemas joint_schemas.json \\
      --transforms_post transforms_post.json \\
      --output transforms.json \\
      --interpolation cubic \\
      --use_masks
  
  # With extrapolation beyond measured frames
  python merge_arti_data.py \\
      --joint_schemas joint_schemas.json \\
      --transforms_post transforms_post.json \\
      --output transforms.json \\
      --extrapolate
        """
    )
    
    parser.add_argument("--joint_schemas", type=str, required=True,
                       help="Path to joint_schemas.json from 4D RANSAC")
    parser.add_argument("--transforms_post", type=str, required=True,
                       help="Path to transforms_post.json from multi-view capture")
    parser.add_argument("--output", type=str, required=True,
                       help="Path to save merged transforms.json")
    parser.add_argument("--use_masks", action="store_true",
                       help="If set, include mask paths in output")
    parser.add_argument("--interpolation", type=str, default="linear",
                       choices=["linear", "nearest", "cubic"],
                       help="Interpolation method for filling frames (default: linear)")
    parser.add_argument("--extrapolate", action="store_true",
                       help="Extrapolate beyond measured frame range (default: clamp)")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not Path(args.joint_schemas).exists():
        print(f"❌ Error: Joint schemas not found: {args.joint_schemas}")
        return 1
    
    if not Path(args.transforms_post).exists():
        print(f"❌ Error: Transforms not found: {args.transforms_post}")
        return 1
    
    # Merge transforms
    merge_transforms(
        joint_schemas_path=args.joint_schemas,
        transforms_post_path=args.transforms_post,
        output_path=args.output,
        mask_dir=args.use_masks,
        interpolation_method=args.interpolation,
        extrapolate=args.extrapolate
    )
    
    return 0


if __name__ == "__main__":
    exit(main())