"""
Merge articulation priors from 4D RANSAC with multi-view transforms.

Takes:
  - joint_schemas.json (from RANSAC: axis, pivot, per-frame θ)
  - multiview/transforms_post.json (camera poses + timestamps)
  
Outputs:
  - multiview/transforms_articulated.json (merged format for training)

Usage:
    python merge_articulation_data.py \
        --joint_schemas output/joint_schemas.json \
        --multiview_transforms dataset/sync_data/multiview/transforms_post.json \
        --output dataset/sync_data/multiview/transforms_articulated.json
"""

import json
import argparse
from pathlib import Path
import numpy as np


def load_joint_schemas(path: str) -> dict:
    """Load joint parameters from RANSAC output."""
    with open(path, 'r') as f:
        data = json.load(f)
    
    if isinstance(data, list) and len(data) > 0:
        return data[0]  # Take first joint
    return data


def load_transforms(path: str) -> dict:
    """Load camera transforms JSON."""
    with open(path, 'r') as f:
        return json.load(f)


def interpolate_articulation(time_val: float, per_frame_data: dict) -> float:
    """
    Interpolate articulation value for a given timestamp.
    
    Args:
        time_val: Normalized time in [0, 1]
        per_frame_data: Dict with 'normalized' frame_idx -> theta mapping
    
    Returns:
        Interpolated theta value
    """
    normalized = per_frame_data['normalized']
    frame_indices = sorted([int(k) for k in normalized.keys()])
    
    if not frame_indices:
        return 0.0
    
    # Map time [0,1] to frame index
    max_frame = frame_indices[-1]
    target_frame = time_val * max_frame
    
    # Find surrounding frames
    lower_idx = int(np.floor(target_frame))
    upper_idx = int(np.ceil(target_frame))
    
    # Clamp to available frames
    lower_idx = max(min(lower_idx, max_frame), 0)
    upper_idx = max(min(upper_idx, max_frame), 0)
    
    # Get theta values
    theta_lower = normalized.get(str(lower_idx), 0.0)
    theta_upper = normalized.get(str(upper_idx), 0.0)
    
    # Linear interpolation
    if lower_idx == upper_idx:
        return theta_lower
    
    alpha = target_frame - lower_idx
    return (1 - alpha) * theta_lower + alpha * theta_upper


def create_articulated_transforms(joint_data: dict, 
                                 multiview_data: dict,
                                 mask_dir: str = None) -> dict:
    """
    Merge joint parameters with multi-view transforms.
    
    Args:
        joint_data: Joint parameters from RANSAC
        multiview_data: Multi-view camera transforms
        mask_dir: Optional directory for mask files
    
    Returns:
        Merged transform data in articulated format
    """
    # Extract articulation metadata
    articulation_metadata = {
        "joint_type": joint_data["joint_type"],
        "joint_axis": joint_data["joint_axis"],
        "joint_pivot": joint_data["joint_pivot"],
        "joint_limits": joint_data["joint_limits"]
    }
    
    # Check if per-frame articulation exists
    has_per_frame = "per_frame_articulation" in joint_data
    per_frame_data = joint_data.get("per_frame_articulation", {})
    
    # Process each frame
    articulated_frames = []
    
    for frame in multiview_data["frames"]:
        time_val = frame.get("time", 0.0)
        
        # Get or interpolate articulation value
        if has_per_frame and per_frame_data:
            theta = interpolate_articulation(time_val, per_frame_data)
        else:
            # Fallback: use joint_angle if present
            theta = frame.get("joint_angle", 0.0)
        
        # Create new frame entry
        new_frame = {
            "file_path": frame["file_path"],
            "transform_matrix": frame["transform_matrix"],
            "articulation_value": round(theta, 6)
        }
        
        # Add depth if present
        if "depth_file_path" in frame:
            new_frame["depth_file_path"] = frame["depth_file_path"]
        
        # Add mask path if directory provided
        if mask_dir:
            # Assume mask has same name as RGB
            rgb_name = Path(frame["file_path"]).name
            mask_name = rgb_name.replace(".png", "_mask.png")
            new_frame["mask_file_path"] = f"{mask_dir}/{mask_name}"
        
        articulated_frames.append(new_frame)
    
    # Build output structure
    output = {
        "camera_model": multiview_data.get("camera_model", "OPENCV"),
        "fl_x": multiview_data["fl_x"],
        "fl_y": multiview_data["fl_y"],
        "cx": multiview_data["cx"],
        "cy": multiview_data["cy"],
        "w": multiview_data["w"],
        "h": multiview_data["h"],
        "k1": multiview_data.get("k1", 0.0),
        "k2": multiview_data.get("k2", 0.0),
        "p1": multiview_data.get("p1", 0.0),
        "p2": multiview_data.get("p2", 0.0),
        "articulation": articulation_metadata,
        "frames": articulated_frames
    }
    
    # Add PLY path if present
    if "ply_file_path" in multiview_data:
        output["ply_file_path"] = multiview_data["ply_file_path"]
    
    return output


def print_summary(output_data: dict):
    """Print summary of merged data."""
    print("\n" + "="*60)
    print("ARTICULATED TRANSFORMS SUMMARY")
    print("="*60)
    
    articulation = output_data["articulation"]
    print(f"Joint Type: {articulation['joint_type']}")
    print(f"Joint Axis: {articulation['joint_axis']}")
    print(f"Joint Pivot: {articulation['joint_pivot']}")
    print(f"Joint Limits: {articulation['joint_limits']}")
    
    print(f"\nTotal Frames: {len(output_data['frames'])}")
    
    # Articulation value statistics
    theta_values = [f["articulation_value"] for f in output_data["frames"]]
    print(f"Articulation Range: [{min(theta_values):.3f}, {max(theta_values):.3f}]")
    print(f"Unique Values: {len(set(theta_values))}")
    
    # Check for masks
    has_masks = any("mask_file_path" in f for f in output_data["frames"])
    print(f"Mask Files: {'Yes' if has_masks else 'No'}")
    
    # Check for depth
    has_depth = any("depth_file_path" in f for f in output_data["frames"])
    print(f"Depth Files: {'Yes' if has_depth else 'No'}")
    
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Merge articulation priors with multi-view transforms"
    )
    
    parser.add_argument(
        "--joint_schemas",
        type=str,
        required=True,
        help="Path to joint_schemas.json from RANSAC"
    )
    
    parser.add_argument(
        "--multiview_transforms",
        type=str,
        required=True,
        help="Path to multiview transforms_post.json"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for merged transforms_articulated.json"
    )
    
    parser.add_argument(
        "--mask_dir",
        type=str,
        default=None,
        help="Directory containing mask files (optional)"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not Path(args.joint_schemas).exists():
        print(f"Error: Joint schemas not found: {args.joint_schemas}")
        return 1
    
    if not Path(args.multiview_transforms).exists():
        print(f"Error: Transforms not found: {args.multiview_transforms}")
        return 1
    
    print("Loading data...")
    joint_data = load_joint_schemas(args.joint_schemas)
    multiview_data = load_transforms(args.multiview_transforms)
    
    print(f"Loaded {len(multiview_data['frames'])} frames")
    print(f"Joint type: {joint_data['joint_type']}")
    
    print("\nMerging articulation data...")
    output_data = create_articulated_transforms(
        joint_data=joint_data,
        multiview_data=multiview_data,
        mask_dir=args.mask_dir
    )
    
    # Save output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nSaved to: {output_path}")
    
    # Print summary
    print_summary(output_data)
    
    return 0


if __name__ == "__main__":
    exit(main())