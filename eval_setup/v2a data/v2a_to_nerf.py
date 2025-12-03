#!/usr/bin/env python3
"""
Convert Video2Articulation view data to NeRF format for TwinSplat pipeline
"""

import numpy as np
import json
import argparse
import shutil
from pathlib import Path
from scipy.spatial.transform import Rotation
import cv2

def quaternion_to_matrix(quat):
    """Convert quaternion (w,x,y,z) to rotation matrix"""
    r = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])  # scipy uses (x,y,z,w)
    return r.as_matrix()

def convert_pose_to_nerf_matrix(pose_tx7):
    """Convert Tx7 pose to 4x4 NeRF transform matrix"""
    translation = pose_tx7[:3]
    quaternion = pose_tx7[3:]  # (w,x,y,z)
    
    # Build 4x4 matrix
    rotation_matrix = quaternion_to_matrix(quaternion)
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = rotation_matrix
    transform_matrix[:3, 3] = translation
    
    return transform_matrix

def load_ground_truth_joints(gt_file, category, object_id):
    """Load ground truth joint parameters"""
    with open(gt_file) as f:
        data = json.load(f)
    
    if category in data and object_id in data[category]:
        return data[category][object_id]
    return {}

def extract_movable_part_mask(segment_path, joint_info):
    """Extract movable part ID from segmentation and joint info"""
    # Load segmentation image
    segment_img = cv2.imread(str(segment_path), cv2.IMREAD_UNCHANGED)
    
    # For now, assume the movable part is the non-background part
    # You might need to refine this based on your specific joint mapping
    unique_parts = np.unique(segment_img)
    
    # Background is typically 0, static parts are 1, movable parts are 2+
    movable_part_ids = unique_parts[unique_parts > 1] if len(unique_parts) > 2 else []
    
    return movable_part_ids

def convert_v2a_data(v2a_path, output_dir, category, object_id, joint_id, gt_joints_file):
    """Convert Video2Articulation data to NeRF format"""
    
    v2a_path = Path(v2a_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    frames_dir = output_dir / "frames"
    depth_dir = output_dir / "depth"
    segment_dir = output_dir / "segment"
    
    frames_dir.mkdir(exist_ok=True)
    depth_dir.mkdir(exist_ok=True)
    segment_dir.mkdir(exist_ok=True)
    
    # Load V2A data
    camera_poses = np.load(v2a_path / "camera_pose.npy")  # Nx7 array
    intrinsics = np.load(v2a_path / "intrinsics.npy")     # 3x3 matrix
    
    # Load ground truth joint parameters
    gt_joints = load_ground_truth_joints(gt_joints_file, category, object_id)
    
    # Get RGB, depth, and segmentation files
    rgb_files = sorted((v2a_path / "rgb").glob("*.png"))
    depth_files = sorted((v2a_path / "depth").glob("*.npy"))
    segment_files = sorted((v2a_path / "segment").glob("*.png"))
    
    print(f"Found {len(rgb_files)} RGB files, {len(depth_files)} depth files")
    
    # Build NeRF transforms.json
    transforms = {
        "camera_model": "OPENCV",
        "fl_x": float(intrinsics[0, 0]),
        "fl_y": float(intrinsics[1, 1]),
        "cx": float(intrinsics[0, 2]),
        "cy": float(intrinsics[1, 2]),
        "w": 480,  # Standard V2A image width
        "h": 480,  # Standard V2A image height
        "frames": []
    }
    
    # Process each frame
    for i, (rgb_file, depth_file) in enumerate(zip(rgb_files, depth_files)):
        # Convert pose to 4x4 matrix
        pose_matrix = convert_pose_to_nerf_matrix(camera_poses[i])
        
        # Frame filename (zero-padded)
        frame_name = f"frame_{i:05d}.png"
        depth_name = f"frame_{i:05d}.npy"
        
        frame_data = {
            "file_path": f"frames/{frame_name}",
            "depth_file_path": f"depth/{depth_name}",
            "transform_matrix": pose_matrix.tolist()
        }
        
        # Add segmentation if available
        if i < len(segment_files):
            segment_name = f"frame_{i:05d}.png"
            frame_data["segment_file_path"] = f"segment/{segment_name}"
            
            # Copy segmentation file
            shutil.copy2(segment_files[i], segment_dir / segment_name)
        
        transforms["frames"].append(frame_data)
        
        # Copy RGB and depth files
        shutil.copy2(rgb_file, frames_dir / frame_name)
        shutil.copy2(depth_file, depth_dir / depth_name)
    
    # Add ground truth joint information
    if gt_joints and "interaction_list" in gt_joints:
        # Find the relevant joint
        relevant_joint = None
        joint_idx = int(joint_id.split('_')[1]) if '_' in joint_id else 0
        
        if joint_idx < len(gt_joints["interaction_list"]):
            relevant_joint = gt_joints["interaction_list"][joint_idx]
        
        if relevant_joint:
            transforms["articulations_gt"] = {
                "joint_0": {
                    "joint_type": relevant_joint["type"],
                    "joint_axis": relevant_joint["joint"]["axis"],
                    "joint_limits": relevant_joint["joint"]["limit"]
                }
            }
    
    # Load ground truth joint values if available
    gt_joint_values_file = v2a_path / "gt_joint_value.npy"
    if gt_joint_values_file.exists():
        gt_joint_values = np.load(gt_joint_values_file)
        transforms["gt_joint_values"] = gt_joint_values.tolist()
    
    # Save transforms.json
    with open(output_dir / "transforms_post.json", "w") as f:
        json.dump(transforms, f, indent=2)
    
    # Create a copy as transforms_reloc.json for pipeline compatibility
    shutil.copy2(output_dir / "transforms_post.json", output_dir / "transforms_reloc.json")
    
    print(f"✓ Converted V2A data to NeRF format")
    print(f"  Frames: {len(transforms['frames'])}")
    print(f"  Output: {output_dir}")
    print(f"  Joint info: {'articulations_gt' in transforms}")

def main():
    parser = argparse.ArgumentParser(description="Convert V2A data to NeRF format")
    parser.add_argument("--v2a_path", type=str, required=True, 
                       help="Path to V2A view directory")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for NeRF format data")
    parser.add_argument("--category", type=str, required=True,
                       help="PartNet category")
    parser.add_argument("--object_id", type=str, required=True,
                       help="PartNet object ID")
    parser.add_argument("--joint_id", type=str, required=True,
                       help="Joint ID")
    parser.add_argument("--gt_joints_file", type=str, 
                       default="new_partnet_mobility_dataset_correct_intr_meta.json",
                       help="Ground truth joints file")
    
    args = parser.parse_args()
    
    convert_v2a_data(
        args.v2a_path,
        args.output_dir, 
        args.category,
        args.object_id,
        args.joint_id,
        args.gt_joints_file
    )

if __name__ == "__main__":
    main()