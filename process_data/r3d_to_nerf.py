"""
R3D to NeRF Format Converter with Anchored Alignment

This script converts Record3D (.r3d) captures to NeRF/3DGS format with support for:
1. Frame subsampling (stride, start, end)
2. Coordinate system conversion (ARKit -> NeRF/COLMAP)
3. Anchored alignment to pre-change reference model
4. Point cloud generation for visualization

Output structure:
dataset_root/
 ├── frames/
 │    ├── frame_00001.jpg
 │    ├── frame_00002.jpg
 │    └── ...
 ├── depth/
 │    ├── frame_00001.png (16-bit depth in mm)
 │    ├── frame_00002.png
 │    └── ...
 └── transforms.json (NeRF format with camera parameters)
"""

import argparse
import json
import shutil
from pathlib import Path
from turtle import down
from typing import Optional, Dict, Any, List, Union
import numpy as np
from regex import F
import torch
from PIL import Image
from tqdm import tqdm
import open3d as o3d

from data_utils import get_posed_rgbd_dataset, get_xyz
from dataset_class import PosedRGBDItem, R3DDataset


def arkit_to_nerf_transform() -> np.ndarray:
    """
    Convert ARKit coordinate system to NeRF/COLMAP coordinate system.
    
    ARKit: +X right, +Y up, +Z backward (right-handed)
    NeRF/COLMAP: +X right, +Y down, +Z forward (right-handed)
    
    Transformation: Rotate 180° around X-axis
    """
    return np.array([
        [1,  0,  0, 0],
        [0, -1,  0, 0],
        [0,  0, -1, 0],
        [0,  0,  0, 1]
    ], dtype=np.float64)


def pose_to_nerf_matrix(pose: np.ndarray) -> np.ndarray:
    """Convert OpenCV-style (R3DDataset) camera pose to NeRF/Blender-style pose."""
    # 180° rotation around X axis (flip Y and Z)
    R_flip = np.diag([1, -1, -1, 1])
    nerf_pose = pose @ R_flip   # post-multiply for camera-to-world
    return nerf_pose



def create_transforms_json(
    dataset: R3DDataset,
    frame_indices: List[int],
    output_dir: Path,
    camera_angle_x: Optional[float] = None
) -> Dict[str, Any]:
    """
    Create transforms.json in NeRF format.
    
    Args:
        dataset: R3D dataset
        frame_indices: List of frame indices to include
        output_dir: Output directory path
        camera_angle_x: Horizontal field of view in radians (optional)
        
    Returns:
        Dictionary containing transform data
    """
    # Get intrinsics from first frame
    sample_item = dataset[frame_indices[0]]
    intrinsics = sample_item.intrinsics.numpy()
    
    # Extract camera parameters
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]
    
    # Get image dimensions
    _, h, w = sample_item.image.shape
    
    # Calculate camera_angle_x (horizontal FOV) if not provided
    if camera_angle_x is None:
        camera_angle_x = 2 * np.arctan(w / (2 * fx))
    
    # Initialize transforms dictionary
    transforms = {
        "camera_angle_x": float(camera_angle_x),
        "fl_x": float(fx),
        "fl_y": float(fy),
        "cx": float(cx),
        "cy": float(cy),
        "w": int(w),
        "h": int(h),
        "frames": [],
        "ply_file_path": "fused_pc.ply"
    }
    
    # Add each frame
    for idx in tqdm(frame_indices, desc="Creating transforms.json"):
        item = dataset[idx]
        pose = item.pose.numpy()
        
        # Convert to NeRF coordinate system
        nerf_pose = pose_to_nerf_matrix(pose)
        
        # Create frame entry
        frame_number = frame_indices.index(idx) + 1
        frame_entry = {
            "file_path": f"frames/frame_{frame_number:05d}.jpg",
            "depth_file_path": f"depth/frame_{frame_number:05d}.png",
            "transform_matrix": nerf_pose.tolist(),
            # "original_index": int(idx)
        }
        
        transforms["frames"].append(frame_entry)
    
    return transforms


def export_frames_and_depth(
    dataset: R3DDataset,
    frame_indices: List[int],
    output_dir: Path,
    depth_scale: float = 1000.0
):
    """
    Export RGB frames and depth maps.
    
    Args:
        dataset: R3D dataset
        frame_indices: List of frame indices to export
        output_dir: Output directory
        depth_scale: Scale factor for depth (default 1000 = convert m to mm)
    """
    frames_dir = output_dir / "frames"
    depth_dir = output_dir / "depth"
    
    frames_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)
    
    for idx in tqdm(frame_indices, desc="Exporting frames and depth"):
        item = dataset[idx]
        frame_number = frame_indices.index(idx) + 1
        
        # Export RGB frame
        rgb_image = item.image.permute(1, 2, 0).numpy()
        rgb_image = (rgb_image * 255).astype(np.uint8)
        rgb_pil = Image.fromarray(rgb_image)
        rgb_path = frames_dir / f"frame_{frame_number:05d}.jpg"
        rgb_pil.save(rgb_path, quality=95)
        
        # Export depth map (16-bit PNG in millimeters)
        depth = item.depth.squeeze(0).numpy()
        
        # Convert to millimeters and handle invalid depths
        depth_mm = depth * depth_scale
        depth_mm[depth < 0] = 0  # Set invalid depths to 0
        depth_mm = np.clip(depth_mm, 0, 65535)  # Clip to 16-bit range
        depth_uint16 = depth_mm.astype(np.uint16)
        
        depth_pil = Image.fromarray(depth_uint16, mode='I;16')
        depth_path = depth_dir / f"frame_{frame_number:05d}.png"
        depth_pil.save(depth_path)


def generate_point_cloud(
    dataset: R3DDataset,
    frame_indices: List[int],
    output_path: Path,
    subsample_ratio: float = 0.1,
    voxel_size: float = 0.01
):
    """
    Generate and save a fused point cloud from selected frames.
    
    Args:
        dataset: R3D dataset
        frame_indices: List of frame indices to use
        output_path: Output path for PLY file
        subsample_ratio: Ratio of points to keep per frame (0-1)
        voxel_size: Voxel size for downsampling
    """
    all_points = []
    all_colors = []
    
    print(f"Generating point cloud from {len(frame_indices)} frames...")
    
    for idx in tqdm(frame_indices, desc="Processing frames for point cloud"):
        item = dataset[idx]
        
        # Get XYZ coordinates
        depth = item.depth.unsqueeze(0)
        mask = item.mask.unsqueeze(0)
        pose = item.pose.unsqueeze(0)
        intrinsics = item.intrinsics.unsqueeze(0)
        
        xyz = get_xyz(depth, mask, pose, intrinsics)
        xyz = xyz.squeeze(0)  # (H, W, 3)
        
        # Get RGB colors
        rgb = item.image.permute(1, 2, 0)  # (H, W, 3)
        
        # Create subsample mask
        valid_mask = ~mask.squeeze(0).squeeze(0)  # (H, W)
        
        if subsample_ratio < 1.0:
            subsample_mask = torch.rand(valid_mask.shape) < subsample_ratio
            valid_mask = valid_mask & subsample_mask
        
        # Extract valid points
        valid_points = xyz[valid_mask].cpu().numpy()
        valid_colors = rgb[valid_mask].cpu().numpy()
        
        if len(valid_points) > 0:
            all_points.append(valid_points)
            all_colors.append(valid_colors)
    
    if not all_points:
        print("Warning: No valid points found!")
        return
    
    # Concatenate all points
    points = np.vstack(all_points)
    colors = np.vstack(all_colors)
    
    print(f"Total points before downsampling: {len(points):,}")
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Voxel downsampling
    if voxel_size > 0:
        print(f"Downsampling with voxel size {voxel_size}...")
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        print(f"Points after downsampling: {len(pcd.points):,}")
    
    # Save point cloud
    output_path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(output_path), pcd)
    print(f"Point cloud saved to {output_path}")


def convert_r3d_to_nerf(
    data_path: Path,
    output_dir: Path,
    stride: int = 1,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
    downsample_factor: Union[int, float] = 1,
    pc_subsample: float = 0.1,
    voxel_downsample: float = 0.01,
    generate_pc: bool = True
):
    """
    Main conversion function.
    
    Args:
        data_path: Path to .r3d file
        output_dir: Output directory
        stride: Frame stride (process every Nth frame)
        start_frame: Starting frame index
        end_frame: Ending frame index (None = all frames)
        pc_subsample: Point cloud subsampling ratio
        voxel_downsample: Voxel downsampling size
        generate_pc: Whether to generate point cloud
    """
    print(f"Loading R3D dataset from {data_path}...")
    dataset = get_posed_rgbd_dataset(key='r3d', path=str(data_path), use_depth_shape=False, downsample_factor=downsample_factor)
    
    total_frames = len(dataset)
    print(f"Total frames in dataset: {total_frames}")
    
    # Determine frame indices to process
    if end_frame is None:
        end_frame = total_frames
    else:
        end_frame = min(end_frame, total_frames)
    
    frame_indices = list(range(start_frame, end_frame, stride))
    print(f"Processing {len(frame_indices)} frames (stride={stride}, start={start_frame}, end={end_frame})")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Export frames and depth
    print("\n=== Exporting frames and depth maps ===")
    export_frames_and_depth(dataset, frame_indices, output_dir)
    
    # Create transforms.json
    print("\n=== Creating transforms.json ===")
    transforms = create_transforms_json(dataset, frame_indices, output_dir)
    
    transforms_path = output_dir / "transforms.json"
    with open(transforms_path, 'w') as f:
        json.dump(transforms, f, indent=2)
    print(f"Transforms saved to {transforms_path}")
    
    # Generate point cloud
    if generate_pc:
        print("\n=== Generating point cloud ===")
        pc_path = output_dir / "fused_pc.ply"
        generate_point_cloud(
            dataset,
            frame_indices,
            pc_path,
            subsample_ratio=pc_subsample,
            voxel_size=voxel_downsample
        )
    
    # Save metadata
    metadata = {
        "source_file": str(data_path),
        "total_frames": total_frames,
        "processed_frames": len(frame_indices),
        "stride": stride,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "coordinate_system": "NeRF/COLMAP (right-handed, +X right, +Y down, +Z forward)"
    }
    
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✓ Conversion complete! Output saved to {output_dir}")
    print(f"  - {len(frame_indices)} RGB frames in ./frames/")
    print(f"  - {len(frame_indices)} depth maps in ./depth/")
    print(f"  - Camera parameters in ./transforms.json")
    if generate_pc:
        print(f"  - Point cloud in ./fused_pc.ply")


def main():
    parser = argparse.ArgumentParser(
        description="Convert Record3D (.r3d) to NeRF format with anchoring support"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to .r3d file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory"
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Process every Nth frame (default: 1)"
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Starting frame index (default: 0)"
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="Ending frame index (default: None = all frames)"
    )
    parser.add_argument(
        "--downsample_factor",
        type=Union[int, float],
        default=2,
        help="Downsample factor for dataset (default: 1)"
    )
    parser.add_argument(
        "--pc_subsample",
        type=float,
        default=0.1,
        help="Point cloud subsampling ratio (default: 0.1)"
    )
    parser.add_argument(
        "--voxel_downsample",
        type=float,
        default=0.01,
        help="Voxel downsampling size in meters (default: 0.01)"
    )
    parser.add_argument(
        "--no_pc",
        action="store_true",
        help="Skip point cloud generation"
    )
    
    args = parser.parse_args()
    
    data_path = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    
    if not data_path.exists():
        print(f"Error: Data file not found: {data_path}")
        return
    
    convert_r3d_to_nerf(
        data_path=data_path,
        output_dir=output_dir,
        stride=args.stride,
        start_frame=args.start,
        end_frame=args.end,
        downsample_factor=args.downsample_factor,
        pc_subsample=args.pc_subsample,
        voxel_downsample=args.voxel_downsample,
        generate_pc=not args.no_pc
    )


if __name__ == "__main__":
    main()