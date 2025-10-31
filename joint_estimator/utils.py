"""
Utility functions for coordinate transformations.
- Camera → World transformations
- OpenCV → NeRF convention conversion
"""
import os
import json
import numpy as np
import torch
from pathlib import Path    

def transform_to_world(points: torch.Tensor, extrinsics: torch.Tensor) -> torch.Tensor:
    """
    Transform 3D points from camera frame to world frame using extrinsic matrix.

    Args:
        points (torch.Tensor): (N, 3) points in camera coordinates
        extrinsics (torch.Tensor): (4, 4) camera-to-world transformation matrix

    Returns:
        torch.Tensor: (N, 3) points in world coordinates
    """
    if points.shape[-1] != 3:
        raise ValueError(f"Expected points with shape (N, 3), got {points.shape}")

    ones = torch.ones((points.shape[0], 1), device=points.device, dtype=points.dtype)
    hom_points = torch.cat([points, ones], dim=1)  # (N, 4)
    world_points = (extrinsics.to(points.device) @ hom_points.T).T[:, :3]
    return world_points


def opencv_to_nerf(points: torch.Tensor) -> torch.Tensor:
    """
    Convert 3D points from OpenCV coordinate convention to NeRF convention.

    OpenCV convention: X right, Y down, Z forward  
    NeRF convention:   X right, Y up,   Z back  

    Args:
        points (torch.Tensor): (N, 3) points in OpenCV coordinates

    Returns:
        torch.Tensor: (N, 3) points in NeRF coordinates
    """
    if points.shape[-1] != 3:
        raise ValueError(f"Expected points with shape (N, 3), got {points.shape}")

    R = torch.tensor([[1,  0,  0],
                      [0, -1,  0],
                      [0,  0, -1]],
                     dtype=points.dtype, device=points.device)
    return points @ R.T


def filter_points_near_joint(points, joint_type, joint_axis, joint_pivot,
                             max_distance=0.3, max_extent=None):
    """
    Filter points to only keep those near the joint axis and within extent.
    
    Args:
        points: [N, 3] tensor of 3D points
        joint_type: "revolute" or "prismatic"
        joint_axis: [3] normalized axis vector
        joint_pivot: [3] point on the axis (pivot for revolute, reference for prismatic)
        max_distance: maximum perpendicular distance from axis (meters)
        max_extent: optional max length along axis (meters) for pruning far-away points
    
    Returns:
        filtered_points: [M, 3] tensor where M <= N
        mask: [N] boolean mask
    """
    # Normalize axis
    axis_normalized = joint_axis / (torch.norm(joint_axis) + 1e-8)

    # Vector from pivot to points
    v = points - joint_pivot.unsqueeze(0)  # [N, 3]

    # Projection length along axis
    proj_length = torch.sum(v * axis_normalized.unsqueeze(0), dim=1)  # [N]

    # Perpendicular distance
    proj_on_axis = proj_length.unsqueeze(1) * axis_normalized.unsqueeze(0)  # [N, 3]
    perpendicular = v - proj_on_axis
    distance = torch.norm(perpendicular, dim=1)  # [N]

    # Radial filter
    mask = distance <= max_distance

    # Longitudinal filter (optional)
    if max_extent is not None:
        mask &= (proj_length >= -max_extent) & (proj_length <= max_extent)

    filtered_points = points[mask]
    return filtered_points, mask


def robust_bbox(points, padding=0.05, q=0.98):
    """
    Compute bounding box using percentiles instead of raw min/max.
    q controls the fraction of points kept (e.g., q=0.98 keeps central 98%).
    """
    import numpy as np
    pts_np = points.cpu().numpy()
    lo = np.quantile(pts_np, (1 - q) / 2, axis=0)
    hi = np.quantile(pts_np, 1 - (1 - q) / 2, axis=0)
    return torch.tensor(lo - padding, dtype=torch.float32), \
           torch.tensor(hi + padding, dtype=torch.float32)

def export_joint_and_inliers(result, inlier_trajectories, extrinsics, output_dir, 
                             filename_prefix="joint", voxel_resolution=64,
                             joint_max_distance=0.5): 
    """Export joint parameters and Object3DSeg-compatible voxel data."""
    os.makedirs(output_dir, exist_ok=True)
    
    if not result.success:
        print("Cannot export: joint estimation failed")
        return
    
    R = extrinsics[:3, :3]
    t = extrinsics[:3, 3]
    
    # === 1. Extract and transform joint parameters ===
    if result.joint_type.value == "hinge":
        hinge_params = result.get_hinge_params()
        
        # Transform pivot to world frame
        pivot_cam = torch.tensor(hinge_params.pivot[None, :], dtype=torch.float32)
        pivot_world = transform_to_world(opencv_to_nerf(pivot_cam), extrinsics)[0]
        
        # Transform axis to world frame
        axis_cam = torch.tensor(hinge_params.axis, dtype=torch.float32)
        axis_world = (R @ opencv_to_nerf(axis_cam[None, :]).T).T[0]
        axis_world = axis_world / (axis_world.norm() + 1e-8)
        
        joint_type_str = "revolute"
        joint_axis_tensor = axis_world
        joint_pivot_tensor = pivot_world
        joint_limits = [
            float(hinge_params.angle_min) if hinge_params.angle_min is not None else 0.0,
            float(hinge_params.angle_max) if hinge_params.angle_max is not None else 0.0
        ]
        
    elif result.joint_type.value == "slider":
        slider_params = result.get_slider_params()
        
        # Transform reference point
        if slider_params.reference_point is not None:
            ref_cam = torch.tensor(slider_params.reference_point[None, :], dtype=torch.float32)
            ref_world = transform_to_world(opencv_to_nerf(ref_cam), extrinsics)[0]
        else:
            ref_world = torch.zeros(3)
        
        # Transform direction
        dir_cam = torch.tensor(slider_params.direction, dtype=torch.float32)
        dir_world = (R @ opencv_to_nerf(dir_cam[None, :]).T).T[0]
        dir_world = dir_world / (dir_world.norm() + 1e-8)
        
        joint_type_str = "prismatic"
        joint_axis_tensor = dir_world
        joint_pivot_tensor = ref_world
        joint_limits = [
            float(slider_params.translation_min),
            float(slider_params.translation_max)
        ]
    
    # === 2. Transform inlier points to world frame ===
    inlier_points = []
    for traj in inlier_trajectories:
        positions = torch.tensor(traj.get_all_positions(), dtype=torch.float32)
        positions_nerf = opencv_to_nerf(positions)
        positions_world = transform_to_world(positions_nerf, extrinsics)
        inlier_points.append(positions_world)
    
    if not inlier_points:
        print("No inlier points to export")
        return
    
    inlier_points = torch.cat(inlier_points, dim=0)
    print(f"Total inlier points: {len(inlier_points)}")
    
    # === 2.5. FILTER POINTS NEAR JOINT AXIS ===
    inlier_points, kept_mask = filter_points_near_joint(
        inlier_points, 
        joint_type_str, 
        joint_axis_tensor, 
        joint_pivot_tensor, 
        max_distance=joint_max_distance,
        max_extent=1.0
    )
    
    print(f"Points near joint axis (within {joint_max_distance}m): {len(inlier_points)} "
          f"({100 * len(inlier_points) / kept_mask.numel():.1f}% kept)")
    
    if len(inlier_points) < 10:
        print("⚠️  Warning: Very few points near joint! Consider increasing joint_max_distance")
    
    # === 3. Compute robust bounding box with padding and percentiles ===
    bbox_min, bbox_max = robust_bbox(inlier_points, padding=0.05, q=0.90)
    print(f"Bounding box (robust): min={bbox_min.tolist()}, max={bbox_max.tolist()}")

    
    # === 4. Voxelize points ===
    voxel_grid = voxelize_points(inlier_points, bbox_min, bbox_max, voxel_resolution)
    occupancy_ratio = voxel_grid.sum().item() / voxel_grid.numel()
    print(f"Voxel occupancy: {voxel_grid.sum().item()}/{voxel_grid.numel()} "
          f"({100 * occupancy_ratio:.2f}%)")
    
    # === 5. Save in Object3DSeg format ===
    obj3dseg_data = {
        'bbox_min': bbox_min.cpu(),
        'bbox_max': bbox_max.cpu(),
        'voxel': voxel_grid.cpu(),
        'tight_bbox': None,
        'mask_dilate_uniform': 0,
        'mask_dilate_top': 0,
        'joint_type': joint_type_str,
        'joint_axis': joint_axis_tensor.cpu(),
        'joint_pivot': joint_pivot_tensor.cpu(),
        'joint_limits': joint_limits,
        'joint_limit_min': torch.tensor(joint_limits[0]),
        'joint_limit_max': torch.tensor(joint_limits[1])
    }

    output_dir = os.path.join(output_dir, "obj_masks")
    os.makedirs(output_dir, exist_ok=True)
    
    voxel_path = os.path.join(output_dir, f"obj_{filename_prefix}.pt")
    torch.save(obj3dseg_data, voxel_path)
    
    print(f"✅ Saved Object3DSeg voxel data: {voxel_path}")
    print(f"   Voxel grid: {voxel_resolution}³")
    print(f"   Joint type: {joint_type_str}")
    print(f"   Joint limits: {joint_limits}")
    
    # === 6. Save raw points for reference ===
    np.save(os.path.join(output_dir, f"obj_{filename_prefix}.npy"),
           inlier_points.cpu().numpy())
    print(f"✅ Saved {inlier_points.shape[0]} filtered inlier points")




def export_joint_and_inliers_tapip3d(result, inlier_trajectories, output_dir, 
                                     filename_prefix="joint", voxel_resolution=32,
                                     joint_max_distance=0.5):
    """
    Export joint parameters and Object3DSeg-compatible voxel data for TAPIP3D.
    
    TAPIP3D trajectories are already in world coordinates (same as sparse_pc.ply),
    so no extrinsics transformation is needed.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if not result.success:
        print("Cannot export: joint estimation failed")
        return
    
    # === 1. Extract joint parameters (already in world frame) ===
    if result.joint_type.value == "hinge":
        hinge_params = result.get_hinge_params()
        
        # Already in world coordinates - no transform needed
        joint_type_str = "revolute"
        joint_axis_tensor = torch.tensor(hinge_params.axis, dtype=torch.float32)
        joint_pivot_tensor = torch.tensor(hinge_params.pivot, dtype=torch.float32)
        joint_limits = [
            float(hinge_params.angle_min) if hinge_params.angle_min is not None else 0.0,
            float(hinge_params.angle_max) if hinge_params.angle_max is not None else 0.0
        ]
        
    elif result.joint_type.value == "slider":
        slider_params = result.get_slider_params()
        
        # Already in world coordinates
        if slider_params.reference_point is not None:
            joint_pivot_tensor = torch.tensor(slider_params.reference_point, dtype=torch.float32)
        else:
            joint_pivot_tensor = torch.zeros(3)
        
        joint_type_str = "prismatic"
        joint_axis_tensor = torch.tensor(slider_params.direction, dtype=torch.float32)
        joint_limits = [
            float(slider_params.translation_min),
            float(slider_params.translation_max)
        ]
    
    # Normalize axis
    joint_axis_tensor = joint_axis_tensor / (joint_axis_tensor.norm() + 1e-8)
    
    # === 2. Extract inlier points (already in world frame) ===
    inlier_points = []
    for traj in inlier_trajectories:
        positions = torch.tensor(traj.get_all_positions(), dtype=torch.float32)
        inlier_points.append(positions)  
    
    if not inlier_points:
        print("No inlier points to export")
        return
    
    inlier_points = torch.cat(inlier_points, dim=0)
    print(f"Total inlier points: {len(inlier_points)}")
    
    # === 3. Filter points near joint axis ===
    inlier_points, kept_mask = filter_points_near_joint(
        inlier_points, 
        joint_type_str, 
        joint_axis_tensor, 
        joint_pivot_tensor, 
        max_distance=joint_max_distance,
        max_extent=1.0
    )
    
    print(f"Points near joint axis (within {joint_max_distance}m): {len(inlier_points)} "
          f"({100 * len(inlier_points) / kept_mask.numel():.1f}% kept)")
    
    if len(inlier_points) < 10:
        print("⚠️  Warning: Very few points near joint! Consider increasing joint_max_distance")
    
    # === 4. Compute robust bounding box ===
    bbox_min, bbox_max = robust_bbox(inlier_points, padding=0.05, q=0.90)
    print(f"Bounding box (robust): min={bbox_min.tolist()}, max={bbox_max.tolist()}")
    
    # === 5. Voxelize points ===
    voxel_grid = voxelize_points(inlier_points, bbox_min, bbox_max, voxel_resolution)
    occupancy_ratio = voxel_grid.sum().item() / voxel_grid.numel()
    print(f"Voxel occupancy: {voxel_grid.sum().item()}/{voxel_grid.numel()} "
          f"({100 * occupancy_ratio:.2f}%)")
    
    # === 6. Save in Object3DSeg format ===
    obj3dseg_data = {
        'bbox_min': bbox_min.cpu(),
        'bbox_max': bbox_max.cpu(),
        'voxel': voxel_grid.cpu(),
        'tight_bbox': None,
        'mask_dilate_uniform': 0,
        'mask_dilate_top': 0,
        'joint_type': joint_type_str,
        'joint_axis': joint_axis_tensor.cpu(),
        'joint_pivot': joint_pivot_tensor.cpu(),
        'joint_limits': joint_limits,
        'joint_limit_min': torch.tensor(joint_limits[0]),
        'joint_limit_max': torch.tensor(joint_limits[1])
    }
    
    output_dir_masks = os.path.join(output_dir, "obj_masks")
    os.makedirs(output_dir_masks, exist_ok=True)
    
    voxel_path = os.path.join(output_dir_masks, f"obj_{filename_prefix}.pt")
    torch.save(obj3dseg_data, voxel_path)
    
    print(f"✅ Saved Object3DSeg voxel data: {voxel_path}")
    print(f"   Voxel grid: {voxel_resolution}³")
    print(f"   Joint type: {joint_type_str}")
    print(f"   Joint limits: {joint_limits}")
    
    # === 7. Save raw points for reference ===
    points_path = os.path.join(output_dir_masks, f"obj_{filename_prefix}.npy")
    np.save(points_path, inlier_points.cpu().numpy())
    print(f"✅ Saved {inlier_points.shape[0]} filtered inlier points: {points_path}")

def voxelize_points(points, bbox_min, bbox_max, resolution=32):
    """Convert point cloud to binary voxel occupancy grid."""
    normalized = (points - bbox_min) / (bbox_max - bbox_min)
    indices = (normalized * (resolution - 1)).long()
    indices = torch.clamp(indices, 0, resolution - 1)
    
    voxel_grid = torch.zeros(resolution, resolution, resolution, dtype=torch.bool)
    for idx in indices:
        voxel_grid[idx[0], idx[1], idx[2]] = True
    
    return voxel_grid


def save_result_to_json(result, per_frame_values: dict, output_path: str, 
                       extrinsics=None, coordinate_system="world"):
    """
    Save joint estimation result with per-frame articulation values.
    
    Args:
        result: JointEstimationResult
        per_frame_values: Per-frame articulation values
        output_path: Output JSON path
        extrinsics: Optional extrinsics for CoTracker mode coordinate transform
        coordinate_system: "camera" or "world" (only matters if extrinsics provided)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    joints_out = []
    
    if result.success:
        if result.joint_type.value == "hinge":
            hinge_params = result.get_hinge_params()
            
            # Transform to world space if needed (CoTracker mode)
            if extrinsics is not None and coordinate_system == "world":
                import torch
                from utils import opencv_to_nerf, transform_to_world
                
                R = extrinsics[:3, :3]
                
                # Transform pivot
                pivot_cam = torch.tensor(hinge_params.pivot[None, :], dtype=torch.float32)
                pivot_world = transform_to_world(opencv_to_nerf(pivot_cam), extrinsics)[0]
                
                # Transform axis
                axis_cam = torch.tensor(hinge_params.axis, dtype=torch.float32)
                axis_world = (R @ opencv_to_nerf(axis_cam[None, :]).T).T[0]
                axis_world = axis_world / (axis_world.norm() + 1e-8)
                
                joint_axis = axis_world.tolist()
                joint_pivot = pivot_world.tolist()
            else:
                # Use as-is (TAPIP3D mode or camera space export)
                joint_axis = hinge_params.axis.tolist()
                joint_pivot = hinge_params.pivot.tolist()
            
            joint_data = {
                "joint_type": "revolute",
                "joint_axis": joint_axis,
                "joint_pivot": joint_pivot,
                "joint_limits": [
                    float(np.degrees(hinge_params.angle_min)),
                    float(np.degrees(hinge_params.angle_max))
                ],
                "per_frame_angles": {
                    int(k): float(np.degrees(v))
                    for k, v in sorted(per_frame_values.items())
                },
                "coordinate_system": coordinate_system
            }
            
        elif result.joint_type.value == "slider":
            slider_params = result.get_slider_params()
            
            # Transform to world space if needed
            if extrinsics is not None and coordinate_system == "world":
                import torch
                from utils import opencv_to_nerf, transform_to_world
                
                R = extrinsics[:3, :3]
                
                # Transform reference point
                if slider_params.reference_point is not None:
                    ref_cam = torch.tensor(slider_params.reference_point[None, :], dtype=torch.float32)
                    ref_world = transform_to_world(opencv_to_nerf(ref_cam), extrinsics)[0]
                    joint_pivot = ref_world.tolist()
                else:
                    joint_pivot = [0.0, 0.0, 0.0]
                
                # Transform direction
                dir_cam = torch.tensor(slider_params.direction, dtype=torch.float32)
                dir_world = (R @ opencv_to_nerf(dir_cam[None, :]).T).T[0]
                dir_world = dir_world / (dir_world.norm() + 1e-8)
                
                joint_axis = dir_world.tolist()
            else:
                joint_axis = slider_params.direction.tolist()
                joint_pivot = (slider_params.reference_point.tolist() 
                              if slider_params.reference_point is not None 
                              else [0.0, 0.0, 0.0])
            
            joint_data = {
                "joint_type": "prismatic",
                "joint_axis": joint_axis,
                "joint_pivot": joint_pivot,
                "joint_limits": [
                    slider_params.translation_min,
                    slider_params.translation_max
                ],
                "per_frame_translations": {
                    int(k): float(v)
                    for k, v in sorted(per_frame_values.items())
                },
                "coordinate_system": coordinate_system
            }
        
        joints_out.append(joint_data)
    
    with open(output_path, 'w') as f:
        json.dump(joints_out, f, indent=2)