"""
Utility functions for coordinate transformations.
- Camera → World transformations
- OpenCV → NeRF convention conversion
"""
import os
import json
import numpy as np
import torch


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


def export_joint_and_inliers(result, inlier_trajectories, extrinsics, output_dir, filename_prefix="joint"):
    os.makedirs(output_dir, exist_ok=True)

    joints_out = []

    if result.success:
        R = extrinsics[:3, :3]  # rotation part only
        t = extrinsics[:3, 3]   # translation part (not needed for axis)

        if result.joint_type.value == "hinge":  # revolute
            hinge_params = result.get_hinge_params()

            # Transform pivot
            pivot_cam = torch.tensor(hinge_params.pivot[None, :], dtype=torch.float32)
            pivot_world = transform_to_world(opencv_to_nerf(pivot_cam), extrinsics)[0].cpu().numpy()

            # Transform axis (direction only)
            axis_cam = torch.tensor(hinge_params.axis, dtype=torch.float32)
            axis_world = (R @ opencv_to_nerf(axis_cam[None, :]).T).T[0].cpu().numpy()
            axis_world = axis_world / (np.linalg.norm(axis_world) + 1e-8)

            joint_data = {
                "joint_type": "revolute",
                "joint_axis": axis_world.tolist(),
                "joint_pivot": pivot_world.tolist(),
                "joint_limits": [
                    float(np.degrees(hinge_params.angle_min)) if hinge_params.angle_min is not None else None,
                    float(np.degrees(hinge_params.angle_max)) if hinge_params.angle_max is not None else None
                ]
            }

            joints_out.append(joint_data)

        elif result.joint_type.value == "slider":  # prismatic
            slider_params = result.get_slider_params()

            # Transform reference point
            if slider_params.reference_point is not None:
                ref_cam = torch.tensor(slider_params.reference_point[None, :], dtype=torch.float32)
                ref_world = transform_to_world(opencv_to_nerf(ref_cam), extrinsics)[0].cpu().numpy()
            else:
                ref_world = [0, 0, 0]

            # Transform direction (axis)
            dir_cam = torch.tensor(slider_params.direction, dtype=torch.float32)
            dir_world = (R @ opencv_to_nerf(dir_cam[None, :]).T).T[0].cpu().numpy()
            dir_world = dir_world / (np.linalg.norm(dir_world) + 1e-8)

            joint_data = {
                "joint_type": "prismatic",
                "joint_axis": dir_world.tolist(),    
                "joint_pivot": ref_world.tolist() if isinstance(ref_world, np.ndarray) else ref_world,
                "joint_limits": [
                    float(slider_params.translation_min),
                    float(slider_params.translation_max)
                ]
            }

            joints_out.append(joint_data)

    schema_path = os.path.join(output_dir, f"{filename_prefix}_schema.json")
    with open(schema_path, 'w') as f:
        json.dump(joints_out, f, indent=2)
    print(f"✅ Saved joint schema in world frame: {schema_path}")

    # --- Inlier points (unchanged) ---
    inlier_points = []
    for traj in inlier_trajectories:
        positions = torch.tensor(traj.get_all_positions(), dtype=torch.float32)
        positions_nerf = opencv_to_nerf(positions)
        positions_world = transform_to_world(positions_nerf, extrinsics)
        inlier_points.append(positions_world)

    if inlier_points:
        inlier_points = torch.cat(inlier_points, dim=0)

        np.save(os.path.join(output_dir, f"{filename_prefix}_inliers_world.npy"),
                inlier_points.cpu().numpy())
        torch.save(inlier_points,
                   os.path.join(output_dir, f"{filename_prefix}_inliers_world.pt"))

        print(f"✅ Saved {inlier_points.shape[0]} inlier points in world frame")
