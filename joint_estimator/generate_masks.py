#!/usr/bin/env python3
"""
Generate 2D masks for articulated objects (prismatic or revolute joints).
De-articulates voxels from t=1 (max limit) back to t=0 (rest pose),
then re-articulates for each frame's joint value.
"""

import json
from pathlib import Path
import torch
import numpy as np
import cv2
import argparse
from tqdm import tqdm


def load_data(data_dir: Path, voxel_subdir="obj_masks", voxel_filename="obj_prismatic.pt"):
    """Load transforms and voxel data."""
    with open(data_dir / "transforms_arti.json", "r") as f:
        T = json.load(f)
    
    pt_path = data_dir / voxel_subdir / voxel_filename
    if not pt_path.exists():
        # Try to find any .pt file in the directory
        pt_files = list((data_dir / voxel_subdir).glob("*.pt"))
        if not pt_files:
            raise FileNotFoundError(f"No .pt files found in {data_dir / voxel_subdir}")
        pt_path = pt_files[0]
        print(f"Using voxel file: {pt_path}")
    
    M = torch.load(pt_path, map_location="cpu")
    return T, M


def occupied_points_from_voxels(bmin, bmax, voxel):
    """Return Nx3 world-space points for all occupied voxels (voxel centers)."""
    nx, ny, nz = voxel.shape
    xs = torch.linspace(bmin[0], bmax[0], nx)
    ys = torch.linspace(bmin[1], bmax[1], ny)
    zs = torch.linspace(bmin[2], bmax[2], nz)
    ii, jj, kk = torch.nonzero(voxel, as_tuple=True)
    pts = torch.stack([xs[ii], ys[jj], zs[kk]], dim=1)  # (N,3)
    return pts


def rotate_around_axis(points, axis, pivot, angle):
    """
    Rotate Nx3 points around axis through pivot by angle (radians).
    Uses Rodrigues' rotation formula.
    """
    axis = axis / torch.norm(axis)
    points_centered = points - pivot.unsqueeze(0)
    angle = torch.as_tensor(angle, dtype=torch.float32, device=points.device)
    
    cos_angle = torch.cos(angle)
    sin_angle = torch.sin(angle)
    
    # Rodrigues' formula: R = I + sin(θ)K + (1-cos(θ))K²
    K = torch.tensor([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ], dtype=torch.float32)
    
    R = torch.eye(3) + sin_angle * K + (1 - cos_angle) * (K @ K)
    
    points_rotated = (R @ points_centered.T).T
    return points_rotated + pivot.unsqueeze(0)


def translate_along_axis(points, axis, displacement):
    """Translate Nx3 points along axis by displacement amount."""
    axis = axis / torch.norm(axis)
    return points + displacement * axis.unsqueeze(0)


def articulate_points(points, joint_type, joint_axis, joint_pivot, joint_value):
    """
    Apply articulation transformation to points.
    
    Args:
        points: Nx3 tensor of 3D points
        joint_type: "revolute" or "prismatic"
        joint_axis: 3D unit vector
        joint_pivot: 3D pivot point (for revolute) or reference point (for prismatic)
        joint_value: angle (radians) for revolute, displacement (meters) for prismatic
    
    Returns:
        Nx3 tensor of articulated points
    """
    if joint_type == "revolute":
        return rotate_around_axis(points, joint_axis, joint_pivot, joint_value)
    elif joint_type == "prismatic":
        return translate_along_axis(points, joint_axis, joint_value)
    else:
        raise ValueError(f"Unknown joint type: {joint_type}")


def de_articulate_from_max(points_t1, joint_type, joint_axis, joint_pivot, joint_limit_max):
    """
    De-articulate points from t=1 (max limit) back to t=0 (rest pose).
    
    Args:
        points_t1: Nx3 points at maximum articulation (t=1)
        joint_type: "revolute" or "prismatic"
        joint_axis: Joint axis
        joint_pivot: Joint pivot/reference point
        joint_limit_max: Maximum joint value (angle or displacement at t=1)
    
    Returns:
        Nx3 points at rest pose (t=0)
    """
    # Apply inverse transformation: negate the joint value
    return articulate_points(points_t1, joint_type, joint_axis, joint_pivot, -joint_limit_max)


def world_to_cam_opencv_from_nerf_c2w_gl(c2w):
    """
    Convert NeRF-style camera-to-world (OpenGL) to OpenCV world-to-camera.
    
    NeRF/instant-ngp: transform_matrix is c2w in OpenGL coords (Y up, Z back).
    Returns R, t such that X_cam = R @ X_world + t in OpenCV coords (Y down, Z forward).
    """
    c2w = torch.tensor(c2w, dtype=torch.float32)
    w2c_gl = torch.inverse(c2w)
    R_gl = w2c_gl[:3, :3]
    t_gl = w2c_gl[:3, 3:4]
    
    # OpenGL to OpenCV: flip Y and Z
    F = torch.diag(torch.tensor([1.0, -1.0, -1.0]))
    R_cv = F @ R_gl
    t_cv = F @ t_gl
    
    return R_cv, t_cv


def project_points_to_mask(Xw, R, t, K, w, h):
    """
    Project 3D points to 2D and create filled convex hull mask.
    
    Args:
        Xw: Nx3 world points
        R, t: OpenCV camera extrinsics (world-to-camera)
        K: 3x3 intrinsics matrix
        w, h: image dimensions
    
    Returns:
        HxW binary mask (uint8)
    """
    # Transform to camera space
    Xc = (R @ Xw.T + t).T  # Nx3
    Z = Xc[:, 2]
    
    # Filter points behind camera
    valid = Z > 1e-3
    if valid.sum() == 0:
        return np.zeros((h, w), dtype=np.uint8)
    
    Xc = Xc[valid]
    
    # Project to image plane
    x = Xc[:, 0] / Xc[:, 2]
    y = Xc[:, 1] / Xc[:, 2]
    u = (K[0, 0] * x + K[0, 2]).cpu().numpy()
    v = (K[1, 1] * y + K[1, 2]).cpu().numpy()
    
    # Keep only points inside image bounds
    inb = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    u, v = u[inb], v[inb]
    
    if len(u) < 3:
        return np.zeros((h, w), dtype=np.uint8)
    
    # Create convex hull mask
    pts = np.stack([u, v], axis=1).astype(np.int32)
    hull = cv2.convexHull(pts)
    
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)
    
    return mask


def main(data_dir="data/gs_t_multi_post", 
         voxel_subdir="obj_masks",
         voxel_filename=None,
         write_overlays=False, 
         update_json=True, 
         backup_json=True,
         close_kernel=9, 
         close_iters=1):
    
    data_dir = Path(data_dir)
    
    # Auto-detect voxel filename if not provided
    if voxel_filename is None:
        pt_files = list((data_dir / voxel_subdir).glob("obj_*.pt"))
        if not pt_files:
            raise FileNotFoundError(f"No obj_*.pt files found in {data_dir / voxel_subdir}")
        voxel_filename = pt_files[0].name
        print(f"Auto-detected voxel file: {voxel_filename}")

    T, M = load_data(data_dir, voxel_subdir, voxel_filename)

    # Intrinsics
    fx, fy = float(T["fl_x"]), float(T["fl_y"])
    cx, cy = float(T["cx"]), float(T["cy"])
    W, H = int(T["w"]), int(T["h"])
    K = torch.tensor([[fx, 0.0, cx],
                      [0.0, fy, cy],
                      [0.0, 0.0, 1.0]], dtype=torch.float32)

    # Extract voxel and joint data
    bmin = M["bbox_min"].float()
    bmax = M["bbox_max"].float()
    voxel = M["voxel"].bool()
    joint_axis = M["joint_axis"].float()
    joint_pivot = M["joint_pivot"].float()
    joint_type = M["joint_type"]
    joint_limit_min = M.get("joint_limit_min", torch.tensor(0.0)).float()
    joint_limit_max = M.get("joint_limit_max", torch.tensor(0.0)).float()

    print(f"\n{'='*60}")
    print(f"Joint Configuration:")
    print(f"  Type: {joint_type}")
    print(f"  Axis: {joint_axis.tolist()}")
    print(f"  Pivot: {joint_pivot.tolist()}")
    print(f"  Limits: [{joint_limit_min.item():.4f}, {joint_limit_max.item():.4f}]")
    print(f"{'='*60}\n")

    # Get 3D points at t=1 (voxelized at max limit)
    points_t1 = occupied_points_from_voxels(bmin, bmax, voxel)
    print(f"Voxel occupancy: {len(points_t1)} points at t=1 (max limit)")

    # De-articulate back to t=0 (rest pose)
    points_rest = de_articulate_from_max(
        points_t1, joint_type, joint_axis, joint_pivot, joint_limit_max
    )
    print(f"De-articulated to rest pose (t=0, joint_value={joint_limit_min.item():.4f})")

    frames = T["frames"]
    print(f"\nProcessing {len(frames)} frames...")

    # Create output directories
    out_pre = data_dir / "masks_pre"
    out_post = data_dir / "masks_post"
    out_pre.mkdir(exist_ok=True)
    out_post.mkdir(exist_ok=True)

    if write_overlays:
        overlay_pre_dir = out_pre / "overlays"
        overlay_post_dir = out_post / "overlays"
        overlay_pre_dir.mkdir(exist_ok=True)
        overlay_post_dir.mkdir(exist_ok=True)

    for i, frame in enumerate(tqdm(frames, desc="Processing frames", unit="frame"), 1):
        img_rel = frame["file_path"]

        joint_value = float(frame.get("joint_angle", 0.0))
        
        c2w = frame["transform_matrix"]
        R, t = world_to_cam_opencv_from_nerf_c2w_gl(c2w)

        # --- PRE mask (rest pose, t=0) ---
        mask_pre = project_points_to_mask(points_rest, R, t, K, W, H)

        # --- POST mask (articulated pose at current joint value) ---
        points_post = articulate_points(
            points_rest, joint_type, joint_axis, joint_pivot, joint_value
        )
        mask_post = project_points_to_mask(points_post, R, t, K, W, H)

        # Apply morphological closing
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
        if mask_pre.any():
            mask_pre = cv2.morphologyEx(mask_pre, cv2.MORPH_CLOSE, kernel, iterations=close_iters)
        if mask_post.any():
            mask_post = cv2.morphologyEx(mask_post, cv2.MORPH_CLOSE, kernel, iterations=close_iters)

        # Save masks
        img_name = Path(img_rel).stem
        mask_pre_rel = f"masks_pre/{img_name}.jpg"
        mask_post_rel = f"masks_post/{img_name}.jpg"

        cv2.imwrite(str(data_dir / mask_pre_rel), mask_pre)
        cv2.imwrite(str(data_dir / mask_post_rel), mask_post)

        # Update JSON
        if update_json:
            frame["mask_pre_path"] = mask_pre_rel
            frame["mask_post_path"] = mask_post_rel

        # Overlays (optional, for debugging)
        if write_overlays:
            img_path = data_dir / img_rel
            if img_path.exists():
                rgb = cv2.imread(str(img_path))
                if rgb is not None and rgb.shape[:2] == (H, W):
                    # Pre overlay (red for rest pose)
                    overlay_pre = rgb.copy()
                    overlay_pre[mask_pre > 0] = [0, 0, 255]  # Red (BGR)
                    result_pre = cv2.addWeighted(rgb, 0.7, overlay_pre, 0.3, 0)
                    cv2.imwrite(str(overlay_pre_dir / f"{img_name}.jpg"), result_pre)
                    
                    # Post overlay (green for articulated pose)
                    overlay_post = rgb.copy()
                    overlay_post[mask_post > 0] = [0, 255, 0]  # Green (BGR)
                    result_post = cv2.addWeighted(rgb, 0.7, overlay_post, 0.3, 0)
                    cv2.imwrite(str(overlay_post_dir / f"{img_name}.jpg"), result_post)

    # Save updated JSON
    if update_json:
        tj = data_dir / "transforms_arti.json"
        if backup_json and tj.exists():
            backup_path = data_dir / "transforms_arti.backup.json"
            backup_path.write_bytes(tj.read_bytes())
            print(f"\n✅ Backup written: {backup_path}")
        
        with open(tj, "w") as g:
            json.dump(T, g, indent=2)
        print(f"✅ Updated {tj} with mask paths")

    print(f"\n{'='*60}")
    print(f"Mask generation complete!")
    print(f"  Pre-masks:  {out_pre}")
    print(f"  Post-masks: {out_post}")
    print(f"  Total: {len(frames)} frame pairs")
    print(f"{'='*60}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Generate 2D masks for articulated objects (prismatic or revolute joints)"
    )
    ap.add_argument("--data_dir", 
                    help="Path containing obj_masks/, rgb/, transforms_arti.json")
    ap.add_argument("--voxel-subdir", default="obj_masks",
                    help="Subdirectory containing voxel .pt file")
    ap.add_argument("--voxel-file", default=None,
                    help="Voxel filename (auto-detected if not specified)")
    ap.add_argument("--overlays", action="store_true",
                    help="Write RGB overlays for debugging")
    ap.add_argument("--no-update-json", action="store_true",
                    help="Do not modify transforms_arti.json")
    ap.add_argument("--backup-json", action="store_true", default=True,
                    help="Write transforms_arti.backup.json before editing")
    ap.add_argument("--close-kernel", type=int, default=9,
                    help="Morphological closing kernel size")
    ap.add_argument("--close-iters", type=int, default=1,
                    help="Morphological closing iterations")
    
    args = ap.parse_args()

    main(
        data_dir=args.data_dir,
        voxel_subdir=args.voxel_subdir,
        voxel_filename=args.voxel_file,
        write_overlays=args.overlays,
        update_json=not args.no_update_json,
        backup_json=args.backup_json,
        close_kernel=args.close_kernel,
        close_iters=args.close_iters
    )