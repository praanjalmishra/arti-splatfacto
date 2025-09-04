#!/usr/bin/env python3
import json
from pathlib import Path
import torch
import numpy as np
import cv2
import json, re
import argparse


def load_data(data_dir: Path):
    with open(data_dir / "transforms_post.json", "r") as f:
        T = json.load(f)
    # load the single .pt in obj_prismatic/
    # pt_path = sorted((data_dir / "obj_prismatic").glob("*.pt"))[0]
    pt_path = data_dir / "obj_masks" / "obj_prismatic.pt"
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

def translate_along_axis(points, axis, displacement):
    """Translate Nx3 points along axis by displacement amount."""
    axis = axis / torch.norm(axis)
    return points + displacement * axis.unsqueeze(0)

def world_to_cam_opencv_from_nerf_c2w_gl(c2w):
    """
    NeRF/instant-ngp style: transform_matrix is camera->world in OpenGL coords.
    We need world->camera in OpenCV coords.
      1) w2c_gl = inverse(c2w_gl)
      2) convert OpenGL cam to OpenCV cam with F = diag([1,-1,-1])
         (x same, y down, z forward)
    Returns 3x3 R (OpenCV), 3x1 t (OpenCV) so that Xc = R*Xw + t
    """
    c2w = torch.tensor(c2w, dtype=torch.float32)
    w2c_gl = torch.inverse(c2w)
    R_gl = w2c_gl[:3, :3]
    t_gl = w2c_gl[:3, 3:4]
    F = torch.diag(torch.tensor([1.0, -1.0, -1.0]))  # GL->CV
    R_cv = F @ R_gl
    t_cv = F @ t_gl
    return R_cv, t_cv

def project_points_cv(Xw, R, t, K, w, h):
    """
    Projects 3D points into a filled silhouette mask.
    """
    Xc = (R @ Xw.T + t).T  # Nx3
    Z = Xc[:, 2]
    valid = Z > 1e-3
    if valid.sum() == 0:
        return np.zeros((h, w), dtype=np.uint8)

    Xc = Xc[valid]
    x = Xc[:, 0] / Xc[:, 2]
    y = Xc[:, 1] / Xc[:, 2]
    u = (K[0, 0] * x + K[0, 2]).cpu().numpy()
    v = (K[1, 1] * y + K[1, 2]).cpu().numpy()

    # Keep only points inside image
    inb = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    u, v = u[inb], v[inb]

    if len(u) < 3:
        return np.zeros((h, w), dtype=np.uint8)

    pts = np.stack([u, v], axis=1).astype(np.int32)
    hull = cv2.convexHull(pts)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)

    return mask

def get_joint_displacement_for_time(joint_value, joint_limit_min, joint_limit_max):
    """
    Convert joint value to displacement for prismatic joint.
    For prismatic joints, joint_value is typically the linear displacement.
    
    Args:
        joint_value: Current joint position (linear displacement)
        joint_limit_min: Minimum joint limit (fully retracted position)
        joint_limit_max: Maximum joint limit (fully extended position)
    
    Returns:
        displacement: Linear displacement from rest position (t=0)
    """
    # For prismatic joints, the joint_value IS the displacement
    # We can clamp it to the joint limits if needed
    displacement = torch.clamp(torch.tensor(joint_value, dtype=torch.float32), 
                              joint_limit_min, joint_limit_max)
    return displacement


# ---------- main ----------
def main(data_dir="data/gs_t_multi_post", out_dir="masks_new",
         write_overlays=False, update_json=True, backup_json=True,
         close_kernel=9, close_iters=1):
    
    data_dir = Path(data_dir)
    out_dir = data_dir / out_dir

    T, M = load_data(data_dir)

    # Intrinsics
    fx, fy = float(T["fl_x"]), float(T["fl_y"])
    cx, cy = float(T["cx"]),  float(T["cy"])
    W, H = int(T["w"]), int(T["h"])
    K = torch.tensor([[fx, 0.0, cx],
                      [0.0, fy, cy],
                      [0.0, 0.0, 1.0]], dtype=torch.float32)

    # 3D object (world, rest pose at t=0)
    bmin = M["bbox_min"].float()
    bmax = M["bbox_max"].float()
    voxel = M["voxel"].bool()
    joint_axis  = M["joint_axis"].float()
    joint_type  = M["joint_type"]
    joint_limit_min = M.get("joint_limit_min", torch.tensor(0.0))
    joint_limit_max = M.get("joint_limit_max", torch.tensor(1.0))

    # Verify this is a prismatic joint
    if joint_type != "prismatic":
        print(f"Warning: Expected prismatic joint, got {joint_type}")

    # Get 3D points from voxel template (rest pose)
    points_t1 = occupied_points_from_voxels(bmin, bmax, voxel)
    joint_limit_max = 0.3  # hardcoded for now

    # Get rest position (t=0, joint_value = 0.0)
    points_rest = translate_along_axis(points_t1, joint_axis, -joint_limit_max)

    frames = T["frames"]
    print(f"Found {len(frames)} frames. Generating prismatic joint masks to subfolders...")

    # Create output directories
    out_pre  = data_dir / "masks_pre"
    out_post = data_dir / "masks_post"
    out_pre.mkdir(exist_ok=True)
    out_post.mkdir(exist_ok=True)

    for i, f in enumerate(frames, 1):
        img_rel = f["file_path"]
        
        # For prismatic joints, we expect joint_value (displacement) instead of joint_angle
        joint_value = float(f.get("joint_angle", 0.0))  # Use joint_value for prismatic
        # if "joint_value" not in f and "joint_angle" in f:
        #     # Fallback to joint_angle if joint_value not present
        #     joint_value = float(f.get("joint_angle", 0.0))
        #     print(f"Warning: Using joint_angle as joint_value for frame {i}")

        c2w = f["transform_matrix"]
        R, t = world_to_cam_opencv_from_nerf_c2w_gl(c2w)

        # --- PRE mask (rest pose, t=0, no displacement) ---
        mask_pre = project_points_cv(points_rest, R, t, K, W, H)

        # --- POST mask (articulated pose at current joint displacement) ---
        # displacement = get_joint_displacement_for_time(joint_value, joint_limit_min, joint_limit_max)
        points_post = translate_along_axis(points_rest, joint_axis, joint_value)
        mask_post = project_points_cv(points_post, R, t, K, W, H)

        # Apply morphological close if non-empty
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
        if mask_post.any():
            mask_post = cv2.morphologyEx(mask_post, cv2.MORPH_CLOSE, kernel, iterations=close_iters)
        if mask_pre.any():
            mask_pre = cv2.morphologyEx(mask_pre, cv2.MORPH_CLOSE, kernel, iterations=close_iters)

        # Save masks (keep same filename as RGB stem)
        img_name = Path(img_rel).stem
        mask_pre_rel  = f"masks_pre/{img_name}.png"
        mask_post_rel = f"masks_post/{img_name}.png"

        cv2.imwrite(str(data_dir / mask_pre_rel), mask_pre)
        cv2.imwrite(str(data_dir / mask_post_rel), mask_post)

        # Update JSON (store both paths)
        if update_json:
            f["mask_pre_path"]  = mask_pre_rel
            f["mask_post_path"] = mask_post_rel

        # Overlays (optional, for debugging)
        if write_overlays:
            img_path = data_dir / img_rel
            if img_path.exists():
                rgb = cv2.imread(str(img_path))
                if rgb is not None and rgb.shape[1] == W and rgb.shape[0] == H:
                    overlay_pre = rgb.copy()
                    overlay_pre[mask_pre > 0] = [255, 0, 0]   # red for pre (t=0)
                    overlay_post = rgb.copy()
                    overlay_post[mask_post > 0] = [0, 255, 0] # green for post (t=t)
                    cv2.imwrite(str(out_pre / f"{img_name}_overlay.png"),
                                cv2.addWeighted(rgb, 0.7, overlay_pre, 0.3, 0))
                    cv2.imwrite(str(out_post / f"{img_name}_overlay.png"),
                                cv2.addWeighted(rgb, 0.7, overlay_post, 0.3, 0))

        if i % 10 == 0:
            print(f"Processed {i}/{len(frames)} frames...")

    # Save updated JSON
    if update_json:
        tj = data_dir / "transforms_post.json"
        if backup_json and tj.exists():
            (data_dir / "transforms_post.backup.json").write_bytes(tj.read_bytes())
            print("Backup written → transforms_post.backup.json")
        with open(tj, "w") as g:
            json.dump(T, g, indent=2)
        print("✅ Updated transforms_post.json with mask_path for all frames.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser("Generate 2D masks for prismatic joints")
    ap.add_argument("data_dir", help="Path containing obj_prismatic/, rgb_new/, transforms_post.json")
    ap.add_argument("--out", default="masks_new", help="Output subfolder name (relative)")
    ap.add_argument("--overlays", action="store_true", help="Write RGB overlays for debugging")
    ap.add_argument("--no-update-json", action="store_true", help="Do not modify transforms_post.json")
    ap.add_argument("--backup-json", action="store_true", help="Write transforms_post.backup.json before editing")
    ap.add_argument("--close-kernel", type=int, default=9, help="Morph close kernel size")
    ap.add_argument("--close-iters", type=int, default=1, help="Morph close iterations")
    args = ap.parse_args()

    main(
        data_dir=args.data_dir,
        out_dir=args.out,
        write_overlays=args.overlays,
        update_json=not args.no_update_json,
        backup_json=args.backup_json,
        close_kernel=args.close_kernel,
        close_iters=args.close_iters
    )