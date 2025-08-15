#!/usr/bin/env python3
import json
from pathlib import Path
from turtle import update
import torch
import numpy as np
import cv2
import json, re
import argparse
# ---------- helpers ----------
def load_data(data_dir: Path):
    with open(data_dir / "transforms_post.json", "r") as f:
        T = json.load(f)
    # load the single .pt in obj_masks/
    pt_path = sorted((data_dir / "obj_masks").glob("*.pt"))[0]
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

def rodrigues_axis_angle(axis, angle):
    """3x3 rotation from unit axis (3,) & scalar angle (rad)."""
    axis = axis / torch.norm(axis)
    x, y, z = axis
    K = torch.tensor([[0, -z,  y],
                      [z,  0, -x],
                      [-y, x,  0]], dtype=torch.float32)
    I = torch.eye(3, dtype=torch.float32)
    c, s = torch.cos(angle), torch.sin(angle)
    return I + s*K + (1-c)*(K@K)

def rotate_about_pivot(points, pivot, axis, angle):
    """Rotate Nx3 points around 'pivot' by 'angle' about 'axis' (all world)."""
    R = rodrigues_axis_angle(axis.float(), torch.as_tensor(angle, dtype=torch.float32))
    pc = points - pivot.unsqueeze(0)
    return (pc @ R.T) + pivot.unsqueeze(0)

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
    Xw: Nx3 world points; R:3x3, t:3x1 (OpenCV, z forward, y down)
    K: 3x3 pixel intrinsics; w,h: image size
    Returns a boolean mask HxW with projected points set True.
    """
    Xc = (R @ Xw.T + t).T  # Nx3
    Z = Xc[:, 2]
    valid = Z > 1e-3
    if valid.sum() == 0:
        return np.zeros((h, w), dtype=np.uint8)

    Xc = Xc[valid]
    x = Xc[:, 0] / Xc[:, 2]
    y = Xc[:, 1] / Xc[:, 2]
    u = K[0,0]*x + K[0,2]
    v = K[1,1]*y + K[1,2]

    # rasterize
    u = torch.round(u).long()
    v = torch.round(v).long()
    inb = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    u, v = u[inb], v[inb]
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[v.numpy(), u.numpy()] = 255
    return mask

def guess_mask_path(rgb_rel: str, masks_dir: str):
    """
    Given 'rgb_new/frame_000066.png', try (in order):
      masks_new/frame_000066_mask.png
      masks_new/frame_000066.png
      masks_new/000066_mask.png
      masks_new/000066.png
    Returns (Path, tried_list)
    """
    rgb_path = Path(rgb_rel)
    stem = rgb_path.stem                    # "frame_000066"
    digits = "".join(re.findall(r"\d+", stem))  # "000066" (if any)

    candidates = [
        # Path(masks_dir) / f"{stem}_mask.png",
        Path(masks_dir) / f"{stem}.png",
    ]
    if digits:
        candidates += [
            # Path(masks_dir) / f"{digits}_mask.png",
            Path(masks_dir) / f"{digits}.png",
        ]
    return candidates
# ---------- main ----------
def main(data_dir="data/gs_t_multi_post", out_dir="masks_new",
         write_overlays=True, update_json=True, backup_json=True,
         flip_angle=True, close_kernel=9, close_iters=8):
    data_dir = Path(data_dir)
    out_dir = data_dir / out_dir
    out_dir.mkdir(exist_ok=True)

    T, M = load_data(data_dir)

    # Intrinsics
    fx, fy = float(T["fl_x"]), float(T["fl_y"])
    cx, cy = float(T["cx"]),  float(T["cy"])
    W, H = int(T["w"]), int(T["h"])
    K = torch.tensor([[fx, 0.0, cx],
                      [0.0, fy, cy],
                      [0.0, 0.0, 1.0]], dtype=torch.float32)

    # 3D object (world, rest pose)
    bmin = M["bbox_min"].float()
    bmax = M["bbox_max"].float()
    voxel = M["voxel"].bool()
    joint_axis  = M["joint_axis"].float()
    joint_pivot = M["joint_pivot"].float()

    Xw0 = occupied_points_from_voxels(bmin, bmax, voxel)
    if Xw0.numel() == 0:
        print("No occupied voxels found."); return

    frames = T["frames"]
    print(f"Found {len(frames)} frames. Generating masks to {out_dir} ...")

    out_subdir = out_dir.relative_to(data_dir).as_posix()

    for i, f in enumerate(frames, 1):
        img_rel = f["file_path"]
        angle = float(f.get("joint_angle", 0.0))
        if flip_angle:
            angle = -angle  # flip rotation direction if needed

        c2w = f["transform_matrix"]

        # rotate articulated object in world
        Xw = rotate_about_pivot(Xw0, joint_pivot, joint_axis, angle)

        # project
        R, t = world_to_cam_opencv_from_nerf_c2w_gl(c2w)
        mask = project_points_cv(Xw, R, t, K, W, H)

        # densify (morph close)
        if mask.any():
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=close_iters)

        # save mask
        img_name = Path(img_rel).stem
        mask_rel = f"{out_subdir}/{img_name}.png"
        cv2.imwrite(str(data_dir / mask_rel), mask)

        # overlay for sanity
        if write_overlays:
            img_path = data_dir / img_rel
            if img_path.exists():
                rgb = cv2.imread(str(img_path))
                if rgb is not None and rgb.shape[1] == W and rgb.shape[0] == H:
                    overlay = rgb.copy()
                    overlay[mask > 0] = [0, 255, 0]
                    blend = cv2.addWeighted(rgb, 0.7, overlay, 0.3, 0)
                    cv2.imwrite(str(out_dir / f"{img_name}_overlay.png"), blend)

        # attach path into JSON
        if update_json:
            f["mask_path"] = mask_rel  # consistent with file_path/depth_file_path (relative)

        if i % 25 == 0 or i == len(frames):
            print(f"  {i}/{len(frames)}")

    # write JSON back
    if update_json:
        tj = data_dir / "transforms_post.json"
        if backup_json and tj.exists():
            (data_dir / "transforms_post.backup.json").write_bytes(tj.read_bytes())
            print("Backup written → transforms_post.backup.json")
        with open(tj, "w") as g:
            json.dump(T, g, indent=2)
        print("✅ Updated transforms_post.json with mask_path for all frames.")

    print(f"✅ Done. Masks in: {out_dir}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser("Generate 2D masks and update transforms JSON")
    ap.add_argument("data_dir", help="Path containing obj_masks/, rgb_new/, transforms_post.json")
    ap.add_argument("--out", default="masks_new", help="Output subfolder name (relative)")
    ap.add_argument("--no-overlays", action="store_true", help="Do not write RGB overlays")
    ap.add_argument("--no-update-json", action="store_true", help="Do not modify transforms_post.json")
    ap.add_argument("--backup-json", action="store_true", help="Write transforms_post.backup.json before editing")
    ap.add_argument("--no-flip-angle", action="store_true", help="Use +angle instead of -angle for articulation")
    ap.add_argument("--close-kernel", type=int, default=9, help="Morph close kernel size")
    ap.add_argument("--close-iters", type=int, default=8, help="Morph close iterations")
    args = ap.parse_args()

    main(
        data_dir=args.data_dir,
        out_dir=args.out,
        write_overlays=not args.no_overlays,
        update_json=not args.no_update_json,
        backup_json=args.backup_json,
        flip_angle=not args.no_flip_angle,
        close_kernel=args.close_kernel,
        close_iters=args.close_iters
    )