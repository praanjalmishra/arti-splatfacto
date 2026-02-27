#!/usr/bin/env python3
"""
aria_to_nerfstudio_depth.py
============================
Converts a processed Project Aria MPS dataset into a Nerfstudio-compatible
depth-supervised 3DGS dataset.

Depth strategy (in priority order):
  1. Leica scan projection  →  metric, dense, static-only  (--leica_scan)
  2. Affine-aligned mono    →  fills holes + dynamic regions from camera_depth/
     Scale/shift estimated per-frame from Leica↔mono overlap on static pixels.
  3. SLAM semidense fallback if no Leica scan provided.

Pipeline:
  1. Load closed_loop_trajectory_aligned poses (globally aligned to Leica frame)
  2. Load camera calibration (intrinsics + T_device_camera extrinsic)
  3. Load Leica scan / SLAM points
  4. For every sampled RGB frame:
       a. Look up nearest aligned pose  →  T_world_device
       b. Compose  T_world_cam = T_world_device @ T_device_camera
       c. Project Leica/SLAM points into camera  →  sparse metric depth map
       d. If --align_mono_depth: affine-align camera_depth .npy, fill holes
  5. Write transforms.json  +  global_points.ply

Usage:
  python aria_to_nerfstudio_depth.py \\
      --data_dir /path/to/aria_recording \\
      --output_dir /path/to/output \\
      --leica_scan /path/to/scan.ply \\
      [--align_mono_depth] \\
      [--max_frames 350] \\
      [--max_output_size 1408] \\
      [--depth_scale 1000.0]
"""

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import open3d as o3d
    HAS_O3D = True
except ImportError:
    HAS_O3D = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

# ── Aria → Nerfstudio/OpenGL convention ──────────────────────────────────────
# Aria: +X right, +Y down,  +Z forward
# NS  : +X right, +Y up,    +Z backward
R_ARIA_TO_NS = np.array([
    [1, 0,  0],
    [0, -1,  0],
    [0,  0, -1],
], dtype=np.float64)
T_ARIA_TO_NS = np.eye(4, dtype=np.float64)
T_ARIA_TO_NS[:3, :3] = R_ARIA_TO_NS


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def quat_to_rotation(qx, qy, qz, qw) -> np.ndarray:
    n = np.sqrt(qx**2 + qy**2 + qz**2 + qw**2)
    qx, qy, qz, qw = qx/n, qy/n, qz/n, qw/n
    return np.array([
        [1 - 2*(qy**2 + qz**2),     2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [    2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2),     2*(qy*qz - qx*qw)],
        [    2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)],
    ], dtype=np.float64)


def make_transform(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3,  3] = t
    return T


# ─────────────────────────────────────────────────────────────────────────────
# 1. Trajectory
# ─────────────────────────────────────────────────────────────────────────────

def load_trajectory(csv_path: Path) -> Tuple[np.ndarray, List[np.ndarray]]:
    timestamps_ns, transforms = [], []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        ts_scale = None
        for row in reader:
            if ts_scale is None:
                raw = float(row["timestamp"])
                if   raw < 1e6:   ts_scale = int(1e9)
                elif raw < 1e9:   ts_scale = int(1e6)
                elif raw < 1e12:  ts_scale = int(1e3)
                else:             ts_scale = 1
                print(f"  Trajectory timestamp scale: x{ts_scale} ns  (first={raw:.3f})")
            ts = int(float(row["timestamp"]) * ts_scale)
            t  = np.array([float(row["tx_world_device"]),
                           float(row["ty_world_device"]),
                           float(row["tz_world_device"])], dtype=np.float64)
            R  = quat_to_rotation(float(row["qx_world_device"]),
                                   float(row["qy_world_device"]),
                                   float(row["qz_world_device"]),
                                   float(row["qw_world_device"]))
            timestamps_ns.append(ts)
            transforms.append(make_transform(R, t))
    return np.array(timestamps_ns, dtype=np.int64), transforms


# ─────────────────────────────────────────────────────────────────────────────
# 2. Calibration
# ─────────────────────────────────────────────────────────────────────────────

def load_calibration(calib_path: Path) -> Dict:
    with open(calib_path) as f:
        raw = json.load(f)

    if "PINHOLE" in raw:
        entry = raw["PINHOLE"]
        print(f"  calib.json: PINHOLE  fx={entry['focal_length'][0]:.2f}")
    elif "K" in raw:
        entry = raw
    else:
        first_key = next(iter(raw))
        entry = raw[first_key]
        print(f"  calib.json: using first entry '{first_key}'")

    K = np.array(entry["K"], dtype=np.float64)
    if K.ndim == 1:
        K = K.reshape(3, 3)

    T_dc = np.array(entry["T_device_camera"], dtype=np.float64)
    if T_dc.ndim == 1:
        T_dc = T_dc.reshape(4, 4)

    w = int(entry.get("w", entry.get("width",  int(K[0, 2] * 2))))
    h = int(entry.get("h", entry.get("height", int(K[1, 2] * 2))))

    return {"K": K, "T_device_camera": T_dc, "width": w, "height": h}


# ─────────────────────────────────────────────────────────────────────────────
# 3. Point cloud loading
# ─────────────────────────────────────────────────────────────────────────────

def load_leica_scan(ply_path: Path, voxel_size: float = 0.01) -> np.ndarray:
    """
    Load Leica dense scan → Nx3 float64, already in world/Leica frame.
    Downsamples to ~1 cm voxels if > 5M points.
    """
    if not HAS_O3D:
        raise ImportError("open3d required: pip install open3d")
    print(f"Loading Leica scan: {ply_path} ...")
    pcd = o3d.io.read_point_cloud(str(ply_path))
    pts = np.asarray(pcd.points, dtype=np.float64)
    print(f"  {len(pts):,} points loaded.")
    if len(pts) > 5_000_000:
        print(f"  Downsampling to {voxel_size*100:.0f} cm voxels ...")
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        pts = np.asarray(pcd.points, dtype=np.float64)
        print(f"  After downsample: {len(pts):,} points.")
    return pts


def load_semidense_points(points_dir: Path) -> np.ndarray:
    """Load Aria MPS semidense SLAM points → Nx3 float64."""
    candidates = (list(points_dir.glob("*.csv.gz")) +
                  list(points_dir.glob("*.csv"))    +
                  list(points_dir.glob("*.ply")))
    if not candidates:
        raise FileNotFoundError(f"No point files in {points_dir}")

    src = candidates[0]
    print(f"  Loading SLAM points from {src.name} ...")

    if src.suffix == ".ply":
        if not HAS_O3D:
            raise ImportError("open3d required")
        pcd = o3d.io.read_point_cloud(str(src))
        return np.asarray(pcd.points, dtype=np.float64)

    import gzip
    XYZ_ALIASES = [
        ("px_world", "py_world", "pz_world"),
        ("px",       "py",       "pz"),
        ("x",        "y",        "z"),
        ("position_x", "position_y", "position_z"),
    ]
    open_fn = gzip.open if src.suffix == ".gz" else open
    rows, xcol, ycol, zcol = [], None, None, None
    with open_fn(src, "rt") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if xcol is None:
                for xc, yc, zc in XYZ_ALIASES:
                    if xc in row:
                        xcol, ycol, zcol = xc, yc, zc
                        print(f"  Point columns: {xcol}, {ycol}, {zcol}")
                        break
                if xcol is None:
                    raise KeyError(f"Cannot find XYZ columns. Got: {list(row.keys())}")
            try:
                rows.append([float(row[xcol]), float(row[ycol]), float(row[zcol])])
            except (KeyError, ValueError):
                continue
    return np.array(rows, dtype=np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Depth projection
# ─────────────────────────────────────────────────────────────────────────────

def project_points_to_depth(
    points_world: np.ndarray,
    T_cam_world:  np.ndarray,   # world → camera (inverse of c2w)
    K:            np.ndarray,
    width:        int,
    height:       int,
    max_depth:    float = 20.0,
) -> np.ndarray:
    """Project Nx3 world points → float32 H×W depth map (0 = no data)."""
    ones  = np.ones((len(points_world), 1), dtype=np.float64)
    P_cam = (T_cam_world @ np.hstack([points_world, ones]).T).T   # Nx4
    X, Y, Z = P_cam[:, 0], P_cam[:, 1], P_cam[:, 2]

    mask    = (Z > 0.01) & (Z < max_depth)
    X, Y, Z = X[mask], Y[mask], Z[mask]
    u       = (K[0, 0] * X / Z + K[0, 2]).astype(np.int32)
    v       = (K[1, 1] * Y / Z + K[1, 2]).astype(np.int32)
    valid   = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    u, v, Z = u[valid], v[valid], Z[valid]

    depth_map = np.zeros((height, width), dtype=np.float32)
    order     = np.argsort(Z)[::-1]          # far → near, closer wins
    depth_map[v[order], u[order]] = Z[order].astype(np.float32)
    return depth_map


# ─────────────────────────────────────────────────────────────────────────────
# 5. Mono depth loading + affine alignment
# ─────────────────────────────────────────────────────────────────────────────

def load_mono_depth_npy(npy_path: Path) -> Optional[np.ndarray]:
    """
    Load camera_depth .npy → float32 H×W in metres.
    Aria camera_depth values are in mm — auto-converts if mean > 100.
    """
    if not npy_path.exists():
        return None
    d = np.load(str(npy_path)).astype(np.float32)
    if d.mean() > 100:
        d = d / 1000.0
    return d


def align_mono_to_leica(
    D_mono:    np.ndarray,   
    D_leica:   np.ndarray,   
    min_valid: int = 50,
) -> Tuple[Optional[np.ndarray], bool]:


    valid = (D_leica > 0) & (D_mono > 0.01)
    n_valid = int(valid.sum())
    if n_valid < min_valid:
        return None, False

    mono_v  = D_mono[valid].astype(np.float64)
    leica_v = D_leica[valid].astype(np.float64)


    lo, hi = np.percentile(mono_v, [1, 99])
    mask = (mono_v >= lo) & (mono_v <= hi)
    mono_v  = mono_v[mask]
    leica_v = leica_v[mask]

    if len(mono_v) < min_valid:
        return None, False

    # Scale-only least squares: s = (m·l) / (m·m)
    denom = np.dot(mono_v, mono_v)
    if denom < 1e-8:
        return None, False

    s = np.dot(mono_v, leica_v) / denom

    if not np.isfinite(s) or s <= 0:
        return None, False

    D_aligned = (s * D_mono.astype(np.float64)).astype(np.float32)
    D_aligned = np.clip(D_aligned, 0, None)

    # print(f"overlap: {n_valid} | scale: {s:.4f}")

    return D_aligned, True

def find_mono_depth_path(depth_dir: Path, rgb_stem: str) -> Optional[Path]:
    """Find camera_depth .npy matching a given RGB frame stem."""
    for candidate in [
        depth_dir / f"{rgb_stem}.npy",
        depth_dir / f"{rgb_stem}_depth.npy",
    ]:
        if candidate.exists():
            return candidate
    # Fallback: match by timestamp digits
    digits = "".join(c for c in rgb_stem if c.isdigit())
    if digits:
        for p in depth_dir.glob(f"*{digits}*.npy"):
            return p
    return None


# ─────────────────────────────────────────────────────────────────────────────
# 6. Depth map I/O
# ─────────────────────────────────────────────────────────────────────────────

def save_depth_map(depth_map: np.ndarray, out_path: Path, scale: float = 1000.0) -> str:
    """Save float32 depth (metres) as uint16 PNG or NPZ. Returns suffix used."""
    if HAS_CV2:
        cv2.imwrite(str(out_path.with_suffix(".png")),
                    (depth_map * scale).astype(np.uint16))
        return ".png"
    np.savez_compressed(str(out_path.with_suffix(".npz")), depth=depth_map)
    return ".npz"


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def list_rgb_images(rgb_dir: Path) -> List[Tuple[int, Path]]:
    result = []
    for p in sorted(rgb_dir.glob("**/*.jpg")) + sorted(rgb_dir.glob("**/*.png")):
        digits = "".join(c for c in p.stem if c.isdigit())
        if digits:
            result.append((int(digits), p))
    result.sort(key=lambda x: x[0])
    return result


def run(args: argparse.Namespace) -> None:
    data_dir   = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "images").mkdir(exist_ok=True)
    (output_dir / "depth").mkdir(exist_ok=True)

    # ── 1. Trajectory ─────────────────────────────────────────────────────────
    traj_aligned = data_dir / "slam" / "closed_loop_trajectory_aligned" / "data.csv"
    traj_default = data_dir / "slam" / "closed_loop_trajectory" / "data.csv"
    traj_path    = traj_aligned if traj_aligned.exists() else traj_default
    print(f"Loading trajectory: {traj_path}")
    ts_ns, T_wd_list = load_trajectory(traj_path)
    print(f"  {len(ts_ns)} poses loaded.")

    # ── 2. Calibration ────────────────────────────────────────────────────────
    calib_path = data_dir / "calib" / "calib.json"
    print(f"Loading calibration: {calib_path}")
    calib  = load_calibration(calib_path)
    K      = calib["K"]
    width  = calib["width"]
    height = calib["height"]

    # Aria RGB sensor is mounted 90° CW around its optical axis — compose correction.
    T_device_rectcam = calib["T_device_camera"]
    R_z_neg90 = np.array([[ 0., 1., 0., 0.],
                           [-1., 0., 0., 0.],
                           [ 0., 0., 1., 0.],
                           [ 0., 0., 0., 1.]], dtype=np.float64)
    T_device_rectcam = T_device_rectcam @ R_z_neg90
    print("  Applied -90 deg Z rotation to T_device_camera.")

    # Output resolution
    if max(width, height) > args.max_output_size:
        sf    = args.max_output_size / max(width, height)
        K_out = K.copy()
        K_out[0] *= sf
        K_out[1] *= sf
        out_w, out_h = int(width * sf), int(height * sf)
    else:
        K_out, out_w, out_h = K.copy(), width, height

    # ── 3. Point cloud ────────────────────────────────────────────────────────
    # closed_loop_trajectory_aligned is already in Leica frame.
    # Leica scan is in Leica frame. No T_wq needed for either.
    points_world: Optional[np.ndarray] = None
    if args.leica_scan:
        points_world = load_leica_scan(Path(args.leica_scan))
    else:
        pts_dir = data_dir / "slam" / "semidense_points"
        if pts_dir.exists():
            print("No --leica_scan provided — falling back to SLAM semidense points.")
            try:
                points_world = load_semidense_points(pts_dir)
                print(f"  {len(points_world):,} SLAM points loaded.")
            except Exception as e:
                print(f"  WARNING: {e}")
        else:
            print("WARNING: No point cloud source. Provide --leica_scan.")

    # ── 4. Mono depth directory ───────────────────────────────────────────────
    mono_depth_dir: Optional[Path] = None
    if args.align_mono_depth:
        candidate = data_dir / "camera_depth"
        if candidate.exists():
            mono_depth_dir = candidate
            print(f"Mono depth: {mono_depth_dir}")
        else:
            print("WARNING: --align_mono_depth set but camera_depth/ not found.")

    # ── 5. Per-frame loop ─────────────────────────────────────────────────────
    all_frames = list_rgb_images(data_dir / "camera_rgb")
    print(f"RGB images found: {len(all_frames)}")

    if len(all_frames) > args.max_frames:
        indices  = [int(x) for x in np.linspace(0, len(all_frames) - 1, args.max_frames)]
        selected = [all_frames[i] for i in indices]
    else:
        selected = all_frames
    print(f"Processing {len(selected)} frames ...\n")

    nerfstudio_frames                = []
    mono_stats: Dict[str, int]       = {"ok": 0, "fail": 0, "missing": 0}
    rel_depth: Optional[str]         = None

    for frame_ts, rgb_path in selected:

        # Nearest pose
        idx            = min(int(np.searchsorted(ts_ns, frame_ts)), len(ts_ns) - 1)
        T_world_device = T_wd_list[idx]

        # Camera transforms
        T_world_cam_aria = T_world_device @ T_device_rectcam   # c2w Aria
        T_world_cam_ns   = T_world_cam_aria @ T_ARIA_TO_NS     # c2w Nerfstudio
        T_cam_world      = np.linalg.inv(T_world_cam_aria)     # world → cam

        # Symlink / copy RGB
        dest_rgb = output_dir / "images" / rgb_path.name
        if not dest_rgb.exists():
            if args.copy_images:
                shutil.copy2(rgb_path, dest_rgb)
            else:
                try:
                    dest_rgb.symlink_to(rgb_path.resolve())
                except FileExistsError:
                    pass

        # Depth
        depth_stem = rgb_path.stem
        depth_out  = output_dir / "depth" / depth_stem
        rel_depth  = None

        if points_world is not None:
            D_leica = project_points_to_depth(
                points_world, T_cam_world, K_out, out_w, out_h,
                max_depth=args.max_depth,
            )
            final_depth = D_leica.copy()

            # Affine-align mono and fill holes (dynamic regions + Leica gaps)
            if mono_depth_dir is not None:
                mono_path = find_mono_depth_path(mono_depth_dir, depth_stem)
                if mono_path is None:
                    mono_stats["missing"] += 1
                else:
                    D_mono = load_mono_depth_npy(mono_path)
                    if D_mono is not None:
                        if D_mono.shape != (out_h, out_w) and HAS_CV2:
                            D_mono = cv2.resize(D_mono, (out_w, out_h),
                                                interpolation=cv2.INTER_LINEAR)
                        D_aligned, ok = align_mono_to_leica(D_mono, D_leica)
                        if ok:
                            fill = (final_depth == 0) & (D_aligned > 0)
                            final_depth[fill] = D_aligned[fill]
                            mono_stats["ok"] += 1
                        else:
                            mono_stats["fail"] += 1

            suffix    = save_depth_map(final_depth, depth_out, scale=args.depth_scale)
            rel_depth = f"depth/{depth_stem}{suffix}"

        frame_entry: Dict = {
            "file_path":        f"images/{rgb_path.name}",
            "transform_matrix": T_world_cam_ns.tolist(),
            "timestamp":        frame_ts,
            "fl_x": float(K_out[0, 0]),
            "fl_y": float(K_out[1, 1]),
            "cx":   float(K_out[0, 2]),
            "cy":   float(K_out[1, 2]),
            "w":    out_w,
            "h":    out_h,
        }
        if rel_depth:
            frame_entry["depth_file_path"]         = rel_depth
            frame_entry["depth_unit_scale_factor"] = 1.0 / args.depth_scale

        nerfstudio_frames.append(frame_entry)

    # ── 6. Global point cloud ─────────────────────────────────────────────────
    ply_rel_path: Optional[str] = None
    if points_world is not None and HAS_O3D:
        ply_out = output_dir / "global_points.ply"
        pcd     = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_world)
        o3d.io.write_point_cloud(str(ply_out), pcd)
        ply_rel_path = "global_points.ply"
        print(f"Saved {ply_out}")

    # ── 7. transforms.json ────────────────────────────────────────────────────
    transforms = {
        "camera_model":            "OPENCV",
        # "has_depth":               rel_depth is not None,
        # "depth_unit_scale_factor": 1.0 / args.depth_scale,
        "frames":                  nerfstudio_frames,
    }
    if ply_rel_path:
        transforms["ply_file_path"] = ply_rel_path

    tf_path = output_dir / "transforms.json"
    tf_path.write_text(json.dumps(transforms, indent=2))

    print(f"\nWrote {tf_path}  ({len(nerfstudio_frames)} frames)")
    if mono_depth_dir is not None:
        print(f"Mono alignment — ok: {mono_stats['ok']}  "
              f"fail: {mono_stats['fail']}  missing: {mono_stats['missing']}")
    print("Done.")


# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert Aria MPS data to depth-supervised Nerfstudio dataset."
    )
    p.add_argument("--data_dir",         required=True,
                   help="Root of the Aria recording directory")
    p.add_argument("--output_dir",       required=True,
                   help="Where to write the Nerfstudio dataset")
    p.add_argument("--leica_scan",       default=None,
                   help="Path to Leica scan .ply — metric depth source (strongly preferred)")
    p.add_argument("--align_mono_depth", action="store_true",
                   help="Affine-align camera_depth/ .npy to Leica and fill holes")
    p.add_argument("--max_frames",       type=int,   default=350)
    p.add_argument("--max_output_size",  type=int,   default=1408,
                   help="Max image dimension (px)")
    p.add_argument("--max_depth",        type=float, default=20.0,
                   help="Depth clip in metres")
    p.add_argument("--depth_scale",      type=float, default=1000.0,
                   help="Multiplier for uint16 PNG saving (1000 = mm)")
    p.add_argument("--copy_images",      action="store_true",
                   help="Copy RGB images instead of symlinking")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())