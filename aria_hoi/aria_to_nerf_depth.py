#!/usr/bin/env python3
"""
aria_to_nerfstudio_depth.py
============================
Converts a processed Project Aria MPS dataset into a Nerfstudio-compatible
depth-supervised 3DGS dataset.

Pipeline:
  1. Load closed_loop_trajectory_aligned poses (globally aligned to Leica frame)
  2. Load camera calibration (intrinsics + T_device_camera extrinsic)
  3. Load semi-dense SLAM points from semidense_points/
  4. For every sampled RGB frame:
       a. Look up nearest aligned pose  →  T_world_device
       b. Compose  T_world_cam = T_world_device @ T_device_camera
       c. Project semi-dense world points into camera  →  depth map (float32 EXR / NPZ)
  5. Optionally apply visual_registration T_wq to bring everything into the
     InLoc / Leica reference frame
  6. Write transforms.json  +  optional global_points.ply

Usage:
  python aria_to_nerfstudio_depth.py \\
      --data_dir /path/to/bathroom_2_1-5_hand_vrs \\
      --output_dir /path/to/output \\
      [--max_frames 350] \\
      [--max_output_size 1408] \\
      [--depth_scale 1000.0] \\
      [--apply_visual_registration] \\
      [--use_hardware_depth]

Outputs (inside output_dir):
  images/          – symlinked or copied RGB frames
  depth/           – per-frame float32 depth maps  (<stem>_depth.npz)
  transforms.json  – Nerfstudio JSON (camera_model FISHEYE624 or OPENCV)
  global_points.ply
"""

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── optional heavy deps ──────────────────────────────────────────────────────
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

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

# ── coordinate-system constant (Aria → Nerfstudio / OpenCV) ──────────────────
# Aria device  : +X right, +Y down,  +Z forward   (right-handed)
# Nerfstudio   : +X right, +Y up,    +Z backward  (OpenGL / Blender)
#
# The standard flip (negate Y and Z) leaves depth points 90° CCW in the image
# for this dataset because the Aria RGB sensor is mounted rotated 90° CW
# relative to the device frame. We compose an additional -90° around Z
# (camera-space) to compensate:
#   T_ARIA_TO_NS = [[0,1,0,0],[1,0,0,0],[0,0,-1,0],[0,0,0,1]]
# which is equivalent to: flip_YZ  @  R_z(-90)
R_ARIA_TO_NS = np.array([
    [1,  0,  0],
    [0, -1,  0],
    [0,  0, -1],
], dtype=np.float64)

T_ARIA_TO_NS = np.eye(4, dtype=np.float64)
T_ARIA_TO_NS[:3, :3] = R_ARIA_TO_NS


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def quat_to_rotation(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Hamilton quaternion → 3×3 rotation matrix."""
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
# 1. Load trajectory
# ─────────────────────────────────────────────────────────────────────────────

def load_trajectory(csv_path: Path) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Returns
    -------
    timestamps_ns : int64 array  shape (N,)
    T_world_device_list : list of 4×4 float64 arrays  len N
    """
    timestamps_ns = []
    transforms = []

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        ts_scale = None
        for row in reader:
            if ts_scale is None:
                # Auto-detect units of 'timestamp' from its magnitude.
                # RGB filenames are device-relative nanoseconds (~1e12 for a 30-min session).
                # We need to normalise whatever unit the CSV uses to nanoseconds.
                #
                # Typical magnitudes:
                #   seconds      : ~1e3  (0–3600 s for a 1-hr recording)
                #   milliseconds : ~1e6
                #   microseconds : ~1e9  (most common in Aria MPS)
                #   nanoseconds  : ~1e12
                raw = float(row["timestamp"])
                if   raw < 1e6:   ts_scale = int(1e9)   # seconds → ns
                elif raw < 1e9:   ts_scale = int(1e6)   # milliseconds → ns
                elif raw < 1e12:  ts_scale = int(1e3)   # microseconds → ns
                else:             ts_scale = 1           # already nanoseconds
                print(f"  Trajectory 'timestamp' first value={raw:.3f} → scale={ts_scale} (×ns)")
            ts = int(float(row["timestamp"]) * ts_scale)
            t = np.array([float(row["tx_world_device"]),
                          float(row["ty_world_device"]),
                          float(row["tz_world_device"])], dtype=np.float64)
            R = quat_to_rotation(float(row["qx_world_device"]),
                                  float(row["qy_world_device"]),
                                  float(row["qz_world_device"]),
                                  float(row["qw_world_device"]))
            timestamps_ns.append(ts)
            transforms.append(make_transform(R, t))

    return np.array(timestamps_ns, dtype=np.int64), transforms


# ─────────────────────────────────────────────────────────────────────────────
# 2. Load calibration
# ─────────────────────────────────────────────────────────────────────────────

def load_calibration(calib_path: Path) -> Dict:
    """
    Returns dict with:
      K                 – 3×3 np array
      T_device_camera   – 4×4 np array  (camera frame → device frame)
      width, height     – int

    Handles two calib.json layouts:
      (a) Flat:   {"K": ..., "T_device_camera": ..., "w": ..., "h": ...}
      (b) Nested: {"PINHOLE": {...}, "NON_PINHOLE": {...}}   ← HOI dataset format
          We always prefer PINHOLE — it has the rectified intrinsics, symmetric
          principal point, and no fisheye distortion, making it correct for
          depth projection and Nerfstudio's OPENCV camera model.
    """
    with open(calib_path) as f:
        raw = json.load(f)

    # ── unwrap nested structure if present ───────────────────────────────────
    if "PINHOLE" in raw:
        # HOI / newer Aria calib format
        entry = raw["PINHOLE"]
        print("  calib.json: using PINHOLE entry "
              f"(fx={entry['focal_length'][0]:.2f}, "
              f"cx={entry['principal_point'][0]:.1f}, "
              f"cy={entry['principal_point'][1]:.1f})")
    elif "K" in raw:
        entry = raw
    else:
        # Try first value if it's a dict of dicts
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

    # pinhole_T_device_camera in this dataset's calib.json is incorrect —
    # empirical testing shows T_device_rectcam = T_device_camera (identity composition).
    # i.e. the rectified pinhole camera frame == the physical camera frame.
    # We return T_device_camera directly as the effective cam→device extrinsic.
    if "pinhole_T_device_camera" in entry:
        print("  NOTE: pinhole_T_device_camera present but not used (empirically incorrect).")

    return {
        "K": K,
        "T_device_camera": T_dc,   # rectified cam → device (identity composition confirmed)
        "width": w,
        "height": h,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. Load semi-dense SLAM points
# ─────────────────────────────────────────────────────────────────────────────

def load_semidense_points(points_dir: Path) -> np.ndarray:
    """
    Loads from semidense_points/ directory.
    Aria MPS stores points in CSV-style files; newer versions use .csv.gz.
    Returns Nx3 float64 array of world-frame XYZ positions.
    """
    candidates = list(points_dir.glob("*.csv.gz")) + \
                 list(points_dir.glob("*.csv"))   + \
                 list(points_dir.glob("*.ply"))

    if not candidates:
        raise FileNotFoundError(f"No point files found in {points_dir}")

    src = candidates[0]
    print(f"  Loading points from {src.name} …")

    if src.suffix in (".ply",):
        if not HAS_O3D:
            raise ImportError("open3d required to load .ply point clouds")
        pcd = o3d.io.read_point_cloud(str(src))
        return np.asarray(pcd.points, dtype=np.float64)

    # CSV / CSV.GZ
    import gzip, io
    open_fn = gzip.open if src.suffix == ".gz" else open
    mode    = "rt"
    rows    = []
    # Column name map — tries each alias in order, uses first that exists
    XYZ_ALIASES = [
        ("px_world", "py_world", "pz_world"),   # HOI / newer Aria MPS format
        ("px",       "py",       "pz"),          # older Aria MPS
        ("x",        "y",        "z"),
        ("position_x", "position_y", "position_z"),
    ]

    with open_fn(src, mode) as f:
        reader = csv.DictReader(f)
        xcol = ycol = zcol = None
        for row in reader:
            # Detect columns from first row
            if xcol is None:
                for xc, yc, zc in XYZ_ALIASES:
                    if xc in row:
                        xcol, ycol, zcol = xc, yc, zc
                        print(f"  Using point columns: {xcol}, {ycol}, {zcol}")
                        break
                if xcol is None:
                    raise KeyError(f"Cannot find XYZ columns. Available: {list(row.keys())}")
            try:
                x = float(row[xcol])
                y = float(row[ycol])
                z = float(row[zcol])
                rows.append([x, y, z])
            except (KeyError, ValueError):
                continue
    return np.array(rows, dtype=np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# 3b. Load hardware depth (camera_depth/)
# ─────────────────────────────────────────────────────────────────────────────

def find_hardware_depth_images(depth_dir: Path) -> Dict[int, Path]:
    """Returns {timestamp_ns: path} for hardware depth images."""
    mapping: Dict[int, Path] = {}
    for p in sorted(depth_dir.glob("**/*")):
        if p.suffix.lower() in (".png", ".jpg", ".exr", ".npy", ".npz"):
            # Aria stores depth images with timestamp in filename
            stem = p.stem
            digits = "".join(c for c in stem if c.isdigit())
            if digits:
                mapping[int(digits)] = p
    return mapping


# ─────────────────────────────────────────────────────────────────────────────
# 4. Per-frame depth projection
# ─────────────────────────────────────────────────────────────────────────────

def project_points_to_depth(
    points_world: np.ndarray,   # Nx3
    T_world_cam: np.ndarray,    # 4×4  world → camera  (inverse of c2w)
    K: np.ndarray,              # 3×3
    width: int,
    height: int,
    max_depth: float = 20.0,
) -> np.ndarray:
    """
    Project 3-D world points into a camera and return a float32 depth image
    (H×W) with 0 = no data.

    T_world_cam  is camera-from-world (i.e. P_cam = T_world_cam @ P_world).
    """
    # Transform to camera frame
    ones = np.ones((len(points_world), 1), dtype=np.float64)
    P_world_h = np.hstack([points_world, ones])          # Nx4
    P_cam_h   = (T_world_cam @ P_world_h.T).T            # Nx4

    X, Y, Z = P_cam_h[:, 0], P_cam_h[:, 1], P_cam_h[:, 2]

    # Keep points in front of camera
    mask = (Z > 0.01) & (Z < max_depth)
    X, Y, Z = X[mask], Y[mask], Z[mask]

    # Project
    u = (K[0, 0] * X / Z + K[0, 2]).astype(np.int32)
    v = (K[1, 1] * Y / Z + K[1, 2]).astype(np.int32)

    # Clip to image bounds
    valid = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    u, v, Z = u[valid], v[valid], Z[valid]

    depth_map = np.zeros((height, width), dtype=np.float32)

    # Paint depth – use minimum Z when multiple points project to same pixel
    # (process from far to near so closer points win)
    order = np.argsort(Z)[::-1]
    depth_map[v[order], u[order]] = Z[order].astype(np.float32)

    return depth_map


def save_depth_map(depth_map: np.ndarray, out_path: Path, scale: float = 1000.0):
    """
    Save depth map.  Default: uint16 PNG  (depth in millimetres, scale=1000).
    Falls back to NPZ if opencv not available.
    """
    if HAS_CV2:
        d_mm = (depth_map * scale).astype(np.uint16)
        cv2.imwrite(str(out_path.with_suffix(".png")), d_mm)
    else:
        np.savez_compressed(str(out_path.with_suffix(".npz")), depth=depth_map)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Visual registration alignment
# ─────────────────────────────────────────────────────────────────────────────

def load_T_wq(json_path: Path) -> np.ndarray:
    with open(json_path) as f:
        raw = json.load(f)
    T = np.array(raw["T_wq"], dtype=np.float64)
    if T.shape == (16,):
        T = T.reshape(4, 4)
    return T


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def list_rgb_images(rgb_dir: Path) -> List[Tuple[int, Path]]:
    """Return [(timestamp_ns, path)] sorted by timestamp."""
    result = []
    for p in sorted(rgb_dir.glob("**/*.jpg")) + sorted(rgb_dir.glob("**/*.png")):  # type: ignore[operator]
        stem = p.stem
        digits = "".join(c for c in stem if c.isdigit())
        if digits:
            result.append((int(digits), p))
    result.sort(key=lambda x: x[0])
    return result


def run(args: argparse.Namespace) -> None:
    data_dir   = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "images").mkdir(exist_ok=True)
    (output_dir / "depth" ).mkdir(exist_ok=True)

    # ── 1. Load aligned trajectory ───────────────────────────────────────────
    traj_aligned = data_dir / "slam" / "closed_loop_trajectory_aligned" / "data.csv"
    traj_default = data_dir / "slam" / "closed_loop_trajectory"          / "data.csv"
    traj_path    = traj_aligned if traj_aligned.exists() else traj_default
    print(f"Loading trajectory: {traj_path}")
    ts_ns, T_wd_list = load_trajectory(traj_path)
    print(f"  {len(ts_ns)} poses loaded.")

    # ── 2. Load calibration ──────────────────────────────────────────────────
    calib_path = data_dir / "calib" / "calib.json"
    print(f"Loading calibration: {calib_path}")
    calib = load_calibration(calib_path)
    K              = calib["K"]
    T_device_rectcam = calib["T_device_camera"]   # cam → device (identity composition confirmed)
    width          = calib["width"]
    height         = calib["height"]
    # The Aria RGB sensor is mounted 90° CW around its optical axis relative
    # to the device frame. T_device_camera accounts for the 3D tilt but not
    # this in-plane rotation. Empirically confirmed (fix_rotation.py, candidate B2):
    # we must compose a -90° rotation around the camera's Z axis.
    R_z_neg90 = np.array([[ 0., 1., 0., 0.],
                           [-1., 0., 0., 0.],
                           [ 0., 0., 1., 0.],
                           [ 0., 0., 0., 1.]], dtype=np.float64)
    T_device_rectcam = T_device_rectcam @ R_z_neg90
    print("  Applied -90° Z rotation to T_device_camera (sensor in-plane correction).")

    # Clamp output size
    scale_factor = 1.0
    if max(width, height) > args.max_output_size:
        scale_factor = args.max_output_size / max(width, height)
        K_out = K.copy()
        K_out[0] *= scale_factor
        K_out[1] *= scale_factor
        out_w = int(width  * scale_factor)
        out_h = int(height * scale_factor)
    else:
        K_out = K.copy()
        out_w, out_h = width, height

    # ── Optional visual registration alignment ───────────────────────────────
    T_wq = np.eye(4, dtype=np.float64)
    if args.apply_visual_registration:
        T_wq_path = data_dir / "visual_registration" / "T_wq.json"
        if T_wq_path.exists():
            T_wq = load_T_wq(T_wq_path)
            print(f"Loaded T_wq (visual registration).")
        else:
            print(f"WARNING: --apply_visual_registration set but {T_wq_path} not found.")

    # ── 3. Load SLAM points ──────────────────────────────────────────────────
    points_world: Optional[np.ndarray] = None
    if not args.use_hardware_depth:
        pts_dir = data_dir / "slam" / "semidense_points"
        if pts_dir.exists():
            print("Loading semi-dense SLAM points …")
            try:
                points_world = load_semidense_points(pts_dir)
                # Apply visual registration to points
                # SLAM points are in raw SLAM world frame.
                # Apply T_wq to bring them into Leica frame,
                # matching the aligned trajectory poses.
                if args.apply_visual_registration:
                    ones = np.ones((len(points_world), 1))
                    pts_h = np.hstack([points_world, ones])
                    points_world = (T_wq @ pts_h.T).T[:, :3]
                print(f"  {len(points_world):,} points loaded.")
            except Exception as e:
                print(f"  WARNING: Could not load semidense points: {e}")
        else:
            print("WARNING: No semidense_points directory found.")

    hw_depth_map: Dict[int, Path] = {}
    if args.use_hardware_depth:
        depth_hw_dir = data_dir / "camera_depth"
        if depth_hw_dir.exists():
            print("Indexing hardware depth images …")
            hw_depth_map = find_hardware_depth_images(depth_hw_dir)
            print(f"  {len(hw_depth_map)} depth images found.")
        else:
            print("WARNING: camera_depth/ not found, falling back to SLAM projection.")

    # ── 4. Enumerate RGB images and subsample ────────────────────────────────
    rgb_dir = data_dir / "camera_rgb"
    print(f"Listing RGB images in {rgb_dir} …")
    all_frames = list_rgb_images(rgb_dir)
    print(f"  {len(all_frames)} RGB images found.")

    total = len(all_frames)
    if total > args.max_frames:
        indices = [int(x) for x in np.linspace(0, total - 1, args.max_frames)]
        selected = [all_frames[i] for i in indices]
    else:
        selected = all_frames
    print(f"  Processing {len(selected)} frames.")

    # ── 5. Build transforms.json frames ──────────────────────────────────────
    nerfstudio_frames = []

    for frame_ts, rgb_path in selected:
        # ── a. Nearest aligned pose ──────────────────────────────────────────
        idx = int(np.searchsorted(ts_ns, frame_ts))
        idx = min(idx, len(ts_ns) - 1)
        T_world_device = T_wd_list[idx]           # 4×4

        # NOTE: closed_loop_trajectory_aligned already has T_wq baked in.
        # Do NOT apply T_wq again to poses — that would double-apply it.

        # ── b. Compose world → rectified pinhole camera transform ─────────────
        # T_device_rectcam : rectified pinhole cam → device frame
        # T_world_device   : device frame → world frame
        # → c2w = T_world_device @ T_device_rectcam
        #
        # This correctly accounts for the rectification rotation baked into
        # camera_rgb/ images (pinhole_T_device_camera from calib.json).
        T_world_cam_aria = T_world_device @ T_device_rectcam  # c2w (Aria convention)

        # Convert Aria device convention → Nerfstudio / OpenGL convention
        # Aria: +X right, +Y down,  +Z forward
        # NS  : +X right, +Y up,    +Z backward
        T_world_cam_ns = T_world_cam_aria @ T_ARIA_TO_NS

        # ── c. Copy / symlink RGB image ───────────────────────────────────────
        dest_rgb = output_dir / "images" / rgb_path.name
        if not dest_rgb.exists():
            if args.copy_images:
                shutil.copy2(rgb_path, dest_rgb)
            else:
                try:
                    dest_rgb.symlink_to(rgb_path.resolve())
                except FileExistsError:
                    pass

        # Resize if needed
        rel_rgb = f"images/{rgb_path.name}"

        # ── d. Generate depth map ─────────────────────────────────────────────
        depth_stem = rgb_path.stem
        depth_out  = output_dir / "depth" / depth_stem

        if args.use_hardware_depth and hw_depth_map:
            # Find nearest hardware depth frame
            hw_ts_arr = np.array(sorted(hw_depth_map.keys()), dtype=np.int64)
            hw_idx    = int(np.searchsorted(hw_ts_arr, frame_ts))
            hw_idx    = min(hw_idx, len(hw_ts_arr) - 1)
            hw_path   = hw_depth_map[hw_ts_arr[hw_idx]]
            # Copy hardware depth
            suffix = hw_path.suffix
            shutil.copy2(hw_path, depth_out.with_suffix(suffix))
            rel_depth = f"depth/{depth_stem}{suffix}"

        elif points_world is not None:
            # Project semi-dense SLAM points into the rectified pinhole camera.
            # T_world_cam_aria is already computed using T_device_rectcam, so
            # its inverse correctly maps world → rectified pinhole camera frame.
            T_cam_world = np.linalg.inv(T_world_cam_aria)
            depth_map   = project_points_to_depth(
                points_world, T_cam_world, K_out, out_w, out_h,
                max_depth=args.max_depth,
            )
            save_depth_map(depth_map, depth_out, scale=args.depth_scale)
            suffix    = ".png" if HAS_CV2 else ".npz"
            rel_depth = f"depth/{depth_stem}{suffix}"
        else:
            rel_depth = None

        frame_entry: Dict = {
            "file_path":        rel_rgb,
            "transform_matrix": T_world_cam_ns.tolist(),
            "timestamp":        frame_ts,
            "fl_x":  float(K_out[0, 0]),
            "fl_y":  float(K_out[1, 1]),
            "cx":    float(K_out[0, 2]),
            "cy":    float(K_out[1, 2]),
            "w":     out_w,
            "h":     out_h,
        }
        if rel_depth:
            frame_entry["depth_file_path"] = rel_depth
            frame_entry["depth_unit_scale_factor"] = 1.0 / args.depth_scale  # convert back to metres

        nerfstudio_frames.append(frame_entry)

    # ── 6. Save global point cloud ────────────────────────────────────────────
    ply_rel_path: Optional[str] = None
    if points_world is not None and HAS_O3D:
        ply_out = output_dir / "global_points.ply"
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_world)
        o3d.io.write_point_cloud(str(ply_out), pcd)
        ply_rel_path = "global_points.ply"
        print(f"Saved {ply_out}")

    # ── 7. Write transforms.json ──────────────────────────────────────────────
    transforms = {
        "camera_model": "OPENCV",          # pinhole; change to FISHEYE624 if using distortion
        "has_depth": rel_depth is not None,
        "depth_unit_scale_factor": 1.0 / args.depth_scale,
        "frames": nerfstudio_frames,
    }
    if ply_rel_path:
        transforms["ply_file_path"] = ply_rel_path

    tf_path = output_dir / "transforms.json"
    tf_path.write_text(json.dumps(transforms, indent=2))
    print(f"\nWrote {tf_path}  ({len(nerfstudio_frames)} frames)")
    print("Done.")


# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert Aria MPS data to depth-supervised Nerfstudio dataset."
    )
    p.add_argument("--data_dir",   required=True, help="Root of the Aria recording directory")
    p.add_argument("--output_dir", required=True, help="Where to write the Nerfstudio dataset")
    p.add_argument("--max_frames",      type=int,   default=350,   help="Max RGB frames to process")
    p.add_argument("--max_output_size", type=int,   default=1408,  help="Max image dimension")
    p.add_argument("--max_depth",       type=float, default=20.0,  help="Max depth in metres (clip)")
    p.add_argument("--depth_scale",     type=float, default=1000.0,
                   help="Multiplier when saving depth as uint16 PNG (1000 → mm)")
    p.add_argument("--apply_visual_registration", action="store_true",
                   help="Apply T_wq from visual_registration/T_wq.json to align to Leica frame")
    p.add_argument("--use_hardware_depth", action="store_true",
                   help="Use camera_depth/ images instead of projecting SLAM points")
    p.add_argument("--copy_images", action="store_true",
                   help="Copy RGB images instead of symlinking")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())