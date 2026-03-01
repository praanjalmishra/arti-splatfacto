"""
ARTi4D → Nerfstudio converter
================================
Converts ARTi4D dataset format to Nerfstudio's transforms.json

Dataset structure expected:
  <scene_dir>/
  ├── rgb/
  │   ├── camera_info.txt
  │   └── rgb_image_<timestamp_ns>.jpg
  ├── depth/
  │   ├── camera_info.txt
  │   └── depth_image_<timestamp_ns>.png
  ├── odom/
  │   └── <scene_name>.csv   (timestamp, x, y, z, qx, qy, qz, qw)
  └── compressed_point_cloud.ply

Coordinate convention (verified working):
  1. Load T from odom (world-from-camera-link)
  2. Apply R_link_to_optical: rotate axes to ROS optical frame
  3. Apply R_flip (180° around Y): correct forward direction
  Depth: saved as float32 .npy in metres (raw PNG is uint16 mm)

Usage:
  python convert_arti4d_to_nerfstudio.py \\
      --scene_dir /path/to/scene \\
      --output_dir /path/to/output \\
      [--max_frames 350]

Train:
  ns-train nerfacto        --data /path/to/output
  ns-train depth-nerfacto  --data /path/to/output
"""

import json
import shutil
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial.transform import Rotation
import cv2


# ──────────────────────────────────────────────────────────────
# Coordinate transforms (verified against compressed_point_cloud.ply)
# ──────────────────────────────────────────────────────────────

# Step 1: camera_link → optical frame  (x=right, y=down, z=forward)
R_LINK_TO_OPTICAL = np.array([
    [ 0,  0,  1],
    [-1,  0,  0],
    [ 0, -1,  0],
], dtype=float)

# Step 2: flip forward direction (180° around Y)
R_FLIP = Rotation.from_euler("y", 180, degrees=True).as_matrix()


def build_c2w(row: pd.Series) -> np.ndarray:
    """
    Convert one odom row → 4×4 camera-to-world matrix (NeRF convention).
    row must have: qx, qy, qz, qw, x, y, z
    """
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat([row.qx, row.qy, row.qz, row.qw]).as_matrix()
    T[:3, 3]  = [row.x, row.y, row.z]
    T[:3, :3] = T[:3, :3] @ R_LINK_TO_OPTICAL @ R_FLIP
    return T


# ──────────────────────────────────────────────────────────────
# Camera intrinsics
# ──────────────────────────────────────────────────────────────
def parse_camera_info(path: Path) -> dict:
    """
    Parse ARTi4D camera_info.txt (ROS CameraInfo text format).
    Returns fx, fy, cx, cy, w, h + full distortion coefficients.
    """
    import re
    text = path.read_text()
    info = {}

    # Width / height — first occurrence (top-level, not ROI zeros)
    for key, out in [("width", "w"), ("height", "h")]:
        m = re.search(rf"^{key}:\s*(\d+)", text, re.MULTILINE)
        if m and int(m.group(1)) > 0:
            info[out] = int(m.group(1))

    # Distortion model
    m = re.search(r"distortion_model:\s*(\S+)", text)
    if m:
        info["distortion_model"] = m.group(1)

    # K matrix: (fx, 0, cx, 0, fy, cy, 0, 0, 1)
    m = re.search(r"^K:\s*\(([^)]+)\)", text, re.MULTILINE)
    if m:
        v = [float(x.strip()) for x in m.group(1).split(",")]
        assert len(v) == 9, f"K matrix should have 9 values, got {len(v)}"
        info.update(fx=v[0], cx=v[2], fy=v[4], cy=v[5])

    # D vector — rational_polynomial order: k1, k2, p1, p2, k3, k4, k5, k6
    m = re.search(r"^D:\s*\(([^)]+)\)", text, re.MULTILINE)
    if m:
        d = [float(x.strip()) for x in m.group(1).split(",")]
        model = info.get("distortion_model", "")
        if model == "rational_polynomial" and len(d) == 8:
            info.update(k1=d[0], k2=d[1], p1=d[2], p2=d[3],
                        k3=d[4], k4=d[5], k5=d[6], k6=d[7])
        elif len(d) >= 4:
            info.update(k1=d[0], k2=d[1], p1=d[2], p2=d[3])
            if len(d) > 4:
                info["k3"] = d[4]

    missing = [k for k in ("fx", "fy", "cx", "cy", "w", "h") if k not in info]
    if missing:
        raise ValueError(f"Could not parse {missing} from {path}\nContents:\n{text}")
    return info


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────
def ts_from_name(name: str) -> int:
    """Extract nanosecond timestamp from rgb_image_<ts>.jpg or depth_image_<ts>.png"""
    return int(Path(name).stem.split("_")[-1])


def uniform_sample(lst, n):
    idx = np.linspace(0, len(lst) - 1, n, dtype=int)
    return [lst[i] for i in idx]



# ──────────────────────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────────────────────
def validate_output(output_dir: Path, transforms: dict, n_check: int = 10):
    """
    Spot-check that output images are readable by PIL (same as Nerfstudio uses).
    Prints a summary so you know training will actually load images.
    """
    from PIL import Image
    frames = transforms["frames"]
    check_frames = frames[::max(1, len(frames)//n_check)][:n_check]

    ok, bad = 0, []
    for f in check_frames:
        img_path = output_dir / f["file_path"]
        try:
            img = Image.open(img_path)
            arr = np.array(img)
            assert arr.ndim == 3 and arr.shape[2] == 3, f"Expected HxWx3, got {arr.shape}"
            assert arr.dtype == np.uint8, f"Expected uint8, got {arr.dtype}"
            ok += 1
        except Exception as e:
            bad.append((f["file_path"], str(e)))

    print(f"\n[validate] {ok}/{len(check_frames)} images OK (spot-check)")
    if bad:
        print(f"  ❌ Failed images:")
        for path, err in bad:
            print(f"     {path}: {err}")
    else:
        print(f"  ✅ All checked images are 8-bit RGB — Nerfstudio should load them fine")
    return len(bad) == 0


# ──────────────────────────────────────────────────────────────
# Main conversion
# ──────────────────────────────────────────────────────────────
def convert(args):
    scene_dir  = Path(args.scene_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Intrinsics ────────────────────────────
    if args.fx is not None:
        cam = dict(fx=args.fx, fy=args.fy, cx=args.cx, cy=args.cy,
                   w=args.width, h=args.height)
        print(f"[intrinsics] Manual override: {cam}")
    else:
        cam = parse_camera_info(scene_dir / "rgb" / "camera_info.txt")
        print(f"[intrinsics] fx={cam['fx']:.2f}  fy={cam['fy']:.2f}  "
              f"cx={cam['cx']:.2f}  cy={cam['cy']:.2f}  "
              f"w={cam['w']}  h={cam['h']}")
        print(f"             distortion_model={cam.get('distortion_model','?')}  "
              f"k1={cam.get('k1',0):.4f}  k2={cam.get('k2',0):.4f}  "
              f"p1={cam.get('p1',0):.6f}  p2={cam.get('p2',0):.6f}")

    # ── Odometry ─────────────────────────────
    odom_csvs = list((scene_dir / "odom").glob("*.csv"))
    assert len(odom_csvs) == 1, f"Expected 1 odom CSV, found: {odom_csvs}"
    odom_df = pd.read_csv(odom_csvs[0]).sort_values("timestamp").reset_index(drop=True)
    odom_ts  = odom_df["timestamp"].values
    print(f"[odom]  {len(odom_df)} poses")

    # ── Point cloud ───────────────────────────
    ply_src  = scene_dir / "compressed_point_cloud.ply"
    ply_dest = output_dir / "point_cloud.ply"
    if ply_src.exists():
        if not ply_dest.exists():
            shutil.copy2(ply_src, ply_dest)
        print(f"[ply]   copied → point_cloud.ply")
        has_ply = True
    else:
        print(f"[ply]   WARNING: {ply_src} not found")
        has_ply = False

    # ── RGB files ────────────────────────────
    rgb_files = sorted((scene_dir / "rgb").glob("rgb_image_*.jpg")) + \
                sorted((scene_dir / "rgb").glob("rgb_image_*.png"))
    rgb_files = sorted(set(rgb_files))
    print(f"[rgb]   {len(rgb_files)} images")

    if args.max_frames and len(rgb_files) > args.max_frames:
        rgb_files = uniform_sample(rgb_files, args.max_frames)
        print(f"[sample] → {len(rgb_files)} frames (uniform)")

    # ── Depth lookup (ts_ns → Path) ──────────
    depth_src_files = sorted((scene_dir / "depth").glob("depth_image_*.png"))
    depth_ts_arr    = np.array([ts_from_name(f.name) for f in depth_src_files])
    print(f"[depth] {len(depth_src_files)} depth maps")

    # ── Output dirs ───────────────────────────
    frames_dir = output_dir / "frames";  frames_dir.mkdir(exist_ok=True)
    depth_dir  = output_dir / "depth";   depth_dir.mkdir(exist_ok=True)

    # ── Build frames ──────────────────────────
    frames  = []
    n_depth = 0

    for i, rgb_path in enumerate(rgb_files):
        ts_ns = ts_from_name(rgb_path.name)
        ts_s  = ts_ns / 1e9

        # Nearest odom pose
        idx_pose = int(np.argmin(np.abs(odom_ts - ts_s)))
        row      = odom_df.iloc[idx_pose]
        T_c2w    = build_c2w(row)

        frame_name = f"frame_{i + 1:05d}.jpg"

        # Read and re-encode as standard 8-bit JPEG
        # (avoids Nerfstudio bugs with 16-bit PNGs or non-standard encodings)
        dst_rgb = frames_dir / frame_name
        if not dst_rgb.exists():
            img = cv2.imread(str(rgb_path), cv2.IMREAD_UNCHANGED)
            if img is None:
                print(f"  [WARN] Could not read {rgb_path}, skipping")
                continue
            # Handle 16-bit → 8-bit
            if img.dtype == np.uint16:
                img = (img / 256).astype(np.uint8)
            elif img.dtype != np.uint8:
                img = img.astype(np.uint8)
            # Handle RGBA → RGB
            if img.ndim == 3 and img.shape[2] == 4:
                img = img[:, :, :3]
            cv2.imwrite(str(dst_rgb), img, [cv2.IMWRITE_JPEG_QUALITY, 95])

        frame = {
            "file_path":        f"frames/{frame_name}",  # .jpg, 8-bit
            "transform_matrix": T_c2w.tolist(),
        }

        # Depth — nearest timestamp, save as float32 .npy in metres
        if len(depth_src_files) > 0:
            idx_d      = int(np.argmin(np.abs(depth_ts_arr - ts_ns)))
            depth_path = depth_src_files[idx_d]

            depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
            depth_m   = depth_raw.astype(np.float32) / 1000.0   # mm → m
            depth_m[depth_raw == 0] = 0.0                        # mark invalid
            depth_m[depth_m > 3.0]  = 0.0                        # cap noise beyond 3m

            # Save as uint16 PNG in millimetres — nerfstudio-data expects 16-bit PNG
            # Use --depth-unit-scale-factor 0.001 at train time to convert back to metres
            depth_mm   = (depth_m * 1000.0).astype(np.uint16)
            depth_name = f"frame_{i + 1:05d}.png"
            cv2.imwrite(str(depth_dir / depth_name), depth_mm)
            frame["depth_file_path"] = f"depth/{depth_name}"
            n_depth += 1

        frames.append(frame)

        if (i + 1) % 50 == 0 or (i + 1) == len(rgb_files):
            print(f"  [{i+1:4d}/{len(rgb_files)}]  {frame_name}", flush=True)

    print(f"[frames] {len(frames)} built  |  {n_depth} with depth")

    # ── transforms.json ───────────────────────
    transforms = {
        "camera_model":  "OPENCV",
        "ply_file_path": "point_cloud.ply" if has_ply else None,
        "fl_x": cam["fx"],
        "fl_y": cam["fy"],
        "cx":   cam["cx"],
        "cy":   cam["cy"],
        "w":    cam["w"],
        "h":    cam["h"],
        # # Full distortion — rational_polynomial (k1-k6, p1, p2)
        # "k1":   cam.get("k1", 0.0),
        # "k2":   cam.get("k2", 0.0),
        # "k3":   cam.get("k3", 0.0),
        # "p1":   cam.get("p1", 0.0),
        # "p2":   cam.get("p2", 0.0),
        "frames": frames,
    }

    out_json = output_dir / "transforms.json"
    with open(out_json, "w") as f:
        json.dump(transforms, f, indent=2)

    validate_output(output_dir, transforms)

    print(f"\n✅  {out_json}")
    print(f"    frames : {len(frames)}")
    print(f"    depth  : {n_depth}  (.npy float32 metres)")
    print(f"    ply    : {'yes' if has_ply else 'no'}")
    print(f"\nTrain:")
    print(f"  ns-train nerfacto       --data {output_dir}")
    print(f"  ns-train depth-nerfacto --data {output_dir}")


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Convert ARTi4D → Nerfstudio transforms.json")
    p.add_argument("--scene_dir",  required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--max_frames", type=int, default=None,
                   help="Uniformly sample N frames (e.g. 350)")
    # Manual intrinsics override
    p.add_argument("--fx",     type=float, default=None)
    p.add_argument("--fy",     type=float, default=None)
    p.add_argument("--cx",     type=float, default=None)
    p.add_argument("--cy",     type=float, default=None)
    p.add_argument("--width",  type=int,   default=None)
    p.add_argument("--height", type=int,   default=None)
    args = p.parse_args()
    convert(args)