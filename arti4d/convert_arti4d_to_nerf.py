"""
ARTi4D → Nerfstudio converter  (with interaction splits)
=========================================================
Converts ARTi4D dataset format to Nerfstudio's transforms.json,
splitting frames into canonical + per-joint articulation subdirectories.

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
  ├── matched_cues.csv        (AXIS_NAME, CUE_START, CUE_END, VERIFICATION)
  └── compressed_point_cloud.ply

Output structure:
  <output_dir>/
  ├── canonical/
  │   ├── frames/frame_00001.jpg  ...
  │   ├── depth/frame_00001.png   ...
  │   ├── point_cloud.ply
  │   └── transforms.json
  ├── articulated_joint0/         (window-drawer-1)
  │   ├── frames/
  │   ├── depth/
  │   └── transforms.json
  ├── articulated_joint1/         (window-drawer-2)
  │   └── ...
  └── ...

Notes:
  - CUE_START / CUE_END are treated as 1-based frame indices matching the
    sorted order of rgb_image_*.jpg files in the rgb/ directory.
  - Frames inside ANY interaction window are excluded from canonical/.
  - Each articulated_joint<N>/ contains only the frames in that window.
  - Depth saved as uint16 PNG (millimetres); use --depth-unit-scale-factor
    0.001 at ns-train time to recover metres.
  - Coordinate convention: odom → camera_link → optical → flip(Y=180°)

Usage:
  python convert_arti4d_to_nerfstudio.py \\
      --scene_dir /path/to/scene \\
      --output_dir /path/to/output \\
      [--cues_csv /path/to/matched_cues.csv]  # default: scene_dir/matched_cues.csv
      [--only_verified]                        # skip non-VERIFIED cues
      [--max_frames 350]                       # uniform-sample canonical only

Train (example):
  ns-train nerfacto       --data /path/to/output/canonical
  ns-train depth-nerfacto --data /path/to/output/articulated_joint0
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
# Coordinate transforms
# ──────────────────────────────────────────────────────────────

R_LINK_TO_OPTICAL = np.array([
    [ 0,  0,  1],
    [-1,  0,  0],
    [ 0, -1,  0],
], dtype=float)

R_FLIP = Rotation.from_euler("y", 180, degrees=True).as_matrix()


def build_c2w(row: pd.Series) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat([row.qx, row.qy, row.qz, row.qw]).as_matrix()
    T[:3, 3]  = [row.x, row.y, row.z]
    T[:3, :3] = T[:3, :3] @ R_LINK_TO_OPTICAL @ R_FLIP
    return T


# ──────────────────────────────────────────────────────────────
# Camera intrinsics
# ──────────────────────────────────────────────────────────────

def parse_camera_info(path: Path) -> dict:
    import re
    text = path.read_text()
    info = {}

    for key, out in [("width", "w"), ("height", "h")]:
        m = re.search(rf"^{key}:\s*(\d+)", text, re.MULTILINE)
        if m and int(m.group(1)) > 0:
            info[out] = int(m.group(1))

    m = re.search(r"distortion_model:\s*(\S+)", text)
    if m:
        info["distortion_model"] = m.group(1)

    m = re.search(r"^K:\s*\(([^)]+)\)", text, re.MULTILINE)
    if m:
        v = [float(x.strip()) for x in m.group(1).split(",")]
        assert len(v) == 9
        info.update(fx=v[0], cx=v[2], fy=v[4], cy=v[5])

    m = re.search(r"^D:\s*\(([^)]+)\)", text, re.MULTILINE)
    if m:
        d    = [float(x.strip()) for x in m.group(1).split(",")]
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
        raise ValueError(f"Could not parse {missing} from {path}")
    return info


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def ts_from_name(name: str) -> int:
    return int(Path(name).stem.split("_")[-1])


def uniform_sample(lst, n):
    idx = np.linspace(0, len(lst) - 1, n, dtype=int)
    return [lst[i] for i in idx]


def base_transforms(cam: dict, has_ply: bool) -> dict:
    return {
        "camera_model":  "OPENCV",
        "ply_file_path": "point_cloud.ply" if has_ply else None,
        "fl_x": cam["fx"],
        "fl_y": cam["fy"],
        "cx":   cam["cx"],
        "cy":   cam["cy"],
        "w":    cam["w"],
        "h":    cam["h"],
        "frames": [],
    }


def write_transforms(out_dir: Path, transforms: dict):
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "transforms.json", "w") as f:
        json.dump(transforms, f, indent=2)


def process_frame(
    i: int,
    rgb_path: Path,
    odom_df: pd.DataFrame,
    odom_ts: np.ndarray,
    depth_src_files: list,
    depth_ts_arr: np.ndarray,
    out_dir: Path,
    frame_label: int,          # 1-based index used in filename
) -> dict | None:
    """
    Read one RGB + depth pair, write to out_dir/frames/ and out_dir/depth/.
    Returns the frame dict for transforms.json, or None on failure.
    """
    ts_ns = ts_from_name(rgb_path.name)
    ts_s  = ts_ns / 1e9

    idx_pose = int(np.argmin(np.abs(odom_ts - ts_s)))
    row      = odom_df.iloc[idx_pose]
    T_c2w    = build_c2w(row)

    frame_name = f"frame_{frame_label:05d}.jpg"
    frames_dir = out_dir / "frames"
    depth_dir  = out_dir / "depth"
    frames_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    dst_rgb = frames_dir / frame_name
    if not dst_rgb.exists():
        img = cv2.imread(str(rgb_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"  [WARN] Could not read {rgb_path}, skipping")
            return None
        if img.dtype == np.uint16:
            img = (img / 256).astype(np.uint8)
        elif img.dtype != np.uint8:
            img = img.astype(np.uint8)
        if img.ndim == 3 and img.shape[2] == 4:
            img = img[:, :, :3]
        cv2.imwrite(str(dst_rgb), img, [cv2.IMWRITE_JPEG_QUALITY, 95])

    frame = {
        "file_path":        f"frames/{frame_name}",
        "transform_matrix": T_c2w.tolist(),
    }

    if len(depth_src_files) > 0:
        idx_d      = int(np.argmin(np.abs(depth_ts_arr - ts_ns)))
        depth_path = depth_src_files[idx_d]
        depth_raw  = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        depth_m    = depth_raw.astype(np.float32) / 1000.0
        depth_m[depth_raw == 0] = 0.0
        depth_m[depth_m > 3.0]  = 0.0
        depth_mm   = (depth_m * 1000.0).astype(np.uint16)
        depth_name = f"frame_{frame_label:05d}.png"
        cv2.imwrite(str(depth_dir / depth_name), depth_mm)
        frame["depth_file_path"] = f"depth/{depth_name}"

    return frame


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def convert(args):
    scene_dir  = Path(args.scene_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Intrinsics ──────────────────────────────────────────
    if args.fx is not None:
        cam = dict(fx=args.fx, fy=args.fy, cx=args.cx, cy=args.cy,
                   w=args.width, h=args.height)
        print(f"[intrinsics] Manual override: {cam}")
    else:
        cam = parse_camera_info(scene_dir / "rgb" / "camera_info.txt")
        print(f"[intrinsics] fx={cam['fx']:.2f}  fy={cam['fy']:.2f}  "
              f"cx={cam['cx']:.2f}  cy={cam['cy']:.2f}  "
              f"w={cam['w']}  h={cam['h']}")

    # ── Odometry ────────────────────────────────────────────
    odom_csvs = list((scene_dir / "odom").glob("*.csv"))
    assert len(odom_csvs) == 1, f"Expected 1 odom CSV, found: {odom_csvs}"
    odom_df = pd.read_csv(odom_csvs[0]).sort_values("timestamp").reset_index(drop=True)
    odom_ts = odom_df["timestamp"].values
    print(f"[odom]  {len(odom_df)} poses")

    # ── Matched cues ────────────────────────────────────────
    cues_csv = Path(args.cues_csv) if args.cues_csv else scene_dir / "matched_cues.csv"
    if not cues_csv.exists():
        raise FileNotFoundError(f"matched_cues.csv not found at {cues_csv}")
    cues_df = pd.read_csv(cues_csv)
    if args.only_verified:
        cues_df = cues_df[cues_df["VERIFICATION"] == "VERIFIED"].reset_index(drop=True)
    print(f"[cues]  {len(cues_df)} interaction windows:")
    for _, row in cues_df.iterrows():
        print(f"         [{row.CUE_START:4d}–{row.CUE_END:4d}]  {row.AXIS_NAME}  ({row.VERIFICATION})")

    # Build a set of all frame indices (1-based) that belong to an interaction
    interaction_frame_indices: set[int] = set()
    for _, row in cues_df.iterrows():
        interaction_frame_indices.update(range(int(row.CUE_START), int(row.CUE_END) + 1))

    # ── Point cloud ─────────────────────────────────────────
    ply_src  = scene_dir / "compressed_point_cloud.ply"
    ply_dest = output_dir / "canonical" / "point_cloud.ply"
    has_ply  = ply_src.exists()
    if has_ply:
        ply_dest.parent.mkdir(parents=True, exist_ok=True)
        if not ply_dest.exists():
            shutil.copy2(ply_src, ply_dest)
        print(f"[ply]   copied → canonical/point_cloud.ply")
    else:
        print(f"[ply]   WARNING: {ply_src} not found")

    # ── RGB files (sorted → 1-based frame index) ────────────
    rgb_files = sorted(
        set(list((scene_dir / "rgb").glob("rgb_image_*.jpg")) +
            list((scene_dir / "rgb").glob("rgb_image_*.png")))
    )
    total_frames = len(rgb_files)
    print(f"[rgb]   {total_frames} images total")

    # ── Depth lookup ────────────────────────────────────────
    depth_src_files = sorted((scene_dir / "depth").glob("depth_image_*.png"))
    depth_ts_arr    = np.array([ts_from_name(f.name) for f in depth_src_files])
    print(f"[depth] {len(depth_src_files)} depth maps")

    # ── Partition frame indices ──────────────────────────────
    # Frame indices here are 1-based to match CUE_START/CUE_END
    canonical_indices     = [i for i in range(1, total_frames + 1)
                              if i not in interaction_frame_indices]
    print(f"\n[split] {len(canonical_indices)} canonical frames  |  "
          f"{len(interaction_frame_indices)} interaction frames  |  "
          f"{len(cues_df)} joints")

    # ── Build canonical/ ────────────────────────────────────
    canonical_dir    = output_dir / "canonical"
    canonical_tf     = base_transforms(cam, has_ply)
    canonical_count  = 0

    # Optionally sub-sample canonical frames
    sampled_canonical = canonical_indices
    if args.max_frames and len(canonical_indices) > args.max_frames:
        sampled_canonical = uniform_sample(canonical_indices, args.max_frames)
        print(f"[sample] canonical → {len(sampled_canonical)} frames (uniform)")

    print(f"\n── Building canonical/ ({len(sampled_canonical)} frames) ──")
    for label, frame_1based in enumerate(sampled_canonical, start=1):
        rgb_path = rgb_files[frame_1based - 1]
        frame    = process_frame(
            i=frame_1based - 1,
            rgb_path=rgb_path,
            odom_df=odom_df,
            odom_ts=odom_ts,
            depth_src_files=depth_src_files,
            depth_ts_arr=depth_ts_arr,
            out_dir=canonical_dir,
            frame_label=label,
        )
        if frame:
            canonical_tf["frames"].append(frame)
            canonical_count += 1
        if label % 50 == 0 or label == len(sampled_canonical):
            print(f"  [{label:4d}/{len(sampled_canonical)}]", flush=True)

    write_transforms(canonical_dir, canonical_tf)
    print(f"[canonical] {canonical_count} frames written → {canonical_dir}")

    # ── Build articulated_joint<N>/ for each cue ────────────
    for joint_idx, (_, cue) in enumerate(cues_df.iterrows()):
        joint_name   = f"articulated_joint_{joint_idx}"
        joint_dir    = output_dir / joint_name
        joint_tf     = base_transforms(cam, has_ply=False)   # no PLY per joint
        joint_count  = 0

        window_indices = list(range(int(cue.CUE_START), int(cue.CUE_END) + 1))
        # Filter to valid frame range
        window_indices = [i for i in window_indices if 1 <= i <= total_frames]

        print(f"\n── Building {joint_name}/ — {cue.AXIS_NAME}  "
              f"[{cue.CUE_START}–{cue.CUE_END}]  ({len(window_indices)} frames) ──")

        for label, frame_1based in enumerate(window_indices, start=1):
            rgb_path = rgb_files[frame_1based - 1]
            frame    = process_frame(
                i=frame_1based - 1,
                rgb_path=rgb_path,
                odom_df=odom_df,
                odom_ts=odom_ts,
                depth_src_files=depth_src_files,
                depth_ts_arr=depth_ts_arr,
                out_dir=joint_dir,
                frame_label=label,
            )
            if frame:
                joint_tf["frames"].append(frame)
                joint_count += 1

        # Store metadata about which joint this represents
        joint_tf["axis_name"]  = cue.AXIS_NAME
        joint_tf["cue_start"]  = int(cue.CUE_START)
        joint_tf["cue_end"]    = int(cue.CUE_END)

        write_transforms(joint_dir, joint_tf)
        print(f"  → {joint_count} frames written  ({joint_dir})")

    # ── Summary ─────────────────────────────────────────────
    print("\n" + "═" * 60)
    print(f"✅  Done!  Output: {output_dir}")
    print(f"    canonical/            {canonical_count} frames")
    for joint_idx, (_, cue) in enumerate(cues_df.iterrows()):
        n = len([i for i in range(int(cue.CUE_START), int(cue.CUE_END) + 1)
                 if 1 <= i <= total_frames])
        print(f"    articulated_joint_{joint_idx}/   {n:4d} frames  ({cue.AXIS_NAME})")
    print()
    print("Train examples:")
    print(f"  ns-train nerfacto       --data {output_dir}/canonical")
    print(f"  ns-train depth-nerfacto --data {output_dir}/articulated_joint0")
    print()
    print("Note: depth is uint16 PNG in millimetres.")
    print("      Pass --depth-unit-scale-factor 0.001 to depth-nerfacto.")


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Convert ARTi4D → Nerfstudio (canonical + articulated splits)"
    )
    p.add_argument("--scene_dir",      required=True,
                   help="Path to ARTi4D scene directory")
    p.add_argument("--output_dir",     required=True,
                   help="Where to write the Nerfstudio datasets")
    p.add_argument("--cues_csv",       default=None,
                   help="Path to matched_cues.csv (default: scene_dir/matched_cues.csv)")
    p.add_argument("--only_verified",  action="store_true",
                   help="Skip cues that are not marked VERIFIED")
    p.add_argument("--max_frames",     type=int, default=None,
                   help="Uniformly sub-sample canonical frames to at most N")
    # Manual intrinsics override
    p.add_argument("--fx",     type=float, default=None)
    p.add_argument("--fy",     type=float, default=None)
    p.add_argument("--cx",     type=float, default=None)
    p.add_argument("--cy",     type=float, default=None)
    p.add_argument("--width",  type=int,   default=None)
    p.add_argument("--height", type=int,   default=None)
    args = p.parse_args()
    convert(args)