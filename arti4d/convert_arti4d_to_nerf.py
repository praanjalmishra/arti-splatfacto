"""
ARTi4D → Nerfstudio converter  (with interaction splits + GT joint info)
=========================================================================
Converts ARTi4D dataset format to Nerfstudio's transforms.json,
splitting frames into canonical + per-joint articulation subdirectories,
and writes a top-level GT_joint_info.json combining scene GT positions/axes
with joint types from metadata.yaml.

Dataset structure expected:
  <arti4d_root>/
  ├── metadata.yaml                          # dataset-wide GT metadata
  └── <scene_name>/                          # e.g. din080
      └── <sequence_name>/                   # e.g. scene_2025-04-11-11-44-32
          ├── rgb/
          │   ├── camera_info.txt
          │   └── rgb_image_<timestamp_ns>.jpg
          ├── depth/
          │   ├── camera_info.txt
          │   └── depth_image_<timestamp_ns>.png
          ├── odom/
          │   └── <sequence_name>.csv   (timestamp, x, y, z, qx, qy, qz, qw)
          ├── matched_cues.csv          (AXIS_NAME, CUE_START, CUE_END, VERIFICATION)
          ├── <sequence_name>.json      (GT joint positions + axes)
          └── compressed_point_cloud.ply

Output structure:
  <output_dir>/
  ├── GT_joint_info.json             ← combined GT: position, axis, joint_type, difficulty
  ├── canonical/
  │   ├── frames/frame_00001.jpg  ...
  │   ├── depth/frame_00001.png   ...
  │   ├── point_cloud.ply
  │   └── transforms.json
  ├── articulated_joint_0/           (e.g. window-drawer-1)
  │   ├── frames/
  │   ├── depth/
  │   └── transforms.json
  ├── articulated_joint_1/
  │   └── ...
  └── ...

GT_joint_info.json schema:
  [
    {
      "joint_dir":    "articulated_joint_0",
      "axis_name":    "window-drawer-1",
      "cue_start":    409,
      "cue_end":      484,
      "verification": "VERIFIED",
      "joint_type":   "PRISMATIC",        // from metadata.yaml (null if not found)
      "difficulty":   "EASY",             // from metadata.yaml (null if not found)
      "position":     [x, y, z],          // GT world-frame position
      "axis":         [ax, ay, az],       // GT unit direction in world frame
    },
    ...
  ]

Notes:
  - CUE_START / CUE_END are treated as 1-based frame indices.
  - Frames inside ANY interaction window are excluded from canonical/.
  - Each articulated_joint_<N>/ contains only the frames in that window.
  - Depth saved as uint16 PNG (millimetres); use --depth-unit-scale-factor
    0.001 at ns-train time to recover metres.
  - Joint positions/axes are in odom world frame = Nerfstudio world frame.
    No additional coordinate transform is needed.

Usage:
  python convert_arti4d_to_nerfstudio.py \\
      --scene_dir   /path/to/arti4d/raw/din080/scene_2025-04-11-11-44-32 \\
      --output_dir  /path/to/output \\
      [--metadata   /path/to/arti4d/raw/metadata.yaml]  \\
      [--cues_csv   /path/to/matched_cues.csv]           \\
      [--only_verified]                                   \\
      [--max_frames 350]

Train (example):
  ns-train nerfacto       --data /path/to/output/canonical
  ns-train depth-nerfacto --data /path/to/output/articulated_joint_0
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
        d     = [float(x.strip()) for x in m.group(1).split(",")]
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
# Metadata (joint_types + difficulty from metadata.yaml)
# ──────────────────────────────────────────────────────────────

def load_metadata(metadata_path: Path, scene_name: str, sequence_name: str) -> tuple[dict, dict]:
    """
    Returns (joint_types, difficulties) dicts keyed by axis_name.
    Both default to empty dicts if metadata is unavailable.
    """
    if metadata_path is None or not metadata_path.exists():
        return {}, {}

    try:
        import yaml
    except ImportError:
        print("[metadata] PyYAML not installed — skipping metadata. "
              "Install with: pip install pyyaml")
        return {}, {}

    with open(metadata_path) as f:
        meta = yaml.safe_load(f)

    joint_types  = {}
    difficulties = {}

    try:
        jt = meta.get("joint_types", {})
        for scene_key, sequences in jt.items():
            if sequence_name in sequences:
                joint_types = dict(sequences[sequence_name])
                break
    except Exception as e:
        print(f"[metadata] Could not parse joint_types: {e}")

    try:
        diff = meta.get("difficulty", {})
        for scene_key, sequences in diff.items():
            if sequence_name in sequences:
                difficulties = dict(sequences[sequence_name])
                break
    except Exception as e:
        print(f"[metadata] Could not parse difficulty: {e}")

    return joint_types, difficulties


# ──────────────────────────────────────────────────────────────
# GT joint positions + axes  (from scene JSON)
# ──────────────────────────────────────────────────────────────

def find_scene_json(scene_dir: Path) -> Path | None:
    """Auto-detect the scene GT JSON."""
    candidates = [f for f in scene_dir.glob("*.json")]
    if not candidates:
        return None
    scene_name = scene_dir.name
    for c in candidates:
        if scene_name in c.stem:
            return c
    return candidates[0]


def load_gt_joints(scene_dir: Path, scene_json_path: Path | None = None) -> dict:
    """Returns dict keyed by axis_name → {position, axis}. Empty if not found."""
    path = scene_json_path or find_scene_json(scene_dir)
    if path is None or not path.exists():
        print(f"[gt]    WARNING: no scene GT JSON found in {scene_dir}")
        return {}
    print(f"[gt]    reading GT joint positions/axes from: {path.name}")
    with open(path) as f:
        return json.load(f)


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
    rgb_path: Path,
    odom_df: pd.DataFrame,
    odom_ts: np.ndarray,
    depth_src_files: list,
    depth_ts_arr: np.ndarray,
    out_dir: Path,
    frame_label: int,
    map1: np.ndarray,
    map2: np.ndarray,
) -> dict | None:
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

        # Rectify RGB
        img_rect = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR)

        cv2.imwrite(str(dst_rgb), img_rect, [cv2.IMWRITE_JPEG_QUALITY, 95])

    frame = {
        "file_path":        f"frames/{frame_name}",
        "transform_matrix": T_c2w.tolist(),
    }

    if len(depth_src_files) > 0:
        idx_d      = int(np.argmin(np.abs(depth_ts_arr - ts_ns)))
        depth_path = depth_src_files[idx_d]
        depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)

        # Rectify depth using same maps
        depth_rect = cv2.remap(
            depth_raw,
            map1,
            map2,
            interpolation=cv2.INTER_NEAREST
        )

        # Convert mm → metres
        depth_m = depth_rect.astype(np.float32) / 1000.0
        depth_m[depth_rect == 0] = 0.0
        depth_m[depth_m > 3.0] = 0.0

        depth_mm = (depth_m * 1000.0).astype(np.uint16)

        depth_name = f"frame_{frame_label:05d}.png"
        cv2.imwrite(str(depth_dir / depth_name), depth_mm)

        frame["depth_file_path"] = f"depth/{depth_name}"

    return frame


# ──────────────────────────────────────────────────────────────
# GT joint info JSON writer
# ──────────────────────────────────────────────────────────────

def build_gt_joint_info(
    cues_df: pd.DataFrame,
    gt_joints: dict,
    joint_types: dict,
    difficulties: dict,
    total_frames: int,
) -> list[dict]:
    """
    Build the list of GT joint dicts to be written to GT_joint_info.json.
    """
    records = []
    for joint_idx, (_, cue) in enumerate(cues_df.iterrows()):
        axis_name  = cue["AXIS_NAME"].strip()
        joint_dir  = f"articulated_joint_{joint_idx}"

        gt = gt_joints.get(axis_name, {})
        position = gt.get("position", None)
        axis_vec = gt.get("axis", None)

        # Normalise axis
        if axis_vec is not None:
            a = np.array(axis_vec, dtype=float)
            norm = np.linalg.norm(a)
            axis_vec = (a / norm).tolist() if norm > 1e-8 else axis_vec

        n_frames = len([i for i in range(int(cue.CUE_START), int(cue.CUE_END) + 1)
                        if 1 <= i <= total_frames])

        record = {
            "joint_dir":    joint_dir,
            "axis_name":    axis_name,
            "cue_start":    int(cue.CUE_START),
            "cue_end":      int(cue.CUE_END),
            "n_frames":     n_frames,
            "verification": cue.get("VERIFICATION", "").strip(),
            "joint_type":   joint_types.get(axis_name, None),
            "difficulty":   difficulties.get(axis_name, None),
            "position":     position,
            "axis":         axis_vec,
        }
        records.append(record)
    return records


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def convert(args):
    scene_dir  = Path(args.scene_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sequence_name = scene_dir.name   # e.g. scene_2025-04-11-11-44-32
    scene_name    = scene_dir.parent.name  # e.g. din080

    # ── Intrinsics ──────────────────────────────────────────
    if args.fx is not None:
        cam = dict(fx=args.fx, fy=args.fy, cx=args.cx, cy=args.cy,
                   w=args.width, h=args.height)
        print(f"[intrinsics] Manual override: {cam}")
    else:
        cam = parse_camera_info(scene_dir / "rgb" / "camera_info.txt")


        # ── Rectification setup ──────────────────────────
        K = np.array([
            [cam["fx"], 0, cam["cx"]],
            [0, cam["fy"], cam["cy"]],
            [0, 0, 1]
        ], dtype=np.float32)

        D = np.array([
            cam.get("k1", 0.0),
            cam.get("k2", 0.0),
            cam.get("p1", 0.0),
            cam.get("p2", 0.0),
            cam.get("k3", 0.0),
            cam.get("k4", 0.0),
            cam.get("k5", 0.0),
            cam.get("k6", 0.0),
        ], dtype=np.float32)

        w, h = cam["w"], cam["h"]

        # Compute rectified intrinsics
        new_K, _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 0)

        # Precompute remap grids
        map1, map2 = cv2.initUndistortRectifyMap(
            K, D, None, new_K, (w, h), cv2.CV_32FC1
        )

        # Replace intrinsics with rectified ones
        cam["fx"] = float(new_K[0, 0])
        cam["fy"] = float(new_K[1, 1])
        cam["cx"] = float(new_K[0, 2])
        cam["cy"] = float(new_K[1, 2])


        print(f"[intrinsics] fx={cam['fx']:.2f}  fy={cam['fy']:.2f}  "
              f"cx={cam['cx']:.2f}  cy={cam['cy']:.2f}  "
              f"w={cam['w']}  h={cam['h']}")

        print(f"[rectified] fx={cam['fx']:.2f}, fy={cam['fy']:.2f}, "
            f"cx={cam['cx']:.2f}, cy={cam['cy']:.2f}")

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

    interaction_frame_indices: set[int] = set()
    for _, row in cues_df.iterrows():
        interaction_frame_indices.update(range(int(row.CUE_START), int(row.CUE_END) + 1))

    # ── Metadata (joint types + difficulty) ─────────────────
    metadata_path = None
    if args.metadata:
        metadata_path = Path(args.metadata)
    else:
        # Auto-detect: walk up to find metadata.yaml
        for parent in [scene_dir.parent.parent, scene_dir.parent.parent.parent]:
            candidate = parent / "metadata.yaml"
            if candidate.exists():
                metadata_path = candidate
                break

    if metadata_path and metadata_path.exists():
        print(f"[meta]  reading: {metadata_path}")
    else:
        print(f"[meta]  WARNING: metadata.yaml not found — joint_type/difficulty will be null")

    joint_types, difficulties = load_metadata(metadata_path, scene_name, sequence_name)
    if joint_types:
        print(f"        {len(joint_types)} joint type(s) loaded")
    if difficulties:
        print(f"        {len(difficulties)} difficulty level(s) loaded")

    # ── GT joint positions + axes ────────────────────────────
    gt_joints = load_gt_joints(scene_dir)

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

    # ── RGB files ────────────────────────────────────────────
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
    canonical_indices = [i for i in range(1, total_frames + 1)
                         if i not in interaction_frame_indices]
    print(f"\n[split] {len(canonical_indices)} canonical frames  |  "
          f"{len(interaction_frame_indices)} interaction frames  |  "
          f"{len(cues_df)} joints")

    # ── Build GT_joint_info.json ─────────────────────────────
    gt_info = build_gt_joint_info(
        cues_df, gt_joints, joint_types, difficulties, total_frames
    )
    gt_info_path = output_dir / "GT_joint_info.json"
    with open(gt_info_path, "w") as f:
        json.dump(gt_info, f, indent=2)
    print(f"\n[GT]    GT_joint_info.json written → {gt_info_path}")
    for rec in gt_info:
        jt   = rec["joint_type"] or "?"
        diff = rec["difficulty"] or "?"
        has_pos = rec["position"] is not None
        print(f"         {rec['joint_dir']:25s}  {rec['axis_name']:30s}  "
              f"{jt:10s}  {diff:5s}  pos={'✓' if has_pos else '✗'}")

    # ── Build canonical/ ────────────────────────────────────
    canonical_dir   = output_dir / "canonical"
    canonical_tf    = base_transforms(cam, has_ply)
    canonical_count = 0

    sampled_canonical = canonical_indices
    if args.max_frames and len(canonical_indices) > args.max_frames:
        sampled_canonical = uniform_sample(canonical_indices, args.max_frames)
        print(f"\n[sample] canonical → {len(sampled_canonical)} frames (uniform)")

    print(f"\n── Building canonical/ ({len(sampled_canonical)} frames) ──")
    for label, frame_1based in enumerate(sampled_canonical, start=1):
        rgb_path = rgb_files[frame_1based - 1]
        frame = process_frame(
            rgb_path=rgb_path,
            odom_df=odom_df,
            odom_ts=odom_ts,
            depth_src_files=depth_src_files,
            depth_ts_arr=depth_ts_arr,
            out_dir=canonical_dir,
            frame_label=label,
            map1=map1,
            map2=map2,
        )
        if frame:
            canonical_tf["frames"].append(frame)
            canonical_count += 1
        if label % 50 == 0 or label == len(sampled_canonical):
            print(f"  [{label:4d}/{len(sampled_canonical)}]", flush=True)

    write_transforms(canonical_dir, canonical_tf)
    print(f"[canonical] {canonical_count} frames written → {canonical_dir}")

    # ── Build articulated_joint_<N>/ ────────────────────────
    for joint_idx, (_, cue) in enumerate(cues_df.iterrows()):
        joint_name  = f"articulated_joint_{joint_idx}"
        joint_dir   = output_dir / joint_name
        joint_tf    = base_transforms(cam, has_ply=False)
        joint_count = 0

        window_indices = [i for i in range(int(cue.CUE_START), int(cue.CUE_END) + 1)
                          if 1 <= i <= total_frames]

        print(f"\n── Building {joint_name}/ — {cue.AXIS_NAME}  "
              f"[{cue.CUE_START}–{cue.CUE_END}]  ({len(window_indices)} frames) ──")

        for label, frame_1based in enumerate(window_indices, start=1):
            rgb_path = rgb_files[frame_1based - 1]
            frame = process_frame(
                rgb_path=rgb_path,
                odom_df=odom_df,
                odom_ts=odom_ts,
                depth_src_files=depth_src_files,
                depth_ts_arr=depth_ts_arr,
                out_dir=joint_dir,   # ✅ FIXED
                frame_label=label,
                map1=map1,
                map2=map2,
            )
            if frame:
                joint_tf["frames"].append(frame)
                joint_count += 1

        # Store metadata in transforms.json for convenience
        axis_name = cue.AXIS_NAME.strip()
        joint_tf["axis_name"]   = axis_name
        joint_tf["cue_start"]   = int(cue.CUE_START)
        joint_tf["cue_end"]     = int(cue.CUE_END)
        joint_tf["joint_type"]  = joint_types.get(axis_name, None)
        joint_tf["difficulty"]  = difficulties.get(axis_name, None)

        write_transforms(joint_dir, joint_tf)
        print(f"  → {joint_count} frames written  ({joint_dir})")

    # ── Summary ─────────────────────────────────────────────
    print("\n" + "═" * 65)
    print(f"✅  Done!  Output: {output_dir}")
    print(f"\n    {'Directory':<28} {'Axis':<30} {'Type':<12} {'Diff':<6} {'Frames'}")
    print(f"    {'─'*28} {'─'*30} {'─'*12} {'─'*6} {'─'*6}")
    print(f"    {'canonical/':<28} {'—':<30} {'—':<12} {'—':<6} {canonical_count}")
    for rec in gt_info:
        n = rec["n_frames"]
        jt   = rec["joint_type"] or "?"
        diff = rec["difficulty"] or "?"
        print(f"    {rec['joint_dir']+'/':<28} {rec['axis_name']:<30} {jt:<12} {diff:<6} {n}")


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Convert ARTi4D → Nerfstudio (canonical + articulated splits + GT)"
    )
    p.add_argument("--scene_dir",     required=True,
                   help="Path to ARTi4D sequence directory "
                        "(e.g. .../din080/scene_2025-04-11-11-44-32)")
    p.add_argument("--output_dir",    required=True,
                   help="Where to write the Nerfstudio datasets")
    p.add_argument("--metadata",      default=None,
                   help="Path to metadata.yaml ")
    p.add_argument("--cues_csv",      default=None,
                   help="Path to matched_cues.csv (default: scene_dir/matched_cues.csv)")
    p.add_argument("--only_verified", action="store_true",
                   help="Skip cues that are not marked VERIFIED")
    p.add_argument("--max_frames",    type=int, default=None,
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