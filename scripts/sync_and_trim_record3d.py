#!/usr/bin/env python3
"""
Synchronize and trim two Record3D sequences using QR codes.
Outputs two separate NeRF datasets with normalized time [0,1].
"""

import cv2
import json
import argparse
import numpy as np
import liblzfse
import open3d as o3d
import subprocess
from pathlib import Path
from datetime import datetime
from qreader import QReader
from scipy.spatial.transform import Rotation
from tqdm import tqdm


# ================================================================
# QR CODE UTILITIES
# ================================================================

def parse_gopro_qr(qr_text: str) -> int:
    """Parse GoPro QR code 'oT<YYMMDDHHMMSS>.mmm' to nanoseconds"""
    if not qr_text.startswith("oT"):
        raise ValueError(f"Invalid QR format: {qr_text}")
    
    timestamp_part = qr_text.split("oT")[1]
    time_main, millis = timestamp_part.split(".")
    
    yy, mm, dd = int(time_main[0:2]), int(time_main[2:4]), int(time_main[4:6])
    hh, mi, ss = int(time_main[6:8]), int(time_main[8:10]), int(time_main[10:12])
    ms = int(millis[:3])
    year = 2000 + yy
    
    dt = datetime(year, mm, dd, hh, mi, ss, ms * 1000)
    return int(dt.timestamp() * 1e9)


def find_first_qr(rgbd_dir: Path) -> tuple:
    """Returns (frame_idx, timestamp_ns) for first valid QR code"""
    qr_reader = QReader(model_size='s', min_confidence=0.4)
    image_files = sorted([f for f in rgbd_dir.glob("*.jpg")], key=lambda x: int(x.stem))
    
    for img_path in tqdm(image_files, desc=f"Scanning {rgbd_dir.parent.name}"):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        decoded = qr_reader.detect_and_decode(img, return_detections=False)
        if decoded and decoded[0]:
            try:
                timestamp_ns = parse_gopro_qr(decoded[0])
                frame_idx = int(img_path.stem)
                print(f"✓ Found QR in frame {frame_idx}: {timestamp_ns} ns")
                return frame_idx, timestamp_ns
            except ValueError as e:
                print(f"  Skipping {img_path.name}: {e}")
    
    raise ValueError(f"No valid QR found in {rgbd_dir}")


# ================================================================
# DEPTH UTILITIES
# ================================================================

def decompress_depth(compressed_data, dw, dh):
    """Decompress LZFSE depth data from Record3D"""
    decompressed = liblzfse.decompress(compressed_data)
    depth = np.frombuffer(decompressed, dtype=np.float32).reshape(dh, dw).copy()
    return depth


def depth_to_pointcloud(depth, rgb, K, c2w, subsample=8):
    """Convert depth map to 3D point cloud with proper alignment"""
    H, W = depth.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    # Resize RGB to match depth if needed
    if rgb.shape[:2] != (H, W):
        rgb = cv2.resize(rgb, (W, H))
    
    # Subsample
    depth_sub = depth[::subsample, ::subsample]
    rgb_sub = rgb[::subsample, ::subsample]
    H_sub, W_sub = depth_sub.shape
    
    u, v = np.meshgrid(np.arange(W_sub), np.arange(H_sub))
    u_orig = u * subsample + subsample / 2.0
    v_orig = v * subsample + subsample / 2.0
    
    valid = (depth_sub > 0.1) & (depth_sub < 5.0)
    
    z = depth_sub[valid]
    x = (u_orig[valid] - cx) * z / fx
    y = (v_orig[valid] - cy) * z / fy
    
    points_cam = np.stack([x, y, z], axis=-1)
    points_cam[:, 1:] *= -1  # Flip Y, Z for OpenCV convention
    
    # Transform to world (c2w is already in OpenGL/NeRF convention)
    R = c2w[:3, :3]
    t = c2w[:3, 3]
    points_world = (R @ points_cam.T).T + t
    
    colors = rgb_sub[valid]
    return points_world, colors


def save_ply(points, colors, output_path):
    """Save point cloud as PLY"""
    with open(output_path, 'w') as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        
        for point, color in zip(points, colors):
            f.write(f"{point[0]} {point[1]} {point[2]} ")
            f.write(f"{int(color[0])} {int(color[1])} {int(color[2])}\n")



def save_side_by_side_preview(static_frames_dir: Path, multi_frames_dir: Path, output_path: Path, 
                               static_transforms: dict, multi_transforms: dict, fps: int = 30):
    """
    Create a side-by-side video preview with normalized timestamps overlaid.
    """
    print(f"\n{'='*60}")
    print("STEP 3: Generating side-by-side preview video")
    print("="*60)
    
    import subprocess
    import tempfile
    
    # Create temporary directory for timestamped frames
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        static_tmp = tmpdir / "static"
        multi_tmp = tmpdir / "multi"
        static_tmp.mkdir()
        multi_tmp.mkdir()
        
        # Process static frames with timestamps
        print("  Adding timestamps to static frames...")
        for i, frame_data in enumerate(static_transforms["frames"], start=1):
            time_val = frame_data["time"]
            frame_path = static_frames_dir / f"frame_{i:05d}.jpg"
            
            if not frame_path.exists():
                continue
            
            img = cv2.imread(str(frame_path))
            if img is None:
                continue
            
            # Add timestamp overlay
            h, w = img.shape[:2]
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = w / 1000  # Scale with image size
            thickness = max(2, int(w / 500))
            
            text = f"Static | t={time_val:.3f}"
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            
            # Black background for text
            cv2.rectangle(img, (10, 10), (text_size[0] + 20, text_size[1] + 30), 
                         (0, 0, 0), -1)
            # White text
            cv2.putText(img, text, (15, text_size[1] + 20), font, font_scale, 
                       (255, 255, 255), thickness, cv2.LINE_AA)
            
            # Save to temp directory
            cv2.imwrite(str(static_tmp / f"frame_{i:05d}.jpg"), img)
        
        # Process multi frames with timestamps
        print("  Adding timestamps to multi frames...")
        for i, frame_data in enumerate(multi_transforms["frames"], start=1):
            time_val = frame_data["time"]
            frame_path = multi_frames_dir / f"frame_{i:05d}.jpg"
            
            if not frame_path.exists():
                continue
            
            img = cv2.imread(str(frame_path))
            if img is None:
                continue
            
            h, w = img.shape[:2]
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = w / 1000
            thickness = max(2, int(w / 500))
            
            text = f"Multi | t={time_val:.3f}"
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            
            cv2.rectangle(img, (10, 10), (text_size[0] + 20, text_size[1] + 30), 
                         (0, 0, 0), -1)
            cv2.putText(img, text, (15, text_size[1] + 20), font, font_scale, 
                       (255, 255, 255), thickness, cv2.LINE_AA)
            
            cv2.imwrite(str(multi_tmp / f"frame_{i:05d}.jpg"), img)
        
        # Create side-by-side video using ffmpeg
        print("  Encoding video...")
        static_pattern = str(static_tmp / "frame_%05d.jpg")
        multi_pattern = str(multi_tmp / "frame_%05d.jpg")
        
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "error",
            "-y",
            "-r", str(fps),
            "-i", multi_pattern,
            "-r", str(fps),
            "-i", static_pattern,
            "-filter_complex",
            "[0:v]setpts=2*PTS[v0];[1:v]setpts=2*PTS[v1];[v0][v1]hstack=inputs=2",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-preset", "veryfast",
            "-crf", "18",
            str(output_path)
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"  ✅ Saved preview: {output_path}")
        except subprocess.CalledProcessError as e:
            print(f"  ❌ Failed to generate preview: {e}")
            print(f"  Error output: {e.stderr.decode() if e.stderr else 'None'}")

# ================================================================
# MAIN SYNC & TRIM LOGIC
# ================================================================

def process_camera(
    data_dir: Path,
    output_dir: Path,
    qr_frame_idx: int,
    qr_time_ns: int,
    action_start_time_ns: int,
    action_end_time_ns: int,
    downsample_factor: int = 1,
    stride: int = 1,
    create_fused_pc: bool = True,
    pc_subsample: int = 8,
    voxel_downsample: float = 0.01
):
    """Process one camera: trim, downsample, stride, and output with normalized time"""
    
    # Load metadata
    with open(data_dir / "metadata", 'r') as f:
        metadata = json.load(f)
    
    frame_timestamps = metadata["frameTimestamps"]
    poses_list = metadata["poses"]
    
    # Parse intrinsics
    K_list = metadata["K"]
    K_raw = np.array([[K_list[0], K_list[1], K_list[2]],
                      [K_list[3], K_list[4], K_list[5]],
                      [K_list[6], K_list[7], K_list[8]]])
    
    if K_raw[2, 0] > 100 or K_raw[2, 1] > 100:
        K = K_raw.T
    else:
        K = K_raw
    
    W = metadata.get("w", 1920)
    H = metadata.get("h", 1440)
    dw = metadata.get("dw", 256)
    dh = metadata.get("dh", 192)
    
    print(f"\n{'='*60}")
    print(f"Processing: {data_dir.name}")
    print(f"  RGB resolution: {W}x{H}")
    print(f"  Depth resolution: {dw}x{dh}")
    print(f"  Total frames: {len(frame_timestamps)}")
    print(f"  Downsample factor: {downsample_factor}x")
    print(f"  Stride: {stride}")
    
    # Convert frame timestamps to global nanoseconds
    qr_relative_time = frame_timestamps[qr_frame_idx]
    global_times_ns = []
    
    for i, rel_time in enumerate(frame_timestamps):
        delta_sec = rel_time - qr_relative_time
        global_time_ns = qr_time_ns + int(delta_sec * 1e9)
        global_times_ns.append(global_time_ns)
    
    # Find frames within action window
    valid_indices = [i for i, t in enumerate(global_times_ns) 
                     if action_start_time_ns <= t <= action_end_time_ns]
    
    if not valid_indices:
        raise ValueError(f"No frames found in action window for {data_dir.name}")
    
    # Apply stride
    valid_indices = valid_indices[::stride]
    
    print(f"  Action window: frames {valid_indices[0]} - {valid_indices[-1]}")
    print(f"  Selected frames after stride: {len(valid_indices)}")
    
    # Normalize times to [0, 1] based on ACTUAL selected frames
    actual_start_time_ns = global_times_ns[valid_indices[0]]
    actual_end_time_ns = global_times_ns[valid_indices[-1]]
    actual_duration_ns = actual_end_time_ns - actual_start_time_ns
    
    normalized_times = [(global_times_ns[i] - actual_start_time_ns) / actual_duration_ns 
                        for i in valid_indices]
    
    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = output_dir / "frames"
    depth_dir = output_dir / "depth"
    frames_dir.mkdir(exist_ok=True)
    depth_dir.mkdir(exist_ok=True)
    
    # Calculate output resolution after downsampling
    output_w = W // downsample_factor
    output_h = H // downsample_factor
    
    # Scale intrinsics for downsampled resolution
    scale = 1.0 / downsample_factor
    K_downsampled = K.copy()
    K_downsampled[0, 0] *= scale  # fx
    K_downsampled[1, 1] *= scale  # fy
    K_downsampled[0, 2] *= scale  # cx
    K_downsampled[1, 2] *= scale  # cy
    
    # Process frames
    rgbd_dir = data_dir / "rgbd"
    frames_json = []
    all_points, all_colors = [], []
    
    F_opencv_to_opengl = np.diag([1, -1, -1, 1])
    
    for new_idx, orig_idx in enumerate(tqdm(valid_indices, desc="Processing frames"), start=1):
        # Load RGB
        rgb_path = rgbd_dir / f"{orig_idx}.jpg"
        rgb = cv2.imread(str(rgb_path))
        if rgb is None:
            print(f"Warning: Could not read {rgb_path.name}")
            continue
        
        # Downsample RGB
        if downsample_factor > 1:
            rgb = cv2.resize(rgb, (output_w, output_h), interpolation=cv2.INTER_AREA)
        
        # Save RGB with new index
        rgb_out = frames_dir / f"frame_{new_idx:05d}.jpg"
        cv2.imwrite(str(rgb_out), rgb)
        
        # Load depth
        depth_path = rgbd_dir / f"{orig_idx}.depth"
        depth = None
        if depth_path.exists():
            with open(depth_path, 'rb') as f:
                compressed = f.read()
            depth = decompress_depth(compressed, dw, dh)
            depth[np.isnan(depth)] = 0.0
            
            # Resize depth to match RGB resolution
            depth_resized = cv2.resize(depth, (W, H), interpolation=cv2.INTER_NEAREST)
            
            # Downsample depth to match downsampled RGB
            if downsample_factor > 1:
                depth_resized = cv2.resize(depth_resized, (output_w, output_h), interpolation=cv2.INTER_NEAREST)
            
            # Save depth as uint16 (mm)
            depth_mm = np.clip(depth_resized * 1000.0, 0, 65535).astype(np.uint16)
            depth_out = depth_dir / f"frame_{new_idx:05d}.png"
            cv2.imwrite(str(depth_out), depth_mm)
            
            # Save depth as numpy for exact values
            depth_npy_out = depth_dir / f"frame_{new_idx:05d}.npy"
            np.save(depth_npy_out, depth_resized.astype(np.float32))
        
        # Get pose and convert to NeRF convention
        pose_raw = poses_list[orig_idx]
        quat = pose_raw[:4]
        trans = pose_raw[4:]
        
        c2w = np.eye(4)
        c2w[:3, :3] = Rotation.from_quat(quat).as_matrix()
        c2w[:3, 3] = trans
        c2w_nerf = F_opencv_to_opengl @ c2w
        
        # Add to frames list
        frames_json.append({
            "file_path": f"frames/frame_{new_idx:05d}.jpg",
            "depth_file_path": f"depth/frame_{new_idx:05d}.png",
            "depth_npy_file_path": f"depth/frame_{new_idx:05d}.npy",
            "transform_matrix": c2w_nerf.tolist(),
            "time": float(normalized_times[new_idx - 1])
        })
        
        # Fused point cloud (sample every 10 frames)
        if create_fused_pc and depth is not None and new_idx % 10 == 1:
            # Use original depth resolution for point cloud (before downsampling)
            depth_for_pc = cv2.resize(depth, (W, H), interpolation=cv2.INTER_NEAREST)
            rgb_for_pc = cv2.imread(str(rgb_path))
            rgb_for_pc = cv2.cvtColor(rgb_for_pc, cv2.COLOR_BGR2RGB)
            
            points, colors = depth_to_pointcloud(depth_for_pc, rgb_for_pc, K, c2w_nerf, subsample=pc_subsample)
            if len(points) > 0:
                all_points.append(points)
                all_colors.append(colors)
    
    # Create fused point cloud with filtering
    if all_points:
        all_points = np.vstack(all_points)
        all_colors = np.vstack(all_colors)
        
        print(f"\n  Fused point cloud processing:")
        print(f"    Total points: {len(all_points):,}")
        
        # Distance filter
        dist = np.linalg.norm(all_points, axis=1)
        mask = (dist > 0.05) & (dist < 5.0)
        all_points = all_points[mask]
        all_colors = all_colors[mask]
        print(f"    After distance filter: {len(all_points):,}")
        
        # Statistical outlier removal
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(all_points)
        pcd.colors = o3d.utility.Vector3dVector(all_colors / 255.0)
        
        pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.0)
        all_points = np.asarray(pcd.points)
        all_colors = (np.asarray(pcd.colors) * 255).astype(np.uint8)
        print(f"    After outlier removal: {len(all_points):,}")
        
        # Voxel downsampling
        if voxel_downsample > 0:
            voxel_indices = np.floor(all_points / voxel_downsample).astype(int)
            _, unique_indices = np.unique(voxel_indices, axis=0, return_index=True)
            all_points = all_points[unique_indices]
            all_colors = all_colors[unique_indices]
            print(f"    After voxel downsampling: {len(all_points):,}")
        
        ply_path = output_dir / "fused_pc.ply"
        save_ply(all_points, all_colors, ply_path)
        print(f"    ✓ Saved: {ply_path}")
    
    # Save transforms.json
    transforms = {
        "camera_model": "OPENCV",
        "fl_x": float(K_downsampled[0, 0]),
        "fl_y": float(K_downsampled[1, 1]),
        "cx": float(K_downsampled[0, 2]),
        "cy": float(K_downsampled[1, 2]),
        "w": int(output_w),
        "h": int(output_h),
        # "ply_file_path": "fused_pc.ply" if all_points is not None and len(all_points) > 0 else None,
        "frames": frames_json
    }
    
    with open(output_dir / "transforms.json", 'w') as f:
        json.dump(transforms, f, indent=2)
    
    print(f"\n  ✓ Saved {len(frames_json)} frames to {output_dir}")
    print(f"  ✓ Output resolution: {output_w}x{output_h}")
    print(f"  ✓ Time range: [{normalized_times[0]:.4f}, {normalized_times[-1]:.4f}]")


def main():
    parser = argparse.ArgumentParser(description="Sync and trim two Record3D sequences")
    parser.add_argument("--multi", required=True, type=Path, help="Moving camera directory")
    parser.add_argument("--static", required=True, type=Path, help="Static camera directory")
    parser.add_argument("--start", required=True, type=int, help="Static frame where action starts")
    parser.add_argument("--end", required=True, type=int, help="Static frame where action ends")
    parser.add_argument("--output_multi", required=True, type=Path, help="Output dir for multi camera")
    parser.add_argument("--output_static", required=True, type=Path, help="Output dir for static camera")
    parser.add_argument("--downsample", type=int, default=1, help="Downsample factor (1=no downsampling, 2=half size)")
    parser.add_argument("--stride", type=int, default=1, help="Use every Nth frame (1=all frames)")
    parser.add_argument("--pc_subsample", type=int, default=8, help="Point cloud subsample factor")
    parser.add_argument("--voxel_size", type=float, default=0.01, help="Voxel downsampling size")
    parser.add_argument("--no_fused_pc", action="store_true", help="Skip fused point cloud generation")
    args = parser.parse_args()
    
    print("="*60)
    print("STEP 1: Finding QR codes for synchronization")
    print("="*60)
    
    multi_qr_idx, multi_qr_ns = find_first_qr(args.multi / "rgbd")
    static_qr_idx, static_qr_ns = find_first_qr(args.static / "rgbd")
    
    delta_ns = multi_qr_ns - static_qr_ns
    print(f"\n✓ Time offset: {delta_ns / 1e9:.6f} seconds")
    
    # Load static metadata to get action window in global time
    with open(args.static / "metadata", 'r') as f:
        static_meta = json.load(f)
    
    static_timestamps = static_meta["frameTimestamps"]
    static_qr_rel_time = static_timestamps[static_qr_idx]
    
    # Convert action window to global nanoseconds
    action_start_rel = static_timestamps[args.start]
    action_end_rel = static_timestamps[args.end]
    
    action_start_ns = static_qr_ns + int((action_start_rel - static_qr_rel_time) * 1e9)
    action_end_ns = static_qr_ns + int((action_end_rel - static_qr_rel_time) * 1e9)
    
    print(f"\n{'='*60}")
    print("STEP 2: Processing cameras")
    print(f"  Action window: {(action_end_ns - action_start_ns) / 1e9:.2f} seconds")
    print("="*60)
    
    # Process both cameras
    process_camera(
        args.static, args.output_static,
        static_qr_idx, static_qr_ns,
        action_start_ns, action_end_ns,
        downsample_factor=args.downsample,
        stride=args.stride,
        create_fused_pc=not args.no_fused_pc,
        pc_subsample=args.pc_subsample,
        voxel_downsample=args.voxel_size
    )
    
    process_camera(
        args.multi, args.output_multi,
        multi_qr_idx, multi_qr_ns,
        action_start_ns, action_end_ns,
        downsample_factor=args.downsample,
        stride=args.stride,
        create_fused_pc=not args.no_fused_pc,
        pc_subsample=args.pc_subsample,
        voxel_downsample=args.voxel_size
    )
    
    print(f"\n{'='*60}")
    print("✅ DONE! Two synchronized NeRF datasets created.")
    print(f"   Multi:  {args.output_multi}")
    print(f"   Static: {args.output_static}")
    print("="*60)


    try:
        preview_output = args.output_multi.parent / "preview_side_by_side.mp4"
        save_side_by_side_preview(
            args.output_static / "frames",
            args.output_multi / "frames",
            preview_output,
            static_transforms=json.load(open(args.output_static / "transforms.json")),
            multi_transforms=json.load(open(args.output_multi / "transforms.json")),
            fps=30
        )
    except Exception as e:
        print(f"⚠️ Preview generation skipped due to error: {e}")


if __name__ == "__main__":
    main()