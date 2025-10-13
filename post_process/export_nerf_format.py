import json
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation
import shutil
import liblzfse
import cv2

def decompress_depth(compressed_data, dw, dh):
    """Decompress LZFSE depth data from Record3D"""
    try:
        decompressed = liblzfse.decompress(compressed_data)
        depth = np.frombuffer(decompressed, dtype=np.float32).reshape(dh, dw)
        return depth
    except Exception as e:
        print(f"Error decompressing depth: {e}")
        return None

def depth_to_pointcloud(depth, rgb, K, c2w, subsample=4, frame_idx=0, rgb_w=None, rgb_h=None):
    """Convert depth map to 3D point cloud - FIXED for intrinsics scaling"""
    H, W = depth.shape
    
    # Extract intrinsics
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    # CRITICAL FIX: Scale intrinsics from RGB resolution to depth resolution
    if rgb_w is not None and rgb_h is not None:
        # Intrinsics are for RGB resolution, scale to depth
        scale_x = W / rgb_w
        scale_y = H / rgb_h
        
        fx_depth = fx * scale_x
        fy_depth = fy * scale_y
        cx_depth = cx * scale_x
        cy_depth = cy * scale_y
        
        if frame_idx == 0:
            print(f"\n  Scaling intrinsics:")
            print(f"    RGB resolution: {rgb_w}x{rgb_h}")
            print(f"    Depth resolution: {W}x{H}")
            print(f"    Scale factors: {scale_x:.4f}, {scale_y:.4f}")
            print(f"    Original: fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")
            print(f"    Scaled:   fx={fx_depth:.1f}, fy={fy_depth:.1f}, cx={cx_depth:.1f}, cy={cy_depth:.1f}")
    else:
        fx_depth, fy_depth = fx, fy
        cx_depth, cy_depth = cx, cy
    
    # Resize RGB to match depth if needed
    if rgb.shape[:2] != (H, W):
        rgb = cv2.resize(rgb, (W, H))
    
    # Subsample for efficiency
    depth_sub = depth[::subsample, ::subsample]
    rgb_sub = rgb[::subsample, ::subsample]
    H_sub, W_sub = depth_sub.shape
    
    # Create pixel grid
    u, v = np.meshgrid(np.arange(W_sub), np.arange(H_sub))
    
    # Scale back to original depth image coordinates (center of subsampled pixels)
    u_orig = u * subsample + subsample / 2.0
    v_orig = v * subsample + subsample / 2.0
    
    # Valid depth mask
    valid = (depth_sub > 0) & (depth_sub < 10.0)
    
    if frame_idx == 0:
        print(f"\n  Frame {frame_idx} reconstruction:")
        print(f"    Depth shape: {depth.shape}")
        print(f"    Valid depth pixels: {valid.sum():,} / {depth_sub.size:,}")
        print(f"    Depth range: [{depth_sub[valid].min():.3f}, {depth_sub[valid].max():.3f}]m")
    
    # Back-project to 3D camera coordinates
    z = depth_sub[valid]
    x = (u_orig[valid] - cx_depth) * z / fx_depth
    y = (v_orig[valid] - cy_depth) * z / fy_depth
    
    # Points in camera frame: shape (N, 3)
    points_cam = np.stack([x, y, z], axis=-1)
    points_cam[:, 1:] *= -1
    
    if frame_idx == 0:
        print(f"    Camera space extent:")
        print(f"      X: [{points_cam[:, 0].min():.3f}, {points_cam[:, 0].max():.3f}]m")
        print(f"      Y: [{points_cam[:, 1].min():.3f}, {points_cam[:, 1].max():.3f}]m")
        print(f"      Z: [{points_cam[:, 2].min():.3f}, {points_cam[:, 2].max():.3f}]m")
    
    # Transform to world coordinates
    R = c2w[:3, :3]
    t = c2w[:3, 3]
    points_world = (R @ points_cam.T).T + t
    
    if frame_idx == 0:
        print(f"    World space extent:")
        print(f"      X: [{points_world[:, 0].min():.3f}, {points_world[:, 0].max():.3f}]m")
        print(f"      Y: [{points_world[:, 1].min():.3f}, {points_world[:, 1].max():.3f}]m")
        print(f"      Z: [{points_world[:, 2].min():.3f}, {points_world[:, 2].max():.3f}]m")
        print(f"    Camera position: [{t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}]m")
        
        # Check if points are reasonable distances from camera
        distances = np.linalg.norm(points_world - t, axis=1)
        print(f"    Distance from camera: [{distances.min():.3f}, {distances.max():.3f}]m (mean: {distances.mean():.3f}m)")
    
    # Get colors
    colors = rgb_sub[valid]
    
    return points_world, colors

def save_ply(points, colors, output_path):
    """Save point cloud as PLY file"""
    output_path = Path(output_path)
    
    with open(output_path, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        
        for point, color in zip(points, colors):
            f.write(f"{point[0]} {point[1]} {point[2]} ")
            f.write(f"{int(color[0])} {int(color[1])} {int(color[2])}\n")

def record3d_to_nerf(data_dir, output_dir, max_frames=-1, stride=10,
                     create_sparse_pc=True, pc_subsample=4, voxel_downsample=0.01,
                     downsample_factor=2):
    """
    Convert Record3D RGBD data to NeRF format for 3DGS training.

    Args:
        data_dir: Path to Record3D data (contains rgb/, depth/, metadata.json)
        output_dir: Output directory for processed data
        max_frames: Maximum number of frames to use (-1 for all)
        stride: Use every Nth frame (e.g., stride=2 uses every other frame)
        create_sparse_pc: Whether to create sparse point cloud from depth
        pc_subsample: Subsample factor for point cloud (higher = fewer points)
        voxel_downsample: Voxel size for downsampling final point cloud
        downsample_factor: Downsample RGB images by this factor (1=no downsampling, 2=half size, 4=quarter size)
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    
    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = output_dir / "images"
    image_dir.mkdir(exist_ok=True)
    
    # Load metadata
    print("Loading metadata...")
    with open(data_dir / "metadata", 'r') as f:
        metadata = json.load(f)
    
    # Parse intrinsics - CRITICAL FIX for Record3D format
    K_list = metadata["K"]
    
    # Record3D stores K in row-major order but may be transposed
    # Standard format should be:
    # [[fx,  0, cx],
    #  [ 0, fy, cy],
    #  [ 0,  0,  1]]
    
    K_raw = np.array([[K_list[0], K_list[1], K_list[2]],
                      [K_list[3], K_list[4], K_list[5]],
                      [K_list[6], K_list[7], K_list[8]]])
    
    print(f"Raw K matrix from metadata:\n{K_raw}")
    
    # Check if K is transposed (cx, cy in wrong position)
    if K_raw[2, 0] > 100 or K_raw[2, 1] > 100:  # cx, cy should be in third column, not row
        print("⚠️  Detected transposed intrinsics matrix, fixing...")
        K = K_raw.T
        print(f"Corrected K matrix:\n{K}")
    else:
        K = K_raw
    
    # Get image dimensions
    W = metadata.get("dw", metadata.get("w", 640))
    H = metadata.get("dh", metadata.get("h", 480))
    
    # Get RGB dimensions if different
    rgb_W = metadata.get("w", W)
    rgb_H = metadata.get("h", H)
    
    print(f"\nDepth dimensions: {W}x{H}")
    print(f"RGB dimensions: {rgb_W}x{rgb_H}")
    print(f"Intrinsics are for resolution: ~{int(K[0, 2]*2)}x{int(K[1, 2]*2)}")
    
    # Parse poses
    poses_list = metadata["poses"]
    num_poses = len(poses_list)
    print(f"Number of poses in metadata: {num_poses}")
    
    if num_poses == 0:
        raise ValueError("No poses found in metadata.json!")
    
    # Get image paths
    rgb_dir = data_dir / "rgbd"
    if not rgb_dir.exists():
        raise ValueError(f"RGB directory not found: {rgb_dir}")
    
    image_files = sorted([f for f in rgb_dir.glob("*.jpg") if f.stem.isdigit()], 
                        key=lambda x: int(x.stem))
    
    if len(image_files) == 0:
        image_files = sorted([f for f in rgb_dir.glob("*.*") if f.stem.isdigit() 
                             and f.suffix.lower() in ['.jpg', '.jpeg', '.png']], 
                            key=lambda x: int(x.stem))
    
    num_images = len(image_files)
    print(f"Number of images found: {num_images}")
    
    if num_images == 0:
        raise ValueError(f"No images found in {rgb_dir}")
    
    # Match number of poses and images
    num_frames = min(num_poses, num_images)
    print(f"Using {num_frames} frames (min of poses and images)")
    
    # Apply stride first
    if stride > 1:
        indices_all = np.arange(num_frames)
        indices_strided = indices_all[::stride]
        print(f"Applied stride={stride}: {len(indices_strided)} frames")
    else:
        indices_strided = np.arange(num_frames)
    
    # Then apply max_frames limit if needed
    if max_frames > 0 and len(indices_strided) > max_frames:
        # Sample uniformly from the strided frames
        indices = np.round(np.linspace(0, len(indices_strided) - 1, max_frames)).astype(int)
        indices = indices_strided[indices]
        print(f"Limited to max_frames={max_frames}")
    else:
        indices = indices_strided
    
    print(f"Final selected frames: {len(indices)}")
    
    image_files = [image_files[i] for i in indices]
    selected_poses = [poses_list[i] for i in indices]
    
    # Copy and optionally downsample images
    print("Copying and processing images...")
    if downsample_factor > 1:
        print(f"  Downsampling images by factor of {downsample_factor}")
    
    copied_paths = []
    for img_file in image_files:
        # Read image
        img = cv2.imread(str(rgb_dir / img_file.name))
        if img is None:
            print(f"Warning: Could not read {img_file.name}")
            continue
        
        # Downsample if requested
        if downsample_factor > 1:
            new_h = img.shape[0] // downsample_factor
            new_w = img.shape[1] // downsample_factor
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Save processed image
        dst = image_dir / img_file.name
        cv2.imwrite(str(dst), img)
        copied_paths.append(f"images/{img_file.name}")
    
    # Process camera poses
    print("Processing camera poses...")
    poses_array = np.array(selected_poses, dtype=np.float32)  # (N, 7)
    
    # Extract quaternions [qx, qy, qz, qw] and translations [tx, ty, tz]
    quats = poses_array[:, :4]  # (N, 4)
    translations = poses_array[:, 4:]  # (N, 3)
    
    print(f"Quaternions shape: {quats.shape}")
    print(f"Translations shape: {translations.shape}")
    print(f"Sample quaternion: {quats[0]}")
    print(f"Sample translation: {translations[0]}")
    
    # Convert quaternions to rotation matrices
    rotations = Rotation.from_quat(quats).as_matrix()  # (N, 3, 3)
    
    # Build camera-to-world matrices (convert from ARKit/OpenCV to Nerfstudio/OpenGL)
    camera_to_worlds = []
    F_opencv_to_opengl = np.diag([1, -1, -1, 1])  # Flip Y and Z

    for i in range(len(rotations)):
        c2w = np.eye(4)
        c2w[:3, :3] = rotations[i]
        c2w[:3, 3] = translations[i]
        
        # Convert coordinate system
        c2w_nerf = F_opencv_to_opengl @ c2w
        camera_to_worlds.append(c2w_nerf)

    camera_to_worlds = np.array(camera_to_worlds)
    print("Converted poses from OpenCV -> OpenGL convention for Nerfstudio")

        
    # Print camera trajectory for verification
    cam_positions = camera_to_worlds[:, :3, 3]
    print(f"Camera trajectory:")
    print(f"  Start position: {cam_positions[0]}")
    print(f"  End position: {cam_positions[-1]}")
    print(f"  Total movement: {np.linalg.norm(cam_positions[-1] - cam_positions[0]):.3f}m")
    
    # Create sparse point cloud from depth if requested
    all_points = []
    all_colors = []
    
    if create_sparse_pc:
        print("\nCreating sparse point cloud from depth data...")
        depth_dir = data_dir / "rgbd"
        
        if not depth_dir.exists():
            print(f"Warning: Depth directory not found: {depth_dir}")
            create_sparse_pc = False
        else:
            success_count = 0
            for idx, img_file in enumerate(image_files):
                # Load RGB
                rgb_path = rgb_dir / img_file.name
                rgb = cv2.imread(str(rgb_path))
                if rgb is None:
                    print(f"Warning: Could not read {img_file.name}")
                    continue
                rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
                
                # Get actual RGB dimensions
                rgb_h_actual, rgb_w_actual = rgb.shape[:2]
                
                # Load and decompress depth
                depth_file = depth_dir / f"{img_file.stem}.depth"
                if depth_file.exists():
                    with open(depth_file, 'rb') as f:
                        compressed_depth = f.read()
                    
                    depth = decompress_depth(compressed_depth, W, H)
                    
                    if depth is not None:
                        if idx == 0:
                            print(f"  RGB actual dimensions: {rgb_w_actual}x{rgb_h_actual}")
                        
                        # Convert to point cloud with proper intrinsics scaling
                        points, colors = depth_to_pointcloud(
                            depth, rgb, K, camera_to_worlds[idx], 
                            subsample=pc_subsample, frame_idx=idx,
                            rgb_w=rgb_w_actual, rgb_h=rgb_h_actual
                        )
                        
                        if len(points) > 0:
                            all_points.append(points)
                            all_colors.append(colors)
                            success_count += 1
                        
                        if (idx + 1) % 100 == 0:
                            print(f"  Processed {idx + 1}/{len(image_files)} frames ({success_count} successful)")
            
            print(f"  Total successful depth frames: {success_count}/{len(image_files)}")
            
            if all_points:
                # Combine all points
                all_points = np.vstack(all_points)
                all_colors = np.vstack(all_colors)
                
                print(f"Total points before downsampling: {len(all_points):,}")
                
                # Print point cloud statistics to verify
                print(f"Point cloud bounds:")
                print(f"  X: [{all_points[:, 0].min():.3f}, {all_points[:, 0].max():.3f}]")
                print(f"  Y: [{all_points[:, 1].min():.3f}, {all_points[:, 1].max():.3f}]")
                print(f"  Z: [{all_points[:, 2].min():.3f}, {all_points[:, 2].max():.3f}]")
                
                # Simple voxel downsampling
                if voxel_downsample > 0:
                    voxel_indices = np.floor(all_points / voxel_downsample).astype(int)
                    _, unique_indices = np.unique(voxel_indices, axis=0, return_index=True)
                    all_points = all_points[unique_indices]
                    all_colors = all_colors[unique_indices]
                    print(f"Points after downsampling: {len(all_points):,}")
                
                # Save point cloud
                ply_path = output_dir / "sparse_pc.ply"
                save_ply(all_points, all_colors, ply_path)
                print(f"✓ Sparse point cloud saved: {ply_path}")
            else:
                print("Warning: No valid depth data found, skipping point cloud generation")
    
    # Build frames
    print("\nBuilding transforms.json...")
    frames = []
    for i, img_path in enumerate(copied_paths):
        frames.append({
            "file_path": img_path,
            "transform_matrix": camera_to_worlds[i].tolist()
        })
    
    # Create transforms.json with downsampled RGB resolution
    # Get original RGB dimensions
    output_w_orig = metadata.get("w", 1920)
    output_h_orig = metadata.get("h", 1440)
    
    # Apply downsampling
    output_w = output_w_orig // downsample_factor
    output_h = output_h_orig // downsample_factor
    
    # Scale intrinsics for downsampled resolution
    scale = 1.0 / downsample_factor
    focal_x = K[0, 0] * scale
    focal_y = K[1, 1] * scale
    cx = K[0, 2] * scale
    cy = K[1, 2] * scale
    
    transforms = {
        "fl_x": float(focal_x),
        "fl_y": float(focal_y),
        "cx": float(cx),
        "cy": float(cy),
        "w": int(output_w),
        "h": int(output_h),
        "camera_model": "OPENCV",
        "frames": frames
    }
    
    print(f"Output transforms.json resolution: {output_w}x{output_h} (downsampled from {output_w_orig}x{output_h_orig})")
    print(f"Intrinsics in transforms.json: fx={focal_x:.1f}, fy={focal_y:.1f}, cx={cx:.1f}, cy={cy:.1f}")
    
    # Add sparse point cloud reference if created
    if create_sparse_pc and len(all_points) > 0:
        transforms["ply_file_path"] = "sparse_pc.ply"
    
    # Save
    with open(output_dir / "transforms_pre.json", 'w') as f:
        json.dump(transforms, f, indent=2)
    
    print(f"\n✓ Processed {len(frames)} frames")
    print(f"✓ Output saved to: {output_dir}")
    print(f"✓ transforms_pre.json created")



    
    return len(frames)


if __name__ == "__main__":
    # Example usage
    DATA_DIR = "/local/home/pmishra/cvg/arti-splatfacto/data_real/pre_static"
    OUTPUT_DIR = "/local/home/pmishra/cvg/arti-splatfacto/data_real/pre_static/pre_static_post"
    
    record3d_to_nerf(
        data_dir=DATA_DIR,
        output_dir=OUTPUT_DIR,
        max_frames=-1,
        create_sparse_pc=True,
        pc_subsample=8,
        voxel_downsample=0.01
    )