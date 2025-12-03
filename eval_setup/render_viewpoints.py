#!/usr/bin/env python3
"""
Enhanced PartNet-Mobility rendering for 3DGS training.

Addresses key requirements:
1. Dense viewpoint sampling for proper 3DGS training
2. Metric depth in .npy format
3. Static pre-change state (joint at initial position)
4. Dynamic post-change sequence with articulation
"""

import sapien.core as sapien
import numpy as np
from PIL import Image
import json
import pickle
import os
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
import argparse
import subprocess
from mesh_utils import parse_partid_to_objs, merge_meshs

def spherical2cartesian(radius: float, theta: float, phi: float) -> np.ndarray:
    """Convert spherical coordinates to cartesian."""
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)
    return np.array([x, y, z])


def create_camera(scene: sapien.Scene, width: int, height: int, fovy: float, name: str) -> sapien.CameraEntity:
    """Create camera with specified parameters."""
    near, far = 0.1, 100
    camera = scene.add_camera(
        name=name,
        width=width,
        height=height,
        fovy=fovy,
        near=near,
        far=far,
    )
    return camera


def set_camera_pose(camera: sapien.CameraEntity, cam_vec: np.ndarray, look_at: np.ndarray = None):
    """Set camera pose looking at target."""
    if look_at is None:
        look_at = np.array([0, 0, 0])
    
    forward = look_at - cam_vec
    forward = forward / np.linalg.norm(forward)
    
    # Handle up vector
    up = np.array([0, 0, 1])
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)
    
    # Create transformation matrix
    mat44 = np.eye(4)
    mat44[:3, :3] = np.stack([forward, -right, up], axis=1)  # OpenGL convention
    mat44[:3, 3] = cam_vec
    
    camera.set_pose(sapien.Pose.from_transformation_matrix(mat44))


def get_rgba_img(camera: sapien.CameraEntity) -> np.ndarray:
    """Get RGBA image from camera."""
    rgba = camera.get_float_texture('Color')
    rgba_img = (rgba * 255).clip(0, 255).astype("uint8")
    return rgba_img


def get_depth_img(camera: sapien.CameraEntity) -> np.ndarray:
    """Get metric depth map (positive distance from camera)."""
    position = camera.get_float_texture('Position')
    depth = -position[..., 2]  # -Z forward
    depth[position[..., 3] >= 1.0] = 0.0  # invalidate background
    return depth.astype(np.float32)



def get_segment_img(camera: sapien.CameraEntity) -> np.ndarray:
    """Get segmentation image."""
    seg_labels = camera.get_uint32_texture('Segmentation')
    return seg_labels[..., 1].astype(np.uint8)  # Actor-level segmentation

def get_point_cloud(camera: sapien.CameraEntity) -> np.ndarray:
    """Extract 3D point cloud in WORLD SPACE from SAPIEN camera outputs."""
    # Position is in OpenGL camera coordinates
    position = camera.get_float_texture('Position')  # [H, W, 4]
    points_cam = position[..., :3].reshape(-1, 3)

    # Mask invalid pixels (render depth = 1 means background)
    valid_mask = position[..., 3].reshape(-1) < 1.0
    points_cam = points_cam[valid_mask]

    # Transform from camera space → world space
    model_matrix = camera.get_model_matrix()  # camera-to-world
    points_world = points_cam @ model_matrix[:3, :3].T + model_matrix[:3, 3]

    # Get RGB colors
    rgba = camera.get_float_texture('Color')  # [H, W, 4]
    rgb = (rgba[..., :3] * 255).clip(0, 255).astype(np.uint8).reshape(-1, 3)
    colors = rgb[valid_mask]

    # Combine
    points_colored = np.concatenate([points_world, colors], axis=1)  # [N, 6]
    return points_colored.astype(np.float32)




def init_scene(ray_tracing: bool = False) -> sapien.Scene:
    """Initialize SAPIEN scene with proper lighting."""
    engine = sapien.Engine()
    if ray_tracing:
        sapien.render_config.camera_shader_dir = "rt"
        sapien.render_config.viewer_shader_dir = "rt"
        sapien.render_config.rt_samples_per_pixel = 256
        sapien.render_config.rt_use_denoiser = True
    
    renderer = sapien.SapienRenderer(offscreen_only=True)
    engine.set_renderer(renderer)
    
    scene = engine.create_scene()
    scene.set_timestep(1 / 100.0)
    
    scene.set_ambient_light([0.5, 0.5, 0.5])  # Slightly brighter for isolated objects
    
    # Disable shadows for cleaner background
    scene.add_directional_light([0, 1, -1], [0.6, 0.6, 0.6], shadow=False)
    scene.add_point_light([2, 2, 2], [0.8, 0.8, 0.8], shadow=False)
    scene.add_point_light([2, -2, 2], [0.8, 0.8, 0.8], shadow=False)
    scene.add_point_light([-2, 0, 2], [0.8, 0.8, 0.8], shadow=False)
    

    ground_material = renderer.create_material()
    ground_material.base_color = np.array([202, 164, 114, 256]) / 256
    ground_material.specular = 0.5
    ground_material.set_diffuse_texture_from_file("/local/home/pmishra/cvg/arti-splatfacto/data_itaco/video2articulation/ground.png")
    scene.add_ground(altitude=0, render_material=ground_material, render_half_size=np.array([10, 10]))
    
    # ADD WALLS WITH TEXTURE
    wall_material = renderer.create_material()
    wall_material.set_diffuse_texture_from_file("/local/home/pmishra/cvg/arti-splatfacto/data_itaco/video2articulation/wall.jpg")
    
    # Helper function to create box walls
    def create_box(pose, half_size, material, name):
        builder = scene.create_actor_builder()
        builder.add_box_collision(half_size=half_size)
        builder.add_box_visual(half_size=half_size, material=material)
        box = builder.build_static(name=name)
        box.set_pose(pose)
        return box
    
    # Create 4 walls
    create_box(sapien.Pose(p=[10, 0, 0]), np.array([0.5, 10, 10]), wall_material, 'wall_front')
    create_box(sapien.Pose(p=[-10, 0, 0]), np.array([0.5, 10, 10]), wall_material, 'wall_back')
    create_box(sapien.Pose(p=[0, 10, 0]), np.array([10, 0.5, 10]), wall_material, 'wall_left')
    create_box(sapien.Pose(p=[0, -10, 0]), np.array([10, 0.5, 10]), wall_material, 'wall_right')
    
    return scene


def generate_dense_viewpoints(object_center: np.ndarray, object_radius: float) -> list:
    """Generate dense viewpoints suitable for 3DGS training."""
    
    viewpoints = []
    
    # Multiple camera distances for better coverage
    distances = [3.0 * object_radius, 4.0 * object_radius, 6.0 * object_radius]
    
    # Dense angular sampling
    phi_angles = np.linspace(np.deg2rad(10), np.deg2rad(80), 6)  # 6 elevation levels
    
    for distance in distances:
        for phi in phi_angles:
            # Number of azimuth views depends on elevation (more views at low elevation)
            if phi < np.deg2rad(30):
                n_azimuth = 24  # 15° spacing
            elif phi < np.deg2rad(60):
                n_azimuth = 16  # 22.5° spacing  
            else:
                n_azimuth = 8   # 45° spacing
            
            theta_angles = np.linspace(0, 2*np.pi, n_azimuth, endpoint=False)
            
            for theta in theta_angles:
                cam_pos = object_center + spherical2cartesian(distance, theta, phi)
                viewpoints.append({
                    'position': cam_pos,
                    'look_at': object_center,
                    'distance': distance,
                    'phi': phi,
                    'theta': theta
                })
    
    print(f"Generated {len(viewpoints)} dense viewpoints for 3DGS training")
    return viewpoints

def save_ply(path: Path, points: np.ndarray):
    """Save point cloud as .ply file (xyz + rgb)."""
    assert points.shape[1] == 6, "Points must be [N, 6] (xyz + rgb)"
    
    with open(path, 'wb') as f:
        # Header
        f.write(b"ply\n")
        f.write(b"format binary_little_endian 1.0\n")
        f.write(f"element vertex {len(points)}\n".encode())
        f.write(b"property float x\n")
        f.write(b"property float y\n")
        f.write(b"property float z\n")
        f.write(b"property uchar red\n")
        f.write(b"property uchar green\n")
        f.write(b"property uchar blue\n")
        f.write(b"end_header\n")
        
        # Data
        for point in points:
            xyz = point[:3].astype(np.float32)
            rgb = point[3:].astype(np.uint8)
            f.write(xyz.tobytes())
            f.write(rgb.tobytes())


# def set_camera_pose(camera, cam_pos, look_at):
#     # original logic from render_interaction_sim.py
#     forward = -(cam_pos - look_at)
#     forward /= np.linalg.norm(forward)

#     left = np.cross([0, 0, 1], forward)
#     left /= np.linalg.norm(left)

#     up = np.cross(forward, left)

#     mat = np.eye(4)
#     mat[:3, :3] = np.stack([forward, left, up], axis=1)
#     mat[:3, 3] = cam_pos

#     camera.set_pose(sapien.Pose.from_transformation_matrix(mat))

def render_static_canonical_state(obj_urdf_path: str, output_dir: Path, object_center: np.ndarray, object_radius: float, bb_min: np.ndarray):
    """Render dense static views of the object in canonical (closed) state."""
    
    print("Rendering static canonical state...")
    scene = init_scene(ray_tracing=False)
    
    # Load object
    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    articulate_obj = loader.load(obj_urdf_path)
    
    # Set to canonical pose (closed state)
    joint_limits = articulate_obj.get_qlimits()
    canonical_qpos = joint_limits[:, 0]  # Use minimum joint positions (closed)
    articulate_obj.set_qpos(canonical_qpos)
    
    aabb_lift = -bb_min[2] + 0.1
    articulate_obj.set_pose(sapien.Pose([0, 0, aabb_lift]))

    # Create camera
    camera = create_camera(scene, 640, 480, np.deg2rad(35), "static_camera")
    object_center_adjusted = object_center + np.array([0, 0, aabb_lift])
    
    # Generate dense viewpoints
    viewpoints = generate_dense_viewpoints(object_center_adjusted, object_radius)
    
    # Output directories
    rgb_dir = output_dir / "pre" / "frames"
    depth_dir = output_dir / "pre" / "depth"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)
    
    # Camera data for transforms.json
    frames = []
    all_points = []
    
    print(f"Rendering {len(viewpoints)} static views...")
    for i, viewpoint in enumerate(tqdm(viewpoints)):
        # Set camera pose
        set_camera_pose(camera, viewpoint['position'], viewpoint['look_at'])
        
        scene.step()
        scene.update_render()
        camera.take_picture()
   
        # Save RGB
        rgba_img = get_rgba_img(camera)
        rgb_img = Image.fromarray(rgba_img[..., :3])  # Drop alpha
        rgb_path = rgb_dir / f"frame_{i+1:05d}.png"
        rgb_img.save(rgb_path)
        
        # Save metric depth as .npy
        depth_img = get_depth_img(camera)
        depth_path = depth_dir / f"depth_{i+1:05d}.npy"
        np.save(depth_path, depth_img)
        
        # Extract point cloud (now in WORLD SPACE)
        pc = get_point_cloud(camera)
        all_points.append(pc)
        
        # ============ FIX: Get camera-to-world transformation ============
        # Get SAPIEN camera pose (camera-to-world)
        camera_pose = camera.get_pose()

        cam_to_world = camera.get_model_matrix()


        c2w = cam_to_world

        
        # Get intrinsics
        intrinsics = camera.get_intrinsic_matrix()
        fl_x = float(intrinsics[0, 0])
        fl_y = float(intrinsics[1, 1])
        cx = float(intrinsics[0, 2])
        cy = float(intrinsics[1, 2])        

        frames.append({
            'file_path': f"frames/{rgb_path.name}",
            'depth_file_path': f"depth/{depth_path.name}",
            'transform_matrix': c2w.tolist()
        })

    # --- Streaming Voxel Hash Merge (now working in world space) ---
    print("Merging and downsampling point clouds in world space...")

    voxel_size = 0.1  # ← REDUCED from 0.1 for finer detail
    voxel_table = {}   # key → (xyz, rgb)

    def voxel_hash(coords):
        """Spatial hash for voxel coordinates."""
        return (
            coords[:, 0].astype(np.int64) * 73856093 ^
            coords[:, 1].astype(np.int64) * 19349663 ^
            coords[:, 2].astype(np.int64) * 83492791
        )

    # Process each frame's point cloud
    for pc in tqdm(all_points, desc="Voxel merging"):
        pc_xyz = pc[:, :3]
        pc_rgb = pc[:, 3:]

        # Compute voxel indices
        coords = np.floor(pc_xyz / voxel_size).astype(np.int32)
        keys = voxel_hash(coords)

        # Insert only 1 point per voxel (first come first serve)
        for k, xyz, rgb in zip(keys, pc_xyz, pc_rgb):
            if k not in voxel_table:
                voxel_table[k] = (xyz, rgb)

    # Convert to final point cloud
    print("Finalizing fused point cloud...")
    downsampled_pc = np.array([
        np.concatenate([xyz, rgb]) 
        for xyz, rgb in voxel_table.values()
    ])

    # Save .ply
    ply_path = output_dir / "pre" / "fused_pc.ply"
    save_ply(ply_path, downsampled_pc)
    print(f"✅ Saved point cloud: {ply_path} ({len(downsampled_pc)} points)")

    transforms_data = {
        "camera_model": "OPENCV",  
        "fl_x": fl_x,
        "fl_y": fl_y,
        "cx": cx,
        "cy": cy,
        "w": 640,
        "h": 480,
        "k1": 0.0,
        "k2": 0.0,
        "p1": 0.0,
        "p2": 0.0,
        "ply_file_path": "fused_pc.ply",
        "frames": frames
    }

    
    with open(output_dir / "pre" / "transforms.json", 'w') as f:
        json.dump(transforms_data, f, indent=2)
    
    print(f"✅ Static canonical state rendered: {len(viewpoints)} views")
    scene = None  # Clean up
    return transforms_data
def render_articulation_sequence(
    obj_urdf_path: str,
    output_dir: Path,
    object_center: np.ndarray,
    object_radius: float,
    joint_id: int,
    joint_type: str,
    n_frames: int = 250,
    joint_axis_origin: np.ndarray = None,
    joint_axis_dir: np.ndarray = None,
    bb_min: np.ndarray = None,
):
    """
    Simple articulation rendering focused on good viewpoint capture.
    """
    import json
    import numpy as np
    from pathlib import Path
    from PIL import Image
    from scipy.spatial.transform import Rotation as R

    scene = init_scene(ray_tracing=False)

    # Load and position object
    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    articulate_obj = loader.load(obj_urdf_path)
    if articulate_obj is None:
        raise RuntimeError(f"Failed to load URDF: {obj_urdf_path}")

    aabb_lift = -bb_min[2] + 0.1
    articulate_obj.set_pose(sapien.Pose([0, 0, aabb_lift]))

    camera = create_camera(scene, 640, 480, np.deg2rad(35), "dynamic_camera")
    
    # Joint setup
    joint_limits = articulate_obj.get_qlimits()
    start_pos, end_pos = joint_limits[joint_id]
    joint_positions = np.linspace(start_pos, end_pos, n_frames)

    joints = articulate_obj.get_active_joints()
    control_joint = next((j for j in joints if j.name == f"joint_{joint_id}"), None)
    if control_joint is None:
        raise ValueError(f"Joint {joint_id} not found")

    child_link = control_joint.get_child_link()
    movable_actor_ids = [child_link.get_id()]

    # ================================================================
    # SIMPLE VIEWPOINT SETUP - LOOK AT MOTION MIDPOINT
    # ================================================================
    
    # Calculate midpoint of articulation motion for better framing
    mid_joint_pos = (start_pos + end_pos) / 2
    
    # Set joint to midpoint to get center of motion range
    qpos_mid = joint_limits[:, 0].copy()
    qpos_mid[joint_id] = mid_joint_pos
    articulate_obj.set_qpos(qpos_mid)
    scene.step()
    
    # Get child link position at motion midpoint
    child_mid_pos = child_link.get_pose().p
    
    # For revolute joints: calculate center between start and end positions
    if joint_type.lower() in ["hinge", "revolute"]:
        # Get child positions at start and end of rotation
        qpos_start = joint_limits[:, 0].copy()
        qpos_start[joint_id] = start_pos
        articulate_obj.set_qpos(qpos_start)
        scene.step()
        child_start_pos = child_link.get_pose().p
        
        qpos_end = joint_limits[:, 0].copy()
        qpos_end[joint_id] = end_pos
        articulate_obj.set_qpos(qpos_end)
        scene.step()
        child_end_pos = child_link.get_pose().p
        
        # Look at center of motion arc
        look_at_point = (child_start_pos + child_end_pos) / 2
        print(f"Revolute joint - looking at motion arc center")
        
    else:  # Prismatic/slider
        # For linear motion, midpoint is the best look-at
        look_at_point = child_mid_pos
        print(f"Prismatic joint - looking at motion midpoint")
    
    # Camera distance - closer but not too close
    camera_distance = max(2.5, 3.5 * object_radius)
    
    # Simple viewpoint pattern: orbit around motion center at fixed elevation
    elevation = np.deg2rad(30)  # Fixed 30-degree elevation
    
    print(f"Camera setup:")
    print(f"  Look-at point (motion center): {look_at_point}")
    print(f"  Joint axis origin: {joint_axis_origin + np.array([0, 0, aabb_lift])}")
    print(f"  Camera distance: {camera_distance:.2f}")
    print(f"  Elevation: 25°")
    print(f"  Arc: ±15° in front (30° total)")
    
    # Reset to initial position for rendering loop
    articulate_obj.set_qpos(joint_limits[:, 0])
    scene.step()

    # Create output directories
    frames_dir = output_dir / "post" / "frames"
    depth_dir = output_dir / "post" / "depth"
    mask_dir = output_dir / "post" / "mask_gt"
    for d in [frames_dir, depth_dir, mask_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # ================================================================
    # SIMPLE RENDER LOOP - SMALL ARC IN FRONT
    # ================================================================
    
    frames = []

    # Small arc parameters
    arc_center = np.deg2rad(180)  # Front view (180° - looking AT object, not away)
    arc_width = np.deg2rad(30)    # ±15° arc (30° total)
    
    for idx, q in enumerate(tqdm(joint_positions, desc="Rendering articulation")):
        frame_id = idx + 1

        # Set joint position
        qpos = joint_limits[:, 0].copy()
        qpos[joint_id] = q
        articulate_obj.set_qpos(qpos)
        scene.step()

        # Small arc in front: smoothly vary azimuth across the arc
        progress = idx / (n_frames - 1)  # 0 to 1
        azimuth = arc_center + (progress - 0.5) * arc_width  # -15° to +15°
        
        # Calculate camera position around motion center
        cam_x = look_at_point[0] + camera_distance * np.cos(elevation) * np.cos(azimuth)
        cam_y = look_at_point[1] + camera_distance * np.cos(elevation) * np.sin(azimuth)
        cam_z = look_at_point[2] + camera_distance * np.sin(elevation)
        
        cam_pos = np.array([cam_x, cam_y, cam_z])
        
        # Always look at motion center point
        set_camera_pose(camera, cam_pos, look_at_point)

        # Render
        scene.update_render()
        camera.take_picture()

        # Save RGB
        rgba = get_rgba_img(camera)[..., :3]
        rgb = Image.fromarray(rgba)
        rgb_path = frames_dir / f"frame_{frame_id:05d}.png"
        rgb.save(rgb_path)

        # Save depth
        depth = get_depth_img(camera)
        depth_path = depth_dir / f"depth_{frame_id:05d}.npy"
        np.save(depth_path, depth)
        
        # Save mask
        try:
            seg_labels = camera.get_uint32_texture("Segmentation")
            actor_seg = seg_labels[..., 1]
            mask = np.zeros_like(actor_seg, np.uint8)
            for aid in movable_actor_ids:
                mask[actor_seg == aid] = 255
        except Exception:
            mask = np.zeros((480, 640), np.uint8)

        mask_path = mask_dir / f"mask_{frame_id:05d}.png"
        Image.fromarray(mask).save(mask_path)

        # Save camera metadata
        cam_pose = camera.get_model_matrix().tolist()
        frames.append({
            "file_path": f"frames/{rgb_path.name}",
            "depth_file_path": f"depth/{depth_path.name}",
            "mask_file_path_gt": f"mask_gt/{mask_path.name}",
            "transform_matrix": cam_pose,
            "joint_angle_gt": float(q),
        })

    # ================================================================
    # SAVE METADATA
    # ================================================================
    
    # Camera intrinsics
    intr = camera.get_intrinsic_matrix()
    fx, fy = float(intr[0, 0]), float(intr[1, 1])
    cx, cy = float(intr[0, 2]), float(intr[1, 2])

    # Joint metadata
    if joint_type.lower() in ["hinge", "revolute"]:
        jt_type = "revolute"
        jlimits = [float(np.rad2deg(start_pos)), float(np.rad2deg(end_pos))]
    elif joint_type.lower() in ["slider", "prismatic"]:
        jt_type = "prismatic"
        jlimits = [float(start_pos), float(end_pos)]
    else:
        jt_type = joint_type.lower()
        jlimits = [float(start_pos), float(end_pos)]

    articulations_gt = [{
        "joint_type_gt": jt_type,
        "joint_axis_gt": joint_axis_dir.tolist() if joint_axis_dir is not None else [0, 0, 1],
        "joint_pivot_gt": joint_axis_origin.tolist() if joint_axis_origin is not None else [0, 0, 0],
        "joint_limits_gt": jlimits,
    }]

    transforms_data = {
        "camera_model": "PINHOLE",
        "fl_x": fx,
        "fl_y": fy,
        "cx": cx,
        "cy": cy,
        "w": 640,
        "h": 480,
        "k1": 0.0,
        "k2": 0.0,
        "p1": 0.0,
        "p2": 0.0,
        "articulations_gt": articulations_gt,
        "frames": frames,
    }

    json_path = output_dir / "post" / "transforms.json"
    with open(json_path, "w") as f:
        json.dump(transforms_data, f, indent=2)

    print(f"✅ Rendered {n_frames} frames with small arc in front of joint")
    scene = None
    return transforms_data

def estimate_object_bounds_and_joint_info(meta_path: Path, joint_id: int = 0):
    """
    Estimate object center, radius, and joint axis information directly from metadata JSON.
    Works with nested PartNet-Mobility metadata structures.
    """

    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    try:
        with open(meta_path, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {meta_path}: {e}")

    bounding_box = None
    joint_axis_origin = None
    joint_axis_dir = None
    joint_type = None

    # --- Traverse to find bounding box and joint info ---
    if isinstance(data, dict):
        for category in data.values():
            if not isinstance(category, dict):
                continue
            for obj_data in category.values():
                if not isinstance(obj_data, dict):
                    continue

                # --- bounding box ---
                if not bounding_box and 'boundingbox' in obj_data:
                    bounding_box = obj_data['boundingbox']

                # --- joint info ---
                if 'interaction_list' in obj_data:
                    for interaction in obj_data['interaction_list']:
                        if interaction.get("id") == joint_id:
                            joint_data = interaction.get("joint", {})
                            joint_type = interaction.get("type", None)
                            if "axis" in joint_data:
                                axis = joint_data["axis"]
                                joint_axis_origin = np.array(axis.get("origin", [0, 0, 0]), dtype=float)
                                joint_axis_dir = np.array(axis.get("direction", [0, 0, 1]), dtype=float)
                            break
                if bounding_box and joint_axis_origin is not None:
                    break
            if bounding_box and joint_axis_origin is not None:
                break

    # --- Compute object bounds ---
    if bounding_box and 'min' in bounding_box and 'max' in bounding_box:
        bb_min = np.array(bounding_box['min'], dtype=float)
        bb_max = np.array(bounding_box['max'], dtype=float)

        object_center = (bb_min + bb_max) / 2
        object_size = bb_max - bb_min
        object_radius = np.linalg.norm(object_size) / 2

        print("✅ Found bounding box in metadata:")
        print(f"   Min: {bb_min}")
        print(f"   Max: {bb_max}")
        print(f"   Center: {object_center}")
        print(f"   Radius: {object_radius:.3f}")
    else:
        print("⚠️  No bounding box found in metadata, using defaults")
        object_center = np.array([0, 0, 0], dtype=float)
        object_radius = 1.0

    # --- Validate and normalize joint info ---
    if joint_axis_origin is not None and joint_axis_dir is not None:
        joint_axis_dir = joint_axis_dir / np.linalg.norm(joint_axis_dir)
        print("✅ Found joint metadata:")
        print(f"   Joint ID: {joint_id}")
        print(f"   Type: {joint_type}")
        print(f"   Axis origin: {joint_axis_origin}")
        print(f"   Axis direction: {joint_axis_dir}")
    else:
        print(f"⚠️  No joint axis info found for joint {joint_id} in {meta_path}")
        joint_axis_origin = np.array([0, 0, 0], dtype=float)
        joint_axis_dir = np.array([0, 0, 1], dtype=float)

    return object_center, object_radius, joint_axis_origin, joint_axis_dir, joint_type, bb_min, bb_max


def extract_gt_meshes_from_objs(
    shape_path: Path,
    output_dir: Path,
    joint_id: int = 0,
    num_samples: int = 100000
):
    """
    Extract GT meshes directly from textured_objs using mobility metadata.
    """
    import pyvista as pv
    
    # Parse which parts are moving
    mobility_file = json.loads((shape_path / 'mobility_v2.json').read_text())
    partid_to_objs = parse_partid_to_objs(shape_path)
    
    # Find moving parts
    moving_objs = set()
    static_objs = set()
    
    for part in mobility_file:
        pid = part['id']
        parent_id = part['parent']
        
        # Get obj files for this part
        partids_in_result = [obj['id'] for obj in part["parts"]]
        objs_for_part = set()
        for partid in partids_in_result:
            objs_for_part |= partid_to_objs[partid]
        
        # Check if this is a moving part (has non-fixed joint)
        if parent_id != -1 and part.get('joint') not in ['fixed', 'junk']:
            # Check if this is the joint we care about
            if pid == joint_id or part.get('name') == f'link_{joint_id}':
                moving_objs |= objs_for_part
            else:
                static_objs |= objs_for_part
        else:
            static_objs |= objs_for_part
    
    # Remove overlap
    static_objs -= moving_objs
    
    gt_dir = output_dir / "gt_meshes"
    gt_dir.mkdir(parents=True, exist_ok=True)
    
    # Merge and save static parts
    if static_objs:
        static_paths = [shape_path / 'textured_objs' / (obj + '.obj') for obj in static_objs]
        merge_meshs(static_paths, gt_dir / "static_parts.ply")
        print(f"✅ Saved static parts: {len(static_objs)} obj files")
    
    # Merge and save moving parts
    if moving_objs:
        moving_paths = [shape_path / 'textured_objs' / (obj + '.obj') for obj in moving_objs]
        merge_meshs(moving_paths, gt_dir / "moving_parts.ply")
        print(f"✅ Saved moving parts: {len(moving_objs)} obj files")
    
    # Merge all for whole object
    all_objs = static_objs | moving_objs
    all_paths = [shape_path / 'textured_objs' / (obj + '.obj') for obj in all_objs]
    merge_meshs(all_paths, gt_dir / "whole_object.ply")
    print(f"✅ Saved whole object: {len(all_objs)} obj files")
    
    return gt_dir

def create_videos(output_dir):
    """Create MP4 videos from rendered frames using FFmpeg"""
    
    pre_frames = output_dir / "pre" / "frames"
    post_frames = output_dir / "post" / "frames"
    
    def make_video(frames_dir, output_path, fps=15):
        """Create video from frame sequence"""
        if not frames_dir.exists():
            print(f"Warning: Frames directory not found: {frames_dir}")
            return
            
        # Count frames
        frame_files = sorted(list(frames_dir.glob("*.png")) + list(frames_dir.glob("*.jpg")))
        if len(frame_files) < 2:
            print(f"Warning: Not enough frames in {frames_dir}")
            return
            
        try:
            cmd = [
                'ffmpeg', '-y',  # -y to overwrite existing files
                '-framerate', str(fps),
                '-pattern_type', 'glob',
                '-i', str(frames_dir / '*.png'),  # Assume PNG frames
                '-c:v', 'libx264',
                '-pix_fmt', 'yuv420p',  # Compatibility format
                '-crf', '23',  # Good quality/size balance
                str(output_path)
            ]
            
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"✅ Video created: {output_path}")
            
        except subprocess.CalledProcessError as e:
            print(f"❌ FFmpeg failed for {output_path}: {e.stderr}")
        except FileNotFoundError:
            print("❌ FFmpeg not found. Please install: apt install ffmpeg")
    
    # Create videos
    make_video(pre_frames, output_dir / "pre_static.mp4", fps=5)  # Slower for static views
    make_video(post_frames, output_dir / "post_articulation.mp4", fps=15)  # Normal speed
    
    print(f"Videos saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Enhanced PartNet rendering for 3DGS training")
    parser.add_argument("--partnet_dir", type=str, required=True,
                        help="Path to PartNet-Mobility object directory")
    parser.add_argument("--meta", type=str,
                        help="Metadata file name (default: meta.json)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for rendered data")
    parser.add_argument("--joint_id", type=int, default=0,
                        help="Joint ID to articulate")
    parser.add_argument("--n_frames", type=int, default=150,
                        help="Number of frames in articulation sequence")

    # -------- NEW ARGUMENTS FOR EVALUATION MODE --------
    parser.add_argument("--eval_mode", action="store_true",
                        help="Generate evaluation dataset (no per-object meta required)")
    parser.add_argument("--eval_config", type=str,
                        help="Path to evaluation config YAML file")
    parser.add_argument("--gt_joints_file", type=str,
                        default="new_partnet_mobility_dataset_correct_intr_meta.json",
                        help="Global JSON containing ground-truth joint parameters")
    # ----------------------------------------------------

    args = parser.parse_args()

    partnet_dir = Path(args.partnet_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    urdf_path = partnet_dir / "mobility.urdf"
    if not urdf_path.exists():
        raise FileNotFoundError(f"URDF not found: {urdf_path}")


    # ------------------------------------------------------------------
    # METADATA HANDLING
    # ------------------------------------------------------------------
    if args.eval_mode:
        print("Running in EVALUATION MODE")

        import json, yaml

        gt_file = Path(args.gt_joints_file)
        if not gt_file.exists():
            raise FileNotFoundError(f"Ground truth joint file not found: {gt_file}")

        with open(gt_file, "r") as f:
            gt_data = json.load(f)

        if args.eval_config is None:
            raise ValueError("--eval_config must be provided in eval_mode")

        with open(args.eval_config, "r") as f:
            eval_cfg = yaml.safe_load(f)

        # Find this object entry in the config
        obj_name = partnet_dir.name  # e.g., "103351"
        entry = None
        for item in eval_cfg.get("datasets", {}).get("partnet_objects", []):
            if str(item["object_id"]) == obj_name:
                entry = item
                break

        if entry is None:
            raise KeyError(f"No object_id {obj_name} found in {args.eval_config}")

        category = entry["category"]
        joint_id = args.joint_id

        if category not in gt_data:
            raise KeyError(f"Category '{category}' not found in {gt_file}")
        if obj_name not in gt_data[category]:
            raise KeyError(f"Object '{obj_name}' not found under category '{category}' in {gt_file}")

        meta = gt_data[category][obj_name]

        # Parse bounding box and joint metadata
        bb_min = np.array(meta["boundingbox"]["min"], dtype=float)
        bb_max = np.array(meta["boundingbox"]["max"], dtype=float)
        object_center = (bb_min + bb_max) / 2
        object_radius = np.linalg.norm(bb_max - bb_min) / 2

        # Use interaction_list
        interaction = next((x for x in meta["interaction_list"] if x["id"] == joint_id), meta["interaction_list"][0])
        joint_type = interaction["type"]
        joint_data = interaction["joint"]
        axis_info = joint_data.get("axis", {})
        joint_axis_origin = np.array(axis_info.get("origin", [0, 0, 0]), dtype=float)
        joint_axis_dir = np.array(axis_info.get("direction", [0, 0, 1]), dtype=float)
        joint_axis_dir /= np.linalg.norm(joint_axis_dir)

        print(f"✅ Loaded GT metadata for {category}/{obj_name}")
        print(f"   Type: {joint_type}")
        print(f"   Axis origin: {joint_axis_origin}")
        print(f"   Axis direction: {joint_axis_dir}")
        print(f"   Radius: {object_radius:.3f}")


        gt_meshes = extract_gt_meshes_from_objs(
            partnet_dir, output_dir, args.joint_id
        )

    else:
        if args.meta is None:
            raise ValueError("--meta must be provided when not in eval_mode")

        meta_dir = Path(args.meta)
        (object_center,
         object_radius,
         joint_axis_origin,
         joint_axis_dir,
         joint_type,
         bb_min,
         bb_max) = estimate_object_bounds_and_joint_info(meta_dir)



    print(f"\nEnhanced PartNet Rendering for 3DGS")
    print(f"Object: {partnet_dir.name}")
    print(f"Joint: {args.joint_id}")
    print(f"Output: {output_dir}")
    print(f"Estimated object center: {object_center}")
    print(f"Estimated object radius: {object_radius}")

    # Step 1: Render static canonical state
    # static_data = render_static_canonical_state(
    #     str(urdf_path), output_dir, object_center, object_radius, bb_min
    # )

    # Step 2: Render articulation sequence
    sequence_data = render_articulation_sequence(
        str(urdf_path), output_dir, object_center, object_radius,
        args.joint_id, joint_type=joint_type, n_frames=args.n_frames,
        joint_axis_origin=joint_axis_origin, joint_axis_dir=joint_axis_dir,
        bb_min=bb_min
    )

    # Save metadata summary
    metadata = {
        "object_id": partnet_dir.name,
        "joint_id": args.joint_id,
        "object_center": object_center.tolist(),
        "object_radius": object_radius,
        # "static_views": len(static_data),
        "sequence_frames": len(sequence_data["frames"]),
        "urdf_path": str(urdf_path),
        "eval_mode": args.eval_mode,
    }

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    create_videos(output_dir)


if __name__ == "__main__":
    main()