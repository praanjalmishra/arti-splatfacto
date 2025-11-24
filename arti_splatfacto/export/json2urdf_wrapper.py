# json2urdf_wrapper.py

import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
import trimesh


def gaussians_to_mesh(
    gaussians: Dict,
    resolution: int = 128,
    method: str = "marching_cubes"
) -> trimesh.Trimesh:
    """
    Convert Gaussian Splatting representation to triangle mesh.
    
    Args:
        gaussians: Dict with means, scales, quats, features_dc, etc.
        resolution: Grid resolution for meshing
        method: "marching_cubes" or "poisson" or "ball_pivoting"
    
    Returns:
        Trimesh object
    """
    means = np.array(gaussians["means"])
    scales = np.array(gaussians["scales"])
    features_dc = np.array(gaussians["features_dc"])
    opacities = np.array(gaussians["opacities"])
    
    if means.size == 0:
        # Return empty mesh
        return trimesh.Trimesh()
    
    # Filter by opacity threshold
    opacity_thresh = 0.1
    valid_mask = opacities.squeeze() > opacity_thresh
    means = means[valid_mask]
    scales = scales[valid_mask]
    features_dc = features_dc[valid_mask]
    
    if len(means) == 0:
        return trimesh.Trimesh()
    
    # Convert SH to RGB
    from nerfstudio.utils.spherical_harmonics import SH2RGB
    import torch
    colors = SH2RGB(torch.from_numpy(features_dc)).numpy()
    colors = (colors * 255).astype(np.uint8)
    
    if method == "marching_cubes":
        # Method 1: Marching Cubes on Gaussian density field
        mesh = gaussians_to_mesh_marching_cubes(means, scales, colors, resolution)
    
    elif method == "poisson":
        # Method 2: Poisson surface reconstruction
        mesh = gaussians_to_mesh_poisson(means, scales, colors)
    
    elif method == "ball_pivoting":
        # Method 3: Ball pivoting
        mesh = gaussians_to_mesh_ball_pivoting(means, scales, colors)
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return mesh


def gaussians_to_mesh_marching_cubes(
    means: np.ndarray,
    scales: np.ndarray,
    colors: np.ndarray,
    resolution: int = 128
) -> trimesh.Trimesh:
    """
    Convert Gaussians to mesh using marching cubes on density field.
    """
    from skimage import measure
    
    # Compute bounding box
    bbox_min = means.min(axis=0) - 3 * scales.max()
    bbox_max = means.max(axis=0) + 3 * scales.max()
    
    # Create grid
    x = np.linspace(bbox_min[0], bbox_max[0], resolution)
    y = np.linspace(bbox_min[1], bbox_max[1], resolution)
    z = np.linspace(bbox_min[2], bbox_max[2], resolution)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    grid_points = np.stack([X, Y, Z], axis=-1)  # [res, res, res, 3]
    
    # Compute density field
    density = np.zeros((resolution, resolution, resolution))
    
    print(f"  Computing density field for {len(means)} Gaussians...")
    for i, (mean, scale) in enumerate(zip(means, scales)):
        # Gaussian density contribution
        diff = grid_points - mean
        # Use average scale for isotropic Gaussian
        sigma = scale.mean()
        dist_sq = np.sum(diff**2, axis=-1)
        density += np.exp(-dist_sq / (2 * sigma**2))
    
    # Marching cubes
    print(f"  Running marching cubes...")
    threshold = 0.5  # Adjust this for mesh quality
    verts, faces, normals, _ = measure.marching_cubes(density, level=threshold)
    
    # Scale vertices to world coordinates
    verts[:, 0] = verts[:, 0] / resolution * (bbox_max[0] - bbox_min[0]) + bbox_min[0]
    verts[:, 1] = verts[:, 1] / resolution * (bbox_max[1] - bbox_min[1]) + bbox_min[1]
    verts[:, 2] = verts[:, 2] / resolution * (bbox_max[2] - bbox_min[2]) + bbox_min[2]
    
    # Create mesh
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
    
    # Assign colors (simple nearest neighbor)
    from scipy.spatial import cKDTree
    tree = cKDTree(means)
    _, nearest_idx = tree.query(verts)
    vertex_colors = colors[nearest_idx]
    mesh.visual.vertex_colors = vertex_colors
    
    return mesh


def gaussians_to_mesh_poisson(
    means: np.ndarray,
    scales: np.ndarray,
    colors: np.ndarray,
) -> trimesh.Trimesh:
    """
    Convert Gaussians to mesh using Poisson surface reconstruction.
    Requires Open3D.
    """
    import open3d as o3d
    
    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(means)
    pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
    
    # Estimate normals
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )
    
    # Poisson reconstruction
    print(f"  Running Poisson reconstruction...")
    mesh_o3d, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=9
    )
    
    # Remove low-density vertices
    vertices_to_remove = densities < np.quantile(densities, 0.01)
    mesh_o3d.remove_vertices_by_mask(vertices_to_remove)
    
    # Convert to trimesh
    vertices = np.asarray(mesh_o3d.vertices)
    faces = np.asarray(mesh_o3d.triangles)
    vertex_colors = np.asarray(mesh_o3d.vertex_colors)
    
    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        vertex_colors=(vertex_colors * 255).astype(np.uint8)
    )
    
    return mesh


def gaussians_to_mesh_ball_pivoting(
    means: np.ndarray,
    scales: np.ndarray,
    colors: np.ndarray,
) -> trimesh.Trimesh:
    """
    Convert Gaussians to mesh using ball pivoting algorithm.
    """
    import open3d as o3d
    
    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(means)
    pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
    
    # Estimate normals
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )
    
    # Ball pivoting
    print(f"  Running ball pivoting...")
    radii = [0.005, 0.01, 0.02, 0.04]
    mesh_o3d = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd,
        o3d.utility.DoubleVector(radii)
    )
    
    # Convert to trimesh
    vertices = np.asarray(mesh_o3d.vertices)
    faces = np.asarray(mesh_o3d.triangles)
    
    # Get colors from nearest neighbors
    from scipy.spatial import cKDTree
    tree = cKDTree(means)
    _, nearest_idx = tree.query(vertices)
    vertex_colors = colors[nearest_idx]
    
    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        vertex_colors=vertex_colors
    )
    
    return mesh


def convert_json_to_meshes(
    json_path: Path,
    output_dir: Path,
    mesh_format: str = "obj",
    meshing_method: str = "poisson",
    resolution: int = 128,
) -> Dict[str, Path]:
    """
    Convert Gaussian JSON to meshes for URDF.
    
    Args:
        json_path: Input JSON path
        output_dir: Output directory for meshes
        mesh_format: "obj" or "ply" or "stl"
        meshing_method: Method for Gaussian to mesh conversion
        resolution: Grid resolution for marching cubes
    
    Returns:
        Dict mapping joint_id to mesh path
    """
    print(f"\n{'='*70}")
    print("CONVERTING GAUSSIANS TO MESHES")
    print(f"{'='*70}")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    meshes_dir = output_dir / "meshes"
    meshes_dir.mkdir(parents=True, exist_ok=True)
    
    mesh_paths = {}
    
    base = json_path.parent  # where model.json + arrays/ live

    for joint_id, joint_data in data["joints"].items():
        print(f"\nConverting {joint_id}...")

        # Load both object (moving) and canonical (inertial) Gaussians
        obj_npz = base / joint_data["object_npz"]
        can_npz = base / joint_data["canonical_npz"]

        if not obj_npz.exists() or not can_npz.exists():
            print(f"  ⚠️  Missing .npz for {joint_id}, skipping")
            continue

        obj_gauss = dict(np.load(obj_npz))
        can_gauss = dict(np.load(can_npz))

        # Generate both meshes
        obj_mesh = gaussians_to_mesh(obj_gauss, resolution, meshing_method)
        can_mesh = gaussians_to_mesh(can_gauss, resolution, meshing_method)

        # Save object (moving) mesh
        if len(obj_mesh.vertices) > 0:
            obj_path = meshes_dir / f"{joint_id}_object.{mesh_format}"
            obj_mesh.export(str(obj_path))
            mesh_paths[f"{joint_id}_object"] = obj_path
            print(f"  ✓ Saved moving mesh: {obj_path} ({len(obj_mesh.vertices)} verts)")
        else:
            print(f"  ⚠️  Empty moving mesh for {joint_id}")

        # Save canonical (inertial) mesh
        if len(can_mesh.vertices) > 0:
            can_path = meshes_dir / f"{joint_id}_canonical.{mesh_format}"
            can_mesh.export(str(can_path))
            mesh_paths[f"{joint_id}_canonical"] = can_path
            print(f"  ✓ Saved canonical mesh: {can_path} ({len(can_mesh.vertices)} verts)")
        else:
            print(f"  ⚠️  Empty canonical mesh for {joint_id}")



    # Convert background
    print(f"\nConverting background...")
    base = json_path.parent

    # Load from .npz file instead of JSON
    if "npz_file" in data["background"]:
        bg_npz = base / data["background"]["npz_file"]
        if bg_npz.exists():
            bg_gaussians = dict(np.load(bg_npz))
        else:
            print(f"  ⚠️  Background npz not found: {bg_npz}")
            bg_gaussians = None
    else:
        print(f"  ⚠️  No background npz_file key found in JSON")
        bg_gaussians = None

    if bg_gaussians is not None:
        # Optionally downsample for speed
        if bg_gaussians["means"].shape[0] > 300_000:
            print("  ⚠️  Background very dense — sampling 100k points for meshing")
            idx = np.random.choice(bg_gaussians["means"].shape[0], 100_000, replace=False)
            for k in bg_gaussians:
                bg_gaussians[k] = bg_gaussians[k][idx]

        bg_mesh = gaussians_to_mesh(bg_gaussians, resolution, meshing_method)

        if len(bg_mesh.vertices) > 0:
            bg_path = meshes_dir / f"background.{mesh_format}"
            bg_mesh.export(str(bg_path))
            mesh_paths["background"] = bg_path
            print(f"  ✓ Saved: {bg_path} ({len(bg_mesh.vertices)} vertices)")
        else:
            print(f"  ⚠️  Empty background mesh")

    
    print(f"\n{'='*70}")
    print(f"MESH CONVERSION COMPLETE")
    print(f"Total meshes: {len(mesh_paths)}")
    print(f"{'='*70}\n")
    
    return mesh_paths

def export_urdf(
    json_path: Path,
    output_dir: Path,
    robot_name: str = "gaussian_robot",
    mesh_format: str = "obj",
    meshing_method: str = "poisson",
    resolution: int = 128,
) -> Path:
    """
    Export URDF from Gaussian JSON (NPZ-based) with automatic mesh generation.
    Each joint exports both:
        - canonical mesh (reference / inertial)
        - object mesh (moving articulated part)
    """
    print(f"\n{'='*70}")
    print("EXPORTING URDF FROM GAUSSIAN MODEL")
    print(f"{'='*70}")

    # Load canonicalized JSON
    with open(json_path, "r") as f:
        data = json.load(f)

    # Convert Gaussians to meshes (object + canonical)
    mesh_paths = convert_json_to_meshes(
        json_path,
        output_dir,
        mesh_format,
        meshing_method,
        resolution,
    )

    print(f"\nGenerating URDF...")
    urdf = '<?xml version="1.0"?>\n'
    urdf += f'<robot name="{robot_name}">\n'
    urdf += '\t<link name="base"/>\n'

    joint_ids = sorted(data["joints"].keys())

    # Create links and joints
    for i, joint_id in enumerate(joint_ids):
        joint_data = data["joints"][joint_id]
        meta = joint_data["metadata"]

        pivot = meta["pivot"]
        axis = meta["axis"]
        joint_type = meta["joint_type"]
        limits = meta["limits"]

        urdf_joint_type = "revolute" if joint_type == "revolute" else "prismatic"
        link_name = f"link_{joint_id}"

        # Look for both canonical and object meshes
        canonical_key = f"{joint_id}_canonical"
        object_key = f"{joint_id}_object"
        canonical_mesh = mesh_paths.get(canonical_key)
        object_mesh = mesh_paths.get(object_key)

        if not canonical_mesh and not object_mesh:
            print(f"  ⚠️  No meshes for {joint_id}, skipping link")
            continue

        urdf += f'\t<link name="{link_name}">\n'

        # Canonical (reference frame)
        if canonical_mesh:
            urdf += f'\t\t<visual>\n'
            urdf += f'\t\t\t<origin xyz="0 0 0"/>\n'
            urdf += f'\t\t\t<geometry>\n'
            urdf += f'\t\t\t\t<mesh filename="meshes/{canonical_mesh.name}" />\n'
            urdf += f'\t\t\t</geometry>\n'
            urdf += f'\t\t</visual>\n'

        # Object (moving geometry)
        if object_mesh:
            urdf += f'\t\t<visual>\n'
            urdf += f'\t\t\t<origin xyz="0 0 0"/>\n'
            urdf += f'\t\t\t<geometry>\n'
            urdf += f'\t\t\t\t<mesh filename="meshes/{object_mesh.name}" />\n'
            urdf += f'\t\t\t</geometry>\n'
            urdf += f'\t\t</visual>\n'

        # Also use same meshes for collision
        if object_mesh:
            urdf += f'\t\t<collision>\n'
            urdf += f'\t\t\t<origin xyz="0 0 0"/>\n'
            urdf += f'\t\t\t<geometry>\n'
            urdf += f'\t\t\t\t<mesh filename="meshes/{object_mesh.name}" />\n'
            urdf += f'\t\t\t</geometry>\n'
            urdf += f'\t\t</collision>\n'
        elif canonical_mesh:
            urdf += f'\t\t<collision>\n'
            urdf += f'\t\t\t<origin xyz="0 0 0"/>\n'
            urdf += f'\t\t\t<geometry>\n'
            urdf += f'\t\t\t\t<mesh filename="meshes/{canonical_mesh.name}" />\n'
            urdf += f'\t\t\t</geometry>\n'
            urdf += f'\t\t</collision>\n'

        urdf += f'\t</link>\n'

        # Create joint connecting to previous link
        parent_link = "base" if i == 0 else f"link_{joint_ids[i-1]}"
        urdf += f'\t<joint name="joint_{joint_id}" type="{urdf_joint_type}">\n'
        urdf += f'\t\t<origin xyz="{pivot[0]} {pivot[1]} {pivot[2]}" rpy="0 0 0"/>\n'
        urdf += f'\t\t<axis xyz="{axis[0]} {axis[1]} {axis[2]}"/>\n'
        urdf += f'\t\t<parent link="{parent_link}"/>\n'
        urdf += f'\t\t<child link="{link_name}"/>\n'
        urdf += f'\t\t<limit lower="{limits[0]}" upper="{limits[1]}" effort="100" velocity="1.0"/>\n'
        urdf += f'\t</joint>\n'

    # Add background (optional, fixed)
    if "background" in mesh_paths:
        mesh_rel = f"meshes/{mesh_paths['background'].name}"
        urdf += f'\t<link name="link_background">\n'
        urdf += f'\t\t<visual>\n'
        urdf += f'\t\t\t<origin xyz="0 0 0"/>\n'
        urdf += f'\t\t\t<geometry>\n'
        urdf += f'\t\t\t\t<mesh filename="{mesh_rel}" />\n'
        urdf += f'\t\t\t</geometry>\n'
        urdf += f'\t\t</visual>\n'
        urdf += f'\t</link>\n'
        urdf += f'\t<joint name="base_to_background" type="fixed">\n'
        urdf += f'\t\t<origin rpy="0 0 0" xyz="0 0 0"/>\n'
        urdf += f'\t\t<parent link="base"/>\n'
        urdf += f'\t\t<child link="link_background"/>\n'
        urdf += f'\t</joint>\n'

    urdf += '</robot>'

    # Save URDF
    urdf_path = output_dir / f"{robot_name}.urdf"
    urdf_path.parent.mkdir(parents=True, exist_ok=True)
    with open(urdf_path, "w") as f:
        f.write(urdf)

    print(f"\n{'='*70}")
    print("URDF EXPORT COMPLETE")
    print(f"URDF:", urdf_path)
    print(f"Meshes:", output_dir / "meshes")
    print(f"{'='*70}\n")

    return urdf_path



if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert Gaussian JSON to URDF")
    parser.add_argument("--input", type=Path, required=True, help="Input canonicalized JSON")
    parser.add_argument("--output", type=Path, required=True, help="Output directory")
    parser.add_argument("--robot-name", type=str, default="gaussian_robot", help="Robot name")
    parser.add_argument("--mesh-format", type=str, default="obj", choices=["obj", "ply", "stl"])
    parser.add_argument("--method", type=str, default="poisson", 
                       choices=["marching_cubes", "poisson", "ball_pivoting"])
    parser.add_argument("--resolution", type=int, default=128, help="Grid resolution for marching cubes")
    
    args = parser.parse_args()
    
    export_urdf(
        args.input,
        args.output,
        args.robot_name,
        args.mesh_format,
        args.method,
        args.resolution
    )