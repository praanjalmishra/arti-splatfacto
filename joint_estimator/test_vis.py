import argparse
import torch
import numpy as np
import open3d as o3d


# ---------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------

def create_voxel_mesh(voxels, bbox_min, bbox_max, color=[0.8, 0.2, 0.8]):
    """Create a single unified mesh from voxel occupancy instead of many cubes."""
    vox = voxels.numpy()
    res = vox.shape[0]

    lengths = bbox_max - bbox_min
    voxel_size = lengths / res

    occupied = np.argwhere(vox > 0)
    mesh = o3d.geometry.TriangleMesh()

    for idx in occupied:
        center = bbox_min + (idx + 0.5) * voxel_size
        cube = o3d.geometry.TriangleMesh.create_box(*voxel_size)
        cube.translate(center - voxel_size / 2)
        mesh += cube

    mesh.paint_uniform_color(color)
    mesh.compute_vertex_normals()
    return mesh


def create_coordinate_frame(origin, axis, scale=0.1):
    """Joint frame: align Z-axis with joint axis."""
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=scale)

    z = np.array([0, 0, 1.0])
    v = axis / np.linalg.norm(axis)
    rot_axis = np.cross(z, v)

    if np.linalg.norm(rot_axis) > 1e-8:
        rot_axis /= np.linalg.norm(rot_axis)
        angle = np.arccos(np.dot(z, v))
        R = o3d.geometry.get_rotation_matrix_from_axis_angle(rot_axis * angle)
        frame.rotate(R, center=np.zeros(3))

    frame.translate(origin)
    return frame


def create_slider_line(origin, direction, bbox_min, bbox_max, color=[1, 0.6, 0]):
    """Infinite-like slider direction line."""
    d = direction / np.linalg.norm(direction)
    span = np.linalg.norm(bbox_max - bbox_min) * 1.2

    p1 = origin - d * span
    p2 = origin + d * span

    pts = [p1, p2]
    lines = [[0, 1]]

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(pts)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([color])

    return line_set


def create_hinge_plane(pivot, axis, size=0.25):
    """Semi-transparent hinge rotation plane."""
    axis = axis / np.linalg.norm(axis)

    tmp = np.array([1, 0, 0]) if abs(axis[0]) < 0.9 else np.array([0, 1, 0])
    u = np.cross(axis, tmp)
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)

    s = size
    corners = [
        pivot + s*u + s*v,
        pivot - s*u + s*v,
        pivot - s*u - s*v,
        pivot + s*u - s*v
    ]

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(corners)
    mesh.triangles = o3d.utility.Vector3iVector([[0, 1, 2], [0, 2, 3]])
    mesh.paint_uniform_color([0.1, 0.6, 1.0])
    mesh.compute_vertex_normals()

    return mesh


# ---------------------------------------------------------------------
# Visualization Function
# ---------------------------------------------------------------------

def visualize_with_open3d(pt_path, npy_path=None, ply_path=None, show_voxels=False):
    data = torch.load(pt_path)

    bbox_min = data["bbox_min"].numpy()
    bbox_max = data["bbox_max"].numpy()
    joint_axis = data["joint_axis"].numpy()
    joint_pivot = data["joint_pivot"].numpy()
    joint_type = data["joint_type"]
    joint_limits = data["joint_limits"]
    vox = data["voxel"] if show_voxels else None

    geoms = []

    # Raw inlier points (.npy)
    if npy_path:
        pts = np.load(npy_path)
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(pts)
        pc.paint_uniform_color([0.1, 0.6, 1.0])
        geoms.append(pc)

    # Sparse PC (.ply)
    if ply_path:
        sparse = o3d.io.read_point_cloud(ply_path)
        geoms.append(sparse)

    # Bounding box
    aabb = o3d.geometry.AxisAlignedBoundingBox(bbox_min, bbox_max)
    aabb.color = (1, 0, 0)
    geoms.append(aabb)

    # Pivot sphere
    sph = o3d.geometry.TriangleMesh.create_sphere(radius=0.015)
    sph.paint_uniform_color([0, 1, 0])
    sph.translate(joint_pivot)
    geoms.append(sph)

    # Joint coordinate frame
    frame = create_coordinate_frame(joint_pivot, joint_axis, scale=0.08)
    geoms.append(frame)

    # Axis line
    axis_line = create_slider_line(joint_pivot, joint_axis, bbox_min, bbox_max)
    geoms.append(axis_line)

    # Hinge plane
    if "revolute" in joint_type.lower():
        hinge_plane = create_hinge_plane(joint_pivot, joint_axis, size=0.15)
        geoms.append(hinge_plane)

    # Voxel mask (optional)
    if vox is not None:
        v = vox.float() > 0.5
        vmesh = create_voxel_mesh(v.cpu(), bbox_min, bbox_max, color=[0.8, 0.2, 0.8])
        geoms.append(vmesh)

    print(f"Joint Type: {joint_type}, Limits: {joint_limits}")

    print("bbox min:", bbox_min)
    print("bbox max:", bbox_max)

    o3d.visualization.draw_geometries(geoms)


# ---------------------------------------------------------------------
# MAIN FUNCTION
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt", type=str, required=True,
                        help="Path to saved obj_xxx.pt file")
    parser.add_argument("--npy", type=str, default=None,
                        help="Optional: path to obj_xxx.npy file for raw inlier points")
    parser.add_argument("--ply", type=str, default=None,
                        help="Optional: sparse_pc.ply path")
    parser.add_argument("--voxels", action="store_true",
                        help="Enable voxel mask visualization")
    args = parser.parse_args()

    visualize_with_open3d(args.pt, args.npy, args.ply, args.voxels)


if __name__ == "__main__":
    main()
