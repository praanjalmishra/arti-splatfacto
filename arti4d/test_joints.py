import json
import torch
import argparse
import numpy as np
import open3d as o3d
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--joint_dir", required=True)
parser.add_argument("--canonical_ply", required=True)
args = parser.parse_args()

joint_dir    = Path(args.joint_dir)
schemas_path = joint_dir / "joint_schemas.json"
mask_path    = next(joint_dir.glob("obj_masks/*.pt"), None)
vox_path     = next(joint_dir.glob("obj_masks/*.npy"), None)
ply_path     = Path(args.canonical_ply)

with open(schemas_path) as f:
    schema = json.load(f)
if isinstance(schema, list):
    schema = schema[0]

axis  = np.array(schema["joint_axis"])
pivot = np.array(schema["joint_pivot"])
jtype = schema["joint_type"]

obj_data = None
if mask_path is not None:
    obj_data = torch.load(mask_path, map_location="cpu")
    if "joint_axis"  in obj_data: axis  = obj_data["joint_axis"].numpy()
    if "joint_pivot" in obj_data: pivot = obj_data["joint_pivot"].numpy()
    if "joint_angle" in obj_data:
        print(f"Joint angle: {np.degrees(float(obj_data['joint_angle'])):.2f}°")

pcd = o3d.io.read_point_cloud(str(ply_path))
pts = np.asarray(pcd.points)
colors = np.ones((len(pts), 3)) * 0.6
pcd.colors = o3d.utility.Vector3dVector(colors)

geoms = [pcd]

vox_pcd = None
if obj_data is not None and "voxel" in obj_data:
    vox      = obj_data["voxel"].numpy()
    raw_min  = obj_data["bbox_min"].numpy()
    raw_max  = obj_data["bbox_max"].numpy()
    bbox_min = np.minimum(raw_min, raw_max)
    bbox_max = np.maximum(raw_min, raw_max)
    idx      = np.argwhere(vox)
    step     = (bbox_max - bbox_min) / np.array(vox.shape)
    vox_pts  = bbox_min + (idx + 0.5) * step
    vox_pcd  = o3d.geometry.PointCloud()
    vox_pcd.points = o3d.utility.Vector3dVector(vox_pts)
    vox_pcd.paint_uniform_color([0.0, 0.6, 1.0])
    print(f"Voxel occupied: {len(vox_pts)} / {vox.size}")
    print(f"bbox: {np.round(bbox_min,3)} → {np.round(bbox_max,3)}")

    bbox = o3d.geometry.AxisAlignedBoundingBox(bbox_min, bbox_max)
    bbox.color = [1.0, 0.5, 0.0]
    geoms.append(bbox)
    geoms.append(vox_pcd)

elif vox_path is not None:
    raw = np.load(vox_path)
    vox_pcd = o3d.geometry.PointCloud()
    vox_pcd.points = o3d.utility.Vector3dVector(raw[:, :3])
    vox_pcd.paint_uniform_color([0.0, 0.6, 1.0])
    geoms.append(vox_pcd)

arrow_len = 0.4
arrow = o3d.geometry.TriangleMesh.create_arrow(
    cylinder_radius=0.008, cone_radius=0.016,
    cylinder_height=arrow_len * 0.8, cone_height=arrow_len * 0.2,
)
arrow.paint_uniform_color([0.0, 1.0, 0.2])

z = np.array([0, 0, 1], dtype=float)
v = np.cross(z, axis)
s = np.linalg.norm(v)
c = np.dot(z, axis)
if s < 1e-6:
    R = np.eye(3) if c > 0 else np.diag([1, -1, -1])
else:
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    R  = np.eye(3) + vx + vx @ vx * ((1 - c) / (s ** 2))

T = np.eye(4)
T[:3, :3] = R
T[:3, 3]  = pivot - R @ np.array([0, 0, arrow_len / 2])
arrow.transform(T)

pivot_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.025)
pivot_sphere.translate(pivot)
pivot_sphere.paint_uniform_color([1.0, 1.0, 0.0])
frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15, origin=pivot)

geoms += [arrow, pivot_sphere, frame]

print(f"Joint type:  {jtype}")
print(f"Joint axis:  {np.round(axis, 4)}")
print(f"Joint pivot: {np.round(pivot, 4)}")

o3d.visualization.draw_geometries(
    geoms,
    window_name=f"Joint: {jtype}  |  axis={np.round(axis,3)}",
    width=1280, height=720,
)