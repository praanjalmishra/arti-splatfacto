import json
import numpy as np
import open3d as o3d

# ---- Load transforms ----
with open("/workspace/data/arti4d/nerf_output/din080_scene1/transforms.json") as f:
    data = json.load(f)

poses = np.array([np.array(f["transform_matrix"]) for f in data["frames"]])
cams = poses[:, :3, 3]

print("Camera min:", cams.min(axis=0))
print("Camera max:", cams.max(axis=0))
print("Camera mean:", cams.mean(axis=0))

# ---- Load PLY ----
pcd = o3d.io.read_point_cloud(
    "/workspace/data/arti4d/nerf_output/din080_scene1/point_cloud.ply"
)
pts = np.asarray(pcd.points)

print("PLY min:", pts.min(axis=0))
print("PLY max:", pts.max(axis=0))
print("PLY mean:", pts.mean(axis=0))

# ---- Distance between camera center and scene center ----
scene_center = pts.mean(axis=0)
cam_center = cams.mean(axis=0)

print("Distance cam_center → scene_center:",
      np.linalg.norm(cam_center - scene_center))

# ---- Forward direction test (first frame) ----
R0 = poses[0][:3, :3]
forward = R0[:, 2]  # optical forward axis
to_scene = scene_center - cams[0]
to_scene /= np.linalg.norm(to_scene)

dot = np.dot(forward, to_scene)
print("Dot(forward, to_scene):", dot)