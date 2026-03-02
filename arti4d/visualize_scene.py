import json
import numpy as np
import open3d as o3d
from pathlib import Path


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def load_point_cloud(ply_path):
    pcd = o3d.io.read_point_cloud(str(ply_path))
    print(f"[PLY] Loaded {len(pcd.points)} points")
    return pcd


def load_camera_frames(transforms_json, scale=0.1):
    with open(transforms_json) as f:
        data = json.load(f)

    frames = []
    for frame in data["frames"]:
        T = np.array(frame["transform_matrix"])
        cam = o3d.geometry.TriangleMesh.create_coordinate_frame(size=scale)
        cam.transform(T)
        frames.append(cam)

    print(f"[POSES] Loaded {len(frames)} camera poses")
    return frames


def create_joint_axis(position, axis, length=0.4):
    """
    Creates:
      - small sphere at pivot
      - arrow showing axis direction
    """

    position = np.array(position)
    axis = np.array(axis)
    axis = axis / np.linalg.norm(axis)

    # Pivot sphere
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.03)
    sphere.paint_uniform_color([1, 0, 0])
    sphere.translate(position)

    # Arrow
    arrow = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=0.01,
        cone_radius=0.02,
        cylinder_height=length * 0.8,
        cone_height=length * 0.2
    )

    # Align arrow to axis
    z = np.array([0, 0, 1])
    v = np.cross(z, axis)
    c = np.dot(z, axis)

    if np.linalg.norm(v) < 1e-8:
        R = np.eye(3)
    else:
        vx = np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])
        R = np.eye(3) + vx + vx @ vx * ((1 - c) / (np.linalg.norm(v) ** 2))

    arrow.rotate(R, center=np.zeros(3))
    arrow.translate(position)
    arrow.paint_uniform_color([0, 1, 0])

    return sphere, arrow


def load_joints(gt_json):
    with open(gt_json) as f:
        joints = json.load(f)

    geometries = []

    for j in joints:
        if j["position"] is None or j["axis"] is None:
            continue

        sphere, arrow = create_joint_axis(
            j["position"],
            j["axis"]
        )
        geometries.extend([sphere, arrow])

    print(f"[JOINTS] Loaded {len(geometries)//2} joints")
    return geometries


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main(output_dir):

    output_dir = Path(output_dir)

    ply_path = output_dir / "canonical" / "point_cloud.ply"
    tf_path = output_dir / "canonical" / "transforms.json"
    gt_path = output_dir / "GT_joint_info.json"

    geometries = []

    # PLY
    if ply_path.exists():
        geometries.append(load_point_cloud(ply_path))

    # Camera poses
    if tf_path.exists():
        geometries.extend(load_camera_frames(tf_path))

    # Joints
    if gt_path.exists():
        geometries.extend(load_joints(gt_path))

    # World frame
    geometries.append(
        o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
    )

    o3d.visualization.draw_geometries(geometries)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    main(args.output_dir)