import json
import numpy as np
from pathlib import Path
from typing import Dict, Optional
import open3d as o3d
from scipy.spatial.transform import Rotation


def normalize(v):
    return v / (np.linalg.norm(v) + 1e-8)


def get_rotation_axis_angle(k, theta):
    k = normalize(k)
    kx, ky, kz = k
    c, s = np.cos(theta), np.sin(theta)
    R = np.zeros((3, 3))
    R[0, 0] = c + kx * kx * (1 - c)
    R[0, 1] = kx * ky * (1 - c) - kz * s
    R[0, 2] = kx * kz * (1 - c) + ky * s
    R[1, 0] = kx * ky * (1 - c) + kz * s
    R[1, 1] = c + ky * ky * (1 - c)
    R[1, 2] = ky * kz * (1 - c) - kx * s
    R[2, 0] = kx * kz * (1 - c) - ky * s
    R[2, 1] = ky * kz * (1 - c) + kx * s
    R[2, 2] = c + kz * kz * (1 - c)
    return R


def save_axis_mesh(k, center, filepath):
    axis = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=0.01, cone_radius=0.02,
        cylinder_height=1.0, cone_height=0.08
    )
    arrow_dir = np.array([0., 0., 1.], dtype=np.float32)
    n = np.cross(arrow_dir, k)
    rad = np.arccos(np.clip(np.dot(arrow_dir, k), -1.0, 1.0))
    if np.linalg.norm(n) > 1e-6:
        R_arrow = get_rotation_axis_angle(n, rad)
        axis.rotate(R_arrow, center=(0, 0, 0))
    axis.translate(center[:3])
    o3d.io.write_triangle_mesh(str(filepath), axis)


def transform_gaussians_npz(npz_path: Path, rotation: np.ndarray,
                            translation: np.ndarray, scale: np.ndarray):
    """Load Gaussian arrays from .npz, apply transform, and overwrite file."""
    data = dict(np.load(npz_path))

    # means
    if "means" in data:
        means = data["means"]
        means = (rotation @ (means * scale).T).T + translation
        data["means"] = means

    # quaternions (assume [x,y,z,w] format)
    if "quats" in data:
        qR = Rotation.from_matrix(rotation).as_quat()
        rot_R = Rotation.from_quat(qR)
        q = data["quats"]
        new_q = np.zeros_like(q)
        for i in range(len(q)):
            qg = Rotation.from_quat(q[i])
            q_new = rot_R * qg
            new_q[i] = q_new.as_quat()
        data["quats"] = new_q

    # scales
    if "scales" in data:
        data["scales"] = data["scales"] * scale

    np.savez_compressed(npz_path, **data)


def compute_centroid_from_npz(model_json: Path, data: dict) -> np.ndarray:
    """Compute centroid using all .npz means."""
    all_means = []
    base = model_json.parent

    for j, jd in data["joints"].items():
        for key in ["object_npz", "canonical_npz"]:
            npz = base / jd[key]
            arr = np.load(npz)
            if "means" in arr:
                all_means.append(arr["means"])

    bg_npz = base / data["background"]["npz_file"]
    arr = np.load(bg_npz)
    if "means" in arr:
        all_means.append(arr["means"])

    if not all_means:
        return np.zeros(3)
    all_means = np.concatenate(all_means, axis=0)
    return np.mean(all_means, axis=0)


def canonicalize_gaussians(
    input_json: Path,
    output_json: Path,
    rotation_matrix: Optional[np.ndarray] = None,
    center_model: bool = True,
    scale: Optional[np.ndarray] = None,
    additional_translation: Optional[np.ndarray] = None,
) -> Path:
    print(f"\n{'='*70}")
    print("CANONICALIZING GAUSSIAN MODEL (.NPZ MODE)")
    print(f"{'='*70}")

    print(f"Loading JSON metadata: {input_json}")
    with open(input_json, "r") as f:
        data = json.load(f)

    if rotation_matrix is None:
        rotation_matrix = np.eye(3)
    if scale is None:
        scale = np.ones(3)

    centroid = compute_centroid_from_npz(input_json, data)
    print(f"Original centroid: {centroid}")
    translation = -centroid if center_model else np.zeros(3)
    if additional_translation is not None:
        translation += additional_translation

    print(f"Applying transform:\nRotation:\n{rotation_matrix}\n"
          f"Translation: {translation}\nScale: {scale}")

    base = input_json.parent
    for jid, jd in data["joints"].items():
        print(f"\nTransforming {jid}...")
        for key in ["object_npz", "canonical_npz"]:
            npz_path = base / jd[key]
            transform_gaussians_npz(npz_path, rotation_matrix, translation, scale)

        # metadata updates
        meta = jd["metadata"]
        pivot = np.array(meta["pivot"]) * scale
        pivot = (rotation_matrix @ pivot) + translation
        meta["pivot"] = pivot.tolist()

        axis = np.array(meta["axis"])
        axis = normalize(rotation_matrix @ axis)
        meta["axis"] = axis.tolist()

        print(f"  New pivot: {meta['pivot']}")
        print(f"  New axis: {meta['axis']}")

    # background
    print("\nTransforming background...")
    bg_npz = base / data["background"]["npz_file"]
    transform_gaussians_npz(bg_npz, rotation_matrix, translation, scale)

    data["meta"]["canonicalized"] = True
    data["meta"]["transformation"] = {
        "rotation": rotation_matrix.tolist(),
        "translation": translation.tolist(),
        "scale": scale.tolist(),
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nSaved canonicalized JSON to: {output_json}")
    print(f"{'='*70}\n")
    return output_json


def export_joint_axes_visualization(json_path: Path, output_dir: Path):
    print("\nExporting joint axis visualizations...")
    with open(json_path, "r") as f:
        data = json.load(f)
    output_dir.mkdir(parents=True, exist_ok=True)
    for jid, jd in data["joints"].items():
        meta = jd["metadata"]
        c = np.array(meta["pivot"], dtype=np.float32)
        a = np.array(meta["axis"], dtype=np.float32)
        save_axis_mesh(a, c, output_dir / f"axis_{jid}.ply")
        save_axis_mesh(-a, c, output_dir / f"axis_{jid}_opposite.ply")
        print(f"  Saved axis visualization for {jid}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Canonicalize Gaussian Splatting model (.npz version)")
    parser.add_argument("--input", type=Path, required=True, help="Input ARTiGS JSON (npz mode)")
    parser.add_argument("--output", type=Path, required=True, help="Output JSON path")
    parser.add_argument("--no-center", action="store_true", help="Do not center model at origin")
    parser.add_argument("--scale", type=float, nargs=3, help="Scale factors [x y z]")
    parser.add_argument("--export-axes", action="store_true", help="Export axis .ply meshes")
    parser.add_argument("--axes-dir", type=Path, default="axes", help="Axis output directory")
    args = parser.parse_args()

    scale = np.array(args.scale) if args.scale else None
    out = canonicalize_gaussians(
        args.input,
        args.output,
        center_model=not args.no_center,
        scale=scale,
    )
    if args.export_axes:
        export_joint_axes_visualization(out, args.axes_dir)
