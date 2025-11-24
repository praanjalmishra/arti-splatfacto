import torch
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class JointMetadata:
    joint_id: str
    joint_type: str
    pivot: List[float]
    axis: List[float]
    limits: List[float]
    n_gaussians: int


def _save_npz_chunk(data: Dict[str, np.ndarray], save_dir: Path, name: str) -> str:
    """Save Gaussian arrays to compressed .npz and return relative path."""
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / f"{name}.npz"
    np.savez_compressed(path, **data)
    return str(path.relative_to(save_dir.parent))


def export_to_artigs_json(
    checkpoint_path: Path,
    output_path: Path,
    config_path: Optional[Path] = None,
) -> Path:
    print(f"\n{'='*70}")
    print("EXPORTING TO ARTIGS JSON FORMAT (NPZ MODE)")
    print(f"{'='*70}")

    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if "pipeline" not in checkpoint or "model_full" not in checkpoint["pipeline"]:
        raise ValueError("Invalid checkpoint format – missing pipeline.model_full")

    model_state = checkpoint["pipeline"]["model_full"]
    joint_metadata_abs = model_state.get("joint_metadata_absolute", {})

    print(f"Checkpoint step: {checkpoint.get('step', 'unknown')}")

    # Detect joints
    joint_ids = sorted(
        {key.split(".")[1] for key in model_state if key.startswith("all_gauss_params_obj.")}
    )
    print(f"Found {len(joint_ids)} joints: {joint_ids}")

    export_data = {
        "meta": {
            "version": "1.1-npz",
            "model_type": "multi_joint_gaussian_splatting",
            "num_joints": len(joint_ids),
            "checkpoint_step": checkpoint.get("step", 0),
        },
        "joints": {},
        "background": {},
    }

    npz_dir = output_path.parent / "arrays"

    # Process each joint
    for joint_id in joint_ids:
        print(f"\nProcessing {joint_id}...")

        # Joint parameters
        joint_params = {}
        for name in ["pivot", "axis_raw", "min_angle", "max_angle"]:
            key = f"all_joint_params.{joint_id}.{name}"
            if key in model_state:
                joint_params[name] = model_state[key].cpu().numpy()

        joint_meta = joint_metadata_abs.get(joint_id, {})
        joint_type = joint_meta.get("type", "revolute")
        limits = (
            joint_meta["limits"].numpy().tolist()
            if "limits" in joint_meta
            else [float(joint_params.get("min_angle", 0.0)), float(joint_params.get("max_angle", 1.0))]
        )

        axis_raw = joint_params.get("axis_raw", np.array([0, 0, 1]))
        axis = axis_raw / (np.linalg.norm(axis_raw) + 1e-8)

        # Gaussian sets
        obj, can = {}, {}
        for name in ["means", "scales", "quats", "features_dc", "features_rest", "opacities"]:
            ok = f"all_gauss_params_obj.{joint_id}.{name}"
            ck = f"all_gauss_params_canon.{joint_id}.{name}"
            if ok in model_state:
                obj[name] = model_state[ok].cpu().numpy()
            if ck in model_state:
                can[name] = model_state[ck].cpu().numpy()

        n_gauss = obj.get("means", np.zeros((0, 3))).shape[0]
        print(f"  Type: {joint_type}")
        print(f"  Pivot: {joint_params.get('pivot', [0,0,0])}")
        print(f"  Axis: {axis}")
        print(f"  Limits: {limits}")
        print(f"  Gaussians: {n_gauss:,}")

        obj_npz = _save_npz_chunk(obj, npz_dir, f"{joint_id}_object")
        can_npz = _save_npz_chunk(can, npz_dir, f"{joint_id}_canonical")

        export_data["joints"][joint_id] = {
            "metadata": {
                "joint_id": joint_id,
                "joint_type": joint_type,
                "pivot": joint_params.get("pivot", np.zeros(3)).tolist(),
                "axis": axis.tolist(),
                "limits": limits,
                "n_gaussians": n_gauss,
            },
            "object_npz": obj_npz,
            "canonical_npz": can_npz,
        }

    # Background
    print(f"\nProcessing background...")
    bg = {
        k.split(".")[-1]: model_state[k].cpu().numpy()
        for k in model_state
        if k.startswith("gauss_params_fixed.")
    }
    n_bg = bg.get("means", np.zeros((0, 3))).shape[0]
    print(f"  Background Gaussians: {n_bg:,}")

    bg_npz = _save_npz_chunk(bg, npz_dir, "background")
    export_data["background"] = {"n_gaussians": n_bg, "npz_file": bg_npz}

    # Save lightweight JSON
    print(f"\nSaving metadata JSON: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(export_data, f, indent=2)

    total = sum(export_data["joints"][j]["metadata"]["n_gaussians"] for j in joint_ids) + n_bg
    print(f"\n{'='*70}")
    print("EXPORT COMPLETE (NPZ MODE)")
    print(f"{'='*70}")
    print(f"Total Gaussians: {total:,}")
    print(f"  object: {total - n_bg:,}")
    print(f"  Background: {n_bg:,}")
    print(f"JSON: {output_path}")
    print(f"Arrays: {npz_dir}")
    print(f"{'='*70}\n")
    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export ArtiSplatfacto to ARTiGS JSON (NPZ split)")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to checkpoint")
    parser.add_argument("--output", type=Path, required=True, help="Output JSON path")
    parser.add_argument("--config", type=Path, help="Optional config.yml path")
    args = parser.parse_args()

    export_to_artigs_json(args.checkpoint, args.output, args.config)
