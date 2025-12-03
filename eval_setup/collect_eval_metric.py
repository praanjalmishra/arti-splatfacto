#!/usr/bin/env python3
import json
import numpy as np
from pathlib import Path
import argparse
import time

def compute_joint_error(estimated_joint, gt_joint):
    """Compute Video2Articulation-style joint estimation errors."""
    est_axis = np.array(estimated_joint.get("joint_axis", [0, 0, 1]), dtype=float)
    est_pivot = np.array(estimated_joint.get("joint_pivot", [0, 0, 0]), dtype=float)
    gt_axis = np.array(gt_joint["joint"]["axis"]["direction"], dtype=float)
    gt_pivot = np.array(gt_joint["joint"]["axis"]["origin"], dtype=float)

    gt_axis = np.array([gt_axis[0], gt_axis[2], gt_axis[1]])
    gt_pivot = np.array([gt_pivot[0], gt_pivot[2], gt_pivot[1]])

    # Normalize
    est_axis /= np.linalg.norm(est_axis)
    gt_axis /= np.linalg.norm(gt_axis)

    # Axis (orientation) error
    axis_err_rad = np.arccos(np.clip(np.abs(np.dot(est_axis, gt_axis)), -1, 1))
    axis_err_deg = np.degrees(axis_err_rad)

    # Position (line-to-line) error
    n = np.cross(est_axis, gt_axis)
    if np.linalg.norm(n) < 1e-6:
        # Parallel axes → use perpendicular distance from one pivot to the other axis
        pos_err = np.linalg.norm(np.cross((est_pivot - gt_pivot), gt_axis))
    else:
        pos_err = np.abs(np.dot(n, (est_pivot - gt_pivot))) / np.linalg.norm(n)


    # Joint type classification correctness
    gt_type = gt_joint.get("type", "").lower()
    est_type = estimated_joint.get("joint_type", "").lower()

    # Normalize synonyms (hinge → revolute, slider → prismatic)
    type_map = {"hinge": "revolute", "revolute": "revolute",
                "slider": "prismatic", "prismatic": "prismatic"}

    gt_type_norm = type_map.get(gt_type, gt_type)
    est_type_norm = type_map.get(est_type, est_type)

    type_correct = (gt_type_norm == est_type_norm)


    return {
        'axis_error_degrees': float(axis_err_deg),
        'pivot_error_distance': float(pos_err),
        'joint_type': gt_type_norm,
        'predicted_joint_type': est_type_norm,
        'type_correct': bool(type_correct)
    }

def collect_metrics(eval_results_file, gt_joints_file):
    """Collect per-object joint estimation metrics."""

    with open(eval_results_file) as f:
        eval_results = json.load(f)

    with open(gt_joints_file) as f:
        gt_data = json.load(f)

    metrics = []

    for result in eval_results:
        if result.get("status") != "success":
            continue

        object_id = result["object_id"]
        category = result["category"]
        output_dir = Path(result["output_dir"])

        # --- Locate GT joint ---
        gt_joint = None
        if category in gt_data and object_id in gt_data[category]:
            gt_entry = gt_data[category][object_id]
            if gt_entry.get("interaction_list"):
                gt_joint = gt_entry["interaction_list"][0]
        if gt_joint is None:
            print(f"⚠️  No GT joint found for {category}/{object_id}")
            continue

        # --- Load estimated joint schema ---
        joint_file = output_dir / "post" / "ransac_joints" / "joint_schemas.json"
        if not joint_file.exists():
            print(f"⚠️  Missing joint_schemas.json for {object_id}")
            continue

        with open(joint_file) as f:
            estimated_joints = json.load(f)

        if not isinstance(estimated_joints, list) or len(estimated_joints) == 0:
            print(f"⚠️  Empty joint schema list for {object_id}")
            continue

        # For now assume single joint per object
        est_joint = estimated_joints[0]

        # --- Compute metrics ---
        joint_err = compute_joint_error(est_joint, gt_joint)

        metrics.append(
            {
                "object_id": object_id,
                "category": category,
                "axis_error_deg": joint_err["axis_error_degrees"],
                "pivot_error": joint_err["pivot_error_distance"],
                "runtime_sec": result.get("runtime_seconds", 0),
                "joint_type": joint_err["joint_type"],
                "predicted_joint_type": joint_err["predicted_joint_type"],
                "type_correct": joint_err["type_correct"],

            }
        )

    return metrics


def compute_summary_stats(metrics):
    """Aggregate summary statistics."""

    if not metrics:
        return {}

    axis_errors = np.array([m["axis_error_deg"] for m in metrics])
    pivot_errors = np.array([m["pivot_error"] for m in metrics])
    runtimes = np.array([m.get("runtime_sec", 0) for m in metrics])
    type_correctness = np.mean([m['type_correct'] for m in metrics]) * 100


    summary = {
        "total_objects": len(metrics),
        "overall": {
            "axis_error_deg": {
                "mean": float(np.mean(axis_errors)),
                "median": float(np.median(axis_errors)),
                "std": float(np.std(axis_errors)),
            },
            "pivot_error": {
                "mean": float(np.mean(pivot_errors)),
                "median": float(np.median(pivot_errors)),
                "std": float(np.std(pivot_errors)),
            },
            "runtime_sec": {
                "mean": float(np.mean(runtimes)),
                "total": float(np.sum(runtimes)),
            },
            "type_correctness": {
                "mean": float(type_correctness),
            },
        },
    }

    # Optional: group by type
    for jt in ["revolute", "prismatic", "unknown"]:
        subset = [m for m in metrics if m["joint_type"] == jt]
        if subset:
            summary[jt] = {
                "count": len(subset),
                "axis_error_deg_mean": float(np.mean([m["axis_error_deg"] for m in subset])),
                "pivot_error_mean": float(np.mean([m["pivot_error"] for m in subset])),
                "type_correctness_mean": float(np.mean([m["type_correct"] for m in subset])) * 100,
            }

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute evaluation metrics for joint estimation results.")
    parser.add_argument("--eval_results", required=True, help="Path to eval_results_*.json file")
    parser.add_argument("--gt_joints", required=True, help="Path to GT joint metadata JSON")
    parser.add_argument("--output", default="metrics_summary.json", help="Output metrics summary JSON")
    args = parser.parse_args()

    metrics = collect_metrics(args.eval_results, args.gt_joints)
    summary = compute_summary_stats(metrics)

    output_data = {
        "summary": summary,
        "detailed_metrics": metrics,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\n✅ Metrics saved to: {args.output}")
    print(f"Processed {len(metrics)} successful objects.")
    if summary:
        print(f"Mean axis error: {summary['overall']['axis_error_deg']['mean']:.2f}°")
        print(f"Mean pivot error: {summary['overall']['pivot_error']['mean']:.4f}")
