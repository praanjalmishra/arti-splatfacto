#!/usr/bin/env python3
"""
Joint estimation evaluation: predicted vs GT.

Metrics:
  - axis_angle_error (deg)     : angle between pred and GT axis, sign-invariant
  - pivot_to_axis_dist (m)     : distance from pred pivot to GT axis LINE (revolute only)
  - joint_type_correct (bool)  : predicted type matches GT type

Usage:
    python eval_joints.py \
        --gt_json   /path/to/nerf_output/din080_scene1/GT_joint_info.json \
        --pred_root /path/to/nerf_output/din080_scene1
"""

import json
import argparse
import numpy as np
from pathlib import Path


def axis_angle_error(pred_axis, gt_axis):
    p = np.array(pred_axis, dtype=float)
    g = np.array(gt_axis,   dtype=float)
    p /= np.linalg.norm(p)
    g /= np.linalg.norm(g)
    cos = np.clip(np.abs(np.dot(p, g)), 0.0, 1.0)  # abs for sign invariance
    return np.degrees(np.arccos(cos))


def pivot_to_axis_distance(pred_pivot, gt_pivot, gt_axis):
    # Distance from pred_pivot to the infinite line defined by (gt_pivot, gt_axis)
    p = np.array(pred_pivot, dtype=float)
    o = np.array(gt_pivot,   dtype=float)
    a = np.array(gt_axis,    dtype=float)
    a /= np.linalg.norm(a)
    diff = p - o
    return np.linalg.norm(diff - np.dot(diff, a) * a)


def load_pred_schema(joint_dir: Path):
    path = joint_dir / "ransac_joints" / "joint_schemas.json"
    if not path.exists():
        return None
    with open(path) as f:
        s = json.load(f)
    return s[0] if isinstance(s, list) else s


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_json",   required=True)
    parser.add_argument("--pred_root", required=True)
    parser.add_argument("--json_out", default=None)
    args = parser.parse_args()

    with open(args.gt_json) as f:
        gt_list = json.load(f)

    pred_root = Path(args.pred_root)

    results = []

    for gt in gt_list:
        joint_dir  = pred_root / gt["joint_dir"]
        axis_name  = gt["axis_name"]
        gt_type    = (gt["joint_type"] or "").lower()
        difficulty = gt["difficulty"] or "?"
        gt_pos     = gt["position"]
        gt_axis    = gt["axis"]

        pred = load_pred_schema(joint_dir)

        if pred is None:
            print(f"  [SKIP] {axis_name}  — no prediction found")
            continue
        if gt_pos is None or gt_axis is None:
            print(f"  [SKIP] {axis_name}  — no GT position/axis")
            continue

        pred_type  = pred.get("joint_type", "").lower()
        pred_axis  = pred.get("joint_axis",  [0, 0, 1])
        pred_pivot = pred.get("joint_pivot", [0, 0, 0])

        type_correct   = (pred_type == gt_type) or \
                         (pred_type == "slider" and gt_type == "prismatic")
        axis_err       = axis_angle_error(pred_axis, gt_axis)
        pivot_dist     = pivot_to_axis_distance(pred_pivot, gt_pos, gt_axis) \
                         if gt_type == "revolute" else None

        results.append({
            "axis_name":    axis_name,
            "difficulty":   difficulty,
            "gt_type":      gt_type,
            "pred_type":    pred_type,
            "type_correct": type_correct,
            "axis_err_deg": axis_err,
            "pivot_dist_m": pivot_dist,
        })

    if not results:
        print("No results to evaluate.")
        return

    # ── Per-joint table ───────────────────────────────────────────────────────
    print(f"\n{'Joint':<30} {'GT':<12} {'Pred':<12} {'Type':>5} {'AxisErr(°)':>10} {'PivotDist(m)':>13} {'Diff'}")
    print("─" * 90)
    for r in results:
        pivot_str = f"{r['pivot_dist_m']:.4f}" if r["pivot_dist_m"] is not None else "  n/a  "
        type_str  = "✓" if r["type_correct"] else "✗"
        print(f"  {r['axis_name']:<28} {r['gt_type']:<12} {r['pred_type']:<12} "
              f"{type_str:>5} {r['axis_err_deg']:>10.2f} {pivot_str:>13}  {r['difficulty']}")

    # ── Aggregate ─────────────────────────────────────────────────────────────
    axis_errs  = [r["axis_err_deg"] for r in results]
    pivot_dists = [r["pivot_dist_m"] for r in results if r["pivot_dist_m"] is not None]
    type_acc   = np.mean([r["type_correct"] for r in results]) * 100

    print(f"\n{'─'*50}")
    print(f"  N evaluated     : {len(results)}")
    print(f"  Type accuracy   : {type_acc:.1f}%")
    print(f"  Axis error      : mean={np.mean(axis_errs):.2f}°  "
          f"med={np.median(axis_errs):.2f}°  "
          f"max={np.max(axis_errs):.2f}°")
    if pivot_dists:
        print(f"  Pivot dist (rev): mean={np.mean(pivot_dists):.4f}m  "
              f"med={np.median(pivot_dists):.4f}m  "
              f"max={np.max(pivot_dists):.4f}m")

    # ── By difficulty ─────────────────────────────────────────────────────────
    for diff in ["EASY", "HARD"]:
        sub = [r for r in results if r["difficulty"] == diff]
        if not sub:
            continue
        errs = [r["axis_err_deg"] for r in sub]
        print(f"\n  [{diff}]  n={len(sub)}  "
              f"axis mean={np.mean(errs):.2f}°  med={np.median(errs):.2f}°")

    # ── By joint type ─────────────────────────────────────────────────────────
    for jt in ["revolute", "prismatic"]:
        sub = [r for r in results if r["gt_type"] == jt]
        if not sub:
            continue
        errs = [r["axis_err_deg"] for r in sub]
        print(f"  [{jt}]  n={len(sub)}  "
              f"axis mean={np.mean(errs):.2f}°  med={np.median(errs):.2f}°")

    print()


    summary = {
        "n": len(results),
        "type_accuracy": float(type_acc),
        "axis_err_mean": float(np.mean(axis_errs)),
        "axis_err_median": float(np.median(axis_errs)),
        "axis_err_max": float(np.max(axis_errs)),
        "pivot_mean": float(np.mean(pivot_dists)) if pivot_dists else None,
        "pivot_median": float(np.median(pivot_dists)) if pivot_dists else None,
        "pivot_max": float(np.max(pivot_dists)) if pivot_dists else None,
    }

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()