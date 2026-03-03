#!/usr/bin/env python3
import json
import numpy as np
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--root", required=True)
parser.add_argument("--out_txt", required=True)
parser.add_argument("--out_csv", required=True)
args = parser.parse_args()

root = Path(args.root)

all_files = list(root.glob("*/*/joint_eval_summary.json"))

if not all_files:
    print("No evaluation files found.")
    exit()

axis_means = []
pivot_means = []
type_accs = []
ns = []

for f in all_files:
    with open(f) as fp:
        s = json.load(fp)

    axis_means.append(s["axis_err_mean"])
    type_accs.append(s["type_accuracy"])
    ns.append(s["n"])

    if s["pivot_mean"] is not None:
        pivot_means.append(s["pivot_mean"])

global_summary = {
    "num_scenes": len(all_files),
    "total_joints": int(sum(ns)),
    "axis_err_mean": float(np.mean(axis_means)),
    "pivot_mean": float(np.mean(pivot_means)) if pivot_means else None,
    "type_accuracy": float(np.mean(type_accs)),
}

# Write TXT (paper-style)
with open(args.out_txt, "w") as f:
    f.write("====================================================\n")
    f.write("Global Joint Estimation Evaluation\n")
    f.write("====================================================\n\n")
    f.write(f"Scenes evaluated : {global_summary['num_scenes']}\n")
    f.write(f"Total joints     : {global_summary['total_joints']}\n\n")
    f.write(f"Axis error mean  : {global_summary['axis_err_mean']:.2f} deg\n")
    if global_summary["pivot_mean"] is not None:
        f.write(f"Pivot dist mean  : {global_summary['pivot_mean']:.4f} m\n")
    f.write(f"Type accuracy    : {global_summary['type_accuracy']:.2f}%\n")

# Write CSV
with open(args.out_csv, "w") as f:
    f.write("num_scenes,total_joints,axis_err_mean,pivot_mean,type_accuracy\n")
    f.write(f"{global_summary['num_scenes']},"
            f"{global_summary['total_joints']},"
            f"{global_summary['axis_err_mean']},"
            f"{global_summary['pivot_mean']},"
            f"{global_summary['type_accuracy']}\n")

print("Global evaluation complete.")