#!/usr/bin/env python3
"""
Batch evaluation renderer for PartNet-Mobility objects.

Reads eval_config YAML and global GT JSON metadata,
then automatically calls render_viewpoints.py for each listed object.
"""

import os
import json
import yaml
import subprocess
from pathlib import Path
from tqdm import tqdm

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Batch evaluation rendering launcher")
    parser.add_argument("--eval_config", type=str, default="eval_setup/configs/eval_config.yaml",
                        help="Path to evaluation config YAML")
    parser.add_argument("--gt_joints_file", type=str, default="eval_setup/new_partnet_mobility_dataset_correct_intr_meta.json",
                        help="Path to ground-truth joint metadata JSON")
    parser.add_argument("--partnet_root", type=str, default="assets",
                        help="Path to root of PartNet-Mobility dataset")
    parser.add_argument("--project_root", type=str, default=".",
                        help="Path to your project root for output placement")
    parser.add_argument("--render_script", type=str, default="eval_setup/render_viewpoints.py",
                        help="Path to render_viewpoints.py")
    parser.add_argument("--n_frames", type=int, default=None,
                        help="Override articulation frame count (optional)")
    args = parser.parse_args()

    eval_cfg = yaml.safe_load(open(args.eval_config))

    render_cfg   = eval_cfg.get("rendering", {})
    n_artic      = render_cfg.get("n_frames_articulation")
    gt_meta = json.load(open(args.gt_joints_file))

    output_root = Path(args.project_root) / eval_cfg["evaluation"]["output_renders"]
    output_root.mkdir(parents=True, exist_ok=True)

    partnet_objects = eval_cfg["datasets"]["partnet_objects"]

    for entry in tqdm(partnet_objects, desc="Rendering objects"):
        category = entry["category"]
        object_id = str(entry["object_id"])
        joint_name = entry.get("joint_id", "joint_0")

        partnet_dir = Path(args.partnet_root) / category / object_id

        if not partnet_dir.exists():
            print(f"⚠️  Skipping {object_id}: path not found {partnet_dir}")
            continue

        if partnet_dir is None or not (partnet_dir / "mobility.urdf").exists():
            print(f"⚠️  Skipping {object_id}: missing URDF in {args.partnet_root}")
            continue
        urdf_path = partnet_dir / "mobility.urdf"
        if not urdf_path.exists():
            print(f"⚠️  Skipping {object_id}: missing URDF at {urdf_path}")
            continue

        # Output directory
        out_dir = output_root / f"{category}_{object_id}"
        out_dir.mkdir(parents=True, exist_ok=True)

        joint_id = 0
        for token in joint_name.split("_"):
            if token.isdigit():
                joint_id = int(token)
                break

        cmd = [
            "python", args.render_script,
            "--partnet_dir", str(partnet_dir),
            "--output_dir", str(out_dir),
            "--eval_mode",
            "--eval_config", args.eval_config,
            "--gt_joints_file", args.gt_joints_file,
            "--joint_id", str(joint_id),
            "--n_frames", str(n_artic)
        ]


        print(f"\n🚀 Launching render for {category}/{object_id} (joint {joint_id})")
        subprocess.run(cmd, check=False)

    print("\n✅ Batch evaluation rendering complete.")

if __name__ == "__main__":
    main()


### usage example:
# python batch_eval_render.py \
#   --eval_config configs/eval_config.yaml \
#   --gt_joints_file data/new_partnet_mobility_dataset_correct_intr_meta.json \
#   --partnet_root data_itaco/video2articulation/partnet-mobility-v0 \
#   --project_root . \
#   --render_script render_viewpoints.py
