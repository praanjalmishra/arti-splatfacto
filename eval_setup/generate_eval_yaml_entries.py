#!/usr/bin/env python3
import json
import argparse

def generate_yaml_entries(gt_json_path, output_yaml_path=None):
    with open(gt_json_path, 'r') as f:
        gt_data = json.load(f)

    entries = []
    for category, objs in gt_data.items():
        if not isinstance(objs, dict):
            continue
        for object_id, obj_info in objs.items():
            if not isinstance(obj_info, dict):
                continue
            if 'interaction_list' not in obj_info:
                continue
            for interaction in obj_info['interaction_list']:
                joint_id = interaction.get('id', 0)
                entries.append({
                    'category': category,
                    'object_id': str(object_id),
                    'joint_id': f"joint_{joint_id}"
                })

    print("datasets:")
    print("  partnet_objects:")
    for e in entries:
        print(f'    - {{category: "{e["category"]}", object_id: "{e["object_id"]}", joint_id: "{e["joint_id"]}"}}')

    if output_yaml_path:
        with open(output_yaml_path, 'w') as f:
            f.write("datasets:\n")
            f.write("  partnet_objects:\n")
            for e in entries:
                f.write(f'    - {{category: "{e["category"]}", object_id: "{e["object_id"]}", joint_id: "{e["joint_id"]}"}}\n')
        print(f"\n✅ YAML entries saved to: {output_yaml_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate eval YAML entries from PartNet-Mobility GT JSON.")
    parser.add_argument("--gt_json", required=True, help="Path to new_partnet_mobility_dataset_correct_intr_meta.json")
    parser.add_argument("--output_yaml", default=None, help="Optional path to save generated YAML entries")
    args = parser.parse_args()

    generate_yaml_entries(args.gt_json, args.output_yaml)
