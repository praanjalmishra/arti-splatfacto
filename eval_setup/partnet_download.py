import sapien.core as sapien
from sapien.asset import download_partnet_mobility
import numpy as np  
import os
import json
import shutil
import yaml
token = os.environ.get("SAPIEN_TOKEN")

def download_part(part_id: int, part_name: str):
    urdf_file = download_partnet_mobility(part_id, token)
    downloaded_dir, filename = os.path.split(urdf_file)
    results_file = os.path.join(downloaded_dir, "result.json")

    if part_name == "":
        part_name = get_object_name(results_file)

    dest_dir = os.path.join("assets", part_name)

    os.makedirs(os.path.dirname(dest_dir), exist_ok=True)

    if os.path.exists(dest_dir):
        shutil.rmtree(dest_dir)

    shutil.move(downloaded_dir, dest_dir)

    print(f"Saved to {dest_dir}")
    return dest_dir


def get_object_name(results_file_path: str):
    with open(results_file_path) as f:
        data = json.load(f)

    obj_class = data[-1]["name"]
    obj_id = os.path.split(os.path.dirname(results_file_path))[1]

    return os.path.join(obj_class, obj_id)


def download_from_yaml(yaml_path: str):
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)

    objects = config["datasets"]["partnet_objects"]

    successes = []
    failures = []

    for item in objects:
        category = item["category"]
        object_id = int(item["object_id"])
        joint_id = item["joint_id"]

        part_name = os.path.join(category, str(object_id))

        print(f"\nDownloading {category} {object_id} ({joint_id})")

        try:
            dest_dir = download_part(object_id, part_name)

            # Ensure output directory exists
            if not os.path.isdir(dest_dir):
                raise RuntimeError("Output directory missing")

            successes.append(object_id)

        except Exception as e:
            print(f"FAILED: {category} {object_id} ({joint_id}) — {e}")
            failures.append(object_id)

    print("\n========== SUMMARY ==========")
    print(f"Total objects : {len(objects)}")
    print(f"Successful    : {len(successes)}")
    print(f"Failed        : {len(failures)}")

    if failures:
        print("Failed object IDs:", failures)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml", type=str, required=True, help="Path to YAML file")
    args = parser.parse_args()

    download_from_yaml(args.yaml)
