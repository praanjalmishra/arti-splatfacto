#!/usr/bin/env python3
"""
Batch convert Video2Articulation view data to post-change NeRF format.
Includes GT geometry extraction using PartNet-Mobility structure.
"""

import os
import json
import yaml
import subprocess
import shutil
from pathlib import Path
from tqdm import tqdm
import numpy as np
from scipy.spatial.transform import Rotation
import pyvista as pv
from typing import List
import traceback

def quaternion_to_matrix(quat):
    """Convert quaternion (w,x,y,z) to rotation matrix"""
    r = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])  # scipy uses (x,y,z,w)
    return r.as_matrix()

def convert_pose_to_matrix(tx7):
    """Convert V2A pose to SAPIEN convention."""
    import numpy as np
    from scipy.spatial.transform import Rotation
    
    t = np.array(tx7[:3])
    qw, qx, qy, qz = tx7[3:]
    
    r = Rotation.from_quat([qx, qy, qz, qw])
    R = r.as_matrix()
    
    T_v2a = np.eye(4)
    T_v2a[:3, :3] = R
    T_v2a[:3, 3] = t
    
    # V2A likely uses standard convention: X=right, Y=up/down, Z=forward/back
    # Try extracting the axes directly
    v2a_x = T_v2a[:3, 0]  # right
    v2a_y = T_v2a[:3, 1]  # up or down
    v2a_z = T_v2a[:3, 2]  # forward or back
    
    # SAPIEN convention: [forward, left, up]
    # If cameras are 180° off, try flipping the forward direction
    sapien_forward = v2a_z          # Try positive Z as forward (NOT -Z)
    sapien_left = -v2a_x            # -right = left
    sapien_up = v2a_y               # Keep Y as up
    
    T_sapien = np.eye(4)
    T_sapien[:3, :3] = np.stack([sapien_forward, sapien_left, sapien_up], axis=1)
    T_sapien[:3, 3] = t
    
    return T_sapien
def parse_partid_to_objs(shape_path: Path):
    """Parse result.json to map part IDs to OBJ files"""
    result_file_path = shape_path / 'result.json'
    result_file = json.loads(result_file_path.read_text())
    partid_to_objs = {}
    
    def parse_part(part):
        pid = part['id']
        partid_to_objs[pid] = set(part.get('objs', set()))
        for child in part.get('children', []):
            parse_part(child)
            childs_objs = partid_to_objs[child['id']]
            partid_to_objs[pid] |= childs_objs

    assert len(result_file) == 1
    parse_part(result_file[0])
    return partid_to_objs

def merge_meshs(meshs_paths: List[Path]):
    """Merge multiple OBJ files into single mesh"""
    meshs_paths = list(set(meshs_paths))
    meshs = []
    
    for mesh_path in meshs_paths:
        try:
            mesh = pv.read(str(mesh_path))
            meshs.append(mesh)
        except FileNotFoundError:
            print(f"    [Warning] {mesh_path} not found.")
        except Exception as e:
            print(f"    [Warning] Failed to load {mesh_path}: {e}")
    
    if not meshs:
        return None
        
    merged_mesh = meshs[0]
    for mesh in meshs[1:]:
        merged_mesh += mesh
    
    return merged_mesh
def extract_geometry_for_joint(partnet_dir, joint_id, output_dir):
    """
    Extract static, movable, and whole geometry for evaluation.
    Saves gt_movable.npy, gt_static.npy, and gt_whole.npy in output_dir/gt_mesh/
    """
    import json
    import numpy as np
    from pathlib import Path

    partnet_dir = Path(partnet_dir)
    output_dir = Path(output_dir)

    print(f"    🔍 Extracting geometry for joint {joint_id}")

    try:
        # Load mobility data
        mobility_file = partnet_dir / "mobility_v2.json"
        if not mobility_file.exists():
            mobility_file = partnet_dir / "mobility.json"  # fallback

        mobility_data = json.loads(mobility_file.read_text())

        # Parse part-to-obj mapping
        partid_to_objs = parse_partid_to_objs(partnet_dir)

        # Build part hierarchy lookup
        part_dict = {p["id"]: p for p in mobility_data}

        # Identify movable vs static parts for this joint
        movable_parts, static_parts = [], []

        for part in mobility_data:
            part_id = part["id"]
            current_part = part
            is_movable = False

            # Traverse up parent hierarchy
            while current_part.get("parent", -1) != -1:
                if current_part["parent"] == joint_id:
                    is_movable = True
                    break
                if current_part["parent"] in part_dict:
                    current_part = part_dict[current_part["parent"]]
                else:
                    break

            # Check if this part is directly controlled by the joint
            if part_id == joint_id:
                is_movable = True

            if is_movable:
                movable_parts.append(part_id)
            else:
                static_parts.append(part_id)

        print(f"    📦 Movable parts: {movable_parts}")
        print(f"    🏠 Static parts: {static_parts}")

        # Prepare output folder
        gt_mesh_dir = output_dir / "gt_mesh"
        gt_mesh_dir.mkdir(parents=True, exist_ok=True)

        geometries = {}

        # ----------------------------------------------------------------------
        # Process movable & static parts
        # ----------------------------------------------------------------------
        for category, part_ids in [("movable", movable_parts), ("static", static_parts)]:
            print(f"    🔨 Processing {category} parts...")

            all_objs = set()
            for part_id in part_ids:
                part_data = next((p for p in mobility_data if p["id"] == part_id), None)
                if not part_data or "parts" not in part_data:
                    continue

                partids_in_result = [obj["id"] for obj in part_data["parts"]]
                for partid_in_result in partids_in_result:
                    if partid_in_result in partid_to_objs:
                        all_objs |= partid_to_objs[partid_in_result]

            if not all_objs:
                print(f"    ⚠️  No OBJ files found for {category} parts")
                continue

            mesh_paths = [partnet_dir / "textured_objs" / (obj + ".obj") for obj in all_objs]
            merged_mesh = merge_meshs(mesh_paths)

            if merged_mesh is None:
                print(f"    ⚠️  No mesh found for {category} parts")
                continue

            try:
                points = np.array(merged_mesh.points)

                # Optional: denser sampling (increase this limit if needed)
                if len(points) > 200000:
                    indices = np.random.choice(len(points), 200000, replace=False)
                    points = points[indices]

                geometries[category] = points
                np.save(gt_mesh_dir / f"gt_{category}.npy", points)
                print(f"    💾 Saved {category}: {len(points)} points")

            except Exception as e:
                print(f"    ❌ Error processing {category}: {e}")

        # ----------------------------------------------------------------------
        # Combine full geometry
        # ----------------------------------------------------------------------
        if "static" in geometries and "movable" in geometries:
            whole_geometry = np.vstack([geometries["static"], geometries["movable"]])
        elif "static" in geometries:
            whole_geometry = geometries["static"]
        elif "movable" in geometries:
            whole_geometry = geometries["movable"]
        else:
            whole_geometry = None

        if whole_geometry is not None:
            np.save(gt_mesh_dir / "gt_whole.npy", whole_geometry)
            print(f"    💾 Saved whole: {len(whole_geometry)} points")

        return True

    except Exception as e:
        import traceback
        print(f"    ❌ GT extraction failed: {type(e).__name__}: {e}")
        traceback.print_exc()
        return False

        
def convert_single_v2a_object(
    v2a_path,
    output_dir,
    category,
    object_id,
    joint_name,
    gt_joints_meta,
    partnet_root=None,
):
    """Convert a single V2A object to the articulation render-style NeRF format."""

    import shutil
    import json
    import numpy as np
    from pathlib import Path
    from PIL import Image

    v2a_path = Path(v2a_path)
    output_dir = Path(output_dir)
    post_dir = output_dir / "post"
    frames_dir = post_dir / "frames"
    depth_dir = post_dir / "depth"
    mask_dir = post_dir / "mask_gt"

    for d in [frames_dir, depth_dir, mask_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------------
    # Load data
    # ----------------------------------------------------------------------
    if not (v2a_path / "camera_pose.npy").exists():
        print(f"  ❌ Missing camera_pose.npy in {v2a_path}")
        return False

    try:
        camera_poses = np.load(v2a_path / "camera_pose.npy")
        intrinsics = np.load(v2a_path / "intrinsics.npy")

        # Handle RGB files (.png or .jpg)
        rgb_dir = v2a_path / "rgb"
        if not rgb_dir.exists():
            for alt in ["sample_rgb", "rgb_reverse"]:
                if (v2a_path / alt).exists():
                    rgb_dir = v2a_path / alt
                    break
        rgb_files = sorted(rgb_dir.glob("*.png")) + sorted(rgb_dir.glob("*.jpg"))

        # Handle depth files (.npy or .npz)
        depth_dir_in = v2a_path / "depth"
        if not depth_dir_in.exists():
            if (v2a_path / "xyz").exists():
                depth_dir_in = v2a_path / "xyz"
        depth_files = (
            sorted(depth_dir_in.glob("*.npy"))
            + sorted(depth_dir_in.glob("*.npz"))
            + sorted(depth_dir_in.glob("*.png"))
            + sorted(depth_dir_in.glob("*.exr"))
        )

        # Handle segment/mask files
        segment_dir = v2a_path / "segment"
        segment_files = sorted(segment_dir.glob("*.npz")) if segment_dir.exists() else []

        print(f"  📊 Found {len(rgb_files)} RGB, {len(depth_files)} depth files")

    except Exception as e:
        print(f"  ❌ Error loading V2A data: {e}")
        return False

    # ----------------------------------------------------------------------
    # Build base transform JSON structure
    # ----------------------------------------------------------------------
    transforms = {
        "camera_model": "PINHOLE",
        "fl_x": float(intrinsics[0, 0]),
        "fl_y": float(intrinsics[1, 1]),
        "cx": float(intrinsics[0, 2]),
        "cy": float(intrinsics[1, 2]),
        "w": 640,
        "h": 480,
        "k1": 0.0,
        "k2": 0.0,
        "p1": 0.0,
        "p2": 0.0,
        "articulations_gt": [],
        "frames": [],
    }

    joint_id = 0
    for token in joint_name.split("_"):
        if token.isdigit():
            joint_id = int(token)
            break

    joint_info = None
    if category in gt_joints_meta and object_id in gt_joints_meta[category]:
        gt_obj = gt_joints_meta[category][object_id]
        if "interaction_list" in gt_obj and joint_id < len(gt_obj["interaction_list"]):
            joint_info = gt_obj["interaction_list"][joint_id]

    if joint_info:
        jt_type = joint_info.get("type", "revolute").lower()
        jt_data = joint_info.get("joint", {})

        # --- Normalize axis ---
        axis_raw = jt_data.get("axis", [0, 0, 1])
        if isinstance(axis_raw, dict):
            jt_axis = axis_raw.get("direction", [0, 0, 1])
            jt_pivot = axis_raw.get("origin", [0, 0, 0])
        else:
            jt_axis = axis_raw
            jt_pivot = jt_data.get("origin", [0, 0, 0])

        # --- Normalize limit ---
        limit_raw = jt_data.get("limit", [0.0, 0.0])
        if isinstance(limit_raw, dict):
            lower = limit_raw.get("lower", 0.0)
            upper = limit_raw.get("upper", 0.0)
            jt_limit = [lower, upper]
        else:
            jt_limit = list(limit_raw)

        # --- Type and unit conversion ---
        if jt_type in ["hinge", "revolute"]:
            jt_type = "revolute"
            jt_limit = [float(np.rad2deg(jt_limit[0])), float(np.rad2deg(jt_limit[1]))]
        elif jt_type in ["slider", "prismatic"]:
            jt_type = "prismatic"
        else:
            jt_type = jt_type

        transforms["articulations_gt"].append(
            {
                "joint_type_gt": jt_type,
                "joint_axis_gt": jt_axis,
                "joint_pivot_gt": jt_pivot,
                "joint_limits_gt": jt_limit,
            }
        )

    # ----------------------------------------------------------------------
    # Optional: Load joint value trajectory
    # ----------------------------------------------------------------------
    gt_joint_file = v2a_path / "gt_joint_value.npy"
    if gt_joint_file.exists():
        gt_values = np.load(gt_joint_file)
    else:
        gt_values = np.zeros(len(rgb_files))

    # ----------------------------------------------------------------------
    # Convert and copy all frames
    # ----------------------------------------------------------------------

    for i, (rgb_file, depth_file) in enumerate(zip(rgb_files, depth_files)):
        if i >= len(camera_poses):
            break

        frame_name = f"frame_{i+1:05d}.jpg"
        depth_name = f"depth_{i+1:05d}.npy"
        mask_name = f"mask_{i+1:05d}.png"

        # Copy RGB
        shutil.copy2(rgb_file, frames_dir / frame_name)

        # Convert or copy depth
        try:
            if depth_file.suffix == ".npz":
                depth_archive = np.load(depth_file)
                key = next(iter(depth_archive.keys()))
                depth = depth_archive[key]
            elif depth_file.suffix == ".npy":
                depth = np.load(depth_file)
            else:
                depth = None  # skip image-based depth
            if depth is not None:
                np.save(depth_dir / depth_name, depth)
        except Exception as e:
            print(f"    ⚠️  Depth load error {depth_file}: {e}")


        # Correctly convert segmentation NPZ → PNG mask
        if i < len(segment_files):
            try:
                seg = np.load(segment_files[i])

                # Try common segmentation keys
                if "actor_id" in seg:
                    mask = seg["actor_id"]
                elif "segmentation" in seg:
                    mask = seg["segmentation"]
                else:
                    key = next(iter(seg.keys()))
                    mask = seg[key]

                # Convert to 8-bit PNG
                mask = mask.astype(np.uint8)
                Image.fromarray(mask).save(mask_dir / mask_name)

            except Exception as e:
                print(f"    ⚠️ Mask conversion error {segment_files[i]}: {e}")
                Image.new("L", (640, 480), 0).save(mask_dir / mask_name)
        else:
            # Fallback empty mask
            Image.new("L", (640, 480), 0).save(mask_dir / mask_name)


        # Camera transform
        pose_matrix = convert_pose_to_matrix(camera_poses[i])

        frame_data = {
            "file_path": f"frames/{frame_name}",
            "depth_file_path": f"depth/{depth_name}",
            "mask_file_path_gt": f"mask_gt/{mask_name}",
            "transform_matrix": pose_matrix.tolist(),
            "joint_angle_gt": float(gt_values[i]) if len(gt_values) > i else 0.0,
        }

        transforms["frames"].append(frame_data)

    # ----------------------------------------------------------------------
    # Save output JSON
    # ----------------------------------------------------------------------
    json_path = post_dir / "transforms.json"
    with open(json_path, "w") as f:
        json.dump(transforms, f, indent=2)

    # ----------------------------------------------------------------------
    # Optionally extract PartNet GT geometry
    # ----------------------------------------------------------------------
    if partnet_root:
        try:
            partnet_dir = Path(partnet_root) / category / object_id
            if partnet_dir.exists():
                extract_geometry_for_joint(partnet_dir, joint_id, post_dir)
        except Exception as e:
            print(f"    ⚠️  Geometry extraction failed: {e}")

    return True


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Batch V2A to NeRF converter")
    parser.add_argument("--eval_config", type=str, default="eval_setup/configs/eval_config.yaml",
                       help="Path to evaluation config YAML")
    parser.add_argument("--gt_joints_file", type=str, 
                       default="eval_setup/new_partnet_mobility_dataset_correct_intr_meta.json",
                       help="Path to ground-truth joint metadata JSON")
    parser.add_argument("--v2a_root", type=str, 
                       default="data_itaco/video2articulation/partnet-mobility-v0",
                       help="Path to Video2Articulation dataset root")
    parser.add_argument("--partnet_root", type=str,
                       default="data_itaco/video2articulation/partnet-mobility-v0", 
                       help="Path to PartNet-Mobility root (for GT geometry)")
    parser.add_argument("--project_root", type=str, default=".",
                       help="Path to project root (where eval renders are)")
    
    args = parser.parse_args()
    
    # Load configs
    eval_cfg = yaml.safe_load(open(args.eval_config))
    gt_meta = json.load(open(args.gt_joints_file))
    
    # Paths
    v2a_root = Path(args.v2a_root)
    partnet_root = Path(args.partnet_root)
    output_root = Path(args.project_root) / eval_cfg["evaluation"]["output_renders"]
    
    partnet_objects = eval_cfg["datasets"]["partnet_objects"]
    
    successful = 0
    failed = 0
    
    for entry in tqdm(partnet_objects, desc="Converting V2A data"):
        category = entry["category"]
        object_id = str(entry["object_id"])
        joint_name = entry.get("joint_id", "joint_0")

        view = entry.get("view")  # fallback if missing
        v2a_path = v2a_root / category / object_id / joint_name / view
        
        # Output path (matches render output structure)
        output_dir = output_root / f"{category}_{object_id}"
        
        if not v2a_path.exists():
            print(f"⚠️  Skipping {object_id}: V2A path not found {v2a_path}")
            failed += 1
            continue
        
        print(f"\n🔄 Converting {category}/{object_id}/{joint_name}")
        
        try:
            success = convert_single_v2a_object(
                v2a_path, output_dir, category, object_id, joint_name, gt_meta, partnet_root
            )
            
            if success:
                print(f"  ✅ Converted successfully")
                successful += 1
            else:
                print(f"  ❌ Conversion failed")
                failed += 1
                
        except Exception as e:
            print(f"  ❌ Error during conversion: {type(e).__name__}: {e}")
            traceback.print_exc()
            failed += 1
    
    print(f"\n📊 V2A Conversion Summary:")
    print(f"  ✅ Successful: {successful}")
    print(f"  ❌ Failed: {failed}")
    print(f"  📁 Output: {output_root}")

if __name__ == "__main__":
    main()