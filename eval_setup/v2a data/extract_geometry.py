#!/usr/bin/env python3
"""
Extract individual part meshes from PartNet-Mobility for evaluation.
Adapted from your preprocessing script to focus on movable vs static parts.
"""

import json
import numpy as np
import pyvista as pv
from pathlib import Path
import point_cloud_utils as pcu

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

def merge_meshs(meshs_paths: list[Path]):
    """Merge multiple OBJ files into single mesh"""
    meshs_paths = list(set(meshs_paths))
    meshs = []
    
    for mesh_path in meshs_paths:
        try:
            mesh = pv.read(str(mesh_path))
            meshs.append(mesh)
        except FileNotFoundError:
            print(f"[Warning] {mesh_path} not found.")
    
    if not meshs:
        return None
        
    merged_mesh = meshs[0]
    for mesh in meshs[1:]:
        merged_mesh += mesh
    
    return merged_mesh

def find_movable_parts(mobility_data, target_joint_id):
    """Find parts that move with the specified joint"""
    movable_part_ids = []
    static_part_ids = []
    
    for part in mobility_data:
        part_id = part['id']
        
        # Check if this part is controlled by target joint
        if part['parent'] == target_joint_id:
            movable_part_ids.append(part_id)
        elif part['parent'] == -1 or part['joint'] == 'fixed':
            static_part_ids.append(part_id)
        else:
            # Check if this part is a child of the movable joint
            # (you might need to traverse the hierarchy here)
            static_part_ids.append(part_id)
    
    return movable_part_ids, static_part_ids

def extract_geometry_for_joint(partnet_dir, joint_id, output_dir):
    """Extract static, movable, and whole geometry for evaluation"""
    
    partnet_dir = Path(partnet_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🔍 Extracting geometry for joint {joint_id}")
    
    # Load mobility data
    mobility_file = partnet_dir / 'mobility_v2.json'
    mobility_data = json.loads(mobility_file.read_text())
    
    # Parse part-to-obj mapping
    partid_to_objs = parse_partid_to_objs(partnet_dir)
    
    # Find movable vs static parts
    movable_parts, static_parts = find_movable_parts(mobility_data, joint_id)
    
    print(f"📦 Movable parts: {movable_parts}")
    print(f"🏠 Static parts: {static_parts}")
    
    # Extract meshes for each category
    geometries = {}
    
    for category, part_ids in [("movable", movable_parts), ("static", static_parts)]:
        print(f"🔨 Processing {category} parts...")
        
        # Collect all OBJ files for these parts
        all_objs = set()
        for part_id in part_ids:
            # Get part data
            part_data = next((p for p in mobility_data if p['id'] == part_id), None)
            if not part_data:
                continue
                
            # Get OBJ files for this part's geometry
            partids_in_result = [obj['id'] for obj in part_data["parts"]]
            for partid_in_result in partids_in_result:
                if partid_in_result in partid_to_objs:
                    all_objs |= partid_to_objs[partid_in_result]
        
        # Load and merge meshes
        mesh_paths = [partnet_dir / 'textured_objs' / (obj + '.obj') for obj in all_objs]
        merged_mesh = merge_meshs(mesh_paths)
        
        if merged_mesh is not None:
            # Sample points from surface
            try:
                points = np.array(merged_mesh.points)
                # Sample uniformly if too many points
                if len(points) > 10000:
                    indices = np.random.choice(len(points), 10000, replace=False)
                    points = points[indices]
                
                geometries[category] = points
                
                # Save as .npy
                np.save(output_dir / f"gt_{category}.npy", points)
                print(f"💾 Saved {category}: {len(points)} points")
                
            except Exception as e:
                print(f"❌ Error processing {category}: {e}")
        else:
            print(f"⚠️  No mesh found for {category} parts")
    
    # Create whole object geometry (combine static + movable)
    if "static" in geometries and "movable" in geometries:
        whole_geometry = np.vstack([geometries["static"], geometries["movable"]])
        np.save(output_dir / "gt_whole.npy", whole_geometry)
        print(f"💾 Saved whole: {len(whole_geometry)} points")
    
    return geometries

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Extract part meshes for evaluation")
    parser.add_argument("--partnet_dir", type=str, required=True,
                       help="Path to PartNet-Mobility object directory")
    parser.add_argument("--joint_id", type=int, required=True,
                       help="Target joint ID")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for GT meshes")
    
    args = parser.parse_args()
    
    extract_geometry_for_joint(args.partnet_dir, args.joint_id, args.output_dir)

if __name__ == "__main__":
    main()