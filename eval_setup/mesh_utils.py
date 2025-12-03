import json
import shutil
import time
# import open3d as o3d
import numpy as np
import pyvista as pv
from rich import print
from glob import glob
from pathlib import Path

from typing import Tuple, List


def parse_partid_to_objs(shape_path:Path):
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

def merge_meshs(meshs_paths: List[Path], output_path:Path):
    meshs_paths = list(set(meshs_paths))
    meshs = []
    for mesh_path in meshs_paths:
        try:
            value = pv.read(str(mesh_path))
            meshs.append(value)
        except FileNotFoundError:
            print(f"[Warning] {mesh_path} not found.")

    merged_mesh = meshs[0]
    for mesh in meshs[1:]:
        merged_mesh += mesh
    merged_mesh.save(str(output_path))
    return merged_mesh
