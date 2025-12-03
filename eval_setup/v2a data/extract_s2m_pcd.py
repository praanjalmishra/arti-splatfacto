import json
from pathlib import Path
import open3d as o3d
import numpy as np

# -------------------------------------------------------------
# Load Shape2Motion mesh mapping (from result_after_merging.json)
# -------------------------------------------------------------
def load_s2m_mapping(partnet_dir):
    mapping_file = Path(partnet_dir) / "result_after_merging.json"
    data = json.load(open(mapping_file))

    part_to_objs = {}

    def traverse(node_list):
        for node in node_list:
            ori_id = node.get("ori_id", None)
            objs = node.get("objs", [])
            if ori_id is not None:
                part_to_objs[ori_id] = objs
            if "children" in node:
                traverse(node["children"])

    traverse(data)
    return part_to_objs


# -------------------------------------------------------------
# Sample mesh for a part using the mapping
# -------------------------------------------------------------
def sample_part_mesh(partnet_dir, semantic_part_id, n_points=5000):
    mapping = load_s2m_mapping(partnet_dir)

    if semantic_part_id not in mapping:
        raise KeyError(f"Part {semantic_part_id} not found in S2M mapping.")

    obj_files = mapping[semantic_part_id]

    full_mesh = o3d.geometry.TriangleMesh()
    mesh_dir = Path(partnet_dir) / "textured_objs"

    for obj in obj_files:
        obj_path = mesh_dir / f"{obj}.obj"
        if not obj_path.exists():
            raise FileNotFoundError(f"Expected mesh not found: {obj_path}")

        mesh = o3d.io.read_triangle_mesh(str(obj_path), enable_post_processing=True)
        mesh.compute_vertex_normals()

        # Merge fragments
        full_mesh += mesh

    return full_mesh.sample_points_uniformly(number_of_points=n_points)


# -------------------------------------------------------------
# Load the full object PCD by merging ALL meshes
# -------------------------------------------------------------
def load_full_object_pcd(partnet_dir, n_points=20000):
    mesh_dir = Path(partnet_dir) / "textured_objs"
    full_mesh = o3d.geometry.TriangleMesh()

    for obj_file in mesh_dir.glob("original-*.obj"):
        mesh = o3d.io.read_triangle_mesh(str(obj_file), enable_post_processing=True)
        mesh.compute_vertex_normals()
        full_mesh += mesh

    return full_mesh.sample_points_uniformly(n_points)


def load_mobility_data(partnet_dir):
    path = Path(partnet_dir) / "mobility_v2.json"
    return json.load(open(path))



def sapien_mesh_to_o3d(mesh):
    verts = np.array(mesh.vertices)
    faces = np.array(mesh.indices).reshape(-1, 3)

    o3d_mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(verts),
        triangles=o3d.utility.Vector3iVector(faces)
    )
    o3d_mesh.compute_vertex_normals()
    return o3d_mesh


def extract_link_mesh_o3d(link):
    merged = o3d.geometry.TriangleMesh()

    for vb in link.get_visual_bodies():
        for shape in vb.get_render_shapes():
            mesh = sapien_mesh_to_o3d(shape.mesh)
            merged += mesh

    return merged