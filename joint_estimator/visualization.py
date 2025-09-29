import json
import torch
import numpy as np
import open3d as o3d

def visualize_joint_and_inliers(ply_path, inliers_pt_path, schema_json_path):
    # --- Load world environment mesh/point cloud
    print(f"Loading world: {ply_path}")
    world_mesh = o3d.io.read_point_cloud(ply_path)
    
    # --- Load inlier points
    print(f"Loading inliers: {inliers_pt_path}")
    inlier_points = torch.load(inliers_pt_path).cpu().numpy()
    inlier_pc = o3d.geometry.PointCloud()
    inlier_pc.points = o3d.utility.Vector3dVector(inlier_points)
    inlier_pc.paint_uniform_color([1, 0, 0])  # red inliers
    
    # --- Load joint schema
    print(f"Loading schema: {schema_json_path}")
    with open(schema_json_path, 'r') as f:
        joints = json.load(f)
    
    geometries = [world_mesh, inlier_pc]
    
    # Axis length heuristic (based on bbox of inliers/world)
    if len(inlier_points) > 0:
        bbox_diag = np.linalg.norm(
            inlier_points.max(axis=0) - inlier_points.min(axis=0)
        )
    else:
        bbox_diag = 1.0
    
    axis_len = 0.3 * bbox_diag  # 30% of bbox diagonal
    
    for joint in joints:
        joint_type = joint["joint_type"]
        axis = np.array(joint["joint_axis"], dtype=np.float32)
        pivot = np.array(joint["joint_pivot"], dtype=np.float32)
        limits = joint.get("joint_limits", [None, None])
        
        # normalize axis
        axis = axis / (np.linalg.norm(axis) + 1e-8)
        
        # --- Draw pivot as sphere
        pivot_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.02 * bbox_diag)
        pivot_sphere.translate(pivot)
        pivot_sphere.paint_uniform_color([0, 1, 0])  # green pivot
        geometries.append(pivot_sphere)
        
        # --- Draw axis as thick line with arrow
        arrow_cylinder = o3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=0.005 * bbox_diag,
            cone_radius=0.01 * bbox_diag,
            cylinder_height=axis_len * 0.8,
            cone_height=axis_len * 0.2
        )
        
        # Create rotation matrix to align arrow with axis
        z_axis = np.array([0, 0, 1])
        rotation_matrix = get_rotation_matrix(z_axis, axis)
        arrow_cylinder.rotate(rotation_matrix, center=[0, 0, 0])
        arrow_cylinder.translate(pivot)
        
        if joint_type == "revolute":
            arrow_cylinder.paint_uniform_color([0, 0, 1])  # blue for revolute
        else:
            arrow_cylinder.paint_uniform_color([1, 1, 0])  # yellow for prismatic
        
        geometries.append(arrow_cylinder)
        
        # --- Add min/max visualization
        if joint_type == "prismatic":
            # For prismatic: show min/max positions as spheres along axis
            if limits[0] is not None:
                min_pos = pivot + axis * limits[0]
                min_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.015 * bbox_diag)
                min_sphere.translate(min_pos)
                min_sphere.paint_uniform_color([1, 0, 1])  # magenta for min
                geometries.append(min_sphere)
            
            if limits[1] is not None:
                max_pos = pivot + axis * limits[1]
                max_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.015 * bbox_diag)
                max_sphere.translate(max_pos)
                max_sphere.paint_uniform_color([0, 1, 1])  # cyan for max
                geometries.append(max_sphere)
        
        elif joint_type == "revolute":
            # For revolute: show min/max angles as lines from pivot
            if limits[0] is not None and limits[1] is not None:
                # Create perpendicular vector to axis for angle visualization
                if abs(axis[2]) < 0.9:
                    perp = np.cross(axis, [0, 0, 1])
                else:
                    perp = np.cross(axis, [1, 0, 0])
                perp = perp / np.linalg.norm(perp)
                
                # Min angle line
                min_angle_vec = rotate_vector_around_axis(perp, axis, limits[0])
                min_end = pivot + min_angle_vec * axis_len * 0.5
                min_line = o3d.geometry.LineSet(
                    points=o3d.utility.Vector3dVector([pivot, min_end]),
                    lines=o3d.utility.Vector2iVector([[0, 1]])
                )
                min_line.paint_uniform_color([1, 0, 1])  # magenta for min angle
                geometries.append(min_line)
                
                # Max angle line
                max_angle_vec = rotate_vector_around_axis(perp, axis, limits[1])
                max_end = pivot + max_angle_vec * axis_len * 0.5
                max_line = o3d.geometry.LineSet(
                    points=o3d.utility.Vector3dVector([pivot, max_end]),
                    lines=o3d.utility.Vector2iVector([[0, 1]])
                )
                max_line.paint_uniform_color([0, 1, 1])  # cyan for max angle
                geometries.append(max_line)
        
        print(f"Added {joint_type} joint at pivot {pivot}, limits={limits}")
    
    # --- Visualize everything
    o3d.visualization.draw_geometries(geometries)

def get_rotation_matrix(v1, v2):
    """Get rotation matrix to rotate v1 to v2"""
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    
    cross = np.cross(v1, v2)
    dot = np.dot(v1, v2)
    
    if np.allclose(cross, 0):
        if dot > 0:
            return np.eye(3)
        else:
            # 180 degree rotation - find any perpendicular vector
            if abs(v1[0]) < 0.9:
                perp = np.cross(v1, [1, 0, 0])
            else:
                perp = np.cross(v1, [0, 1, 0])
            perp = perp / np.linalg.norm(perp)
            return 2 * np.outer(perp, perp) - np.eye(3)
    
    # Rodrigues' rotation formula
    k = cross / np.linalg.norm(cross)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    angle = np.arccos(np.clip(dot, -1, 1))
    
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
    return R

def rotate_vector_around_axis(vector, axis, angle):
    """Rotate vector around axis by angle (in radians)"""
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    
    # Rodrigues' rotation formula
    rotated = (vector * cos_angle + 
               np.cross(axis, vector) * sin_angle + 
               axis * np.dot(axis, vector) * (1 - cos_angle))
    
    return rotated

if __name__ == "__main__":
    # visualize_joint_and_inliers(
    #     ply_path="output/sparse_pc.ply",
    #     inliers_pt_path="output/prismatic_inliers_world.pt",
    #     schema_json_path="output/prismatic_schema.json"
    # )

    visualize_joint_and_inliers(
        ply_path="output_rev/sparse_pc.ply",
        inliers_pt_path="output_rev/revolute_inliers_world.pt",
        schema_json_path="output_rev/revolute_schema.json"
    )