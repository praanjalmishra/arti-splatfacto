"""
Localize post-change images against pre-change scene reconstruction
"""

from pathlib import Path
import json
import numpy as np
import pycolmap
import h5py
import cv2
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from hloc import extract_features, match_features, pairs_from_retrieval
from hloc.localize_sfm import QueryLocalizer
from hloc.utils.io import get_keypoints
from collections import defaultdict
from hloc.utils import viz_3d
import plotly.graph_objects as go

def localize_post_images(
    pre_sfm_dir,           # Path to pre-change reconstruction (from script 1)
    post_image_dir,        # Path to post-change images
    post_transforms_file,  # Where to save results
    num_retrieval=10,      # Number of reference images to match
    ransac_thresh=12.0     # RANSAC threshold in pixels
):
    """
    Localize post-change images against pre-change reconstruction
    
    Args:
        pre_sfm_dir: Directory with pre-change hloc outputs (contains sfm/, features.h5, etc)
        post_image_dir: Directory with post-change frames
        post_transforms_file: Output path for transforms.json
        num_retrieval: Number of pre-change images to retrieve per query
        ransac_thresh: RANSAC threshold
    """
    
    pre_sfm_dir = Path(pre_sfm_dir)
    post_image_dir = Path(post_image_dir)
    post_transforms_file = Path(post_transforms_file)
    
    # Output paths for post-change features
    post_outputs = post_image_dir.parent / "post_hloc_outputs"
    post_outputs.mkdir(exist_ok=True)
    
    post_features = post_outputs / "post_features.h5"
    post_global = post_outputs / "post_global.h5"
    post_matches = post_outputs / "post_matches.h5"
    pairs_file = post_outputs / "post_pairs.txt"
    
    # Paths to pre-change data
    pre_features = pre_sfm_dir / "features.h5"
    pre_global = pre_sfm_dir / "global-descriptors.h5"
    pre_model = pre_sfm_dir / "sfm" / "reconstruction"  # Fixed path
    
    print("\n=== Step 1: Extract features from post-change images ===")
    feature_conf = {
        'output': 'feats-superpoint',
        'model': {
            'name': 'superpoint',
            'nms_radius': 3,
            'max_keypoints': 8192,
            'keypoint_threshold': 0.005
        },
        'preprocessing': {
            'grayscale': True,
            'resize_max': 1600,
        }
    }
    
    extract_features.main(
        conf=feature_conf,
        image_dir=post_image_dir,
        export_dir=post_outputs,
        feature_path=post_features
    )
    
    print("\n=== Step 2: Extract global descriptors ===")
    global_conf = extract_features.confs["netvlad"]
    extract_features.main(
        conf=global_conf,
        image_dir=post_image_dir,
        export_dir=post_outputs,
        feature_path=post_global
    )
    
    print("\n=== Step 3: Retrieve similar pre-change images ===")
    pairs_from_retrieval.main(
        descriptors=post_global,
        output=pairs_file,
        num_matched=num_retrieval,
        db_descriptors=pre_global,
        db_model=pre_model
    )
    
    print("\n=== Step 4: Match features ===")
    matcher_conf = {
        'output': 'matches-superglue',
        'model': {
            'name': 'superglue',
            'weights': 'outdoor',
            'sinkhorn_iterations': 50,
            'match_threshold': 0.2,
        }
    }
    
    match_features.main(
        conf=matcher_conf,
        pairs=pairs_file,
        features=post_features,
        export_dir=post_outputs,
        matches=post_matches,
        features_ref=pre_features
    )
    
    print("\n=== Step 5: Localize post-change images ===")
    
    # Load pre-change reconstruction
    reconstruction = pycolmap.Reconstruction(str(pre_model))
    
    # Get camera from reconstruction
    camera = list(reconstruction.cameras.values())[0]
    
    # Setup localizer
    conf = {
        "estimation": {"ransac": {"max_error": ransac_thresh}},
        "refinement": {"refine_focal_length": False, "refine_extra_params": False},
    }
    localizer = QueryLocalizer(reconstruction, conf)
    
    # Get post-change image names
    post_images = sorted([p.name for p in post_image_dir.iterdir() 
                         if p.suffix in ['.jpg', '.png', '.jpeg']])
    
    # Load matches from h5 file
    import h5py
    matches_data = {}
    
    print("Loading matches from h5 file...")
    with h5py.File(post_matches, 'r') as f:
        # Each query image is a top-level group
        for query_name in f.keys():
            matches_data[query_name] = {}
            query_group = f[query_name]
            
            # Each reference image is a subgroup
            for ref_name in query_group.keys():
                matches_data[query_name][ref_name] = query_group[ref_name]['matches0'][()]
    
    print(f"Loaded matches for {len(matches_data)} query images")
    if len(matches_data) > 0:
        first_query = list(matches_data.keys())[0]
        print(f"Example: {first_query} has matches with {len(matches_data[first_query])} reference images")
    
    # Build name to ID mapping
    db_name_to_id = {img.name: img_id for img_id, img in reconstruction.images.items()}
    
    # Localize each image
    poses = {}
    failed = []
    
    for query_name in post_images:
        print(f"Localizing {query_name}...")
        
        if query_name not in matches_data:
            failed.append(query_name)
            print(f"  ✗ No matches found")
            continue
        
        # Get reference images for this query
        ref_names = list(matches_data[query_name].keys())
        ref_ids = [db_name_to_id[name] for name in ref_names if name in db_name_to_id]
        
        if not ref_ids:
            failed.append(query_name)
            print(f"  ✗ No valid reference images")
            continue
        
        try:
            # Prepare matches in the format expected by pose_from_cluster
            kpq = get_keypoints(post_features, query_name)
            kpq += 0.5  # COLMAP coordinates
            
            kp_idx_to_3D = defaultdict(list)
            kp_idx_to_3D_to_db = defaultdict(lambda: defaultdict(list))
            num_matches = 0
            
            for i, db_id in enumerate(ref_ids):
                image = reconstruction.images[db_id]
                if image.num_points3D() == 0:
                    continue
                
                points3D_ids = np.array([p.point3D_id if p.has_point3D() else -1
                                         for p in image.points2D])
                
                # Get matches for this reference image
                ref_name = image.name
                if ref_name not in matches_data[query_name]:
                    continue
                    
                matches = matches_data[query_name][ref_name]
                
                # Convert matches to pairs of indices
                valid_matches = []
                for idx_q, idx_r in enumerate(matches):
                    if idx_r >= 0 and idx_r < len(points3D_ids):
                        if points3D_ids[idx_r] != -1:
                            valid_matches.append([idx_q, idx_r])
                
                if not valid_matches:
                    continue
                    
                valid_matches = np.array(valid_matches)
                num_matches += len(valid_matches)
                
                for idx_q, idx_r in valid_matches:
                    id_3D = points3D_ids[idx_r]
                    kp_idx_to_3D_to_db[idx_q][id_3D].append(i)
                    if id_3D not in kp_idx_to_3D[idx_q]:
                        kp_idx_to_3D[idx_q].append(id_3D)
            
            if num_matches == 0:
                failed.append(query_name)
                print(f"  ✗ No valid 2D-3D correspondences")
                continue
            
            # Prepare data for localization
            idxs = list(kp_idx_to_3D.keys())
            mkp_idxs = [i for i in idxs for _ in kp_idx_to_3D[i]]
            mp3d_ids = [j for i in idxs for j in kp_idx_to_3D[i]]
            
            ret = localizer.localize(kpq, mkp_idxs, mp3d_ids, camera)
            ret['camera'] = {
                'model': camera.model_name,
                'width': camera.width,
                'height': camera.height,
                'params': camera.params,
            }
            
            if ret['success']:
                qvec = ret['qvec']
                tvec = ret['tvec']
                
                # Convert to transform matrix (w2c)
                R = Rotation.from_quat([qvec[1], qvec[2], qvec[3], qvec[0]]).as_matrix()
                w2c = np.eye(4)
                w2c[:3, :3] = R
                w2c[:3, 3] = tvec
                
                # Convert to c2w (camera to world)
                c2w = np.linalg.inv(w2c)
                
                # Apply coordinate system fix (OpenCV to OpenGL)
                fix_rot = np.diag([1, -1, -1, 1])
                c2w = c2w @ fix_rot
                
                poses[query_name] = c2w
                
                print(f"  ✓ Success: {ret['num_inliers']} inliers")
            else:
                failed.append(query_name)
                print(f"  ✗ Failed to localize")
                
        except Exception as e:
            failed.append(query_name)
            print(f"  ✗ Error: {e}")
    
    print(f"\n=== Results ===")
    print(f"Successfully localized: {len(poses)}/{len(post_images)} images")
    if failed:
        print(f"Failed images: {failed}")
    
    # Save to transforms.json
    if poses:
        save_transforms_json(
            poses,
            post_transforms_file,
            camera
        )
        print(f"\n✓ Saved transforms to: {post_transforms_file}")
    
    # Visualize localized poses
    if poses:
        print("\n=== Step 7: Visualization ===")
        print(reconstruction.summary())
        
        # 3D visualization (using hloc.utils.viz_3d)
        fig = viz_3d.init_figure()
        viz_3d.plot_reconstruction(
            fig,
            reconstruction,
            color="rgba(255,0,0,0.5)",
            name="pre-change reconstruction",
            points=len(reconstruction.points3D) > 0,
            cameras=True,
            cs=0.1,
            points_rgb=True
        )
        
        # Add post-change localized cameras
        camera_centers = np.array([pose[:3, 3] for pose in poses.values()])
        fig.add_trace(
            go.Scatter3d(
                x=camera_centers[:, 0],
                y=camera_centers[:, 1],
                z=camera_centers[:, 2],
                mode='markers',
                marker=dict(size=4, color='blue'),
                name='post-change cameras'
            )
        )
        
        fig.update_layout(
            scene=dict(
                aspectmode='data',
                camera=dict(
                    up=dict(x=0, y=-1, z=0),
                    eye=dict(x=1.5, y=1.5, z=1.5)
                ),
            ),
            title="Camera Localization: Pre-change (red) vs Post-change (blue)"
        )
        
        html_path = post_outputs / "localization_viz.html"
        fig.write_html(str(html_path), auto_open=True)
        print(f"✓ 3D visualization saved to: {html_path}")
    

    
    return poses


# def visualize_localization(reconstruction, poses, image_dir, output_dir):
#     """Visualize localized camera poses"""
    
#     print("Creating 3D visualization...")
    
#     # Initialize figure
#     fig = viz_3d.init_figure()
    
#     # Plot reference reconstruction (pre-change scene)
#     viz_3d.plot_reconstruction(
#         fig,
#         reconstruction,
#         color="rgba(255,0,0,0.5)",
#         name="pre-change (reference)",
#         points=len(reconstruction.points3D) > 0,
#         cameras=True,
#         cs=0.1,
#         points_rgb=True
#     )
    
#     # Plot localized post-change cameras
#     camera_centers = []
#     camera_directions = []
    
#     for img_name, c2w in poses.items():
#         # Camera center in world coordinates
#         center = c2w[:3, 3]
#         camera_centers.append(center)
        
#         # Camera direction (negative z-axis in camera frame)
#         direction = c2w[:3, :3] @ np.array([0, 0, -1])
#         camera_directions.append(direction)
    
#     camera_centers = np.array(camera_centers)
#     camera_directions = np.array(camera_directions)
    
#     # Add post-change camera centers as blue points
#     fig.add_trace(go.Scatter3d(
#         x=camera_centers[:, 0],
#         y=camera_centers[:, 1],
#         z=camera_centers[:, 2],
#         mode='markers',
#         marker=dict(size=4, color='blue'),
#         name='post-change cameras'
#     ))
    
#     # Add camera direction arrows (sample a few)
#     sample_indices = np.linspace(0, len(poses)-1, min(20, len(poses)), dtype=int)
    
#     for idx in sample_indices:
#         center = camera_centers[idx]
#         direction = camera_directions[idx]
        
#         # Create arrow from center pointing in camera direction
#         arrow_length = 0.2
#         end_point = center + direction * arrow_length
        
#         fig.add_trace(go.Scatter3d(
#             x=[center[0], end_point[0]],
#             y=[center[1], end_point[1]],
#             z=[center[2], end_point[2]],
#             mode='lines',
#             line=dict(color='cyan', width=2),
#             showlegend=False,
#             hoverinfo='skip'
#         ))
    
#     fig.update_layout(
#         scene=dict(
#             aspectmode='data',
#             camera=dict(
#                 up=dict(x=0, y=-1, z=0),
#                 eye=dict(x=1.5, y=1.5, z=1.5)
#             ),
#         ),
#         title="Camera Localization: Pre-change (red) vs Post-change (blue)"
#     )
    
#     html_path = output_dir / "localization_viz.html"
#     fig.write_html(str(html_path), auto_open=True)
#     print(f"✓ 3D visualization saved to: {html_path}")
    
#     # Also create a simple 2D visualization of a few matches
#     print("\nCreating 2D match visualization...")
#     visualize_matches_2d(reconstruction, poses, image_dir, output_dir)


# def visualize_matches_2d(reconstruction, poses, post_image_dir, output_dir, num_viz=3):
#     """Visualize feature matches for a few image pairs"""

#     # Sample a few images to visualize
#     post_images = list(poses.keys())
#     sample_images = post_images[::len(post_images)//min(num_viz, len(post_images))][:num_viz]
    
#     for query_name in sample_images:
#         # Load query image
#         query_path = post_image_dir / query_name
#         query_img = cv2.imread(str(query_path))
#         if query_img is None:
#             continue
#         query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
        
#         # Find a reference image with good overlap
#         # (We'll just use the first one from reconstruction for simplicity)
#         ref_img_data = list(reconstruction.images.values())[0]
#         ref_name = ref_img_data.name
        
#         # Try to find reference image in pre-change directory
#         # (assuming it's in the same relative structure)
#         ref_path = post_image_dir.parent.parent / "pre_static" / "pre_static_1_post" / "frames" / ref_name
        
#         if ref_path.exists():
#             ref_img = cv2.imread(str(ref_path))
#             if ref_img is not None:
#                 ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
                
#                 # Create side-by-side visualization
#                 h1, w1 = query_img.shape[:2]
#                 h2, w2 = ref_img.shape[:2]
#                 h = max(h1, h2)
                
#                 # Resize if needed
#                 scale = 800 / max(w1, w2)
#                 if scale < 1:
#                     query_img = cv2.resize(query_img, None, fx=scale, fy=scale)
#                     ref_img = cv2.resize(ref_img, None, fx=scale, fy=scale)
                
#                 fig, axes = plt.subplots(1, 2, figsize=(15, 7))
#                 axes[0].imshow(query_img)
#                 axes[0].set_title(f'Post-change: {query_name}')
#                 axes[0].axis('off')
                
#                 axes[1].imshow(ref_img)
#                 axes[1].set_title(f'Pre-change: {ref_name}')
#                 axes[1].axis('off')
                
#                 plt.tight_layout()
#                 viz_path = output_dir / f"match_viz_{query_name[:-4]}.png"
#                 plt.savefig(str(viz_path), dpi=150, bbox_inches='tight')
#                 plt.close()
                
#                 print(f"✓ Saved 2D visualization: {viz_path}")
    
#     print(f"\n✓ Visualization complete!")


def save_transforms_json(poses, output_path, camera):
    """Save poses to transforms.json format"""
    
    transforms = {
        "camera_model": "PINHOLE",
        "fl_x": camera.params[0],
        "fl_y": camera.params[1],
        "cx": camera.params[2],
        "cy": camera.params[3],
        "w": camera.width,
        "h": camera.height,
        "k1": 0.0,
        "k2": 0.0,
        "p1": 0.0,
        "p2": 0.0,
        "frames": []
    }
    
    for img_name, c2w in poses.items():
        frame = {
            "file_path": f"frames/{img_name}",
            "transform_matrix": c2w.tolist()
        }
        transforms["frames"].append(frame)
    
    with open(output_path, 'w') as f:
        json.dump(transforms, f, indent=2)

if __name__ == "__main__":
    # Example usage
    pre_sfm_dir = Path("data_real/pre_static/pre_static_1_post/hloc_outputs")
    post_image_dir = Path("data_real/sync/prismatic/multi_sync/frames")
    post_transforms_file = Path("data_real/sync/prismatic/multi_sync/transforms_aligned.json")

    poses = localize_post_images(
        pre_sfm_dir=pre_sfm_dir,
        post_image_dir=post_image_dir,
        post_transforms_file=post_transforms_file,
        num_retrieval=10,
        ransac_thresh=12.0
    )