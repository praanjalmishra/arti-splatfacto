"""
Localize post-change images against pre-change COLMAP reconstruction,
update poses in existing transforms.json, and visualize results.
"""

import json
import numpy as np
from pathlib import Path
from hloc import extract_features, match_features, pairs_from_retrieval
from hloc.localize_sfm import QueryLocalizer
import pycolmap
from hloc.utils.io import get_keypoints
from scipy.spatial.transform import Rotation
from collections import defaultdict
from hloc.utils import viz_3d
import plotly.graph_objects as go
import h5py
import argparse
import open3d as o3d

def localize_and_update_json(
    pre_sfm_dir,           # path to pre-change reconstruction (hloc outputs)
    post_image_dir,        # directory of post-change images
    arkit_transforms_path,   # old transforms.json (with depth paths, etc.)
    new_transforms_path,   # where to save updated transforms.json
    num_retrieval=20,
    ransac_thresh=10.0
):
    """
    Localize post-change images and update existing transforms.json
    """
    pre_sfm_dir = Path(pre_sfm_dir)
    post_image_dir = Path(post_image_dir)
    arkit_transforms_path = Path(arkit_transforms_path)
    new_transforms_path = Path(new_transforms_path)

    # Load reconstruction and camera
    print("Loading pre-change reconstruction...")
    reconstruction = pycolmap.Reconstruction(str(pre_sfm_dir / "sfm" / "reconstruction"))
    camera = list(reconstruction.cameras.values())[0]
    print(f"Loaded reconstruction with {len(reconstruction.images)} images, {len(reconstruction.points3D)} points")

    # Setup output directory
    post_outputs = post_image_dir.parent / "post_hloc_outputs"
    post_outputs.mkdir(exist_ok=True)

    # Define output paths
    post_features = post_outputs / "features.h5"
    post_global = post_outputs / "global-descriptors.h5"
    post_matches = post_outputs / "matches.h5"
    pairs_file = post_outputs / "pairs.txt"

    # Paths to pre-change data
    pre_features = pre_sfm_dir / "features.h5"
    pre_global = pre_sfm_dir / "global-descriptors.h5"
    pre_model = pre_sfm_dir / "sfm" / "reconstruction"

    # === Step 1: Extract features ===
    print("\n=== Step 1: Extract local features ===")
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

    # === Step 3: Image retrieval ===
    print("\n=== Step 3: Retrieve similar pre-change images ===")
    pairs_from_retrieval.main(
        descriptors=post_global,
        output=pairs_file,
        num_matched=num_retrieval,
        db_descriptors=pre_global,
        db_model=pre_model
    )

    # === Step 4: Feature matching ===
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

    # === Step 5: Setup localizer ===
    print("\n=== Step 5: Localize post-change images ===")
    conf = {
        "estimation": {"ransac": {"max_error": ransac_thresh}},
        "refinement": {"refine_focal_length": False, "refine_extra_params": False}
    }
    localizer = QueryLocalizer(reconstruction, conf)

    # Get image names
    post_images = sorted([p.name for p in post_image_dir.iterdir() 
                         if p.suffix.lower() in [".jpg", ".png", ".jpeg"]])
    db_name_to_id = {img.name: img_id for img_id, img in reconstruction.images.items()}

    # Load matches
    print("Loading matches from h5 file...")
    with h5py.File(post_matches, "r") as f:
        matches_data = {}
        for query_name in f.keys():
            matches_data[query_name] = {}
            for ref_name in f[query_name].keys():
                matches_data[query_name][ref_name] = f[query_name][ref_name]['matches0'][()]
    
    print(f"Loaded matches for {len(matches_data)} query images")

    # Localize each image
    poses = {}
    failed = []
    
    for query_name in post_images:
        if query_name not in matches_data:
            failed.append(query_name)
            continue
            
        ref_names = list(matches_data[query_name].keys())
        ref_ids = [db_name_to_id[n] for n in ref_names if n in db_name_to_id]
        
        if not ref_ids:
            failed.append(query_name)
            continue

        # Get keypoints
        kpq = get_keypoints(post_features, query_name) + 0.5  # COLMAP coordinates
        
        # Build 2D-3D correspondences
        kp_idx_to_3D = defaultdict(list)
        
        for ref_id in ref_ids:
            image = reconstruction.images[ref_id]
            if image.num_points3D() == 0:
                continue
                
            points3D_ids = np.array([p.point3D_id if p.has_point3D() else -1 
                                    for p in image.points2D])
            ref_name = image.name
            
            if ref_name not in matches_data[query_name]:
                continue
                
            matches = matches_data[query_name][ref_name]
            
            for idx_q, idx_r in enumerate(matches):
                if idx_r >= 0 and idx_r < len(points3D_ids):
                    id_3D = points3D_ids[idx_r]
                    if id_3D != -1 and id_3D not in kp_idx_to_3D[idx_q]:
                        kp_idx_to_3D[idx_q].append(id_3D)

        if not kp_idx_to_3D:
            failed.append(query_name)
            continue

        # Prepare for localization
        idxs = list(kp_idx_to_3D.keys())
        mkp_idxs = [i for i in idxs for _ in kp_idx_to_3D[i]]
        mp3d_ids = [j for i in idxs for j in kp_idx_to_3D[i]]

        # Localize
        ret = localizer.localize(kpq, mkp_idxs, mp3d_ids, camera)
        
        if not ret['success']:
            failed.append(query_name)
            continue

        # Convert pose
        qvec, tvec = ret['qvec'], ret['tvec']
        R = Rotation.from_quat([qvec[1], qvec[2], qvec[3], qvec[0]]).as_matrix()
        w2c = np.eye(4)
        w2c[:3, :3] = R
        w2c[:3, 3] = tvec
        c2w = np.linalg.inv(w2c)
        
        fix_rot = np.diag([1, -1, -1, 1])
        poses[query_name] = c2w @ fix_rot
        
        print(f"✓ {query_name}: {ret['num_inliers']} inliers")

    print(f"\n=== Localization Results ===")

    if args.align_arkit:
        print("\n=== Step 5b: Aligning ARKit trajectory to pre-change SfM frame ===")

        # Load ARKit poses from old JSON
        with open(arkit_transforms_path, "r") as f:
            old_json = json.load(f)

        arkit_centers, hloc_centers = [], []
        name_to_index = {}

        for frame in old_json["frames"]:
            name = Path(frame["file_path"]).name
            if name in poses:
                T_arkit = np.array(frame["transform_matrix"])
                arkit_centers.append(T_arkit[:3, 3])
                hloc_centers.append(poses[name][:3, 3])
                name_to_index[name] = len(arkit_centers) - 1

        if len(arkit_centers) >= 4:
            # Create PointCloud objects
            src_cloud = o3d.geometry.PointCloud()
            src_cloud.points = o3d.utility.Vector3dVector(np.array(arkit_centers))
            
            dst_cloud = o3d.geometry.PointCloud()
            dst_cloud.points = o3d.utility.Vector3dVector(np.array(hloc_centers))
            
            # Create correspondence indices
            n_points = len(arkit_centers)
            correspondences = o3d.utility.Vector2iVector(
                np.array([[i, i] for i in range(n_points)], dtype=np.int32)
            )
            
            reg = o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=True)
            transform = reg.compute_transformation(src_cloud, dst_cloud, correspondences)

            print("✓ Estimated ARKit→SfM alignment transform:")
            print(transform)

            for frame in old_json["frames"]:
                name = Path(frame["file_path"]).name
                T = np.array(frame["transform_matrix"])
                T_aligned = transform @ T
                frame["transform_matrix"] = T_aligned.tolist()
                # Update poses dict so visualization shows aligned positions
                if name in poses:
                    poses[name] = T_aligned

            # Save aligned version
            with open(new_transforms_path, "w") as f:
                json.dump(old_json, f, indent=2)

            print(f"Saved globally aligned ARKit poses to {new_transforms_path}")
        else:
            print("Not enough matched frames to estimate ARKit alignment. Skipping.")


    print(f"Successfully localized: {len(poses)}/{len(post_images)} images")
    if failed:
        print(f"Failed: {len(failed)} images")

    
    with open(arkit_transforms_path, "r") as f:
        old_json = json.load(f)

    file_to_pose = {f"frames/{name}": mat.tolist() for name, mat in poses.items()}
    
    updated = 0
    for frame in old_json["frames"]:
        fp = frame["file_path"]
        if fp in file_to_pose:
            frame["transform_matrix"] = file_to_pose[fp]
            updated += 1
    
    with open(new_transforms_path, "w") as f:
        json.dump(old_json, f, indent=2)
    
    print(f"Saved updated transforms to: {new_transforms_path}")

    if args.visualize:
        
        fig = viz_3d.init_figure()
        
        # Plot pre-change reconstruction
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

        if poses:
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
            
            # Add camera direction arrows (sample)
            sample_indices = np.linspace(0, len(poses)-1, min(20, len(poses)), dtype=int)
            pose_list = list(poses.values())
            
            for idx in sample_indices:
                pose = pose_list[idx]
                center = pose[:3, 3]
                direction = pose[:3, :3] @ np.array([0, 0, -1])
                
                arrow_length = 0.2
                end_point = center + direction * arrow_length
                
                fig.add_trace(
                    go.Scatter3d(
                        x=[center[0], end_point[0]],
                        y=[center[1], end_point[1]],
                        z=[center[2], end_point[2]],
                        mode='lines',
                        line=dict(color='cyan', width=2),
                        showlegend=False,
                        hoverinfo='skip'
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
    
    return poses, old_json


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Localize post-change images against a pre-change COLMAP reconstruction and update transforms.json."
    )
    parser.add_argument("--pre_sfm_dir", type=str, required=True,
                        help="Path to pre-change reconstruction (hloc outputs, contains sfm/)")
    parser.add_argument("--post_image_dir", type=str, required=True,
                        help="Path to post-change image directory (frames/)")
    parser.add_argument("--arkit_transforms_path", type=str, required=True,
                        help="Path to original transforms.json file")
    parser.add_argument("--new_transforms_path", type=str, required=True,
                        help="Output path for updated transforms.json file")
    parser.add_argument("--num_retrieval", type=int, default=10,
                        help="Number of retrieval matches for localization (default: 10)")
    parser.add_argument("--ransac_thresh", type=float, default=12.0,
                        help="RANSAC reprojection threshold (default: 12.0)")

    parser.add_argument("--align_arkit", action="store_true",
                        help="Align old ARKit transforms to pre-change HLoc coordinate frame instead of replacing poses")
    parser.add_argument("--visualize", action="store_true",
                        help="Generate 3D visualization of localization results")
    args = parser.parse_args()

    poses, updated_json = localize_and_update_json(
        pre_sfm_dir=args.pre_sfm_dir,
        post_image_dir=args.post_image_dir,
        arkit_transforms_path=args.arkit_transforms_path,
        new_transforms_path=args.new_transforms_path,
        num_retrieval=args.num_retrieval,
        ransac_thresh=args.ransac_thresh
    )

    print("\n✓ Done! Updated transforms saved with all original metadata preserved.")