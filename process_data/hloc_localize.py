"""
Localize post-change images against pre-change COLMAP reconstruction,
align ARKit trajectory to SfM frame using Umeyama algorithm, and visualize results.

- Uses Umeyama's closed-form similarity alignment (rotation + translation + scale)

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


def umeyama_alignment(src_points, dst_points):
    """
    Estimate similarity transform (rotation, translation, scale) using Umeyama algorithm.
    
    This is a closed-form solution that finds the optimal similarity transformation
    (rotation R, translation t, scale c) that minimizes:
        || dst - (c * R @ src + t) ||^2
    
    Args:
        src_points: Nx3 array of source points (ARKit camera centers)
        dst_points: Nx3 array of destination points (HLoc camera centers)
        
    Returns:
        4x4 transformation matrix [c*R | t; 0 0 0 1]
    
    Reference: Umeyama, "Least-squares estimation of transformation parameters
               between two point patterns", PAMI 1991
    """
    assert src_points.shape == dst_points.shape
    n, m = src_points.shape
    assert m == 3, "Points must be 3D"
    
    # Compute centroids
    src_mean = np.mean(src_points, axis=0)
    dst_mean = np.mean(dst_points, axis=0)
    
    # Center the point sets
    src_centered = src_points - src_mean
    dst_centered = dst_points - dst_mean
    
    # Compute covariance matrix
    H = src_centered.T @ dst_centered / n
    
    # SVD
    U, S, Vt = np.linalg.svd(H)
    
    # Compute rotation
    R = Vt.T @ U.T
    
    # Handle reflection case (ensure det(R) = 1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # Compute scale
    src_var = np.var(src_points, axis=0).sum()
    scale = np.trace(np.diag(S)) / src_var if src_var > 1e-10 else 1.0
    
    # Compute translation
    t = dst_mean - scale * R @ src_mean
    
    # Construct 4x4 transformation matrix
    T = np.eye(4)
    T[:3, :3] = scale * R
    T[:3, 3] = t
    
    return T


def compute_alignment_errors(src_poses, dst_poses, transform, common_names):
    """
    Compute alignment errors for camera centers and orientations.
    
    Args:
        src_poses: dict of source poses (ARKit)
        dst_poses: dict of destination poses (HLoc)
        transform: 4x4 alignment matrix
        common_names: list of common image names
        
    Returns:
        dict with error statistics
    """
    position_errors = []
    rotation_errors = []
    
    for name in common_names:
        # Position error
        src_center = src_poses[name][:3, 3]
        dst_center = dst_poses[name][:3, 3]
        
        src_center_hom = np.append(src_center, 1.0)
        src_center_aligned = (transform @ src_center_hom)[:3]
        
        pos_error = np.linalg.norm(src_center_aligned - dst_center)
        position_errors.append(pos_error)
        
        # Rotation error (angular difference in degrees)
        src_pose_aligned = transform @ src_poses[name]
        R_src = src_pose_aligned[:3, :3]
        R_dst = dst_poses[name][:3, :3]
        
        # Normalize to pure rotation (remove any scale)
        R_src = R_src / np.linalg.det(R_src)**(1/3)
        R_dst = R_dst / np.linalg.det(R_dst)**(1/3)
        
        R_diff = R_dst.T @ R_src
        trace = np.clip((np.trace(R_diff) - 1) / 2, -1, 1)
        angle_error = np.arccos(trace) * 180 / np.pi
        rotation_errors.append(angle_error)
    
    return {
        'position': {
            'mean': np.mean(position_errors),
            'median': np.median(position_errors),
            'max': np.max(position_errors),
            'std': np.std(position_errors),
            'values': position_errors
        },
        'rotation': {
            'mean': np.mean(rotation_errors),
            'median': np.median(rotation_errors),
            'max': np.max(rotation_errors),
            'std': np.std(rotation_errors),
            'values': rotation_errors
        }
    }


def localize_and_update_json(
    pre_sfm_dir,
    post_image_dir,
    arkit_transforms_path,
    new_transforms_path,
    num_retrieval=20,
    ransac_thresh=10.0,
    align_arkit=False,
    visualize=False
):
    """
    Localize post-change images and align ARKit trajectory to pre-change SfM frame.
    
    Two modes:
    1. align_arkit=False: Replace ARKit poses with HLoc poses (only for localized frames)
    2. align_arkit=True: Compute alignment transform and apply to ALL ARKit poses
    """
    pre_sfm_dir = Path(pre_sfm_dir)
    post_image_dir = Path(post_image_dir)
    arkit_transforms_path = Path(arkit_transforms_path)
    new_transforms_path = Path(new_transforms_path)

    # Load reconstruction and camera
    print("=" * 60)
    print("Loading pre-change reconstruction...")
    print("=" * 60)
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
    print("\n" + "=" * 60)
    print("STEP 1: Extract local features")
    print("=" * 60)
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

    print("\n" + "=" * 60)
    print("STEP 2: Extract global descriptors")
    print("=" * 60)
    global_conf = extract_features.confs["netvlad"]
    extract_features.main(
        conf=global_conf,
        image_dir=post_image_dir,
        export_dir=post_outputs,
        feature_path=post_global
    )

    # === Step 3: Image retrieval ===
    print("\n" + "=" * 60)
    print("STEP 3: Retrieve similar pre-change images")
    print("=" * 60)
    pairs_from_retrieval.main(
        descriptors=post_global,
        output=pairs_file,
        num_matched=num_retrieval,
        db_descriptors=pre_global,
        db_model=pre_model
    )

    # === Step 4: Feature matching ===
    print("\n" + "=" * 60)
    print("STEP 4: Match features")
    print("=" * 60)
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
    print("\n" + "=" * 60)
    print("STEP 5: Localize post-change images")
    print("=" * 60)
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
    hloc_poses = {}  # Stores HLoc-localized poses in SfM frame
    failed = []
    inlier_counts = {}
    
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

        # Convert pose: COLMAP convention to OpenCV/NeRF convention
        qvec, tvec = ret['qvec'], ret['tvec']
        
        # COLMAP quaternion is [w, x, y, z], scipy expects [x, y, z, w]
        R = Rotation.from_quat([qvec[1], qvec[2], qvec[3], qvec[0]]).as_matrix()
        
        # Construct world-to-camera
        w2c = np.eye(4)
        w2c[:3, :3] = R
        w2c[:3, 3] = tvec
        
        # Invert to get camera-to-world
        c2w = np.linalg.inv(w2c)
        
        # Apply coordinate frame fix (COLMAP uses [right, down, forward], convert to [right, up, back])
        fix_rot = np.diag([1, -1, -1, 1])
        hloc_poses[query_name] = c2w @ fix_rot
        inlier_counts[query_name] = ret['num_inliers']
        
        print(f"✓ {query_name}: {ret['num_inliers']} inliers")

    print("\n" + "=" * 60)
    print("LOCALIZATION RESULTS")
    print("=" * 60)
    print(f"Successfully localized: {len(hloc_poses)}/{len(post_images)} images")
    if failed:
        print(f"Failed: {len(failed)} images")
        if len(failed) <= 10:
            print(f"Failed images: {failed}")
    
    if inlier_counts:
        inlier_values = list(inlier_counts.values())
        print(f"\nInlier statistics:")
        print(f"  Mean: {np.mean(inlier_values):.1f}")
        print(f"  Median: {np.median(inlier_values):.1f}")
        print(f"  Min: {np.min(inlier_values)}")
        print(f"  Max: {np.max(inlier_values)}")

    # Load ARKit transforms
    with open(arkit_transforms_path, "r") as f:
        old_json = json.load(f)

    # Build ARKit pose dictionary
    arkit_poses = {}
    for frame in old_json["frames"]:
        name = Path(frame["file_path"]).name
        arkit_poses[name] = np.array(frame["transform_matrix"])

    # === Step 6: Alignment or Direct Replacement ===
    if align_arkit:
        print("\n" + "=" * 60)
        print("STEP 6: Align ARKit trajectory to SfM frame using Umeyama")
        print("=" * 60)
        
        # Find common frames
        common_names = sorted(set(arkit_poses.keys()) & set(hloc_poses.keys()))
        
        if len(common_names) < 3:
            print(f"ERROR: Need at least 3 matched poses for alignment, got {len(common_names)}")
            print("Falling back to direct pose replacement for localized frames only")
            align_arkit = False
        else:
            print(f"Computing alignment from {len(common_names)} matched poses...")
            
            # Extract camera centers
            arkit_centers = np.array([arkit_poses[name][:3, 3] for name in common_names])
            hloc_centers = np.array([hloc_poses[name][:3, 3] for name in common_names])
            
            # Compute Umeyama alignment
            alignment_transform = umeyama_alignment(arkit_centers, hloc_centers)
            
            print("\nEstimated ARKit → SfM alignment transform:")
            print(alignment_transform)
            
            # Extract scale from transform
            scale = np.linalg.det(alignment_transform[:3, :3])**(1/3)
            print(f"\nEstimated scale factor: {scale:.6f}")
            
            # Compute and report errors
            errors = compute_alignment_errors(arkit_poses, hloc_poses, alignment_transform, common_names)
            
            print(f"\nAlignment error statistics:")
            print(f"Position errors (meters):")
            print(f"  Mean:   {errors['position']['mean']:.4f}")
            print(f"  Median: {errors['position']['median']:.4f}")
            print(f"  Max:    {errors['position']['max']:.4f}")
            print(f"  Std:    {errors['position']['std']:.4f}")
            
            print(f"\nRotation errors (degrees):")
            print(f"  Mean:   {errors['rotation']['mean']:.2f}")
            print(f"  Median: {errors['rotation']['median']:.2f}")
            print(f"  Max:    {errors['rotation']['max']:.2f}")
            print(f"  Std:    {errors['rotation']['std']:.2f}")
            
            # Apply alignment to ALL ARKit poses
            for frame in old_json["frames"]:
                T_arkit = np.array(frame["transform_matrix"])
                T_aligned = alignment_transform @ T_arkit
                frame["transform_matrix"] = T_aligned.tolist()
            
            # Update hloc_poses for visualization (show aligned ARKit for all frames)
            aligned_poses = {}
            for name, T_arkit in arkit_poses.items():
                aligned_poses[name] = alignment_transform @ T_arkit
            
            print(f"\n✓ Applied alignment to all {len(old_json['frames'])} frames")

    if not align_arkit:
        # Direct replacement: update only localized frames
        print("\n" + "=" * 60)
        print("STEP 6: Replace ARKit poses with HLoc poses (localized frames only)")
        print("=" * 60)
        
        file_to_pose = {f"frames/{name}": mat.tolist() for name, mat in hloc_poses.items()}
        
        updated = 0
        for frame in old_json["frames"]:
            fp = frame["file_path"]
            if fp in file_to_pose:
                frame["transform_matrix"] = file_to_pose[fp]
                updated += 1
        
        print(f"Updated {updated} frames with HLoc poses")
        aligned_poses = hloc_poses  # For visualization

    # Save results
    with open(new_transforms_path, "w") as f:
        json.dump(old_json, f, indent=2)
    
    print(f"\n✓ Saved updated transforms to: {new_transforms_path}")

    # === Step 7: Visualization ===
    if visualize:
        print("\n" + "=" * 60)
        print("STEP 7: Generate 3D visualization")
        print("=" * 60)
        
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

        if aligned_poses:
            camera_centers = np.array([pose[:3, 3] for pose in aligned_poses.values()])
            
            fig.add_trace(
                go.Scatter3d(
                    x=camera_centers[:, 0],
                    y=camera_centers[:, 1],
                    z=camera_centers[:, 2],
                    mode='markers',
                    marker=dict(size=4, color='blue'),
                    name='post-change cameras (aligned)'
                )
            )
            
            # Add camera direction arrows (sample subset)
            n_arrows = min(20, len(aligned_poses))
            sample_indices = np.linspace(0, len(aligned_poses)-1, n_arrows, dtype=int)
            pose_list = list(aligned_poses.values())
            
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
            
            # Highlight successfully localized cameras
            if hloc_poses:
                hloc_centers = np.array([hloc_poses[name][:3, 3] for name in hloc_poses.keys()])
                fig.add_trace(
                    go.Scatter3d(
                        x=hloc_centers[:, 0],
                        y=hloc_centers[:, 1],
                        z=hloc_centers[:, 2],
                        mode='markers',
                        marker=dict(size=6, color='lime', symbol='diamond'),
                        name=f'HLoc localized ({len(hloc_poses)})'
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
            title=f"Camera Localization: Pre-change (red) vs Post-change (blue)<br>"
                  f"{'Umeyama Alignment' if align_arkit else 'Direct Replacement'}"
        )

        html_path = post_outputs / "localization_viz.html"
        fig.write_html(str(html_path), auto_open=True)
        print(f"✓ 3D visualization saved to: {html_path}")
    
    return hloc_poses, old_json


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Localize post-change images against a pre-change COLMAP reconstruction and update transforms.json.",
        formatter_class=argparse.RawDescriptionHelpFormatter
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
                        help="RANSAC reprojection threshold in pixels (default: 12.0)")
    parser.add_argument("--align_arkit", action="store_true",
                        help="Align entire ARKit trajectory to SfM frame using Umeyama (recommended)")
    parser.add_argument("--visualize", action="store_true",
                        help="Generate 3D visualization of localization results")
    
    args = parser.parse_args()

    poses, updated_json = localize_and_update_json(
        pre_sfm_dir=args.pre_sfm_dir,
        post_image_dir=args.post_image_dir,
        arkit_transforms_path=args.arkit_transforms_path,
        new_transforms_path=args.new_transforms_path,
        num_retrieval=args.num_retrieval,
        ransac_thresh=args.ransac_thresh,
        align_arkit=args.align_arkit,
        visualize=args.visualize
    )

    print("\n" + "=" * 60)
    print("✓ COMPLETE!")
    print("=" * 60)
    print(f"Updated transforms saved to: {args.new_transforms_path}")
    print(f"Mode: {'Umeyama Alignment (all frames)' if args.align_arkit else 'Direct Replacement (localized only)'}")