"""
Refine ARKit Poses with HLOC Triangulation & Bundle Adjustment

This script:
1. Loads ARKit poses from transforms.json (NeRF format)
2. Creates a reference COLMAP model from ARKit poses
3. Uses HLOC features/matches to triangulate 3D points
4. Runs bundle adjustment to refine poses while preserving metric scale
5. Detects and smooths ARKit pose spikes
6. Exports refined reconstruction

Usage:
    python refine_arkit_poses.py --data_dir /path/to/data --output_dir refined_outputs
"""

import argparse
import json
import logging
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from numpy.linalg import norm
from scipy.spatial.transform import Rotation

import os
os.environ["GLOG_logtostderr"] = "1"
os.environ["GLOG_minloglevel"] = "2"

import pyceres
import pycolmap

from hloc import extract_features, match_features, pairs_from_sequential
from hloc.triangulation import create_db_from_model, import_features, import_matches

def setup_logging(output_dir: Path):
    """Setup logging configuration."""
    log_file = output_dir / "refinement.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger('refine_arkit')


def parse_args():
    parser = argparse.ArgumentParser(
        description="Refine ARKit poses with HLOC triangulation and bundle adjustment"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to data directory containing frames/ and transforms.json"
    )
    parser.add_argument(
        "--transforms",
        type=str,
        default="transforms_arkit.json",
        help="Transform JSON file name (default: transforms.json)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="refined_outputs",
        help="Output directory name (default: refined_outputs)"
    )
    parser.add_argument(
        "--skip_features",
        action="store_true",
        help="Skip feature extraction/matching if already done"
    )
    parser.add_argument(
        "--max_keypoints",
        type=int,
        default=1024,
        help="Maximum keypoints per image (default: 1024)"
    )
    parser.add_argument(
        "--num_ba_iterations",
        type=int,
        default=4,
        help="Number of bundle adjustment iterations (default: 4)"
    )
    parser.add_argument(
        "--spike_threshold",
        type=float,
        default=0.3,
        help="Translation threshold for spike detection in meters (default: 0.3)"
    )
    return parser.parse_args()


def create_reference_model_from_transforms(
    transforms_file: Path,
    image_dir: Path,
    logger: logging.Logger
) -> pycolmap.Reconstruction:
    """Create COLMAP reference model from NeRF transforms.json."""
    
    logger.info(f"Loading transforms from: {transforms_file}")
    with open(transforms_file, 'r') as f:
        data = json.load(f)
    
    # Get camera parameters
    w = int(data.get('w', 1920))
    h = int(data.get('h', 1440))
    fx = float(data.get('fl_x', w))
    fy = float(data.get('fl_y', w))
    cx = float(data.get('cx', w / 2))
    cy = float(data.get('cy', h / 2))
    
    logger.info(f"Camera parameters: {w}x{h}, fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")
    
    # Create temporary directory for text format
    temp_dir = Path("temp_colmap_model")
    temp_dir.mkdir(exist_ok=True)
    
    # Write cameras.txt
    with open(temp_dir / "cameras.txt", 'w') as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"1 PINHOLE {w} {h} {fx} {fy} {cx} {cy}\n")
    
    # Write images.txt
    image_names = []
    with open(temp_dir / "images.txt", 'w') as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("# POINTS2D[] as (X, Y, POINT3D_ID)\n")
        
        for idx, frame in enumerate(data['frames'], start=1):
            img_path = Path(frame['file_path'])
            img_name = img_path.name
            full_img_path = image_dir / img_name
            
            if not full_img_path.exists():
                logger.warning(f"Image not found: {full_img_path}")
                continue
            
            image_names.append(img_name)
            
            # Convert camera-to-world to world-to-camera
            c2w = np.array(frame['transform_matrix'])
            w2c = np.linalg.inv(c2w)
            
            R = w2c[:3, :3]
            t = w2c[:3, 3]
            
            # Convert rotation to quaternion (COLMAP uses qw, qx, qy, qz order)
            quat = Rotation.from_matrix(R).as_quat()  # Returns [qx, qy, qz, qw]
            qw, qx, qy, qz = quat[3], quat[0], quat[1], quat[2]
            
            f.write(f"{idx} {qw} {qx} {qy} {qz} {t[0]} {t[1]} {t[2]} 1 {img_name}\n")
            f.write("\n")  # Empty line for POINTS2D
    
    # Create empty points3D.txt
    with open(temp_dir / "points3D.txt", 'w') as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
    
    # Load reconstruction from text format
    reconstruction = pycolmap.Reconstruction(str(temp_dir))
    logger.info(f"Created reference model with {len(reconstruction.images)} images")
    
    # Clean up temporary directory
    shutil.rmtree(temp_dir)
    
    return reconstruction, image_names


def detect_and_smooth_pose_spikes(
    reconstruction: pycolmap.Reconstruction,
    spike_threshold: float,
    logger: logging.Logger
) -> Dict[int, Dict]:
    """
    Detect and smooth ARKit pose spikes (sudden jumps from loop closure).
    Returns dictionary with spike info per image.
    """
    
    arkit_info = {}
    prev_pose = None
    prev_pose_smoothed = None
    prev_diff_pose = None
    has_spikes = False
    new_cam_from_world = {}
    
    for image_id in sorted(reconstruction.images.keys()):
        image = reconstruction.images[image_id]
        pose = image.cam_from_world.inverse()  # world-to-camera -> camera-to-world
        
        arkit_info[image_id] = {
            "arkit_spike": False,
            "cam_from_world": deepcopy(image.cam_from_world)
        }
        
        if prev_pose is None:
            prev_pose = pose
            prev_pose_smoothed = pose
            prev_diff_pose = pycolmap.Rigid3d()
            new_cam_from_world[image_id] = image.cam_from_world
            continue
        
        # Compute relative pose
        diff_pose = prev_pose.inverse() * pose
        
        # Check for spike (sudden large translation change)
        if prev_diff_pose is not None:
            trans_diff = norm(diff_pose.translation - prev_diff_pose.translation)
            
            if trans_diff > spike_threshold:
                logger.info(f"SPIKE detected at image {image_id}: translation diff = {trans_diff:.4f}m")
                logger.info(f"  Previous diff: {prev_diff_pose.translation}")
                logger.info(f"  Current diff: {diff_pose.translation}")
                
                # Smooth the spike: use previous differential
                diff_pose.rotation = pycolmap.Rotation3d()  # Reset rotation
                diff_pose.translation = np.array(prev_diff_pose.translation)
                
                # Limit maximum movement
                if norm(diff_pose.translation) > 0.05:
                    diff_pose.translation *= 0.05 / norm(diff_pose.translation)
                
                logger.info(f"  Smoothed diff: {diff_pose.translation}")
                arkit_info[image_id]["arkit_spike"] = True
                has_spikes = True
        
        # Update smoothed pose
        pose_smoothed = prev_pose_smoothed * diff_pose
        prev_pose_smoothed = pose_smoothed
        prev_pose = pose
        prev_diff_pose = diff_pose
        new_cam_from_world[image_id] = pose_smoothed.inverse()
    
    # Apply smoothed poses if spikes were detected
    if has_spikes:
        logger.info(f"Applying smoothed poses to {len(new_cam_from_world)} images")
        for image_id, image in reconstruction.images.items():
            if image_id in new_cam_from_world:
                image.cam_from_world = new_cam_from_world[image_id]
                arkit_info[image_id]["cam_from_world"] = deepcopy(image.cam_from_world)
    
    return arkit_info


def run_triangulation(
    database_path: Path,
    image_dir: Path,
    reference_model: pycolmap.Reconstruction,
    mapper_options: Dict[str, Any],
    num_ba_iterations: int,
    logger: logging.Logger
) -> pycolmap.Reconstruction:
    """Run triangulation and bundle adjustment."""
    
    min_num_matches = 15
    ignore_watermarks = True
    image_names = set()
    
    database = pycolmap.Database(database_path)
    database_cache = pycolmap.DatabaseCache.create(
        database, min_num_matches, ignore_watermarks, image_names
    )
    
    # Create reconstruction from reference model
    reconstruction = pycolmap.Reconstruction()
    for img in reference_model.images.values():
        if database_cache.exists_image(img.image_id):
            reconstruction.add_image(img)
    for cam in reference_model.cameras.values():
        if database_cache.exists_camera(cam.camera_id):
            reconstruction.add_camera(cam)
    
    logger.info(f"Starting triangulation with {len(reconstruction.images)} images")
    
    # Initialize mapper
    mapper = pycolmap.IncrementalMapper(database_cache)
    mapper.begin_reconstruction(reconstruction)
    
    # Triangulation options
    tri_options = pycolmap.IncrementalTriangulatorOptions()
    tri_options.re_min_ratio = 0.8
    tri_options.re_max_angle_error = 8.0
    tri_options.re_max_trials = 3
    
    # Triangulate each image
    for image_id in reconstruction.reg_image_ids():
        try:
            mapper.triangulate_image(tri_options, image_id)
        except IndexError as e:
            logger.error(f"Error triangulating image {image_id}: {e}")
            continue
    
    mapper.complete_and_merge_tracks(tri_options)
    logger.info(f"Initial triangulation: {reconstruction.num_points3D()} 3D points")
    
    # Bundle adjustment options
    ba_options = pycolmap.BundleAdjustmentOptions()
    ba_options.refine_focal_length = False
    ba_options.refine_principal_point = False
    ba_options.refine_extra_params = False
    ba_options.refine_extrinsics = True
    ba_options.solver_options.max_num_iterations = 100
    ba_options.solver_options.gradient_tolerance = 1.0
    ba_options.solver_options.logging_type = pyceres.LoggingType.SILENT
    ba_options.solver_options.minimizer_progress_to_stdout = False
    
    sorted_image_ids = sorted(reconstruction.reg_image_ids())
    retriangulated = False
    ba_iterations_remaining = num_ba_iterations
    
    # Bundle adjustment loop
    while ba_iterations_remaining > 0:
        mapper.observation_manager.filter_observations_with_negative_depth()
        num_observations = reconstruction.compute_num_observations()
        
        logger.info(f"Bundle adjustment iteration {num_ba_iterations - ba_iterations_remaining + 1}/{num_ba_iterations}")
        
        # Configure bundle adjustment
        ba_config = pycolmap.BundleAdjustmentConfig()
        for image_id in sorted_image_ids:
            ba_config.add_image(image_id)
        
        # Fix 7-DOF (fix first pose completely, fix one translation component of second pose)
        ba_config.set_constant_cam_pose(sorted_image_ids[0])
        ba_config.set_constant_cam_positions(sorted_image_ids[1], [0])
        
        # Run bundle adjustment
        summary = pycolmap.bundle_adjustment(reconstruction, ba_options)
        print(summary)
        # summary = pycolmap.bundle_adjustment(reconstruction, ba_options)

        
        # logger.info(f"BA Summary: {summary.FullReport()}")
        
        # Filter and merge
        num_changed = 0
        num_changed += mapper.complete_and_merge_tracks(tri_options)
        num_changed += mapper.filter_points(mapper_options)
        
        changed_ratio = num_changed / num_observations if num_observations > 0 else 0
        logger.info(f"Changed observations: {changed_ratio:.4f}")
        
        ba_iterations_remaining -= 1
        
        # Retriangulate after first successful BA
        if not retriangulated:
            logger.info("Retriangulating...")
            num_retriangulated = mapper.retriangulate(tri_options)
            logger.info(f"Retriangulated {num_retriangulated} observations")
            retriangulated = True
            
            # Ensure at least 2 more iterations after retriangulation
            additional_iterations = max(0, 2 - ba_iterations_remaining)
            num_ba_iterations += additional_iterations
            ba_iterations_remaining += additional_iterations
    
    # Extract colors
    logger.info("Extracting colors from images...")
    reconstruction.extract_colors_for_all_images(image_dir)
    
    mapper.end_reconstruction(False)
    
    return reconstruction


def process_features_and_matching(
    image_names: list,
    image_dir: Path,
    output_dir: Path,
    max_keypoints: int,
    logger: logging.Logger
):
    """Extract features and match them."""
    
    features_path = output_dir / "features.h5"
    matches_path = output_dir / "matches.h5"
    pairs_path = output_dir / "pairs.txt"
    global_features_path = output_dir / "global_features.h5"
    
    # Feature extraction configuration
    feature_conf = extract_features.confs["aliked-n16"]
    feature_conf["model"]["max_num_keypoints"] = max_keypoints
    feature_conf["model"]["detection_threshold"] = 0.3
    feature_conf["model"]["nms_radius"] = 4
    feature_conf["preprocessing"]["resize_max"] = 1024
    feature_conf["output"] = features_path
    
    logger.info(f"Extracting features with max {max_keypoints} keypoints per image")
    extract_features.main(
        feature_conf,
        image_dir,
        output_dir,
        feature_path=features_path,
        as_half=True,
        image_list=image_names
    )
    
    # Global features for loop closure
    logger.info("Extracting global descriptors for loop closure")
    global_feature_conf = extract_features.confs["eigenplaces"]
    global_feature_conf["output"] = global_features_path
    extract_features.main(
        global_feature_conf,
        image_dir,
        output_dir,
        feature_path=global_features_path,
        as_half=True,
        image_list=image_names[::5]  # Sample every 5th image
    )
    
    # Generate pairs with loop closure
    logger.info("Generating image pairs with sequential + loop closure")
    pairs_from_sequential.main(
        pairs_path,
        image_names,
        features=None,
        window_size=3,
        quadratic_overlap=True,
        use_loop_closure=True,
        retrieval_path=global_features_path,
        retrieval_interval=5,
        num_loc=10,
        min_retrieval_distance=50
    )
    
    # Feature matching
    logger.info("Matching features")
    matcher_conf = match_features.confs["aliked+lightglue"]
    matcher_conf["model"]["compile_network"] = False
    match_features.main(
        matcher_conf,
        pairs_path,
        features=features_path,
        matches=matches_path
    )
    
    return features_path, matches_path, pairs_path


def triangulate_with_known_poses(
    sfm_dir: Path,
    reference_model_path: Path,
    image_dir: Path,
    pairs_path: Path,
    features_path: Path,
    matches_path: Path,
    mapper_options: Dict[str, Any],
    num_ba_iterations: int,
    logger: logging.Logger
) -> pycolmap.Reconstruction:
    """Triangulate 3D points using known camera poses."""
    
    sfm_dir.mkdir(parents=True, exist_ok=True)
    database_path = sfm_dir / "database.db"
    
    # Load reference model
    reference_model = pycolmap.Reconstruction(reference_model_path)
    
    # Create database from model
    logger.info("Creating database from reference model")
    image_ids = create_db_from_model(reference_model, database_path)
    
    # Import features and matches
    logger.info("Importing features and matches")
    import_features(image_ids, database_path, features_path)
    import_matches(
        image_ids,
        database_path,
        pairs_path,
        matches_path,
        min_match_score=None,
        skip_geometric_verification=True
    )
    
    # Run triangulation and bundle adjustment
    reconstruction = run_triangulation(
        database_path,
        image_dir,
        reference_model,
        mapper_options,
        num_ba_iterations,
        logger
    )
    
    logger.info(f"Final reconstruction: {reconstruction.summary()}")
    return reconstruction


def export_refined_transforms(
    reconstruction: pycolmap.Reconstruction,
    original_transforms: Path,
    output_path: Path,
    logger: logging.Logger
):
    """Export refined poses back to NeRF transforms.json format."""
    
    with open(original_transforms, 'r') as f:
        data = json.load(f)
    
    # Create mapping from image name to frame index
    name_to_frame = {}
    for idx, frame in enumerate(data['frames']):
        img_path = Path(frame['file_path'])
        name_to_frame[img_path.name] = idx
    
    # Update poses
    for image in reconstruction.images.values():
        if image.name in name_to_frame:
            frame_idx = name_to_frame[image.name]
            
            # Convert world-to-camera back to camera-to-world
            w2c = image.cam_from_world
            c2w = w2c.inverse()
            
            # Convert to 4x4 matrix
            transform_matrix = np.eye(4)
            transform_matrix[:3, :3] = c2w.rotation.matrix()
            transform_matrix[:3, 3] = c2w.translation
            
            data['frames'][frame_idx]['transform_matrix'] = transform_matrix.tolist()
    
    # Save refined transforms
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    logger.info(f"Saved refined transforms to: {output_path}")


def main():
    args = parse_args()
    
    # Setup paths
    data_dir = Path(args.data_dir)
    output_dir = data_dir / args.output_dir
    output_dir.mkdir(exist_ok=True, parents=True)
    
    image_dir = data_dir / "frames"
    transforms_file = data_dir / args.transforms
    
    assert image_dir.exists(), f"Image directory not found: {image_dir}"
    assert transforms_file.exists(), f"Transforms file not found: {transforms_file}"
    
    # Setup logging
    logger = setup_logging(output_dir)
    logger.info("=" * 80)
    logger.info("ARKit Pose Refinement with HLOC")
    logger.info("=" * 80)
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Output directory: {output_dir}")
    
    # Step 1: Create reference model from ARKit poses
    logger.info("\n=== Step 1: Creating reference model from ARKit poses ===")
    reference_model, image_names = create_reference_model_from_transforms(
        transforms_file, image_dir, logger
    )
    
    # Step 2: Detect and smooth pose spikes
    logger.info("\n=== Step 2: Detecting and smoothing pose spikes ===")
    arkit_info = detect_and_smooth_pose_spikes(
        reference_model, args.spike_threshold, logger
    )
    
    # Save reference model
    ref_model_path = output_dir / "reference_model"
    ref_model_path.mkdir(exist_ok=True)
    reference_model.write_text(str(ref_model_path))
    logger.info(f"Reference model saved to: {ref_model_path}")
    
    # Step 3: Feature extraction and matching
    if not args.skip_features:
        logger.info("\n=== Step 3: Feature extraction and matching ===")
        features_path, matches_path, pairs_path = process_features_and_matching(
            image_names, image_dir, output_dir, args.max_keypoints, logger
        )
    else:
        logger.info("\n=== Step 3: Skipping feature extraction (using existing) ===")
        features_path = output_dir / "features.h5"
        matches_path = output_dir / "matches.h5"
        pairs_path = output_dir / "pairs.txt"
    
    # Step 4: Triangulation and bundle adjustment
    logger.info("\n=== Step 4: Triangulation and bundle adjustment ===")
    sfm_dir = output_dir / "sfm"
    mapper_options = pycolmap.IncrementalMapperOptions()
    
    refined_reconstruction = triangulate_with_known_poses(
        sfm_dir,
        ref_model_path,
        image_dir,
        pairs_path,
        features_path,
        matches_path,
        mapper_options,
        args.num_ba_iterations,
        logger
    )
    
    # Step 5: Export results
    logger.info("\n=== Step 5: Exporting results ===")
    
    # Save COLMAP model
    final_model_path = output_dir / "refined_model"
    final_model_path.mkdir(exist_ok=True)
    refined_reconstruction.write_text(str(final_model_path))
    logger.info(f"Refined COLMAP model saved to: {final_model_path}")
    
    # Export sparse point cloud
    ply_path = output_dir / "sparse_pointcloud.ply"
    refined_reconstruction.export_PLY(str(ply_path))
    logger.info(f"Sparse point cloud exported to: {ply_path}")
    
    # Export refined transforms
    refined_transforms_path = output_dir / "transforms_refined.json"
    export_refined_transforms(
        refined_reconstruction,
        transforms_file,
        refined_transforms_path,
        logger
    )
    
    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("REFINEMENT COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Number of images: {len(refined_reconstruction.images)}")
    logger.info(f"Number of 3D points: {refined_reconstruction.num_points3D()}")
    logger.info(f"Refined poses saved to: {refined_transforms_path}")
    logger.info(f"All outputs saved to: {output_dir}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()