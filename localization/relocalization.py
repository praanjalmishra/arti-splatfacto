"""
Query image relocalization pipeline.
Localizes post-change images in pre-change coordinate frame using PnP.
"""

from pathlib import Path
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
from collections import defaultdict

import numpy as np
import h5py
import pycolmap
from tqdm import tqdm

from hloc.localize_sfm import QueryLocalizer
from hloc.utils.io import get_keypoints

from .config import RelocalizationConfig, SceneConfig, FeatureConfig, LocalizationConfig
from .io_utils import (
    load_arkit_transforms,
    load_colmap_model,
    save_transforms,
    get_camera_intrinsics,
    create_pycolmap_camera,
    get_image_name_to_id
)
from .features import (
    extract_all_features,
    match_for_localization,
    get_image_list
)
from .pose_utils import colmap_to_opengl_pose


@dataclass
class LocalizationResult:
    """Results from relocalization pipeline."""
    
    reference_model_path: Path
    query_features_path: Path
    query_global_desc_path: Path
    matches_path: Path
    pairs_path: Path
    output_transforms_path: Path
    
    num_query_images: int
    num_localized: int
    num_failed: int
    failure_reasons: Dict[str, int]
    localized_poses: Dict[str, Dict]  # image_name -> {qvec, tvec, num_inliers}
    
    @property
    def success_rate(self) -> float:
        """Localization success rate."""
        return self.num_localized / self.num_query_images if self.num_query_images > 0 else 0.0
    
    def summary(self) -> str:
        """Return a summary string of the localization."""
        summary = (
            f"Relocalization Summary:\n"
            f"  Query Images: {self.num_query_images}\n"
            f"  Successfully Localized: {self.num_localized} ({self.success_rate:.1%})\n"
            f"  Failed: {self.num_failed}\n"
        )
        
        if self.failure_reasons:
            summary += "\nTop Failure Reasons:\n"
            for reason, count in sorted(self.failure_reasons.items(), key=lambda x: -x[1])[:3]:
                summary += f"  - {reason}: {count}\n"
        
        summary += f"\nOutput: {self.output_transforms_path}"
        return summary


def names_to_pair(name0: str, name1: str) -> str:
    """Convert two image names to HLoc pair format."""
    return '_'.join((name0.replace('/', '-'), name1.replace('/', '-')))


def localize_single_image(
    query_name: str,
    query_keypoints: np.ndarray,
    retrieval_pairs: Dict[str, List[str]],
    matches_h5: h5py.File,
    reference_reconstruction: pycolmap.Reconstruction,
    reference_name_to_id: Dict[str, int],
    localizer: QueryLocalizer,
    camera: pycolmap.Camera
) -> Tuple[Optional[Dict], str]:
    """
    Localize a single query image using PnP.
    
    Args:
        query_name: Query image name
        query_keypoints: Query image keypoints (Nx2)
        retrieval_pairs: Dict mapping query names to reference names
        matches_h5: Open h5py file with matches
        reference_reconstruction: Reference COLMAP reconstruction
        reference_name_to_id: Mapping from reference image name to ID
        localizer: QueryLocalizer instance
        camera: Camera model
    
    Returns:
        Tuple of (result_dict, status_message)
        result_dict is None if localization failed
    """
    # COLMAP coordinate convention
    kpq = query_keypoints + 0.5
    
    # Get retrieved reference images
    if query_name not in retrieval_pairs:
        return None, "No retrieval pairs"
    
    ref_names = retrieval_pairs[query_name]
    ref_ids = [reference_name_to_id[name] for name in ref_names if name in reference_name_to_id]
    
    if len(ref_ids) == 0:
        return None, "No valid reference IDs"
    
    # Collect 2D-3D correspondences
    kp_idx_to_3D = defaultdict(list)
    total_matches = 0
    valid_matches = 0
    
    for ref_id in ref_ids:
        ref_image = reference_reconstruction.images[ref_id]
        
        if ref_image.num_points3D() == 0:
            continue
        
        # Get 3D point IDs for this reference image
        points3D_ids = np.array([
            p.point3D_id if p.has_point3D() else -1
            for p in ref_image.points2D
        ])
        
        # Try different pair naming conventions
        pair_names = [
            f'{query_name}/{ref_image.name}',
            f'{query_name}_{ref_image.name}',
            names_to_pair(query_name, ref_image.name)
        ]
        
        matches = None
        for pair_name in pair_names:
            if pair_name in matches_h5:
                group = matches_h5[pair_name]
                # Try matches0 first (standard HLoc format)
                if 'matches0' in group:
                    matches = group['matches0'][:]
                elif 'matches' in group:
                    matches = group['matches'][:]
                
                if matches is not None and len(matches) > 0:
                    total_matches += len(matches)
                    break
        
        if matches is None or len(matches) == 0:
            continue
        
        # Filter matches with valid 3D points
        # matches format: [N] where value is ref_idx or -1 for no match
        if matches.ndim == 1:
            valid_indices = np.where(matches != -1)[0]
            if len(valid_indices) == 0:
                continue
            matches_filtered = np.stack([valid_indices, matches[valid_indices]], axis=1)
        else:  # [N, 2] format
            matches_filtered = matches
        
        # Check which matches have valid 3D points
        valid_mask = points3D_ids[matches_filtered[:, 1]] != -1
        matches_filtered = matches_filtered[valid_mask]
        
        valid_matches += len(matches_filtered)
        
        # Store 2D-3D correspondences
        for query_idx, ref_idx in matches_filtered:
            point3D_id = points3D_ids[ref_idx]
            if point3D_id not in kp_idx_to_3D[query_idx]:
                kp_idx_to_3D[query_idx].append(point3D_id)
    
    # Need at least 4 points for PnP
    if len(kp_idx_to_3D) < 4:
        return None, f"Insufficient 2D-3D: {len(kp_idx_to_3D)} (matches: {total_matches}, valid: {valid_matches})"
    
    # Prepare data for PnP
    mkp_idxs = [i for i in kp_idx_to_3D.keys() for _ in kp_idx_to_3D[i]]
    mp3d_ids = [j for i in kp_idx_to_3D.keys() for j in kp_idx_to_3D[i]]
    
    # Run PnP localization
    ret = localizer.localize(kpq, mkp_idxs, mp3d_ids, camera)
    
    if not ret['success']:
        return None, f"PnP failed with {len(kp_idx_to_3D)} 2D-3D correspondences"
    
    result = {
        'qvec': ret['qvec'],
        'tvec': ret['tvec'],
        'num_inliers': ret['num_inliers']
    }
    
    return result, f"Success: {ret['num_inliers']}/{len(mkp_idxs)} inliers"


def run_relocalization(
    pre_dir: Path,
    post_dir: Path,
    config: Optional[RelocalizationConfig] = None
) -> LocalizationResult:
    """
    Run complete relocalization pipeline.
    
    Pipeline:
    1. Extract features from query (post) images
    2. Retrieve similar reference (pre) images
    3. Match features between query and reference
    4. Localize query images using PnP+RANSAC
    5. Save query poses in reference coordinate frame
    
    Args:
        pre_dir: Path to reference (pre-change) scene directory
        post_dir: Path to query (post-change) scene directory
        config: Optional RelocalizationConfig (uses defaults if None)
    
    Returns:
        LocalizationResult with paths and statistics
    """
    pre_dir = Path(pre_dir)
    post_dir = Path(post_dir)
    
    # Create config if not provided
    if config is None:
        config = RelocalizationConfig.from_dirs(pre_dir, post_dir)
    
    print("=" * 80)
    print("QUERY IMAGE RELOCALIZATION PIPELINE")
    print("=" * 80)
    print(f"Reference Scene: {pre_dir}")
    print(f"Query Scene: {post_dir}")
    print()
    
    # Get paths
    pre_colmap_model = config.reference_scene.get_colmap_ba_dir()
    pre_transforms = config.reference_scene.scene_dir / 'transforms_colmap_metric.json'
    pre_features = config.reference_scene.get_hloc_outputs_dir() / 'features.h5'
    pre_global_desc = config.reference_scene.get_hloc_outputs_dir() / 'global-descriptors.h5'
    
    post_images_dir = config.query_scene.images_dir
    post_outputs = config.query_scene.get_hloc_outputs_dir()
    post_outputs.mkdir(exist_ok=True, parents=True)
    
    post_features = post_outputs / 'features.h5'
    post_global_desc = post_outputs / 'global-descriptors.h5'
    post_matches = post_outputs / 'matches.h5'

    print("=" * 80)
    print("STEP 1: Extracting features from query images")
    print("=" * 80)
    
    post_image_list = get_image_list(post_images_dir)
    print(f"Found {len(post_image_list)} query images")
    
    post_features_h5, post_global_h5 = extract_all_features(
        image_dir=post_images_dir,
        output_dir=post_outputs,
        feature_conf=config.features.get_feature_conf(),
        global_conf=config.features.get_global_conf(),
        image_list=post_image_list
    )

    print("=" * 80)
    print("STEP 2: Retrieving similar reference images and matching")
    print("=" * 80)
    
    pairs_file, matches_h5, retrieval_pairs = match_for_localization(
        query_images_dir=post_images_dir,
        query_features_h5=post_features_h5,
        query_global_h5=post_global_h5,
        reference_features_h5=pre_features,
        reference_global_h5=pre_global_desc,
        reference_model=pre_colmap_model,
        output_dir=post_outputs,
        matcher_conf=config.features.get_matcher_conf(),
        num_matched=config.features.num_matched
    )
    print()
    

    print("=" * 80)
    print("STEP 3: Localizing query cameras in reference frame")
    print("=" * 80)
    
    pre_reconstruction = load_colmap_model(pre_colmap_model)
    pre_data = load_arkit_transforms(pre_transforms)
    pre_name_to_id = get_image_name_to_id(pre_reconstruction)
    
    fx, fy, cx, cy, w, h = get_camera_intrinsics(pre_data)
    camera = create_pycolmap_camera(fx, fy, cx, cy, w, h)
    
    localizer = QueryLocalizer(
        pre_reconstruction,
        config.localization.get_localization_conf()
    )
    
    matches_h5_file = h5py.File(matches_h5, 'r')
    print(f"Matches file contains {len(matches_h5_file.keys())} pairs")
    
    localized_poses = {}
    failure_reasons = defaultdict(int)
    
    print("\nLocalizing images...")
    for query_name in tqdm(post_image_list, desc="Localizing"):
        kpq = get_keypoints(post_features_h5, query_name)
        
        # Localize
        result, msg = localize_single_image(
            query_name=query_name,
            query_keypoints=kpq,
            retrieval_pairs=retrieval_pairs,
            matches_h5=matches_h5_file,
            reference_reconstruction=pre_reconstruction,
            reference_name_to_id=pre_name_to_id,
            localizer=localizer,
            camera=camera
        )
        
        if result is not None:
            localized_poses[query_name] = result
        else:
            failure_reasons[msg] += 1
    
    matches_h5_file.close()
    
    num_localized = len(localized_poses)
    num_failed = len(post_image_list) - num_localized
    
    print(f"\nSuccessfully localized {num_localized}/{len(post_image_list)} query images")
    
    if failure_reasons:
        print("\nTop failure reasons:")
        for reason, count in sorted(failure_reasons.items(), key=lambda x: -x[1])[:3]:
            print(f"  - {reason}: {count}")
    print()
    

    print("=" * 80)
    print("STEP 4: Saving query poses in reference coordinate frame")
    print("=" * 80)
    
    # Create frames for transforms.json
    frames = []
    for img_name in sorted(localized_poses.keys()):
        pose_data = localized_poses[img_name]
        
        # Convert COLMAP pose to transform matrix
        w2c = np.eye(4)
        w2c[:3, :3] = pycolmap.qvec_to_rotmat(pose_data['qvec'])
        w2c[:3, 3] = pose_data['tvec']
        
        c2w = colmap_to_opengl_pose(w2c)
        
        frame = {
            'file_path': f'./frames/{img_name}',
            'transform_matrix': c2w.tolist(),
            'num_inliers': int(pose_data['num_inliers'])
        }
        
        # Add depth path if exists
        if config.query_scene.depth_dir.exists():
            depth_name = Path(img_name).stem + '.npy'
            frame['depth_file_path'] = f'depth/{depth_name}'
        
        frames.append(frame)
    
    # Save transforms
    camera_params = {
        'fl_x': fx,
        'fl_y': fy,
        'cx': cx,
        'cy': cy,
        'w': w,
        'h': h
    }
    
    output_transforms_path = config.query_scene.scene_dir / 'transforms_reloc.json'
    save_transforms(frames, camera_params, output_transforms_path)
    
    print(f"\n✓ Query images localized in reference coordinate frame")
    print(f"✓ Metric scale consistency preserved")
    print(f"✓ Ready for direct pre-post comparison")
    print()
    
    # ========================================================================
    # Create result
    # ========================================================================
    result = LocalizationResult(
        reference_model_path=pre_colmap_model,
        query_features_path=post_features_h5,
        query_global_desc_path=post_global_h5,
        matches_path=matches_h5,
        pairs_path=pairs_file,
        output_transforms_path=output_transforms_path,
        num_query_images=len(post_image_list),
        num_localized=num_localized,
        num_failed=num_failed,
        failure_reasons=dict(failure_reasons),
        localized_poses=localized_poses
    )
    
    print("=" * 80)
    print("RELOCALIZATION COMPLETE!")
    print("=" * 80)
    print(result.summary())
    print()
    
    return result


# Convenience function
def relocalize_images(pre_dir: str, post_dir: str, **kwargs) -> LocalizationResult:
    """
    Convenience function for quick relocalization.
    
    Args:
        pre_dir: Path to reference scene directory
        post_dir: Path to query scene directory
        **kwargs: Additional config parameters
    
    Returns:
        LocalizationResult
    
    Example:
        result = relocalize_images('data/pre', 'data/post', num_matched=15)
    """
    
    ref_scene = SceneConfig(scene_dir=Path(pre_dir))
    query_scene = SceneConfig(scene_dir=Path(post_dir))
    
    # Extract config parameters
    feature_kwargs = {}
    loc_kwargs = {}
    
    for key, value in kwargs.items():
        if key in ['max_keypoints', 'match_threshold', 'num_matched']:
            feature_kwargs[key] = value
        elif key in ['ransac_max_error', 'min_inliers']:
            loc_kwargs[key] = value
    
    config = RelocalizationConfig(
        reference_scene=ref_scene,
        query_scene=query_scene,
        features=FeatureConfig(**feature_kwargs) if feature_kwargs else FeatureConfig(),
        localization=LocalizationConfig(**loc_kwargs) if loc_kwargs else LocalizationConfig()
    )
    
    return run_relocalization(Path(pre_dir), Path(post_dir), config)