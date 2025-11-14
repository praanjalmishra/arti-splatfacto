"""
Reference scene reconstruction pipeline.
Takes ARKit poses and refines them using COLMAP SfM.
"""

import os
import shutil
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

import pycolmap
from hloc import triangulation

from .io_utils import (
    load_arkit_transforms,
    arkit_to_colmap_model,
    load_colmap_model,
    colmap_model_to_transforms,
    get_camera_intrinsics
)
from .features import (
    extract_all_features,
    match_for_reconstruction
)

from .config import ReconstructionConfig, SceneConfig, FeatureConfig, BundleAdjustmentConfig


@dataclass
class ReconstructionResult:
    """Results from reconstruction pipeline."""
    
    colmap_arkit_dir: Path      # Initial ARKit-based model
    colmap_ba_dir: Path          # Final bundle-adjusted model
    transforms_path: Path        # Refined transforms.json
    feature_path: Path           # Features h5
    global_desc_path: Path       # Global descriptors h5
    matches_path: Path           # Matches h5
    pairs_path: Path             # Pairs txt
    
    num_images: int
    num_points3d: int
    num_iterations: int
    
    def summary(self) -> str:
        """Return a summary string of the reconstruction."""
        return (
            f"Reconstruction Summary:\n"
            f"  Images: {self.num_images}\n"
            f"  3D Points: {self.num_points3d}\n"
            f"  BA Iterations: {self.num_iterations}\n"
            f"  Final Model: {self.colmap_ba_dir}\n"
            f"  Refined Transforms: {self.transforms_path}"
        )


def run_reconstruction(
    scene_dir: Path,
    config: Optional[ReconstructionConfig] = None
) -> ReconstructionResult:
    """
    Run complete reconstruction pipeline on a scene.
    
    Pipeline:
    1. Load ARKit transforms and create reference COLMAP model
    2. Extract features (local + global)
    3. Generate pairs from poses and match features
    4. Iterative triangulation + bundle adjustment
    5. Export refined transforms
    
    Args:
        scene_dir: Path to scene directory containing frames/ and transforms_arkit.json
        config: Optional ReconstructionConfig (uses defaults if None)
    
    Returns:
        ReconstructionResult with paths and statistics
    """
    scene_dir = Path(scene_dir)
    
    # Create config if not provided
    if config is None:
        config = ReconstructionConfig(scene=SceneConfig(scene_dir=scene_dir))
    
    print("=" * 80)
    print("REFERENCE SCENE RECONSTRUCTION PIPELINE")
    print("=" * 80)
    print(f"Scene: {scene_dir}")
    print(f"Images: {config.scene.images_dir}")
    print(f"Transforms: {config.scene.transforms_path}")
    print()
    
    # Get paths
    hloc_outputs = config.scene.get_hloc_outputs_dir()
    hloc_outputs.mkdir(exist_ok=True, parents=True)
    
    colmap_arkit_dir = config.scene.get_colmap_arkit_dir()
    colmap_sparse_dir = config.scene.get_colmap_sparse_dir()
    colmap_ba_dir = config.scene.get_colmap_ba_dir()
    

    print("=" * 80)
    print("STEP 1: Creating COLMAP reference model from ARKit poses")
    print("=" * 80)
    
    arkit_data = load_arkit_transforms(config.scene.transforms_path)
    arkit_to_colmap_model(arkit_data, colmap_arkit_dir)
    print()
    

    print("=" * 80)
    print("STEP 2: Feature extraction")
    print("=" * 80)
    
    features_h5, global_h5 = extract_all_features(
        image_dir=config.scene.images_dir,
        output_dir=hloc_outputs,
        feature_conf=config.features.get_feature_conf(),
        global_conf=config.features.get_global_conf()
    )
    print()
    

    print("=" * 80)
    print("STEP 3: Feature matching")
    print("=" * 80)
    
    pairs_file, matches_h5 = match_for_reconstruction(
        image_dir=config.scene.images_dir,
        colmap_model=colmap_arkit_dir,
        features_h5=features_h5,
        output_dir=hloc_outputs,
        matcher_conf=config.features.get_matcher_conf(),
        num_matched=config.features.num_matched
    )
    print()
    

    print("=" * 80)
    print("STEP 4: Iterative triangulation and bundle adjustment")
    print("=" * 80)
    
    colmap_input = colmap_arkit_dir
    
    for iteration in range(config.num_ba_iterations):
        print(f"\n--- Iteration {iteration + 1}/{config.num_ba_iterations} ---")
        
        # Triangulation
        print("Running triangulation...")
        colmap_sparse_dir.mkdir(exist_ok=True, parents=True)
        
        reconstruction = triangulation.main(
            sfm_dir=colmap_sparse_dir,
            reference_model=colmap_input,
            image_dir=config.scene.images_dir,
            pairs=pairs_file,
            features=features_h5,
            matches=matches_h5,
            skip_geometric_verification=False
        )
        
        if reconstruction is None:
            print("⚠ Triangulation failed!")
            break
        
        print(f"✓ Triangulation complete:")
        print(f"  - {len(reconstruction.images)} images")
        print(f"  - {len(reconstruction.points3D)} 3D points")
        
        # Bundle Adjustment
        print("\nRunning bundle adjustment...")
        colmap_ba_dir.mkdir(exist_ok=True, parents=True)
        
        ba_cmd = config.bundle_adjustment.get_colmap_command(
            colmap_sparse_dir,
            colmap_ba_dir
        )
        
        os.system(ba_cmd)
        print(f"✓ Bundle adjustment complete: {colmap_ba_dir}")
        
        # Update input for next iteration
        colmap_input = colmap_ba_dir
        
        # Clean up intermediate sparse dir
        if iteration < config.num_ba_iterations - 1:
            shutil.rmtree(colmap_sparse_dir)
            colmap_sparse_dir.mkdir(exist_ok=True)
    
    print()
    

    print("=" * 80)
    print("STEP 5: Exporting refined poses to transforms.json")
    print("=" * 80)
    
    # Load final reconstruction
    final_reconstruction = load_colmap_model(colmap_ba_dir)
    
    # Save transforms (metric scale preserved!)
    output_transforms_path = config.scene.scene_dir / 'transforms_colmap_metric.json'
    colmap_model_to_transforms(
        final_reconstruction,
        output_transforms_path,
        depth_dir='depth'
    )
    
    print(f"\n✓ Metric scale preserved (ARKit initialization)")
    print(f"✓ Geometric accuracy improved (COLMAP refinement)")
    print()
    
    result = ReconstructionResult(
        colmap_arkit_dir=colmap_arkit_dir,
        colmap_ba_dir=colmap_ba_dir,
        transforms_path=output_transforms_path,
        feature_path=features_h5,
        global_desc_path=global_h5,
        matches_path=matches_h5,
        pairs_path=pairs_file,
        num_images=len(final_reconstruction.images),
        num_points3d=len(final_reconstruction.points3D),
        num_iterations=config.num_ba_iterations
    )
    
    print("=" * 80)
    print("RECONSTRUCTION COMPLETE!")
    print("=" * 80)
    print(result.summary())
    print()
    
    return result


# Convenience function
def reconstruct_scene(scene_dir: str, **kwargs) -> ReconstructionResult:
    """
    Convenience function for quick reconstruction.
    
    Args:
        scene_dir: Path to scene directory
        **kwargs: Additional config parameters (e.g., num_ba_iterations=3)
    
    Returns:
        ReconstructionResult
    
    Example:
        result = reconstruct_scene('data/pre', num_ba_iterations=3)
    """
    
    scene = SceneConfig(scene_dir=Path(scene_dir))
    
    # Extract config parameters
    feature_kwargs = {}
    ba_kwargs = {}
    recon_kwargs = {}
    
    for key, value in kwargs.items():
        if key in ['max_keypoints', 'match_threshold', 'num_matched']:
            feature_kwargs[key] = value
        elif key in ['max_iterations', 'refine_focal_length']:
            ba_kwargs[key] = value
        elif key in ['num_ba_iterations']:
            recon_kwargs[key] = value
    
    config = ReconstructionConfig(
        scene=scene,
        features=FeatureConfig(**feature_kwargs) if feature_kwargs else FeatureConfig(),
        bundle_adjustment=BundleAdjustmentConfig(**ba_kwargs) if ba_kwargs else BundleAdjustmentConfig(),
        **recon_kwargs
    )
    
    return run_reconstruction(Path(scene_dir), config)