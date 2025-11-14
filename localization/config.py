"""
Configuration classes and all hyperparameters for the localization pipeline.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any


@dataclass
class FeatureConfig:
    """Configuration for feature extraction and matching."""
    
    # SuperPoint feature extraction
    max_keypoints: int = 8192
    nms_radius: int = 3
    keypoint_threshold: float = 0.005
    resize_max: int = 1600
    
    # SuperGlue matching
    match_threshold: float = 0.2
    sinkhorn_iterations: int = 50
    
    # Image retrieval
    num_matched: int = 10  # Number of similar images to retrieve
    
    def get_feature_conf(self) -> Dict[str, Any]:
        """Get HLoc feature extraction config dict."""
        return {
            'output': 'feats-superpoint',
            'model': {
                'name': 'superpoint',
                'nms_radius': self.nms_radius,
                'max_keypoints': self.max_keypoints,
                'keypoint_threshold': self.keypoint_threshold
            },
            'preprocessing': {
                'grayscale': True,
                'resize_max': self.resize_max,
            }
        }
    
    def get_matcher_conf(self) -> Dict[str, Any]:
        """Get HLoc matcher config dict."""
        return {
            'output': 'matches-superglue',
            'model': {
                'name': 'superglue',
                'weights': 'outdoor',
                'sinkhorn_iterations': self.sinkhorn_iterations,
                'match_threshold': self.match_threshold,
            }
        }
    
    def get_global_conf(self) -> Dict[str, Any]:
        """Get HLoc global descriptor config dict."""
        from hloc import extract_features
        return extract_features.confs["netvlad"]


@dataclass
class LocalizationConfig:
    """Configuration for PnP localization."""
    
    # RANSAC parameters
    ransac_max_error: float = 2.0  # Reprojection error threshold (pixels)
    min_inliers: int = 10  # Minimum inliers for successful localization
    
    # Refinement options (keep False to preserve metric scale)
    refine_focal_length: bool = False
    refine_principal_point: bool = False
    refine_extra_params: bool = False
    
    def get_localization_conf(self) -> Dict[str, Any]:
        """Get HLoc localization config dict."""
        return {
            "estimation": {
                "ransac": {"max_error": self.ransac_max_error}
            },
            "refinement": {
                'refine_focal_length': self.refine_focal_length,
                'refine_extra_params': self.refine_extra_params
            }
        }


@dataclass
class BundleAdjustmentConfig:
    """Configuration for COLMAP bundle adjustment."""
    
    max_iterations: int = 50
    
    # Refinement flags (keep False to preserve metric scale from ARKit)
    refine_focal_length: bool = False
    refine_principal_point: bool = False
    refine_extra_params: bool = False
    
    def get_colmap_command(self, input_path: Path, output_path: Path) -> str:
        """Generate COLMAP bundle adjustment command."""
        return (
            f"colmap bundle_adjuster "
            f"--input_path {input_path} "
            f"--output_path {output_path} "
            f"--BundleAdjustment.refine_focal_length {int(self.refine_focal_length)} "
            f"--BundleAdjustment.refine_extra_params {int(self.refine_extra_params)} "
            f"--BundleAdjustment.max_num_iterations {self.max_iterations}"
        )


@dataclass
class SceneConfig:
    """Configuration for a scene (pre or post)."""
    
    scene_dir: Path
    images_dir: Optional[Path] = None
    transforms_path: Optional[Path] = None
    depth_dir: Optional[Path] = None
    
    def __post_init__(self):
        """Set default paths based on scene_dir."""
        self.scene_dir = Path(self.scene_dir)
        
        if self.images_dir is None:
            self.images_dir = self.scene_dir / 'frames'
        else:
            self.images_dir = Path(self.images_dir)
        
        if self.transforms_path is None:
            self.transforms_path = self.scene_dir / 'transforms_arkit.json'
        else:
            self.transforms_path = Path(self.transforms_path)
        
        if self.depth_dir is None:
            self.depth_dir = self.scene_dir / 'depth'
        else:
            self.depth_dir = Path(self.depth_dir)
    
    def get_hloc_outputs_dir(self) -> Path:
        """Get HLoc outputs directory."""
        return self.scene_dir / 'hloc_outputs'
    
    def get_colmap_arkit_dir(self) -> Path:
        """Get COLMAP ARKit reference model directory."""
        return self.scene_dir / 'colmap_arkit'
    
    def get_colmap_sparse_dir(self) -> Path:
        """Get COLMAP sparse reconstruction directory."""
        return self.scene_dir / 'colmap_sparse'
    
    def get_colmap_ba_dir(self) -> Path:
        """Get COLMAP bundle adjustment directory."""
        return self.scene_dir / 'colmap_ba'


@dataclass
class ReconstructionConfig:
    """Complete configuration for reference scene reconstruction."""
    
    scene: SceneConfig
    features: FeatureConfig = field(default_factory=FeatureConfig)
    bundle_adjustment: BundleAdjustmentConfig = field(default_factory=BundleAdjustmentConfig)
    
    # Number of triangulation + BA iterations
    num_ba_iterations: int = 2
    
    @classmethod
    def from_scene_dir(cls, scene_dir: Path, **kwargs):
        """Create config from scene directory path."""
        scene = SceneConfig(scene_dir=scene_dir)
        return cls(scene=scene, **kwargs)


@dataclass
class RelocalizationConfig:
    """Complete configuration for query image relocalization."""
    
    reference_scene: SceneConfig 
    query_scene: SceneConfig      
    
    features: FeatureConfig = field(default_factory=FeatureConfig)
    localization: LocalizationConfig = field(default_factory=LocalizationConfig)
    
    @classmethod
    def from_dirs(cls, pre_dir: Path, post_dir: Path, **kwargs):
        """Create config from pre and post directory paths."""
        reference_scene = SceneConfig(scene_dir=pre_dir)
        query_scene = SceneConfig(scene_dir=post_dir)
        return cls(reference_scene=reference_scene, query_scene=query_scene, **kwargs)


# Convenience function
def create_reconstruction_config(scene_dir: str) -> ReconstructionConfig:
    return ReconstructionConfig.from_scene_dir(Path(scene_dir))


def create_relocalization_config(pre_dir: str, post_dir: str) -> RelocalizationConfig:
    return RelocalizationConfig.from_dirs(Path(pre_dir), Path(post_dir))