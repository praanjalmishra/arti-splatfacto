#!/usr/bin/env python3
"""
V2A Dataset Loader for ArtiSplatFacto Pipeline
Loads V2A format data with state=0 (closed) and state=1 (open)
"""

import json
import os
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple
import torch
from PIL import Image


@dataclass
class V2AFrame:
    """Single frame from V2A dataset"""
    frame_id: int
    file_path: str
    state: int  # 0=closed, 1=open
    time: float
    transform_matrix: np.ndarray  # 4x4 camera-to-world
    
    @property
    def is_canonical(self) -> bool:
        """Check if this is a canonical (closed) frame"""
        return self.state == 0
    
    @property
    def is_articulated(self) -> bool:
        """Check if this is an articulated (open) frame"""
        return self.state == 1


@dataclass
class V2AScene:
    """Complete V2A scene data"""
    scene_path: Path
    scene_name: str
    
    # Camera intrinsics
    focal_x: float
    focal_y: float
    cx: float
    cy: float
    width: int
    height: int
    fov_x: float
    fov_y: float
    
    # Frame data
    frames: List[V2AFrame]
    
    # GT data paths
    gt_joint_info_path: Path
    gt_whole_mesh_path: Path
    gt_moving_mesh_path: Path
    gt_static_mesh_path: Path
    point_cloud_path: Path
    
    def __post_init__(self):
        """Organize frames by state"""
        self.canonical_frames = [f for f in self.frames if f.is_canonical]
        self.articulated_frames = [f for f in self.frames if f.is_articulated]
        
        print(f"✓ Loaded scene: {self.scene_name}")
        print(f"  - Total frames: {len(self.frames)}")
        print(f"  - Canonical (state=0): {len(self.canonical_frames)}")
        print(f"  - Articulated (state=1): {len(self.articulated_frames)}")
    
    def get_frame_paths(self, state: int = None) -> List[Path]:
        """Get image paths for specified state"""
        if state is None:
            frames = self.frames
        elif state == 0:
            frames = self.canonical_frames
        elif state == 1:
            frames = self.articulated_frames
        else:
            raise ValueError(f"Invalid state: {state}. Must be 0 or 1")
        
        return [self.scene_path / f.file_path for f in frames]
    
    def load_image(self, frame: V2AFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load RGBA image and extract RGB + alpha mask
        Returns:
            rgb: (H, W, 3) uint8
            mask: (H, W) bool - True for foreground object
        """
        img_path = self.scene_path / frame.file_path
        rgba = np.array(Image.open(img_path))
        
        rgb = rgba[..., :3]
        mask = rgba[..., 3] > 127  # Threshold alpha channel
        
        return rgb, mask
    
    def load_depth(self, frame: V2AFrame) -> np.ndarray:
        """
        Load depth map (in millimeters)
        Returns:
            depth: (H, W) float32 in meters
        """
        # Convert image path to depth path
        # images/000001.png -> depth/000001.png
        depth_path = str(self.scene_path / frame.file_path)
        depth_path = depth_path.replace('/images/', '/depth/')
        
        depth_img = Image.open(depth_path)
        depth_mm = np.array(depth_img, dtype=np.float32)
        depth_m = depth_mm / 1000.0  # Convert mm to meters
        
        return depth_m
    
    def get_camera_matrix(self) -> np.ndarray:
        """Get 3x3 camera intrinsic matrix"""
        K = np.array([
            [self.focal_x, 0, self.cx],
            [0, self.focal_y, self.cy],
            [0, 0, 1]
        ], dtype=np.float32)
        return K
    
    def load_gt_joint_info(self) -> Dict:
        """Load ground truth joint parameters"""
        with open(self.gt_joint_info_path, 'r') as f:
            joint_info = json.load(f)
        return joint_info[0] if isinstance(joint_info, list) else joint_info
    


    def export_canonical_transforms(self, output_path: Path):
        """
        Export transforms.json with only canonical frames.
        Compatible with nerfstudio format.
        """

        transforms_dict = {
            "camera_angle_x": self.fov_x,
            "camera_angle_y": self.fov_y,
            "fl_x": self.focal_x,
            "fl_y": self.focal_y,
            "cx": self.cx,
            "cy": self.cy,
            "w": self.width,
            "h": self.height,
            "frames": []
        }

        for idx, frame in enumerate(self.canonical_frames):
            frame_dict = {
                "file_path": f"frames/frame_{idx:05d}.png",
                "depth_file_path": f"depth/frame_{idx:05d}.png",
                "transform_matrix": frame.transform_matrix.tolist()
            }
            transforms_dict["frames"].append(frame_dict)

        with open(output_path, 'w') as f:
            json.dump(transforms_dict, f, indent=2)

        print(f"✓ Exported canonical transforms to: {output_path}")
        print(f"  - {len(self.canonical_frames)} frames")



def load_v2a_scene(scene_path: str) -> V2AScene:
    """
    Load a V2A scene
    
    Args:
        scene_path: Path to scene directory (e.g., data/v2a/7265_joint_0_bg_view_0)
    
    Returns:
        V2AScene object with all data loaded
    """
    scene_path = Path(scene_path)
    
    if not scene_path.exists():
        raise FileNotFoundError(f"Scene path not found: {scene_path}")
    
    # Load transforms.json
    transforms_path = scene_path / "transforms.json"
    with open(transforms_path, 'r') as f:
        transforms = json.load(f)
    
    # Parse camera intrinsics
    camera_params = {
        'focal_x': transforms['focal_x'],
        'focal_y': transforms['focal_y'],
        'cx': transforms['cx'],
        'cy': transforms['cy'],
        'width': transforms['w'],
        'height': transforms['h'],
        'fov_x': transforms['camera_angle_x'],
        'fov_y': transforms['camera_angle_y'],
    }
    
    # Parse frames
    frames = []
    for idx, frame_data in enumerate(transforms['frames']):
        frame = V2AFrame(
            frame_id=idx,
            file_path=frame_data['file_path'],
            state=frame_data['state'],
            time=frame_data['time'],
            transform_matrix=np.array(frame_data['transform_matrix'], dtype=np.float32)
        )
        frames.append(frame)
    
    # GT data paths
    gt_dir = scene_path / "gt"
    
    scene = V2AScene(
        scene_path=scene_path,
        scene_name=scene_path.name,
        frames=frames,
        gt_joint_info_path=gt_dir / "mobility_v2.json",
        gt_whole_mesh_path=gt_dir / "whole_mesh.ply",
        gt_moving_mesh_path=gt_dir / "part_1.ply",
        gt_static_mesh_path=gt_dir / "part_0.ply",
        point_cloud_path=scene_path / "point_cloud.ply",
        **camera_params
    )
    
    return scene


def load_v2a_dataset(data_root: str) -> List[V2AScene]:
    """
    Load all V2A scenes from a directory
    
    Args:
        data_root: Root directory containing V2A scenes
    
    Returns:
        List of V2AScene objects
    """
    data_root = Path(data_root)
    scene_dirs = sorted([d for d in data_root.iterdir() if d.is_dir()])
    
    scenes = []
    for scene_dir in scene_dirs:
        try:
            scene = load_v2a_scene(scene_dir)
            scenes.append(scene)
        except Exception as e:
            print(f"⚠ Failed to load {scene_dir.name}: {e}")
    
    print(f"\n✓ Loaded {len(scenes)} scenes from {data_root}")
    return scenes


# ============================================================================
# Usage Examples
# ============================================================================

if __name__ == "__main__":
    
    # Example 1: Load single scene

    DATA_ROOT = os.environ.get("VIDEOART_DATA")

    if DATA_ROOT is None:
        raise ValueError("VIDEOART_DATA environment variable not set")

    scene = load_v2a_scene(
        os.path.join(DATA_ROOT, "v2a/7265_joint_0_bg_view_0")
    )    
    # Access frame data
    print(f"\nCamera intrinsics:")
    print(f"  Focal: ({scene.focal_x:.1f}, {scene.focal_y:.1f})")
    print(f"  Resolution: {scene.width}x{scene.height}")
    
    # Get canonical frames
    print(f"\nCanonical frames: {len(scene.canonical_frames)}")
    print(f"First canonical frame: {scene.canonical_frames[0].file_path}")
    
    # Load an image
    frame = scene.canonical_frames[0]
    rgb, mask = scene.load_image(frame)
    print(f"\nLoaded image: {rgb.shape}")
    print(f"Mask coverage: {mask.sum() / mask.size * 100:.1f}%")
    
    # Load depth
    depth = scene.load_depth(frame)
    print(f"Depth range: {depth[depth > 0].min():.3f}m - {depth.max():.3f}m")
    
    # Export canonical transforms for nerfstudio
    output_dir = Path("outputs/v2a_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    scene.export_canonical_transforms(output_dir / "transforms_canonical.json")
    
    # Load GT joint info
    joint_info = scene.load_gt_joint_info()
    print(f"\nGT Joint type: {joint_info['joint']}")
