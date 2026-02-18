#!/usr/bin/env python3
"""
Prepare V2A scene for ArtiSplatFacto pipeline
Sets up directory structure and exports data
Handles RGBA → RGB conversion for nerfstudio compatibility
"""

import os
import shutil
import json
from pathlib import Path
from .dataloader import load_v2a_scene
import argparse
from PIL import Image
import numpy as np


def extract_rgb_and_mask(
    rgba_path: Path,
    rgb_output: Path,
    mask_output: Path = None,
    mode: str = "rgba",  # "rgb", "rgba", or "composite"
    bg_color=(255, 255, 255),
):
    """
    Extract or process RGBA image.

    Args:
        rgba_path: Path to input image (RGB or RGBA)
        rgb_output: Path to save processed image
        mask_output: Optional path to save mask (binary PNG)
        mode:
            "rgb"       -> strip alpha
            "rgba"      -> keep 4 channels
            "composite" -> composite over background
        bg_color: Background color for compositing
    """
    rgba = np.array(Image.open(rgba_path))

    # If image has alpha
    if rgba.ndim == 3 and rgba.shape[-1] == 4:
        rgb = rgba[..., :3]
        alpha = rgba[..., 3:4] / 255.0

        # Save mask if requested
        if mask_output is not None:
            mask = (rgba[..., 3] > 127).astype(np.uint8) * 255
            Image.fromarray(mask, mode="L").save(mask_output)

        if mode == "rgb":
            output_img = rgb

        elif mode == "rgba":
            output_img = rgba  # keep all 4 channels

        elif mode == "composite":
            bg = np.ones_like(rgb) * np.array(bg_color, dtype=np.uint8)
            output_img = (rgb * alpha + bg * (1 - alpha)).astype(np.uint8)

        else:
            raise ValueError(f"Invalid mode: {mode}")

    else:
        # Already RGB
        output_img = rgba

    # Save image
    img = Image.fromarray(output_img)
    if rgb_output.suffix.lower() == ".jpg":
        img.save(rgb_output, quality=95)
    else:
        img.save(rgb_output)



def prepare_v2a_scene(scene_path: str, output_root: str, copy_images: bool = True):
    """
    Prepare a V2A scene for training
    
    Args:
        scene_path: Path to V2A scene
        output_root: Root directory for outputs
        copy_images: If True, copy images. If False, use symlinks (not recommended for RGBA→RGB conversion).
    """
    
    # Load scene
    scene = load_v2a_scene(scene_path)
    
    # Create output directory structure
    output_dir = Path(output_root) / scene.scene_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"Preparing V2A scene: {scene.scene_name}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*80}\n")
    

    # ========================================================================
    # 1. Setup canonical data for 3DGS training
    # ========================================================================

    canonical_dir = output_dir / "canonical"
    canonical_dir.mkdir(exist_ok=True)

    frames_dir = canonical_dir / "frames"
    frames_dir.mkdir(exist_ok=True)

    masks_dir = canonical_dir / "masks_gt"  
    masks_dir.mkdir(exist_ok=True)       

    depth_dir = canonical_dir / "depth"
    depth_dir.mkdir(exist_ok=True)

    print("Step 1: Setting up canonical frames...")

    for idx, frame in enumerate(scene.canonical_frames, start=1):
        # Extract RGB from RGBA
        src_img = scene.scene_path / frame.file_path
        dst_img = frames_dir / f"frame_{idx:05d}.png"
        dst_mask = masks_dir / f"frame_{idx:05d}.png"  
        
        # Extract RGB and mask from RGBA
        extract_rgb_and_mask(src_img, dst_img, mask_output=dst_mask)  
        
        # Copy depth
        depth_src = str(scene.scene_path / frame.file_path).replace('/images/', '/depth/')
        depth_dst = depth_dir / f"frame_{idx:05d}.png"
        
        if Path(depth_src).exists():
            shutil.copy2(depth_src, depth_dst)

    # Export canonical transforms
    canonical_transforms = {
        "camera_angle_x": scene.fov_x,
        "camera_angle_y": scene.fov_y,
        "fl_x": scene.focal_x,
        "fl_y": scene.focal_y,
        "cx": scene.cx,
        "cy": scene.cy,
        "w": scene.width,
        "h": scene.height,
        "ply_file_path": "point_cloud.ply",

        "frames": []
    }

    for idx, frame in enumerate(scene.canonical_frames, start=1):
        frame_dict = {
            "file_path": f"frames/frame_{idx:05d}.png",
            "mask_gt_path": f"masks_gt/frame_{idx:05d}.png",        
            "depth_file_path": f"depth/frame_{idx:05d}.png",
            "transform_matrix": frame.transform_matrix.tolist()
        }
        canonical_transforms["frames"].append(frame_dict)

    
    transforms_path = canonical_dir / "transforms.json"
    with open(transforms_path, 'w') as f:
        json.dump(canonical_transforms, f, indent=2)
    
    print(f"  ✓ Exported canonical transforms")
    
    # Copy point cloud (useful for initialization)
    if scene.point_cloud_path.exists():
        shutil.copy2(scene.point_cloud_path, canonical_dir / "point_cloud.ply")
        print(f"  ✓ Copied point cloud")
    
    # ========================================================================
    # 2. Setup articulated frames data
    # ========================================================================
    
    articulated_dir = output_dir / "articulated"
    articulated_dir.mkdir(exist_ok=True)
    
    frames_dir = articulated_dir / "frames"
    frames_dir.mkdir(exist_ok=True)
    
    masks_dir = articulated_dir / "masks_gt"
    masks_dir.mkdir(exist_ok=True)
    
    depth_dir = articulated_dir / "depth"
    depth_dir.mkdir(exist_ok=True)
    
    print("\nStep 2: Setting up articulated frames...")
    
    # Process articulated frames: extract RGB + mask
    for idx, frame in enumerate(scene.articulated_frames, start=1):
        src_img = scene.scene_path / frame.file_path
        dst_img = frames_dir / f"frame_{idx:05d}.png"
        dst_mask = masks_dir / f"frame_{idx:05d}.png"
        
        # Extract RGB and mask from RGBA
        extract_rgb_and_mask(src_img, dst_img, mask_output=dst_mask)
        
        # Copy depth
        depth_src = str(scene.scene_path / frame.file_path).replace('/images/', '/depth/')
        depth_dst = depth_dir / f"frame_{idx:05d}.png"
        if Path(depth_src).exists():
            shutil.copy2(depth_src, depth_dst)
    
    print(f"  ✓ Converted {len(scene.articulated_frames)} RGBA → RGB articulated frames")
    print(f"  ✓ Extracted {len(scene.articulated_frames)} masks from alpha channel")
    
    # Export articulated transforms (with camera poses and states)
    arti_transforms = {
        "camera_angle_x": scene.fov_x,
        "camera_angle_y": scene.fov_y,
        "fl_x": scene.focal_x,
        "fl_y": scene.focal_y,
        "cx": scene.cx,
        "cy": scene.cy,
        "w": scene.width,
        "h": scene.height,
        "frames": []
    }
    
    for idx, frame in enumerate(scene.articulated_frames, start=1):
        frame_dict = {
            "file_path": f"frames/frame_{idx:05d}.png",
            "mask_gt_path": f"masks_gt/frame_{idx:05d}.png",
            "depth_file_path": f"depth/frame_{idx:05d}.png",
            "state": frame.state,
            "time": frame.time,
            "transform_matrix": frame.transform_matrix.tolist()
        }
        arti_transforms["frames"].append(frame_dict)
    
    with open(articulated_dir / "transforms.json", 'w') as f:
        json.dump(arti_transforms, f, indent=2)
    
    print(f"  ✓ Exported articulated transforms")
    
    # ========================================================================
    # 3. Copy ground truth data
    # ========================================================================
    
    gt_dir = output_dir / "gt"
    gt_dir.mkdir(exist_ok=True)
    
    print("\nStep 3: Copying ground truth data...")
    
    gt_files = {
        "joint_info": scene.gt_joint_info_path,
        "whole_mesh": scene.gt_whole_mesh_path,
        "moving_mesh": scene.gt_moving_mesh_path,
        "static_mesh": scene.gt_static_mesh_path,
    }
    
    for name, src_path in gt_files.items():
        if src_path.exists():
            dst_path = gt_dir / src_path.name
            shutil.copy2(src_path, dst_path)
            print(f"  ✓ Copied {name}")
    
    # ========================================================================
    # 4. Create scene metadata
    # ========================================================================
    
    print("\nStep 4: Creating scene metadata...")
    
    metadata = {
        "scene_name": scene.scene_name,
        "source_path": str(scene.scene_path),
        "total_frames": len(scene.frames),
        "canonical_frames": len(scene.canonical_frames),
        "articulated_frames": len(scene.articulated_frames),
        "camera": {
            "fl_x": scene.focal_x,
            "fl_y": scene.focal_y,
            "cx": scene.cx,
            "cy": scene.cy,
            "width": scene.width,
            "height": scene.height,
            "fov_x": scene.fov_x,
            "fov_y": scene.fov_y,
        },
        "directories": {
            "canonical": "canonical",
            "articulated": "articulated",
            "ground_truth": "gt",
        }
    }
    
    with open(output_dir / "scene_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"  ✓ Saved scene metadata")
    
    
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare V2A scene for evaluation")
    parser.add_argument("--scene", type=str, required=True,
                        help="Path to V2A scene directory")
    parser.add_argument("--output", type=str, default="outputs/v2a_eval",
                        help="Output root directory")
    
    args = parser.parse_args()
    
    # Always copy images (can't symlink when converting RGBA→RGB)
    prepare_v2a_scene(args.scene, args.output, copy_images=True)