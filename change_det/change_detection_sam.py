#!/usr/bin/env python3
"""
3DGS-based Change Detection

Compares the last captured frame (t=1) with the rendered frame from 
the pre-trained static 3DGS model to detect articulated regions.
"""

import argparse
import datetime
import json
import os
from pathlib import Path

import cv2
import torch
from tqdm import tqdm

from change_det.utils.sam_refine import refine_change_detection_masks
from change_det.utils.debug_utils import debug_image_pairs, debug_depth_pairs
from change_det.utils.img_utils import overlay_mask_on_image
from change_det.utils.io import read_transforms, save_masks, params_to_cameras
from change_det.utils.render_utils import render_cameras
from change_det.utils.image_diff import image_diff_sam2_with_depth
from nerfstudio.utils.eval_utils import eval_setup
import yaml

DEFAULT_CONFIG = {
    "area_threshold": 0.01,
    "cd_kernel_ratio": 0.01,
    "depth_weight": 0.3,
    "num_positive_points": 20,
    "num_negative_points": 20,
    "min_iou": 0.5,
}

def apply_dataparser_transform(poses: torch.Tensor, dataparser_json: Path) -> torch.Tensor:
    """Apply Nerfstudio dataparser_transforms.json to query poses."""
    with open(dataparser_json) as f:
        dp = json.load(f)
    T = torch.eye(4, dtype=torch.float32)
    T[:3, :] = torch.tensor(dp["transform"], dtype=torch.float32)
    scale = float(dp["scale"])
    out = poses.clone()
    out[:, :3, :3] = scale * (T[:3, :3] @ poses[:, :3, :3])
    out[:, :3,  3] = scale * (T[:3, :3] @ poses[:, :3, 3].unsqueeze(-1)).squeeze(-1) + scale * T[:3, 3]
    return out


class ChangeDetector:
    def __init__(self, pretrained_config: Path, output_dir: Path, debug: bool = False):
        self.pretrained_config = pretrained_config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.debug = debug

        if debug:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.debug_dir = self.output_dir / "cd_debug" / timestamp
            self.debug_dir.mkdir(parents=True, exist_ok=True)
            print(f"[DEBUG] Outputs will be saved to: {self.debug_dir}")
        else:
            self.debug_dir = None

        _, self.pipeline, _, _ = eval_setup(pretrained_config, test_mode="inference")
        print("[INFO] 3DGS model loaded successfully")

        # Auto-locate dataparser_transforms.json next to config.yml
        self.dataparser_json = Path(pretrained_config).parent / "dataparser_transforms.json"
        assert self.dataparser_json.exists(), \
            f"dataparser_transforms.json not found at {self.dataparser_json}"
        print(f"[INFO] Dataparser transform loaded: {self.dataparser_json}")

    def load_last_frame(self, transforms_json: Path):
        """
        Load the last frame from the dynamic sequence.
        Applies the dataparser transform so poses align with the trained model.
        """
        result = read_transforms(transforms_json, keep_alpha=True)
        if len(result) == 8:
            color_images, depth_images, img_fnames, c2w, K, dist_params, _, _ = result
        else:
            color_images, depth_images, img_fnames, c2w, K, dist_params, _ = result

        assert dist_params.sum() < 1e-6, "Images must be undistorted before change detection"

        # Align poses to trained model's coordinate frame
        c2w_transformed = apply_dataparser_transform(c2w, self.dataparser_json)
        dist_zeros = torch.zeros(len(c2w), 4)
        cameras = params_to_cameras(c2w_transformed, K.cpu(), dist_zeros,
                                    color_images.shape[-2], color_images.shape[-1])

        last_idx = len(color_images) // 2
        print(f"[INFO] Last frame index: {last_idx}")
        print(f"[INFO] Frame name: {img_fnames[last_idx]}")

        rgb_captured   = color_images[last_idx:last_idx+1].to(self.device)
        depth_captured = depth_images[last_idx:last_idx+1].to(self.device)
        camera_last    = cameras[last_idx:last_idx+1]

        if rgb_captured.shape[1] == 4:
            print("[INFO] Captured image is RGBA - compositing onto black background")
            rgb_captured = rgb_captured[:, :3] * rgb_captured[:, 3:4]

        return rgb_captured, depth_captured, camera_last, img_fnames[last_idx]

    def render_at_last_frame(self, camera):
        """Render the pre-trained 3DGS model at the last frame's viewpoint."""
        print("[INFO] Rendering pre-trained 3DGS at last frame viewpoint...")

        rgb_rendered, depth_rendered = render_cameras(
            self.pipeline, camera, device=self.device, return_rgba=True
        )

        if rgb_rendered.shape[1] == 4:
            print("[INFO] Rendered image is RGBA - compositing onto black background")
            rgb_rendered = rgb_rendered[:, :3] * rgb_rendered[:, 3:4]

        return rgb_rendered, depth_rendered
    
    def detect_changes(
        self, 
        rgb_rendered, 
        rgb_captured, 
        depth_rendered, 
        depth_captured,
        config: dict
    ):
        """
        Detect changed regions by comparing rendered and captured images.
        
        Args:
            rgb_rendered: Rendered RGB from static model [1, 3, H, W]
            rgb_captured: Captured RGB [1, 3, H, W]
            depth_rendered: Rendered depth [1, 1, H, W]
            depth_captured: Captured depth [1, 1, H, W]
            config: Detection configuration parameters
            
        Returns:
            masks_changed: Binary masks of changed regions [N, 1, H, W]
        """
        print("[INFO] Running change detection...")
        
        # Compute change masks using SAM2 + depth
        masks_changed, masks_all = image_diff_sam2_with_depth(
            rgb_rendered,
            rgb_captured,
            depth_rendered,
            depth_captured,
            debug_dir=self.debug_dir,
            threshold=config["area_threshold"],
            kernel_ratio=config["cd_kernel_ratio"],
            depth_weight=config["depth_weight"],
        )
        
        print(f"[INFO] Detected {masks_changed.size(0)} changed regions")
        
        if masks_changed.numel() == 0:
            print("[WARNING] No changes detected!")
            return masks_changed
        
        return masks_changed
    
    def refine_masks(self, rgb_captured, masks_changed, config: dict):
        """
        Refine change detection masks using SAM2.
        
        Args:
            rgb_captured: Captured RGB image [3, H, W]
            masks_changed: Initial change masks [N, 1, H, W]
            config: Refinement configuration
            
        Returns:
            refined_masks: Refined binary masks [N, 1, H, W] in range [0, 255]
        """
        if masks_changed.numel() == 0:
            print("[INFO] No masks to refine")
            return masks_changed
        
        print("[INFO] Refining masks with SAM2...")
        
        refined_masks, iou_scores = refine_change_detection_masks(
            image_captured=rgb_captured.squeeze(0),  # [3, H, W]
            masks_changed=masks_changed,  # [N, 1, H, W]
            device=self.device,
            num_positive_points=config["num_positive_points"],
            num_negative_points=config["num_negative_points"],
            debug=self.debug
        )
        
        print(f"[INFO] Refined {len(refined_masks)} masks")
        print(f"[INFO] IoU scores: {iou_scores}")
        
        refined_masks_uint8 = (refined_masks * 255.0).to(torch.uint8)
        
        return refined_masks_uint8
    
    def save_results(self, masks, rgb_captured, frame_name):
        """
        Save detection results: masks and overlays.
        """
        if masks.numel() == 0:
            print("[INFO] No masks to save")
            return

        print("[INFO] Saving results...")

        # Save all masks
        mask_dir = self.output_dir / "cd_mask"
        mask_dir.mkdir(exist_ok=True)
        masks_normalized = masks.float() / 255.0
        mask_paths = [str(mask_dir / f"mask_{i:02d}.png") for i in range(len(masks))]
        save_masks(masks_normalized, mask_paths)

        # Save overlays if in debug mode
        if self.debug and self.debug_dir is not None:
            overlay_dir = self.debug_dir / "overlays"
            overlay_dir.mkdir(exist_ok=True)

            for i, mask in enumerate(masks):
                mask_normalized = mask.float() / 255.0
                overlay = overlay_mask_on_image(rgb_captured, mask_normalized)
                overlay_path = overlay_dir / f"overlay_{i:02d}.png"
                cv2.imwrite(
                    str(overlay_path),
                    cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
                )

            print(f"  Saved overlays to: {overlay_dir}")

    
    def run(self, transforms_json: Path, config: dict = None):
        """
        Run the complete CD pipeline.
        
        Args:
            transforms_json: Path to transforms.json
            config: Configuration dictionary (optional)
            
        Returns:
            masks: Detected change masks [N, 1, H, W]
        """
        
        # Load last frame
        rgb_captured, depth_captured, camera_last, frame_name = \
            self.load_last_frame(transforms_json)
        
        # Render pre-trained model at last frame
        rgb_rendered, depth_rendered = self.render_at_last_frame(camera_last)
        

        
        # Detect changes
        masks_changed = self.detect_changes(
            rgb_rendered, rgb_captured,
            depth_rendered, depth_captured,
            config
        )
        
        # Refine masks
        masks_refined = self.refine_masks(rgb_captured, masks_changed, config)
        
        # Step 5: Save results
        self.save_results(masks_refined, rgb_captured, frame_name)
        
        # Cleanup
        print("[INFO] Cleaning up GPU memory...")
        del self.pipeline
        torch.cuda.empty_cache()
        
        print("[INFO] Change detection complete!")
        return masks_refined
    
    def __del__(self):
        """Cleanup on deletion"""
        if hasattr(self, 'pipeline'):
            del self.pipeline
        torch.cuda.empty_cache()




def load_config(config_path: Path = None) -> dict:
    config = DEFAULT_CONFIG.copy()
    if config_path and config_path.exists():
        with open(config_path, 'r') as f:
            if config_path.suffix in [".yaml", ".yml"]:
                user_cfg = yaml.safe_load(f)
            else:
                user_cfg = json.load(f)
        config.update(user_cfg)
    return config


def main():
    """Main entry point for change detection"""
    parser = argparse.ArgumentParser(
        description="3DGS change detection for articulated objects"
    )
    parser.add_argument(
        "--config", "-c", required=True, type=str,
        help="Path to config.yml of pre-trained 3DGS model"
    )
    parser.add_argument(
        "--output", "-o", required=True, type=str,
        help="Output directory for detection results"
    )
    parser.add_argument(
        "--transform", "-t", required=True, type=str,
        help="Path to transforms.json with time-annotated frames"
    )
    parser.add_argument(
        "--params", "-p", type=str, default=None,
        help="Path to detection parameters JSON"
    )
    parser.add_argument(
        "--debug", "-d", action="store_true",
        help="Enable debug mode with visualizations"
    )
    
    args = parser.parse_args()
    
    config_path = Path(args.config)
    transforms_path = Path(args.transform)
    output_dir = Path(args.output)
    
    assert config_path.exists(), f"Config not found: {config_path}"
    assert transforms_path.exists(), f"Transforms not found: {transforms_path}"
    
    params_path = Path(args.params) if args.params else None
    detection_config = load_config(params_path)
    
    print("=" * 80)
    print("3DGS Change Detection - Last Frame Analysis")
    print("=" * 80)
    print(f"Pre-trained model: {config_path}")
    print(f"Transforms: {transforms_path}")
    print(f"Output directory: {output_dir}")
    print(f"Debug mode: {args.debug}")
    print(f"Detection config: {detection_config}")
    print("=" * 80)
    
    detector = ChangeDetector(
        pretrained_config=config_path,
        output_dir=output_dir,
        debug=args.debug
    )
    
    masks = detector.run(
        transforms_json=transforms_path,
        config=detection_config
    )
    
    print("=" * 80)
    print(f"Detected {masks.size(0)} changed regions")
    print(f"Results saved to: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()