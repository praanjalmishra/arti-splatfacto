import torch
from typing import Dict
import os
import torchvision.utils as vutils


def decode_id_map(id_map: torch.Tensor, n_obj: int, n_canon: int) -> Dict[str, torch.Tensor]:
    """
    Decode the ID map to extract masks for different Gaussian types.
    
    Args:
        id_map: (H, W, 3) tensor with encoded IDs
        n_obj: Number of object Gaussians
        n_canon: Number of canonical Gaussians
    
    Returns:
        Dictionary with decoded masks and statistics
    """
    H, W = id_map.shape[:2]
    
    # Extract channel values and convert back to IDs
    red_vals = (id_map[..., 0] * 255).round().long()
    green_vals = (id_map[..., 1] * 255).round().long()
    blue_vals = (id_map[..., 2] * 255).round().long()
    
    # Create masks for each type
    obj_mask = (red_vals > 0) & (green_vals == 0) & (blue_vals == 0)
    canon_mask = (red_vals == 0) & (green_vals > 0) & (blue_vals == 0)
    bg_mask = (red_vals == 0) & (green_vals == 0) & (blue_vals > 0)
    
    # Count pixels for each type
    obj_pixels = obj_mask.sum().item()
    canon_pixels = canon_mask.sum().item()
    bg_pixels = bg_mask.sum().item()
    total_pixels = H * W

    if obj_pixels > 0:
        # Simple check: count object pixels in border regions
        H, W = obj_mask.shape
        border_width = 20
        
        # Create border mask (pixels near edges)
        border_mask = torch.zeros_like(obj_mask)
        border_mask[:border_width, :] = 1  # Top
        border_mask[-border_width:, :] = 1  # Bottom
        border_mask[:, :border_width] = 1  # Left
        border_mask[:, -border_width:] = 1  # Right
        
        # Count object pixels in border regions (potential floaters)
        border_obj_pixels = (obj_mask & border_mask).sum().item()
        border_ratio = border_obj_pixels / obj_pixels if obj_pixels > 0 else 0
        

    return {
        "obj_mask": obj_mask.float(),
        "canon_mask": canon_mask.float(), 
        "bg_mask": bg_mask.float(),
        "obj_pixel_count": obj_pixels,
        "canon_pixel_count": canon_pixels,
        "bg_pixel_count": bg_pixels,
    }


def save_debug_id_maps(trainer, batch):
    """Save ID maps for debugging floater Gaussians"""
    
    debug_dir = "/local/home/pmishra/cvg/arti-splatfacto/arti_debug"  
    os.makedirs(debug_dir, exist_ok=True)
    
    try:
        camera = getattr(trainer, '_current_camera', None)
        if camera is None:
            print("No camera found in batch for debug saving")
            return
            
        # Render ID map
        debug_outputs = trainer.get_outputs(camera, render_id_map=True)
        
        id_map = debug_outputs["id_map"]
        obj_mask = debug_outputs["obj_mask"] 
        canon_mask = debug_outputs["canon_mask"]
        
        vutils.save_image(id_map.permute(2,0,1), f"{debug_dir}/step{trainer.step:06d}_id_map.png")
        vutils.save_image(obj_mask.unsqueeze(0), f"{debug_dir}/step{trainer.step:06d}_obj_mask.png") 
        vutils.save_image(canon_mask.unsqueeze(0), f"{debug_dir}/step{trainer.step:06d}_canon_mask.png")
        
        normal_outputs = trainer.get_outputs(camera, render_id_map=False)
        rgb_rendered = normal_outputs["rgb"]
        vutils.save_image(rgb_rendered.permute(2,0,1), f"{debug_dir}/step{trainer.step:06d}_rgb_rendered.png")
        
        gt_img = trainer.get_gt_img(batch["image"])
        if trainer._get_downscale_factor() > 1:
            import torchvision.transforms.functional as TF
            d = trainer._get_downscale_factor()
            newsize = (gt_img.shape[0] // d, gt_img.shape[1] // d)
            gt_img = TF.resize(gt_img.permute(2, 0, 1), newsize, antialias=None).permute(1, 2, 0)
        
        vutils.save_image(gt_img.permute(2,0,1), f"{debug_dir}/step{trainer.step:06d}_gt_image.png")
        
        
    except Exception as e:
        print(f"Failed to save debug ID maps: {e}")

def depth_debug(step: int, depth_out: torch.Tensor, depth_gt: torch.Tensor, mask: torch.Tensor = None,
                out_dir: str = "/local/home/pmishra/cvg/arti-splatfacto/debug_depth"):
    """
    Save depth debug visualizations: predicted, ground truth, diff, and mask (optional).
    """
    os.makedirs(out_dir, exist_ok=True)

    # Print ranges
    print(f"Step {step} - Depth Analysis:")
    print(f"  Predicted depth range: {depth_out.min():.4f} to {depth_out.max():.4f}")
    print(f"  Ground truth depth range: {depth_gt.min():.4f} to {depth_gt.max():.4f}")
    print(f"  Depth ratio (pred/gt): {depth_out.mean() / depth_gt.mean():.4f}")

    # Normalize to [0,1]
    def norm(x):
        return (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else torch.zeros_like(x)

    depth_out_norm, depth_gt_norm = norm(depth_out), norm(depth_gt)

    # Save images
    vutils.save_image(depth_out_norm.permute(2, 0, 1), f"{out_dir}/step{step:06d}_pred_depth.png")
    vutils.save_image(depth_gt_norm.permute(2, 0, 1), f"{out_dir}/step{step:06d}_gt_depth.png")
    vutils.save_image(torch.abs(depth_out_norm - depth_gt_norm).permute(2, 0, 1),
                      f"{out_dir}/step{step:06d}_depth_diff.png")

    if mask is not None:
        vutils.save_image(mask.float().permute(2, 0, 1), f"{out_dir}/step{step:06d}_mask.png")
