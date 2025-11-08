import torch
from typing import Dict
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


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

    border_obj_pixels = 0
    border_ratio = 0.0
    
    if obj_pixels > 0:
        # Simple check: count object pixels in border regions
        border_width = 20
        
        # Create border mask (pixels near edges)
        border_mask = torch.zeros_like(obj_mask)
        border_mask[:border_width, :] = True
        border_mask[-border_width:, :] = True
        border_mask[:, :border_width] = True
        border_mask[:, -border_width:] = True
        
        border_obj_pixels = (obj_mask & border_mask).sum().item()
        border_ratio = border_obj_pixels / obj_pixels

    return {
        "obj_mask": obj_mask.float(),
        "canon_mask": canon_mask.float(), 
        "bg_mask": bg_mask.float(),
        "obj_pixel_count": obj_pixels,
        "canon_pixel_count": canon_pixels,
        "bg_pixel_count": bg_pixels,
        "border_obj_pixel_count": border_obj_pixels,
        "border_ratio": border_ratio,
    }


def save_debug_id_maps(trainer, batch):
    """Save ID maps for debugging floater Gaussians in a compact combined image"""
    debug_dir = Path("debug_id_maps")
    debug_dir.mkdir(exist_ok=True, parents=True)
    
    try:
        camera = getattr(trainer, '_current_camera', None)
        if camera is None:
            print("⚠ No camera found in batch for debug saving")
            return
        
        debug_outputs = trainer.get_outputs(camera, render_id_map=True)
        id_map = debug_outputs["id_map"]
        obj_mask = debug_outputs["obj_mask"] 
        canon_mask = debug_outputs["canon_mask"]
        
        normal_outputs = trainer.get_outputs(camera, render_id_map=False)
        rgb_rendered = normal_outputs["rgb"]
        
        gt_img = trainer.get_gt_img(batch["image"])
        if trainer._get_downscale_factor() > 1:
            import torchvision.transforms.functional as TF
            d = trainer._get_downscale_factor()
            newsize = (gt_img.shape[0] // d, gt_img.shape[1] // d)
            gt_img = TF.resize(gt_img.permute(2, 0, 1), newsize, antialias=None).permute(1, 2, 0)
        
        # Convert to numpy
        to_numpy = lambda x: x.detach().cpu().numpy()
        id_map_np = to_numpy(id_map)
        obj_mask_np = to_numpy(obj_mask)
        canon_mask_np = to_numpy(canon_mask)
        rgb_rendered_np = to_numpy(rgb_rendered)
        gt_img_np = to_numpy(gt_img)
        
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(2, 3, hspace=0.25, wspace=0.15)
        
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(gt_img_np)
        ax1.set_title("Ground Truth", fontsize=12, fontweight='bold')
        ax1.axis('off')
        
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(rgb_rendered_np)
        ax2.set_title("Rendered RGB", fontsize=12, fontweight='bold')
        ax2.axis('off')
        
        ax3 = fig.add_subplot(gs[0, 2])
        diff_rgb = np.abs(gt_img_np - rgb_rendered_np).mean(axis=-1)
        im3 = ax3.imshow(diff_rgb, cmap='hot')
        ax3.set_title(f"Difference (MAE: {diff_rgb.mean():.4f})", fontsize=12, fontweight='bold')
        ax3.axis('off')
        # plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
        
        ax4 = fig.add_subplot(gs[1, 0])
        if len(id_map_np.shape) == 2 or (len(id_map_np.shape) == 3 and id_map_np.shape[-1] == 1):
            id_map_vis = id_map_np.squeeze() if len(id_map_np.shape) == 3 else id_map_np
            im4 = ax4.imshow(id_map_vis, cmap='tab20')
        else:
            im4 = ax4.imshow(id_map_np)
        unique_ids = len(np.unique(id_map_np))
        ax4.set_title(f"ID Map ({unique_ids} objects)", fontsize=12, fontweight='bold')
        ax4.axis('off')
        
        ax5 = fig.add_subplot(gs[1, 1])
        im5 = ax5.imshow(obj_mask_np, cmap='viridis', vmin=0, vmax=1)
        ax5.set_title(f"Object Mask ({obj_mask_np.mean():.1%})", fontsize=12, fontweight='bold')
        ax5.axis('off')
        # plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)
        
        ax6 = fig.add_subplot(gs[1, 2])
        im6 = ax6.imshow(canon_mask_np, cmap='plasma', vmin=0, vmax=1)
        ax6.set_title(f"Canonical Mask ({canon_mask_np.mean():.1%})", fontsize=12, fontweight='bold')
        ax6.axis('off')
        # plt.colorbar(im6, ax=ax6, fraction=0.046, pad=0.04)
        
        fig.suptitle(f"Debug Visualization - Step {trainer.step:06d}", 
                     fontsize=16, fontweight='bold')
        
        output_path = debug_dir / f"step{trainer.step:06d}_combined_debug.png"
        plt.savefig(output_path, dpi=120, bbox_inches='tight')
        plt.close(fig)
        
        # print(f"✓ Saved debug to {output_path}")
        
    except Exception as e:
        print(f"✗ Failed to save debug ID maps: {e}")
        import traceback
        traceback.print_exc()


def save_depth_debug(step: int, depth_out: torch.Tensor, depth_gt: torch.Tensor,
                     mask: torch.Tensor = None, scale: float = None, shift: float = None):
    """
    Save simple depth debug visualization as a combined image with consistent color scaling.
    """
    debug_dir = Path("debug_depth")
    debug_dir.mkdir(exist_ok=True, parents=True)

    # Apply mask if given
    if mask is not None:
        valid_mask = (mask > 0.5)
        depth_out_masked = depth_out * valid_mask
        depth_gt_masked = depth_gt * valid_mask
    else:
        depth_out_masked = depth_out
        depth_gt_masked = depth_gt
        valid_mask = torch.ones_like(depth_gt, dtype=torch.bool)

    # Print ranges
    print(f"Step {step} - Depth Debug:")
    print(f"  Pred range: {depth_out[valid_mask].min():.4f} - {depth_out[valid_mask].max():.4f}")
    print(f"  GT range:   {depth_gt[valid_mask].min():.4f} - {depth_gt[valid_mask].max():.4f}")

    # Apply alignment if given
    if scale is not None and shift is not None:
        depth_out_aligned = scale * depth_out_masked + shift
        print(f"  Aligned range: {depth_out_aligned[valid_mask].min():.4f} - {depth_out_aligned[valid_mask].max():.4f}")
        print(f"  Scale: {scale:.4f}, Shift: {shift:.4f}")
    else:
        depth_out_aligned = depth_out_masked

    vmin = torch.min(torch.stack([depth_gt, depth_out, depth_out_aligned])).item()
    vmax = torch.max(torch.stack([depth_gt, depth_out, depth_out_aligned])).item()

    def norm_shared(x):
        return (x - vmin) / (vmax - vmin + 1e-8)

    depth_gt_norm = norm_shared(depth_gt_masked)
    depth_out_norm = norm_shared(depth_out_masked)
    depth_aligned_norm = norm_shared(depth_out_aligned)
    depth_diff = torch.abs(depth_aligned_norm - depth_gt_norm)

    # Convert to numpy for plotting
    def to_np(x): return x.squeeze().detach().cpu().numpy()
    depth_gt_np = to_np(depth_gt_norm)
    depth_out_np = to_np(depth_out_norm)
    depth_aligned_np = to_np(depth_aligned_norm)
    depth_diff_np = to_np(depth_diff)

    # Plot
    num_plots = 5 if mask is not None else 4
    fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 5))

    im0 = axes[0].imshow(depth_gt_np, cmap='turbo', vmin=0, vmax=1)
    axes[0].set_title("GT Depth"); axes[0].axis('off')
    # plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(depth_out_np, cmap='turbo', vmin=0, vmax=1)
    axes[1].set_title("Pred (Raw)"); axes[1].axis('off')
    # plt.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(depth_aligned_np, cmap='turbo', vmin=0, vmax=1)
    axes[2].set_title("Pred (Aligned)"); axes[2].axis('off')
    # plt.colorbar(im2, ax=axes[2])

    im3 = axes[3].imshow(depth_diff_np, cmap='hot')
    axes[3].set_title(f"Diff (mean: {depth_diff[valid_mask].mean():.4f})")
    axes[3].axis('off')
    plt.colorbar(im3, ax=axes[3])

    if mask is not None:
        im4 = axes[4].imshow(mask.squeeze().cpu().numpy(), cmap='gray', vmin=0, vmax=1)
        axes[4].set_title("Mask"); axes[4].axis('off')
        plt.colorbar(im4, ax=axes[4])

    fig.suptitle(f"Depth Debug - Step {step:06d}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(debug_dir / f"step{step:06d}_depth_combined.png", dpi=150, bbox_inches='tight')
    plt.close(fig)


def save_normal_debug(step: int, normals_gt, normals_pred, mask=None):
    """Helper to save normal visualization for debugging"""
    debug_dir = Path("debug_normals")
    debug_dir.mkdir(exist_ok=True, parents=True)

    normals_gt_vis = (normals_gt + 1) / 2
    normals_pred_vis = (normals_pred + 1) / 2

    if mask is not None:
        normals_gt_vis = normals_gt_vis * mask
        normals_pred_vis = normals_pred_vis * mask

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(normals_gt_vis.detach().cpu().numpy())
    axes[0].set_title("GT Normals", fontsize=12, fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(normals_pred_vis.detach().cpu().numpy())
    axes[1].set_title("Predicted Normals", fontsize=12, fontweight='bold')
    axes[1].axis('off')

    diff = torch.abs(normals_gt_vis - normals_pred_vis).mean(dim=-1)
    im2 = axes[2].imshow(diff.detach().cpu().numpy(), cmap='hot')
    axes[2].set_title(f"Difference (mean: {diff.mean():.4f})", fontsize=12, fontweight='bold')
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    fig.suptitle(f"Normal Debug - Step {step:06d}", fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(debug_dir / f"normals_step_{step:06d}.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # print(f"✓ Saved normal debug to {debug_dir / f'normals_step_{step:06d}.png'}")