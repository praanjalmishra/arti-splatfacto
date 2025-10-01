#!/usr/bin/env python3
"""
Stage 1: Single-view mask refinement with embedding similarity and SAM validation.

Processes projected masks in mini-batches:
1. Extract EfficientSAM embeddings (batched for efficiency)
2. Refine boundaries using embedding similarity (FG vs BG prototypes)
3. Validate with SAM scoring (only keep masks with score > threshold)

Output: High-confidence refined masks for subset of views
"""

import argparse
from pathlib import Path
import json
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from tqdm import tqdm

# Import from effsam_utils (assumes it's in same directory or PYTHONPATH)
from effsam_utils import effsam_embedding, effsam_refine_masks


def load_image(path: Path, device='cuda') -> torch.Tensor:
    """Load RGB image as (1, 3, H, W) tensor."""
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img).float() / 255.0
    img = img.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
    return img.to(device)


def load_mask(path: Path, device='cuda') -> torch.Tensor:
    """Load binary mask as (H, W) bool tensor."""
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Cannot load mask: {path}")
    mask = torch.from_numpy(mask > 0).bool()
    return mask.to(device)


def refine_mask_with_embeddings(
    embedding: torch.Tensor,  # (C, H, W) on GPU
    projected_mask: torch.Tensor,  # (H, W) bool on GPU
    erosion_size: int = 20,
    dilation_size: int = 50,
    similarity_margin: float = 0.0
) -> torch.Tensor:
    """
    Refine mask boundaries using embedding similarity.
    
    Args:
        embedding: (C, H, W) pixel embeddings on GPU
        projected_mask: (H, W) binary mask on GPU
        erosion_size: kernel size for confident FG region
        dilation_size: kernel size for safe BG region
        similarity_margin: confidence margin (fg_sim > bg_sim + margin)
    
    Returns:
        refined_mask: (H, W) bool tensor on GPU
    """
    C, H, W = embedding.shape
    
    # Create confident regions (use CPU for morphological ops, then back to GPU)
    mask_np = projected_mask.cpu().numpy().astype(np.uint8) * 255
    
    kernel_erode = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (erosion_size, erosion_size)
    )
    kernel_dilate = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (dilation_size, dilation_size)
    )
    
    mask_eroded = cv2.erode(mask_np, kernel_erode) > 0
    mask_dilated = cv2.dilate(mask_np, kernel_dilate) > 0
    mask_boundary = mask_dilated & ~mask_eroded
    
    # Convert back to GPU tensors
    mask_eroded = torch.from_numpy(mask_eroded).to(embedding.device)
    mask_dilated = torch.from_numpy(mask_dilated).to(embedding.device)
    mask_boundary = torch.from_numpy(mask_boundary).to(embedding.device)
    
    # Sample embeddings from confident regions
    fg_embeddings = embedding[:, mask_eroded].T  # (N_fg, C)
    bg_embeddings = embedding[:, ~mask_dilated].T  # (N_bg, C)
    
    # Check if we have enough samples
    if len(fg_embeddings) < 10 or len(bg_embeddings) < 10:
        # Not enough samples - return original mask
        return projected_mask
    
    # Compute prototypes (centroids)
    fg_proto = fg_embeddings.mean(dim=0)  # (C,)
    bg_proto = bg_embeddings.mean(dim=0)  # (C,)
    
    # Normalize prototypes for cosine similarity
    fg_proto = F.normalize(fg_proto.unsqueeze(0), dim=1)  # (1, C)
    bg_proto = F.normalize(bg_proto.unsqueeze(0), dim=1)  # (1, C)
    
    # Get boundary embeddings
    boundary_embeddings = embedding[:, mask_boundary].T  # (N_boundary, C)
    
    if len(boundary_embeddings) == 0:
        # No boundary pixels - return original
        return projected_mask
    
    # Normalize boundary embeddings
    boundary_embeddings = F.normalize(boundary_embeddings, dim=1)  # (N_boundary, C)
    
    # Compute cosine similarities
    fg_sim = (boundary_embeddings @ fg_proto.T).squeeze()  # (N_boundary,)
    bg_sim = (boundary_embeddings @ bg_proto.T).squeeze()  # (N_boundary,)
    
    # Classify boundary pixels with optional margin
    boundary_is_fg = (fg_sim > bg_sim + similarity_margin)
    
    # Build refined mask
    refined_mask = projected_mask.clone()
    
    # Get boundary pixel coordinates
    boundary_coords = mask_boundary.nonzero()  # (N_boundary, 2)
    
    # # Assign boundary pixels
    # for idx, is_fg in enumerate(boundary_is_fg):
    #     y, x = boundary_coords[idx]
    #     refined_mask[y, x] = is_fg

    refined_mask[boundary_coords[:, 0], boundary_coords[:, 1]] = boundary_is_fg
    
    return refined_mask


def validate_mask_with_sam(
    rgb: torch.Tensor,  # (1, 3, H, W)
    refined_mask: torch.Tensor,  # (H, W) bool
    expand: float = 0.05
) -> float:
    """
    Validate refined mask using SAM scoring.
    
    Args:
        rgb: (1, 3, H, W) RGB image
        refined_mask: (H, W) bool mask
        expand: bbox expansion ratio
    
    Returns:
        sam_score: confidence score from SAM (0-1)
    """
    # Convert to format expected by effsam_refine_masks
    mask_tensor = refined_mask.float().unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    
    # Get SAM's score via bbox prompt
    _, scores = effsam_refine_masks(rgb, mask_tensor, expand=expand)
    
    return scores[0]

def process_batch(
    rgb_paths: list,
    mask_paths: list,
    output_dir: Path,
    device: str = 'cuda',
    erosion_size: int = 20,
    dilation_size: int = 50,
    min_sam_score: float = 0.95,
    write_overlays: bool = False,
    overlay_dir: Path = None
) -> dict:
    """
    Process a batch of frames.
    
    Returns:
        stats: dict with processing statistics
    """
    batch_size = len(rgb_paths)
    
    # Load batch
    rgbs = [load_image(p, device) for p in rgb_paths]  # List of (1, 3, H, W)
    masks = [load_mask(p, device) for p in mask_paths]
    
    # Extract embeddings individually (effsam_embedding doesn't support batching)
    embeddings = []
    with torch.no_grad():
        for rgb in rgbs:
            emb = effsam_embedding(rgb, upsample=True)  # (1, C, H, W)
            embeddings.append(emb.squeeze(0))  # Store as (C, H, W)
    
    stats = {
        'processed': 0,
        'high_conf': 0,
        'low_conf': 0,
        'scores': []
    }
    
    # Refine each mask individually
    for i in range(batch_size):
        frame_name = rgb_paths[i].stem
        
        # Embedding-based refinement
        refined_mask = refine_mask_with_embeddings(
            embeddings[i], masks[i], erosion_size, dilation_size
        )
        
        # Validate with SAM
        sam_score = validate_mask_with_sam(
            rgbs[i], refined_mask, expand=0.05  # rgbs[i] is already (1, 3, H, W)
        )
        sam_score = float(sam_score)
        stats['processed'] += 1
        stats['scores'].append(sam_score)
        
        if sam_score >= min_sam_score:
            stats['high_conf'] += 1
            
            # Save refined mask
            mask_np = refined_mask.cpu().numpy().astype(np.uint8) * 255
            output_path = output_dir / f"{frame_name}.png"
            cv2.imwrite(str(output_path), mask_np)
            
            # Optional: save overlay
            if write_overlays and overlay_dir is not None:
                rgb_np = (rgbs[i].squeeze(0).cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                rgb_np = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)
                overlay = rgb_np.copy()
                overlay[mask_np > 0] = [0, 255, 0]  # Green
                result = cv2.addWeighted(rgb_np, 0.7, overlay, 0.3, 0)
                
                # Add score text
                cv2.putText(result, f"Score: {sam_score:.3f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                cv2.imwrite(str(overlay_dir / f"{frame_name}.png"), result)
        else:
            stats['low_conf'] += 1
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Stage 1: Single-view mask refinement with SAM validation"
    )
    parser.add_argument("data_dir", type=str,
                       help="Data directory containing rgb/, masks_*_projected/")
    parser.add_argument("--pose", type=str, choices=['pre', 'post'], required=True,
                       help="Which pose to process (pre or post)")
    parser.add_argument("--batch-size", type=int, default=4,
                       help="Batch size for processing (default: 4)")
    parser.add_argument("--erosion-size", type=int, default=20,
                       help="Erosion kernel size for confident FG")
    parser.add_argument("--dilation-size", type=int, default=50,
                       help="Dilation kernel size for safe BG")
    parser.add_argument("--min-sam-score", type=float, default=0.90,
                       help="Minimum SAM score threshold")
    parser.add_argument("--overlays", action="store_true",
                       help="Write debug overlays")
    parser.add_argument("--device", type=str, default='cuda',
                       help="Device (cuda or cpu)")
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    pose = args.pose
    
    # Setup paths
    rgb_dir = data_dir / "rgb"
    mask_input_dir = data_dir / f"masks_{pose}"
    mask_output_dir = data_dir / f"masks_{pose}_refined"
    mask_output_dir.mkdir(exist_ok=True)
    
    overlay_dir = None
    if args.overlays:
        overlay_dir = data_dir / "debug_overlays" / f"{pose}_stage1"
        overlay_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all frames
    mask_files = sorted(mask_input_dir.glob("*.png"))
    print(f"\n{'='*60}")
    print(f"Stage 1: Refining {pose} masks")
    print(f"Input: {mask_input_dir}")
    print(f"Output: {mask_output_dir}")
    print(f"Total frames: {len(mask_files)}")
    print(f"Batch size: {args.batch_size}")
    print(f"Min SAM score: {args.min_sam_score}")
    print(f"{'='*60}\n")
    
    # Process in batches
    global_stats = {
        'total_processed': 0,
        'total_high_conf': 0,
        'total_low_conf': 0,
        'all_scores': []
    }
    
    for i in tqdm(range(0, len(mask_files), args.batch_size), desc=f"Refining {pose} masks", unit="frame"):
        batch_mask_files = mask_files[i:i+args.batch_size]
        batch_rgb_files = [rgb_dir / f"{m.stem}.png" for m in batch_mask_files]

        batch_stats = process_batch(
            batch_rgb_files, batch_mask_files, mask_output_dir,
            device=args.device,
            erosion_size=args.erosion_size,
            dilation_size=args.dilation_size,
            min_sam_score=args.min_sam_score,
            write_overlays=args.overlays,
            overlay_dir=overlay_dir
        )

        # Accumulate stats
        global_stats['total_processed'] += batch_stats['processed']
        global_stats['total_high_conf'] += batch_stats['high_conf']
        global_stats['total_low_conf'] += batch_stats['low_conf']
        global_stats['all_scores'].extend(batch_stats['scores'])

        torch.cuda.empty_cache()


    
    # Summary
    avg_score = np.mean(global_stats['all_scores'])
    pass_rate = global_stats['total_high_conf'] / global_stats['total_processed'] * 100
    
    print(f"\n{'='*60}")
    print(f"Stage 1 Complete - {pose} masks")
    print(f"{'='*60}")
    print(f"Total processed:     {global_stats['total_processed']}")
    print(f"High confidence:     {global_stats['total_high_conf']} ({pass_rate:.1f}%)")
    print(f"Low confidence:      {global_stats['total_low_conf']}")
    print(f"Average SAM score:   {avg_score:.3f}")
    print(f"Output directory:    {mask_output_dir}")
    if args.overlays:
        print(f"Debug overlays:      {overlay_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()