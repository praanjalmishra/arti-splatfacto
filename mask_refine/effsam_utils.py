import numpy as np
import torch
import os
import sys
    
sys.path.append(os.path.expanduser("~/cvg/EfficientSAM"))
from efficient_sam.build_efficient_sam import (
    build_efficient_sam_vits, build_efficient_sam_vitt
)
from tqdm import tqdm


# Load EfficientSAM model
device = "cuda" if torch.cuda.is_available() else "cpu"
model_type = "vit_t"
if model_type == "vit_s":
    effsam = build_efficient_sam_vits()
elif model_type == "vit_t":
    effsam = build_efficient_sam_vitt()
else:
    raise ValueError("Invalid model type")
effsam.eval()
effsam.to(device)


def compute_2D_bbox(points):
    """
    Compute bboxes for a batch of 2D points

    Args:
        points (NxMx2 Tensor): 2D points

    Returns:
        bboxes (Nx4 Tensor): 2D bboxes (xyxy)
    """
    assert len(points.shape) == 3
    mins, _ = torch.min(points, dim=1)
    maxs, _= torch.max(points, dim=1)
    bboxes = torch.cat((mins, maxs), dim=1)
    return bboxes


def expand_2D_bbox(bboxes, percent=0.05):
    """
    Expand 2D bboxes by a certain percentage

    Args:
        bboxes (Nx4 tensor): 2D bboxes
        H, W (int): Image height and width
        percent (float): percentage to expand (xyxy)

    Returns:
        expanded_bboxes (Nx4 tensor): expanded bboxes (xyxy)
    """
    assert bboxes.shape[-1] == 4
    bboxes = bboxes.float()
    # Calculate width and height of each box
    widths = bboxes[:, 2] - bboxes[:, 0]
    heights = bboxes [:, 3] - bboxes[:, 1]
    # Calculate the expansion for width and height
    expand_width = widths * percent
    expand_height = heights * percent
    # Adjust the bbax coordinates 
    expanded_bboxes = bboxes.clone()
    expanded_bboxes[:, 0] -= expand_width / 2
    expanded_bboxes[:, 1] -= expand_height / 2
    expanded_bboxes[:, 2] += expand_width / 2
    expanded_bboxes[:, 3] += expand_height / 2
    expanded_bboxes [:, 3] += expand_height / 2
    return expanded_bboxes


# def effsam_predict(rgbs, bboxes=None, points=None):
#     """
#     Query SAM model with bboxes prompts

#     Args:
#         rgbs: (N, 3, H, W) RGB images
#         bboxes (N, 4): Bbox prompts (xyxy)
#         points (N, K, 2): Positive point prompts

#     Returns:
#         masks (N, 1, H, W): Image masks
#         scores (N-list): Confidence scores
#     """
#     if rgbs.shape[1] == 1:
#         rgbs = rgbs.repeat(1, 3, 1, 1)
#     elif rgbs.shape[1] == 3:
#         pass
#     else:
#         raise ValueError("RGB images are of shape (N, 3, H, W)")
#     assert bboxes is None or bboxes.shape[-1] == 4, \
#         "bbox prompts are of shape (N, 4)"
#     assert bboxes is None or bboxes.shape[0] == rgbs.shape[0], \
#         "Image bbox batch mismatch"
#     device = rgbs.device
#     pts, pts_labels = [], []
#     if bboxes is not None:
#         # Uncomment to debug
#         # debug_bbox_prompts(rgbs, bboxes, "/home/ziqi/Desktop/test/")
#         bbox_pts = bboxes.reshape(bboxes.shape[0], 1, -1, 2)
#         # Make labels for bbox points: 2 for top-left, 3 for bottom-right
#         labels = torch.tensor([2, 3]).to(bbox_pts.device).reshape(1, 1, -1)
#         pts.append(bbox_pts)
#         pts_labels.append(labels)
#     if points is not None:
#         assert points.shape[0] == rgbs.shape[0]
#         # Uncomment to debug
#         # debug_point_prompts(rgbs, points, "/home/ziqi/Desktop/test/")
#         pts.append(points[:, None, :, :])
#         pts_labels.append(torch.ones(1, 1, points.shape[1]).to(device))
#     pts = torch.cat(pts, dim=2)
#     pts_label = torch.cat(pts_labels, dim=2)
#     masks, scores = [], []
#     for rgb, bbox_pt in tqdm(zip(rgbs, pts), desc="EffSAM"):
#         rgb = rgb.to(device)
#         bbox_pt = bbox_pt.to(device)
#         logits, iou = effsam(rgb[None, ...], bbox_pt[None, ...], pts_label)
#         sorted_ids = torch.argsort(iou, dim=-1, descending=True)
#         iou = torch.take_along_dim(iou, sorted_ids, dim=2)
#         logits = torch.take_along_dim(
#             logits, sorted_ids[..., None, None], dim=2
#         )
#         mask = torch.ge(logits[0, 0, 0, :, :], 0)
#         masks.append(mask)
#         scores.append(iou[0, 0, 0].item())
#     masks = torch.stack(masks, dim=0).unsqueeze(1)
#     return masks, scores


def effsam_predict(rgbs, bboxes=None, points=None, top_k=1, device=None):
    """
    Query EfficientSAM model with bbox and/or point prompts.

    Args:
        rgbs (N, 3, H, W): RGB images
        bboxes (N, 4): Bbox prompts (xyxy), optional
        points (N, K, 2): Positive point prompts, optional
        top_k (int): Number of top masks to return per image
        device (torch.device): Target device (default: rgbs.device)

    Returns:
        masks (N, top_k, H, W): Predicted masks
        scores (N, top_k): Confidence scores
    """
    if rgbs.ndim != 4:
        raise ValueError("Expected input (N, 3, H, W)")
    if rgbs.shape[1] == 1:  # Grayscale → expand to 3-channels
        rgbs = rgbs.repeat(1, 3, 1, 1)
    elif rgbs.shape[1] != 3:
        raise ValueError("Images must have 1 or 3 channels")

    N = rgbs.shape[0]
    device = device or rgbs.device
    rgbs = rgbs.to(device)

    # Collect prompts
    all_pts, all_labels = [], []
    if bboxes is not None:
        if bboxes.shape[0] != N or bboxes.shape[1] != 4:
            raise ValueError("Bboxes must be of shape (N, 4)")
        bbox_pts = bboxes.reshape(N, 1, -1, 2).to(device)
        bbox_labels = torch.tensor([2, 3], device=device).reshape(1, 1, -1).repeat(N, 1, 1)
        all_pts.append(bbox_pts)
        all_labels.append(bbox_labels)

    if points is not None:
        if points.shape[0] != N:
            raise ValueError("Points must have shape (N, K, 2)")
        point_pts = points[:, None, :, :].to(device)
        point_labels = torch.ones((N, 1, points.shape[1]), device=device, dtype=torch.long)
        all_pts.append(point_pts)
        all_labels.append(point_labels)

    if not all_pts:
        raise ValueError("At least one of bboxes or points must be provided")

    pts = torch.cat(all_pts, dim=2)      # (N, 1, P, 2)
    pts_label = torch.cat(all_labels, dim=2)  # (N, 1, P)

    # Run EfficientSAM once on the whole batch
    with torch.no_grad():
        logits, iou = effsam(rgbs, pts, pts_label)  # logits: (N, 1, M, H, W), iou: (N, 1, M)

    # Sort by IoU and keep top_k
    sorted_ids = torch.argsort(iou, dim=-1, descending=True)
    sorted_logits = torch.gather(
        logits, 2, sorted_ids[..., None, None].expand(-1, -1, -1, logits.shape[-2], logits.shape[-1])
    )
    sorted_iou = torch.gather(iou, 2, sorted_ids)

    # Take top-k
    top_logits = sorted_logits[:, 0, :top_k]  # (N, top_k, H, W)
    top_scores = sorted_iou[:, 0, :top_k]     # (N, top_k)

    # Convert logits to binary masks
    masks = (top_logits >= 0).float()  # (N, top_k, H, W)

    return masks, top_scores


def effsam_batch_predict(rgb, bboxes):
    """
    Multi-bbox batch predict with EfficientSAM

    Args:
        rgb (1, 3, H, W): RGB images
        bboxes (N, 4): Bbox prompts (xyxy)
    
    Returns:
        masks (N, 1, H, W): Image masks
        scores (N-list): Confidence scores
    """
    assert rgb.shape[:2] == (1, 3)
    assert len(bboxes.shape) == 2 and bboxes.shape[-1] == 4

    bbox_pts = bboxes.reshape(1, bboxes.shape[0], 2, 2)
    # Make labels for bbox points: 2 for top-left, 3 for bottom-right
    labels = torch.tensor([2, 3]).to(bbox_pts.device)
    labels = labels[None, None].repeat(1, bboxes.shape[0], 1)
    logits, iou = effsam(rgb, bbox_pts, labels)
    sorted_ids = torch.argsort(iou, dim=-1, descending=True)
    iou = torch.take_along_dim(iou, sorted_ids, dim=2)
    logits = torch.take_along_dim(logits, sorted_ids[..., None, None], dim=2)
    masks = torch.ge(logits[0, :, 0, :, :], 0)
    masks = masks.unsqueeze(1)
    scores = iou[0, :, 0].tolist()
    return masks, scores


def effsam_embedding(rgb, upsample=True):
    """
    Get pixel-aligned image embeddings
    @param rgb (HxWx3 np.array or 1x3xHxW tensor): Image
    @param upsample (bool): Whether to upsample the features
    @return features (1xCxHxW tensor): Pixel-aligned image embeddings
    """
    if isinstance(rgb, np.ndarray):
        if rgb.dtype == np.uint8:
            rgb = rgb.astype(np.float32) / 255.0
        rgb = torch.from_numpy(rgb).permute(2, 0, 1).to(device)[None, ...]
    elif isinstance(rgb, torch.Tensor):
        assert rgb.dim() == 4, "Input tensor should be 1x1xHxW"
        rgb = rgb.to(device)
    features = effsam.get_image_embeddings(rgb).detach()
    if upsample:
        features = torch.nn.functional.interpolate(
            features, rgb.shape[-2:], mode="bilinear", align_corners=False
        )
    return features


def get_effsam_embedding_in_masks(rgbs, masks):
    """
    Get pixel-aligned image embeddings in masks
    @param rgbs (Nx3xHxW): RGB images
    @param masks (N-list of Mx1xHxW): 2D object masks
    @return features (N-list of MxK): Per-view per-object embedding vectors
    """
    from nerfstudio.utils.img_utils import median_high_dim
    assert rgbs.shape[1] == 3
    device = rgbs.device
    embeddings = []
    for rgb, mask in zip(rgbs, masks):
        embedding = effsam_embedding(rgb[None], upsample=False)
        if len(mask) == 0:
            embeddings.append(torch.zeros(0, embedding.shape[1]).to(device))
            continue
        assert mask.shape[1] == 1
        mask = torch.nn.functional.interpolate(
            mask.float(), embedding.shape[-2:], mode="nearest"
        ).bool()
        embed_vecs = []
        for m in mask:
            embedding_map = embedding[0, :, m.squeeze()].permute(1, 0)
            embedding_vec = median_high_dim(embedding_map)
            embed_vecs.append(embedding_vec)
        embed_vecs = torch.stack(embed_vecs, dim=0)
        embeddings.append(embed_vecs)
    return embeddings


def effsam_refine_masks(rgb, masks, expand=0.1):
    """
    Use SAM to refine the masks on a RGB image

    Args:
        rgbs: (1, 3, H, W) RGB images
        masks: (M, 1, H, W) Image masks
        expand (float): How much we expand the extracted bbox as prompt (%) 

    Returns:
        masks_refined (M, 1, H, W): Refined image masks   
        scores (M-list): Confidence scores     
    """
    assert rgb.shape[1] == 3
    assert len(masks.shape) == 4
    bboxes = []
    for mask in masks:
        point_coords = torch.nonzero(mask.squeeze())[:, [1, 0]]
        bbox = compute_2D_bbox(point_coords.unsqueeze(0))
        bbox = expand_2D_bbox(bbox, expand)
        bboxes.append(bbox)
    bboxes = torch.cat(bboxes, dim=0)
    masks_refined, scores = effsam_predict(
        rgb.repeat(masks.shape[0], 1, 1, 1), bboxes
    )
    return masks_refined, scores