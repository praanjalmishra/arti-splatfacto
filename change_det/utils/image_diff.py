import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from change_det.utils.img_utils import image_align
from change_det.utils.io import save_imgs, save_embedding_pca
from joint_estimator.dinov2_utils import load_dinov2_model, compute_dinov2_similarity
from mask_refine.effsam_utils import effsam_embedding
from change_det.utils.sam2_utils import sam_embedding
from lightglue import viz2d



def image_diff_dinov2(
    capture, render, debug_dir,
    threshold=1e-2, kernel_ratio=0.05, device="cuda"
):
    """
    DINOv2-based semantic image differencing for robust change detection.

    Args:
        capture (1x3xHxW): Captured image tensor
        render  (1x3xHxW): Rendered image tensor
        processor, model: from load_dinov2_model()
        threshold (float): Fractional area threshold to ignore small regions
        kernel_ratio (float): Gaussian blur kernel fractional size (no blur if <= 0)
        device (str): "cuda" or "cpu"
    Returns:
        masks (Nx1xHxW): Significant change region masks
        masks_all (Nx1xHxW): All detected change region masks
    """
    H, W = capture.shape[-2:]
    render, align_mask = image_align(capture, render)
    device = torch.device(device)

    cap_np = capture[0].permute(1, 2, 0).cpu().numpy()
    ren_np = render[0].permute(1, 2, 0).cpu().numpy()
    align_mask = align_mask.squeeze().cpu().numpy().astype(np.uint8)

    if kernel_ratio > 0:
        k = int(W * kernel_ratio)
        k = k + 1 if k % 2 == 0 else k
        cap_np = cv2.GaussianBlur(cap_np, (k, k), 0)
        ren_np = cv2.GaussianBlur(ren_np, (k, k), 0)

    dinov2_processor, dinov2_model = load_dinov2_model(device=device) 

    # --- compute DINOv2 patch-level similarity map ---
    sim = compute_dinov2_similarity(cap_np, ren_np, dinov2_model, dinov2_processor, device=device)
    sim = cv2.resize(sim, (W, H), interpolation=cv2.INTER_LANCZOS4)
    sim = cv2.GaussianBlur(sim, (5, 5), 0)
    sim = (sim - sim.min()) / (sim.max() - sim.min() + 1e-8)
    sim_map = (sim * 255).astype(np.uint8)

    if debug_dir is not None:
        cv2.imwrite(f"{debug_dir}/dinov2_similarity.png", sim_map)

    # --- threshold & contour extraction ---
    _, thresh = cv2.threshold(sim_map, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    thresh = thresh * align_mask

    if debug_dir is not None:
        cv2.imwrite(f"{debug_dir}/dinov2_thresh.png", thresh)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    masks, masks_all = [], []
    for contour in contours:
        area = cv2.contourArea(contour)
        mask = np.zeros((H, W), dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, (255, 255, 255), thickness=cv2.FILLED)
        mask_t = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).to(device)
        masks_all.append(mask_t)
        if area >= threshold * H * W:
            masks.append(mask_t)

    masks = torch.cat(masks, dim=0) if masks else torch.zeros((0,1,H,W), device=device)
    masks_all = torch.cat(masks_all, dim=0) if masks_all else torch.zeros((0,1,H,W), device=device)

    print(f"[INFO] (DINOv2) Found {len(masks)} significant and {len(masks_all)} total change masks")
    print(f"[DEBUG] capture size: {capture.shape[-2:]}, mask size: {masks[0].shape[-2:]}")
    return masks, masks_all


def image_diff_effsam(capture, render, debug_dir, threshold=1e-2, kernel_ratio=0.1):
    """
    EfficientSAM-based image differencing for change detection.

    Args:
        capture (1x3xHxW): Captured image tensor
        render  (1x3xHxW): Rendered image tensor
        threshold (float): Fractional area threshold to ignore small regions
        kernel_ratio (float): Gaussian blur kernel fractional size (no blur if <= 0)

    Returns:
        masks (Nx1xHxW): Significant change region masks
        masks_all (Nx1xHxW): All detected change region masks
    """

    os.makedirs(debug_dir, exist_ok=True)

    H, W = capture.shape[-2:]
    device = render.device

    print(f"[INFO] Performing EfficientSAM-based image differencing on images of size {H}x{W}")
    render, align_mask = image_align(capture, render)
    capture = capture[0].permute(1, 2, 0).cpu().numpy()  # (H, W, 3)
    render = render[0].permute(1, 2, 0).cpu().numpy()
    align_mask = align_mask.squeeze().cpu().numpy().astype(np.uint8)


    if kernel_ratio > 0:
        kernel_size = int(W * kernel_ratio)
        kernel_size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size
        capture = cv2.GaussianBlur(capture, (kernel_size, kernel_size), 0)
        render = cv2.GaussianBlur(render, (kernel_size, kernel_size), 0)

    if debug_dir is not None:
        viz2d.plot_images([capture, render])
        viz2d.save_plot(f"{debug_dir}/debug_input_pair.png")
        plt.close()

    emb1 = effsam_embedding(capture)  # (1,C,H',W')
    emb2 = effsam_embedding(render)   # (1,C,H',W')

    # Normalize feature maps
    norm1 = torch.nn.functional.normalize(emb1, p=2, dim=1)
    norm2 = torch.nn.functional.normalize(emb2, p=2, dim=1)

    # Save feature visualization (optional)
    if debug_dir is not None:
        save_imgs(norm1[:, :3], [f"{debug_dir}/feat1.png"])
        save_imgs(norm2[:, :3], [f"{debug_dir}/feat2.png"])


    with torch.no_grad():
        similarity_map = torch.nn.functional.cosine_similarity(norm1, norm2, dim=1)
        similarity_map = similarity_map.squeeze(0).cpu().numpy()  # (H, W)
        similarity_map = (similarity_map * 255).astype(np.uint8)

    if debug_dir is not None:
        cv2.imwrite(f"{debug_dir}/similarity_map.png", similarity_map)

    thresh = cv2.threshold(
        similarity_map, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    )[1]
    thresh = thresh * align_mask  # ignore black misalignment areas

    if debug_dir is not None:
        cv2.imwrite(f"{debug_dir}/thresh.png", thresh)

    contours, _ = cv2.findContours(
        thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    masks, masks_all = [], []
    for contour in contours:
        area = cv2.contourArea(contour)
        mask = np.zeros((H, W), dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, (255, 255, 255), thickness=cv2.FILLED)
        mask_t = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        masks_all.append(mask_t)
        if area >= threshold * H * W:
            masks.append(mask_t)


    if len(masks) > 0:
        masks = torch.cat(masks, dim=0).to(device)        # (N,1,H,W)
    else:
        masks = torch.zeros((0, 1, H, W), device=device)

    if len(masks_all) > 0:
        masks_all = torch.cat(masks_all, dim=0).to(device)  # (N_all,1,H,W)
    else:
        masks_all = torch.zeros((0, 1, H, W), device=device)

    # Sanity checks
    # assert masks.ndim == 4 and masks.shape[1] == 1, f"Got {masks.shape}"
    # assert masks_all.ndim == 4 and masks_all.shape[1] == 1, f"Got {masks_all.shape}"

    # if self.debug_dir is not None and len(masks) > 0:
    #     for i, mask in enumerate(masks):
    #         cv2.imwrite(
    #             f"{self.debug_dir}/mask_{i}.png",
    #             mask.squeeze().cpu().numpy() * 255
    #         )

    print(f"[INFO] Found {len(masks)} masks of changed regions")
    print(f"[INFO] Found {len(masks_all)} masks of all changed regions")

    return masks, masks_all




def image_diff_sam2_with_depth(capture_rgb, render_rgb, capture_depth, render_depth, 
                                 debug_dir, threshold=1e-2, kernel_ratio=0.1,
                                 depth_weight=0.1, merge_nearby=True, merge_distance=30):
    """
    Enhanced SAM2-based image differencing with depth information and smart merging.
    Combines RGB features and depth to detect changes, then merges nearby regions.
    
    Args:
        capture_rgb (1x3xHxW): Captured RGB image tensor
        render_rgb  (1x3xHxW): Rendered RGB image tensor
        capture_depth (1x1xHxW): Captured depth map tensor
        render_depth  (1x1xHxW): Rendered depth map tensor
        debug_dir (str): Directory to save debug visualizations
        threshold (float): Fractional area threshold to ignore small regions
        kernel_ratio (float): Gaussian blur kernel fractional size
        depth_weight (float): Weight for depth difference (0-1)
        merge_nearby (bool): Merge nearby regions that likely belong together
        merge_distance (int): Maximum distance in pixels to merge regions
        
    Returns:
        masks (Nx1xHxW): Significant change region masks (merged)
        masks_all (Nx1xHxW): All detected change region masks
    """
    # os.makedirs(debug_dir, exist_ok=True)
    H, W = capture_rgb.shape[-2:]
    device = render_rgb.device
    
    print(f"[INFO] Performing SAM2+Depth image differencing on {H}x{W} images")
    
    # Align images
    render_rgb, align_mask = image_align(capture_rgb, render_rgb)
    
    # Convert RGB to numpy
    capture_np = capture_rgb[0].permute(1, 2, 0).cpu().numpy()
    render_np = render_rgb[0].permute(1, 2, 0).cpu().numpy()
    align_mask = align_mask.squeeze().cpu().numpy().astype(np.uint8)
    
    # Process depth maps
    capture_depth_np = capture_depth[0, 0].cpu().numpy()  # Already (H, W)
    render_depth_np = render_depth[0, 0].cpu().numpy()    # Already (H, W)
    
    # Ensure 2D
    capture_depth_np = np.squeeze(capture_depth_np)
    render_depth_np = np.squeeze(render_depth_np)
    
    # Optional blur for RGB
    if kernel_ratio > 0:
        kernel_size = int(W * kernel_ratio)
        kernel_size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size
        capture_np = cv2.GaussianBlur(capture_np, (kernel_size, kernel_size), 0)
        render_np = cv2.GaussianBlur(render_np, (kernel_size, kernel_size), 0)
    
    if debug_dir is not None:
        viz2d.plot_images([capture_np, render_np])
        viz2d.save_plot(f"{debug_dir}/debug_input_pair.png")
        plt.close()
    
    # === SAM2 FEATURE EXTRACTION ===
    print("[INFO] Extracting SAM2 features...")
    emb_capture = sam_embedding(capture_np)
    emb_render = sam_embedding(render_np)
    
    norm_capture = torch.nn.functional.normalize(emb_capture, p=2, dim=1)
    norm_render = torch.nn.functional.normalize(emb_render, p=2, dim=1)
    
    # if debug_dir is not None:
    #     save_imgs(norm_capture[:, :3], [f"{debug_dir}/feat_capture.png"])
    #     save_imgs(norm_render[:, :3], [f"{debug_dir}/feat_render.png"])

    if debug_dir is not None:
        save_embedding_pca(emb_capture, f"{debug_dir}/feat_capture_pca.png")
        save_embedding_pca(emb_render, f"{debug_dir}/feat_render_pca.png")
    
    # === BIDIRECTIONAL FEATURE SIMILARITY ===
    print("[INFO] Computing bidirectional feature similarity...")
    with torch.no_grad():
        # Forward: what appears (revealed area)
        sim_forward = torch.nn.functional.cosine_similarity(norm_render, norm_capture, dim=1)
        sim_forward = sim_forward.squeeze(0).cpu().numpy()
        if sim_forward.shape != (H, W):
            sim_forward = cv2.resize(sim_forward, (W, H), interpolation=cv2.INTER_LINEAR)
        
        # Backward: what disappears (door)
        sim_backward = torch.nn.functional.cosine_similarity(norm_capture, norm_render, dim=1)
        sim_backward = sim_backward.squeeze(0).cpu().numpy()
        if sim_backward.shape != (H, W):
            sim_backward = cv2.resize(sim_backward, (W, H), interpolation=cv2.INTER_LINEAR)
    
    # === DEPTH DIFFERENCE ===
    print("[INFO] Computing depth difference...")
    depth_diff = np.abs(capture_depth_np - render_depth_np)
    
    # Normalize depth difference (handle invalid depths)
    valid_mask = (capture_depth_np > 0) & (render_depth_np > 0)
    depth_diff_norm = np.zeros_like(depth_diff)
    if valid_mask.sum() > 0:
        depth_diff_valid = depth_diff[valid_mask]
        depth_max = np.percentile(depth_diff_valid, 95)  # Use 95th percentile for robustness
        depth_diff_norm = np.clip(depth_diff / (depth_max + 1e-8), 0, 1)
    
    if debug_dir is not None:
        def safe_imwrite(path, img):
            """Safely write image with shape validation"""
            if img.ndim >= 2 and img.shape[0] > 0 and img.shape[1] > 0 and img.shape[0] < 100000 and img.shape[1] < 100000:
                try:
                    cv2.imwrite(path, img)
                except Exception as e:
                    print(f"[WARNING] Failed to save {path}: {e}")
            else:
                print(f"[WARNING] Skipping {path} - invalid dimensions: {img.shape}")
        
        safe_imwrite(f"{debug_dir}/depth_capture.png", 
                    (capture_depth_np / (capture_depth_np.max() + 1e-8) * 255).astype(np.uint8))
        safe_imwrite(f"{debug_dir}/depth_render.png", 
                    (render_depth_np / (render_depth_np.max() + 1e-8) * 255).astype(np.uint8))
        safe_imwrite(f"{debug_dir}/depth_diff.png", (depth_diff_norm * 255).astype(np.uint8))
    
    # === RGB DIFFERENCE ===
    rgb_diff = np.abs(capture_np - render_np).mean(axis=2)
    rgb_diff_norm = rgb_diff / (rgb_diff.max() + 1e-8)
    
    # Squeeze any extra dimensions first
    sim_forward = np.squeeze(sim_forward)
    sim_backward = np.squeeze(sim_backward)
    rgb_diff_norm = np.squeeze(rgb_diff_norm)
    depth_diff_norm = np.squeeze(depth_diff_norm)
    align_mask = np.squeeze(align_mask)

    
    # === COMBINE ALL CUES ===
    print("[INFO] Combining all difference cues...")
    # Convert similarities to dissimilarities
    dissim_forward = 1 - sim_forward
    dissim_backward = 1 - sim_backward
    
    # Weighted combination
    rgb_weight = 0.25
    feature_weight = 0.5 - depth_weight / 2  # Adjust feature weight based on depth weight
    
    combined_dissimilarity = (
        feature_weight * 0.5 * dissim_forward +      # Revealed area
        feature_weight * 0.5 * dissim_backward +     # Door movement
        rgb_weight * rgb_diff_norm +                  # Appearance change
        depth_weight * depth_diff_norm                # Depth change
    )
    
    # Convert back to similarity for visualization
    combined_similarity = 1 - combined_dissimilarity
    similarity_map = (combined_similarity * 255).astype(np.uint8)
    
    # Verify final shape
    
    if debug_dir is not None:
        # Save with shape validation
        def safe_imwrite(path, img):
            """Safely write image with shape validation"""
            if img.shape[0] > 0 and img.shape[1] > 0 and img.shape[0] < 100000 and img.shape[1] < 100000:
                try:
                    cv2.imwrite(path, img)
                except Exception as e:
                    print(f"[WARNING] Failed to save {path}: {e}")
            else:
                print(f"[WARNING] Skipping {path} - invalid dimensions: {img.shape}")
        
        safe_imwrite(f"{debug_dir}/sim_forward.png", (sim_forward * 255).astype(np.uint8))
        safe_imwrite(f"{debug_dir}/sim_backward.png", (sim_backward * 255).astype(np.uint8))
        safe_imwrite(f"{debug_dir}/rgb_diff.png", (rgb_diff_norm * 255).astype(np.uint8))
        safe_imwrite(f"{debug_dir}/similarity_combined.png", similarity_map)
    
    # === ADAPTIVE THRESHOLDING ===
    print("[INFO] Thresholding combined similarity map...")
    # Use Otsu thresholding
    thresh_val, thresh = cv2.threshold(
        similarity_map, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    )
    print(f"[INFO] Otsu threshold value: {thresh_val}")
    
    # Morphological operations to clean up
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_small)  # Close small holes
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_small)   # Remove small noise
    
    # Apply alignment mask
    thresh = thresh * align_mask
    
    if debug_dir is not None:
        cv2.imwrite(f"{debug_dir}/thresh_initial.png", thresh)
    
    # === MERGE NEARBY REGIONS ===
    if merge_nearby:
        print(f"[INFO] Merging regions within {merge_distance} pixels...")
        
        # Dilate to connect nearby regions
        kernel_merge = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, 
            (merge_distance, merge_distance)
        )
        thresh_dilated = cv2.dilate(thresh, kernel_merge)
        
        # Fill holes in dilated mask
        kernel_fill = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
        thresh_merged = cv2.morphologyEx(thresh_dilated, cv2.MORPH_CLOSE, kernel_fill)
        
        if debug_dir is not None:
            cv2.imwrite(f"{debug_dir}/thresh_dilated.png", thresh_dilated)
            cv2.imwrite(f"{debug_dir}/thresh_merged.png", thresh_merged)
        
        thresh = thresh_merged
    
    if debug_dir is not None:
        cv2.imwrite(f"{debug_dir}/thresh_final.png", thresh)
    
    # === FIND CONTOURS ===
    print("[INFO] Finding contours...")
    contours, _ = cv2.findContours(
        thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    print(f"[INFO] Found {len(contours)} contours")
    
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    # === CREATE MASKS ===
    masks, masks_all = [], []
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        mask = np.zeros((H, W), dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, (255, 255, 255), thickness=cv2.FILLED)
        mask_t = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        masks_all.append(mask_t)
        
        if area >= threshold * H * W:
            masks.append(mask_t)
            print(f"[INFO] Mask {i}: area={int(area)} pixels ({area/(H*W)*100:.2f}%)")
            if debug_dir is not None:
                cv2.imwrite(f"{debug_dir}/mask_{i}_area_{int(area)}.png", mask)
    
    if len(masks) > 0:
        masks = torch.cat(masks, dim=0).to(device)
    else:
        masks = torch.zeros((0, 1, H, W), device=device)
    
    if len(masks_all) > 0:
        masks_all = torch.cat(masks_all, dim=0).to(device)
    else:
        masks_all = torch.zeros((0, 1, H, W), device=device)
    
    print(f"[INFO] Found {len(masks)} significant change masks")
    print(f"[INFO] Found {len(masks_all)} total change masks")
    
    
    return masks, masks_all