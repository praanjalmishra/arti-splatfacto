import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from change_det.utils.img_utils import image_align
from change_det.utils.io import save_imgs
from joint_estimator.dinov2_utils import load_dinov2_model, compute_dinov2_similarity
from mask_refine.effsam_utils import effsam_embedding
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
    sim = cv2.resize(sim, (W, H), interpolation=cv2.INTER_CUBIC)
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
    H, W = capture.shape[-2:]
    device = render.device


    render, align_mask = image_align(capture, render)
    capture = capture[0].permute(1, 2, 0).cpu().numpy()  # (H, W, 3)
    render = render[0].permute(1, 2, 0).cpu().numpy()
    align_mask = align_mask.squeeze().cpu().numpy().astype(np.uint8)


    if kernel_ratio > 0:
        kernel_size = int(W * kernel_ratio)
        kernel_size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size
        capture = cv2.GaussianBlur(capture, (kernel_size, kernel_size), 0)
        render = cv2.GaussianBlur(render, (kernel_size, kernel_size), 0)

    # Optional debugging
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
    assert masks.ndim == 4 and masks.shape[1] == 1, f"Got {masks.shape}"
    assert masks_all.ndim == 4 and masks_all.shape[1] == 1, f"Got {masks_all.shape}"

    # if self.debug_dir is not None and len(masks) > 0:
    #     for i, mask in enumerate(masks):
    #         cv2.imwrite(
    #             f"{self.debug_dir}/mask_{i}.png",
    #             mask.squeeze().cpu().numpy() * 255
    #         )

    print(f"[INFO] Found {len(masks)} masks of changed regions")
    print(f"[INFO] Found {len(masks_all)} masks of all changed regions")

    return masks, masks_all
