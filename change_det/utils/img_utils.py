# 2D image and mask operation functions
import cv2
import numpy as np
import os
import torch

from glob import glob
from lightglue import LightGlue, SuperPoint
from lightglue.utils import rbd
from nerfstudio.data.datamanagers.full_images_datamanager import _undistort_image
from change_det.utils.io import params_to_cameras
from change_det.utils.effsam_utils import compute_2D_bbox
from PIL import Image, ImageOps
from scipy.ndimage import binary_dilation, generate_binary_structure
from torchvision.ops import RoIAlign, RoIPool
from tqdm import tqdm


def in_image(points2d, H, W):
    """
    Check if 2D points are in image

    Args:
        points2d (..., 2): 2D points
        H, W (int): Image height and width

    Returns:
        in_image_mask (...): Mask for points in image
    """
    assert points2d.shape[-1] == 2
    in_image_mask = (points2d[..., 0] >= 0) & (points2d[..., 0] < W) & \
        (points2d[..., 1] >= 0) & (points2d[..., 1] < H)
    return in_image_mask


def undistort_images(cams, imgs):
    """
    Undistort images (Images will become smaller after undistortion)

    Args:
        cams (NeRFStudio Cameras): Cameras
        imgs (Nx3xHxW): Images

    Returns:
        new_cams (Nx3x3): New cameras
        undistorted_imgs (Nx3xH'xW'): Undistorted images
    """
    assert cams.device == imgs.device, \
        f"Cameras on {cams.device} but images on {imgs.device}"
    imgs = imgs.permute(0, 2, 3, 1).cpu().numpy()
    res = [
        _undistort_image(
            cams[i], cams[i].distortion_params.numpy(), {}, imgs[i],
            cams[i].get_intrinsics_matrices().numpy()
        ) for i in tqdm(range(len(cams)), "Undistort images")
    ]
    undist_imgs = [r[1] for r in res]
    undist_imgs = torch.from_numpy(np.array(undist_imgs)).permute(0, 3, 1, 2)
    new_Ks = [r[0] for r in res]
    new_Ks = torch.from_numpy(np.array(new_Ks))
    new_dist_params = torch.zeros((len(cams), 4))
    new_poses = cams.camera_to_worlds.clone()
    new_poses[:, 0:3, 1:3] = -new_poses[:, 0:3, 1:3] # OpenGL to OpenCV
    H, W = undist_imgs.shape[-2:]
    new_cams = params_to_cameras(
        new_poses, new_Ks, new_dist_params, H, W
    )
    return new_cams, undist_imgs


# def extract_depths_at_pixels(pixels, depth):
#     """
#     Extract depth values for pixel coordinates from the depth map.

#     Parameters
#         pixels: (N, 2) Pixel coordinates
#         depth: (1, 1, H, W) Depth map of the image

#     Returns:
#         depths: (N, 1) Depth values at pixels
#     """
#     assert pixels.shape[-1] == 2
#     pixel_indices = pixels.long()
#     depth_values = depth[0, 0, pixel_indices[:, 1], pixel_indices[:, 0]]
#     return depth_values.reshape(-1, 1)

def extract_depths_at_pixels(pixels, depth):
    """
    Extract depth values for pixel coordinates from the depth map.

    Args:
        pixels (N, 2): Pixel coordinates (x, y)
        depth (1, 1, H, W) or (1, 1, 1, H, W): Depth map

    Returns:
        depths (N, 1): Depth values at each pixel
    """
    # Handle extra singleton dimension
    if depth.ndim == 5 and depth.shape[2] == 1:
        depth = depth.squeeze(2)
    elif depth.ndim != 4:
        raise ValueError(f"Expected depth shape (1,1,H,W), got {depth.shape}")

    _, _, H, W = depth.shape
    pixels = pixels.to(depth.device)

    if pixels.ndim == 1 and pixels.shape[0] == 2:
        pixels = pixels.unsqueeze(0)

    x = pixels[:, 0].round().long().clamp(0, W - 1)
    y = pixels[:, 1].round().long().clamp(0, H - 1)

    depth_values = depth[0, 0, y, x].unsqueeze(-1)
    return depth_values



def dilate_masks(masks, kernel_size=3):
    """
    Dilate binary masks using scipy's binary_dilation.
    
    Args:
        masks (Nx1xHxW): Binary masks
        kernel_size (int): Number of dilation iterations

    Returns:
        dilated_masks (Nx1xHxW): Dilated masks        
    """
    masks_np = masks.cpu().numpy()
    dilated_masks_np=[]
    for mask_np in tqdm(masks_np, desc="Dilate images"):
        dilated_mask_np = binary_dilation(
            mask_np[0], iterations=kernel_size,
            structure=generate_binary_structure(2, 3)
        )
        dilated_masks_np.append(dilated_mask_np)
    dilated_masks = torch.from_numpy(np.array(dilated_masks_np)).to(masks)
    return dilated_masks.unsqueeze(1)


def erode_masks(masks, kernel_size=3):
    """
    Erode binary masks using scipy's binary_erosion.
    
    Args:
        masks (Nx1xHxW): Binary masks
        kernel_size (int): Number of erosion iterations

    Returns:
        eroded_masks (Nx1xHxW): Eroded masks        
    """
    masks_np = masks.cpu().numpy()
    eroded_masks_np=[]
    for mask_np in tqdm(masks_np, desc="Erode images"):
        eroded_mask_np = cv2.erode(
            mask_np[0].astype(np.uint8),
            np.ones((kernel_size, kernel_size), np.uint8),
            iterations=1
        )
        eroded_masks_np.append(eroded_mask_np)
    eroded_masks = torch.from_numpy(np.array(eroded_masks_np)).to(masks)
    return eroded_masks.unsqueeze(1)


def invert_mask(mask_file, output_file):
    """Invert colors of a mask image.

    Args:
      mask_file: String, path to the mask image.
      output_file: String, path to the output mask image.
    """

    # Load the mask image
    mask = Image.open(mask_file)

    # Invert the colors of the mask image
    inverted_mask = ImageOps.invert(mask)

    # Save the inverted mask
    inverted_mask.save(output_file)



def split_masks(masks, threshold=1e-2):
    """
    Safely split disconnected masks in a batch of masks.

    Args:
        masks (Nx1xHxW): Binary masks tensor (0/1 or 0–255)
        threshold (float): Fractional area threshold to filter small components

    Returns:
        (Mx1xHxW): Split masks (M >= N)
    """
    if masks is None or masks.numel() == 0:
        print("[WARN] split_masks: empty input tensor")
        return torch.zeros(0, 1, 1, 1, device=masks.device if masks is not None else "cuda")

    assert len(masks.shape) == 4 and masks.shape[1] == 1, f"Invalid shape: {masks.shape}"

    H, W = masks.shape[-2:]
    device = masks.device

    # Ensure safe range and dtype before converting to numpy
    masks = masks.detach().float()
    if masks.max() <= 1.0:
        masks = (masks * 255).byte()
    else:
        masks = masks.byte()

    masks_np = masks.cpu().numpy()

    split_masks_np = []

    for mask_np in tqdm(masks_np, desc="Split masks"):
        mask_np = mask_np[0]  # (H, W)
        num_labels, labels = cv2.connectedComponents(mask_np.astype(np.uint8))
        for i in range(1, num_labels):
            component_mask = (labels == i).astype(np.uint8)
            if component_mask.sum() > threshold * H * W:
                split_masks_np.append(component_mask)

    # Handle no components safely
    if len(split_masks_np) == 0:
        return torch.zeros(0, 1, H, W, device=device, dtype=torch.float32)

    # Stack and move back to GPU
    split_masks_tensor = torch.from_numpy(np.stack(split_masks_np)).float().unsqueeze(1).to(device)
    return split_masks_tensor

def masks_to_focus(masks, kernel_ratio=0.15):
    """
    Dilate masks and only keep the new pixels masked out

    Args:
        masks (Nx1xHxW): Binary masks
        kernel_ratio (int): Ratio of the kernel size to the image size

    Returns:
        focus_masks (Nx1xHxW): Dilated masks
    """
    assert masks.shape[1] == 1, "Masks shape must be Nx1xHxW"
    masks_np = masks.cpu().numpy()
    kernel_size = int(masks.shape[-1] * kernel_ratio)
    focus_masks_np = []
    for mask_np in tqdm(masks_np, desc="Dilate to focus"):
        dilated_mask_np = binary_dilation(
            mask_np[0], iterations=kernel_size,
            structure=generate_binary_structure(2, 3)
        )
        # Keep only the new pixels white
        focus_mask_np = np.where(mask_np[0] == 1, 0, dilated_mask_np)
        # Invert the mask to keep only the new pixels masked out
        focus_mask_np = 1 - focus_mask_np
        focus_masks_np.append(focus_mask_np)
    focus_masks = torch.from_numpy(np.array(focus_masks_np)).to(masks)
    return focus_masks.unsqueeze(1)


def median_high_dim(vectors, num_samples=50):
    """
    Compute the median of high-dimensional vectors

    Args:
        vectors (N, D): High-dimensional vectors

    Returns:
        median (D): Median vector
    """
    assert vectors.dim() == 2, "Vectors must be NxD"
    # Sample points in mask to avoid OOM
    if vectors.shape[0] > num_samples:
        sampled_indices = torch.randperm(vectors.shape[0])[:num_samples]
        vectors = vectors[sampled_indices]
    # Median vector is the vector that has the minimum dist to other points
    dist_matrix = torch.cdist(vectors, vectors, p=2)
    sum_distances = dist_matrix.sum(dim=0)
    min_index = sum_distances.argmin()
    median = vectors[min_index]
    return median


def masks_median_points(masks, num_samples=50):
    """
    Find median points in the given masks

    Args:
        masks (N, 1, H, W): Binary masks

    Returns:
        median_points (N, 2): Median pixel coords
    """
    median_points = torch.zeros(masks.shape[0], 2).to(masks.device)
    for i, mask in enumerate(masks):
        assert mask.sum() > 0, f"The mask {i} has no white pixels"
        # Get indices of all ones in the masks
        y, x = torch.nonzero(masks[i, 0], as_tuple=True)
        yx = torch.stack((y, x), dim=1).float()
        median_points[i] = median_high_dim(yx, num_samples=num_samples)
    return median_points


def points2D_to_mask(points2D, valid2D, H, W):
    """
    Creates a mask from a set of 2D points.
    TODO: Make this more efficient
    
    Args:
        points2D (N, M, 2): 2D points
        valid2D (N, M): Mask for valid 2D points
        H (int): Height of the mask
        W (int): width of the mask

    Returns:
        mask (N, 1, H, W): Binary mask
    """
    assert points2D.shape[-1] == 2
    masks = []
    for idx, pts2D in enumerate(tqdm(points2D, desc="points to masks")):
        # Filter points that are outside the image
        valid_pts2D = pts2D[valid2D[idx]]
        mask = np.zeros((H, W), dtype=np.uint8)
        if len(valid_pts2D) != 0: # If no points in image, all-black mask
            valid_pts2D = valid_pts2D.cpu().numpy().astype(np.int32)
            hull = cv2.convexHull(valid_pts2D)
            cv2.fillPoly(mask, [hull], 1)
        mask = torch.from_numpy(mask).bool()[None, None, :, :]
        masks.append(mask)
    masks = torch.cat(masks, dim=0)
    return masks.to(points2D.device)


def points2D_to_point_masks(points2D, valid2D, H, W, kernel_size=3):
    """
    Creates a point mask from a set of 2D points.

    Args:
        points2D (N, M, 2): 2D points
        valid2D (N, M): Mask for valid 2D points
        H (int): Height of the mask
        W (int): width of the mask

    Returns:
        mask (N, 1, H, W): Binary mask
    """
    assert points2D.shape[-1] == 2
    masks = []
    for idx, pts2D in enumerate(tqdm(points2D, desc="points to pt masks")):
        # Filter points that are outside the image
        valid_pts2D = pts2D[valid2D[idx]]
        mask = np.zeros((H, W), dtype=np.uint8)
        if len(valid_pts2D) != 0: # If no points in image, all-black mask
            valid_pts2D = valid_pts2D.detach().cpu().numpy().astype(np.int32)
            is_in_img = in_image(valid_pts2D, H, W)
            valid_pts2D = valid_pts2D[is_in_img]
            mask[valid_pts2D[:, 1], valid_pts2D[:, 0]] = 255
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = torch.from_numpy(mask).bool()[None, None, :, :]
        masks.append(mask)
    masks = torch.cat(masks, dim=0)
    return masks.to(points2D.device)


def filter_features_with_mask(feats, masks):
    """
    Filter out features that are not in the mask

    Args:
        feats (dict): SuperPoint feature positions and descriptors
        masks (1x1xHxW): Binary masks

    Returns:
        feats (dict): SuperPoints within the mask
    """
    keypoints = feats['keypoints']
    valid_indices = masks[
        0, 0, keypoints[..., 1].long(), keypoints[..., 0].long()
    ] > 0.5
    feats['keypoints']= keypoints[valid_indices].unsqueeze(0)
    feats['keypoint_scores'] = \
        feats['keypoint_scores'][valid_indices].unsqueeze(0)
    feats['descriptors'] = \
        feats['descriptors'][valid_indices].unsqueeze(0)
    return feats


def image_matching(
    imgs1, imgs2, masks1=None, masks2=None, flip=False,
    extractor=None, matcher=None
):
    """
    Match imgs1 and imgs2 using their respective masks
    and return the keypoints and matches.

    Parameters:
        imgs1, imgs2 (N, 3, H, W): Images
        masks1, masks2 (N, 1, H, W or None): Masks
        flip (bool): Flip images in case img match is sensitive to rotation

    Returns:
        kp_list1 (N-list of M1x2 tensor): Keypoints on image 1
        kp_list2 (N-list of M2x2 tensor): Keypoints on image 2
        matches_list (N-list of Mx2 tensor): Matched keypoint IDs
    """
    device = imgs1.device
    # Load extractor and matcher modules
    if extractor is None:
        extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)
    if matcher is None:
        matcher = LightGlue(features='superpoint').eval().to(device)

    if flip:
        # Rotate images by 180 degrees if flip is True
        imgs1 = torch.flip(imgs1, [2, 3])
        if masks1 is not None:
            masks1 = torch.flip(masks1, [2, 3])

    # Initialize Lists to store results for each pair
    kp_list1, kp_list2, matches_list = [], [], []
    # Process each image pair individually
    for i in range(imgs1.size(0)):
        # Extract features
        feats1 = extractor.extract(imgs1[i])
        feats2 = extractor.extract(imgs2[i])
        # Ensure keypoints are within image
        feats1["keypoints"].clamp_(min=0)
        feats2["keypoints"].clamp_(min=0)
        # Filter keypoints using masks
        if masks1 is not None:
            feats1 = filter_features_with_mask(
                feats1, masks1[i:i+1].to(device)
            )
        if masks2 is not None:
            feats2 = filter_features_with_mask(
                feats2, masks2[i:i+1].to(device)
            )
        # Match the extracted features
        matches = matcher({'image0': feats1, 'image1': feats2})
        feats1, feats2, matches = [
            rbd(x) for x in [feats1, feats2, matches]
        ]
        if flip:
            H, W = imgs1.shape[2], imgs1.shape[3]
            feats1['keypoints'][:, 0] = W - feats1['keypoints'][:, 0] - 1
            feats1['keypoints'][:, 1] = H - feats1['keypoints'][:, 1] - 1
        # Append results to lists
        kp_list1.append(feats1['keypoints'])
        kp_list2.append(feats2['keypoints'])
        matches_list.append(matches ['matches'])
    return kp_list1, kp_list2, matches_list


def image_align(img1, img2):
    """
    Align image2 to image1 using homography estimation

    Args:
        img1, img2 (1x3xHxW): Images to align

    Returns:
        img2_aligned (1x3xHxW): Aligned image
        align_mask (1x1xHxW): Mask for the aligned image, True for valid pixels
    """
    H, W = img1.shape[-2:]
    kp1, kp2, matches = image_matching(img1, img2)

    num_matches = matches[0].shape[0]
    print(f"[image_align] Matches found: {num_matches}")

    if num_matches <= 4:
        print("⚠️ Not enough matches to compute homography — skipping alignment")
        return img2, torch.ones_like(img1[:, :1, :, :], dtype=torch.bool)

    kp1 = kp1[0].cpu().numpy()
    kp2 = kp2[0].cpu().numpy()
    matches = matches[0].cpu().numpy()

    src_pts = np.float32([kp1[m[0]] for m in matches])
    dst_pts = np.float32([kp2[m[1]] for m in matches])
    M, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

    img2_np = img2[0].permute(1, 2, 0).cpu().numpy()
    img2_aligned = cv2.warpPerspective(img2_np, M, (W, H))
    img2_aligned = torch.from_numpy(img2_aligned).permute(2, 0, 1).unsqueeze(0).to(img1.device)

    align_mask_np = np.ones((H, W), dtype=np.uint8)
    align_mask_np = cv2.warpPerspective(align_mask_np, M, (W, H))
    align_mask = torch.from_numpy(align_mask_np).unsqueeze(0).unsqueeze(0).bool().to(img1.device)

    return img2_aligned, align_mask



def batch_crop_resize(
    img, rois, out_H, out_W, aligned=True, interpolation="bilinear"
):
    """
    Crop and resize images

    Args:
        img: [BxCxHxW] Batch of images
        rois: [Bx4] Region of Interest [[x1, y1, x2, y1], ...]
        out_H: [int] Output size height
        out_W: [int] Output size width

    Returns:
        cropped: [BxCxH'xW'] Cropped image batch
    """
    assert len(img.shape) >= 3 and img.shape[-3] == 3, \
        "Error: Image size must be (*, 3, H, W)"
    assert rois.shape[-1] == 4, "Error: Bboxes should be Bx4"
    roi_idx = torch.arange(rois.size(0)).view(-1, 1).to(rois)
    rois = torch.cat((roi_idx, rois), dim=-1)
    # Crop and resize
    output_size = (out_H, out_W)
    if interpolation == "bilinear":
        op = RoIAlign(output_size, 1.0, 0, aligned=aligned)
    elif interpolation == "nearest":
        op = RoIPool(output_size, 1.0)  #
    else:
        raise ValueError(f"Wrong interpolation type: {interpolation}")
    return op(img, rois)


def crop_imgs_w_masks(images, masks, resize=(256, 256)):
    """
    Crop the images to the smallest bbox containing the masks

    Args:
        images (N, 3, H, W): Images
        masks (N, 1, H, W): Image masks

    Returns:
        images_cropped (N, 3, H', W'): Cropped images
    """
    # Masks to bboxes
    bboxes = []
    for mask in masks:
        point_coords = torch.nonzero(mask.squeeze())[:, [1, 0]]
        bbox = compute_2D_bbox(point_coords.unsqueeze(0)).float()
        bboxes.append(bbox)
    bboxes = torch.cat(bboxes, dim=0)
    imgs_cropped = batch_crop_resize(images, bboxes, *resize)
    return imgs_cropped


def rgb2rgba(rgb_files, mask_files, output_folder):
    """
    Convert RGB images to RGBA format using binary masks

    Args:
        rgb_files (list): List of RGB image paths
        mask_files (list): List of mask image paths
        output_folder (str): Output folder for RGBA images
    """
    # Ensure output folder exists
    assert os.path.isdir(output_folder), "Output folder does not exist"
    assert len(rgb_files) == len(mask_files), "Number of images do not match"
    # Iterate through all images in the RGB folder
    for rgb_img_path, mask_img_path in tqdm(zip(rgb_files, mask_files), desc="RGB2RGBA"):
        # Load the RGB(A) image and its corresponding mask
        rgb_image = Image.open(rgb_img_path).convert("RGBA")
        mask_image = Image.open(mask_img_path).convert("L")

        # Prepare the new alpha channel
        # Note: mask image is assumed to be in grayscale ('L' mode)
        # Convert mask to 'L' mode if it's not, ensuring compatibility
        alpha_channel = mask_image.point(lambda p: 255 if p > 0 else 0)

        # Combine original RGB channels with the new alpha channel
        rgb_channels = rgb_image.split()[:3]  # Ignore existing alpha if present
        new_image = Image.merge("RGBA", rgb_channels + (alpha_channel,))

        # Save the modified image
        rgb_img_basename = os.path.basename(rgb_img_path)
        output_path = os.path.join(output_folder, rgb_img_basename)
        new_image.save(output_path)


def get_interest_region(rgbs, masks, iteration=10):
    """
    Obtain regions of interest for a masked RGB image (iNeRF)

    Args:
        rgbs (Nx3xHxW): Captured RGB images
        masks (Nx1xHxW): Binary masks
        iteration (int): Number of dilation iterations

    Returns:
        masks_interest (Nx1xHxW): Interest region masks
    """
    device = rgbs.device
    extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)

    masks_interest = torch.zeros_like(masks).to(device)
    for ii, (rgb, mask) in enumerate(zip(rgbs, masks)):
        feats = extractor.extract(rgb)
        # Ensure keypoints are within image
        feats["keypoints"].clamp_(min=0)
        # Filter out keypoints outside the mask
        feats = filter_features_with_mask(feats, mask.unsqueeze(0).to(device))
        # Get feature point masks
        kpts = feats['keypoints'].squeeze(0).long()
        masks_interest[ii, 0, kpts[..., 1], kpts[..., 0]] = 1
    # Dilate the feature point masks
    masks_interest = dilate_masks(masks_interest, kernel_size=iteration)
    return masks_interest



def overlay_mask_on_image(rgb, mask, alpha=0.5, color=(255, 0, 0)):
    """
    Overlay a binary mask on an RGB image.

    Args:
        rgb (torch.Tensor | np.ndarray): (1,3,H,W), (3,H,W), or (H,W,3)
            RGB image in [0,1] or uint8.
        mask (torch.Tensor | np.ndarray): (1,H,W) or (H,W) binary mask.
        alpha (float): Transparency of overlay color.
        color (tuple[int, int, int]): (R,G,B) color for the overlay.

    Returns:
        np.ndarray: (H,W,3) uint8 image with mask overlay.
    """
    # --- Convert RGB to uint8 numpy (H,W,3) ---
    if isinstance(rgb, torch.Tensor):
        rgb = rgb.detach().cpu()
        if rgb.ndim == 4:
            rgb = rgb[0]  # (1,3,H,W) → (3,H,W)
        rgb_np = rgb.permute(1, 2, 0).numpy()
        if rgb_np.max() <= 1.0:
            rgb_np = (rgb_np * 255).astype(np.uint8)
        else:
            rgb_np = rgb_np.astype(np.uint8)
    else:
        rgb_np = rgb.copy()
        if rgb_np.dtype != np.uint8:
            if rgb_np.max() <= 1.0:
                rgb_np = (rgb_np * 255).astype(np.uint8)
            else:
                rgb_np = rgb_np.astype(np.uint8)

    # --- Convert mask to binary np.uint8 (H,W) ---
    if isinstance(mask, torch.Tensor):
        mask_np = mask.detach().cpu().squeeze().numpy()
    else:
        mask_np = np.squeeze(mask)
    mask_np = (mask_np > 0).astype(np.uint8)

    # --- Create overlay ---
    overlay = rgb_np.copy()
    color_arr = np.array(color, dtype=np.uint8).reshape(1, 1, 3)

    # Vectorized blending
    mask_3d = mask_np[..., None]
    overlay = np.where(
        mask_3d,
        ((1 - alpha) * overlay + alpha * color_arr).astype(np.uint8),
        overlay
    )

    return overlay


if __name__ == "__main__":
    from nerfstudio.utils.io import read_masks
    from matplotlib import pyplot as plt

    # Load a mask image
    masks = read_masks(["/home/ziqi/Desktop/test/Liar2/masks_new/mask_00001.png"])
    sampled_median_points = masks_median_points(masks, 100)
    N = masks.shape[0]

    fig, axes = plt.subplots(1, 1, figsize=(20, 4))
    axes.imshow(masks[0, 0], cmap='gray')
    median_y, median_x = sampled_median_points[0]
    axes.scatter(
        median_x, median_y, color='red', s=50, label='Sampled Median Point'
    )
    axes.set_title(f'Mask {0+1}')
    axes.axis('off')
    axes.legend()
    plt.savefig("/home/ziqi/Desktop/test/mask_median_points.png")