import matplotlib.pyplot as plt
import numpy as np
import os

from matplotlib import patches
from mpl_toolkits.mplot3d import Axes3D
from lightglue import viz2d
from PIL import Image
from tqdm import tqdm


def debug_point_prompts(images, points, debug_dir):
    """
    Debugging point prompts for the SAM model

    Args:
        images (Nx3xHxW): Images
        points (NxKx2): 2D points
        debug_dir (str): Directory to save the images
    """
    assert images.shape[0] == points.shape[0]
    assert points.shape[-1] == 2
    assert os.path.isdir(debug_dir)
    target_images = images.permute(0, 2, 3, 1).cpu().numpy()
    target_points_np = points.cpu().numpy()
    for i, img in enumerate(target_images):
        plt.imshow(img)
        plt.scatter(
            target_points_np[i, :, 0], target_points_np[i, :, 1],
            c='r', marker='o'
        )
        plt.title(f"Target View {i}")
        plt.axis("off")
        plt.savefig(f"{debug_dir}/debug_points_{i}.png")
        plt.close()


def debug_bbox_prompts(images, bboxes, debug_dir):
    """
    Debugging bounding box prompts for the SAM model

    Args:
        images (Nx3xHxW): Images
        bboxes (Nx4): Bounding boxes
        debug_dir (str): Directory to save the images
    """
    assert images.shape[0] == bboxes.shape[0]
    assert os.path.isdir(debug_dir)
    target_images = images.permute(0, 2, 3, 1).cpu().numpy()
    bboxes_np = bboxes.cpu().numpy()
    for i, img in enumerate(target_images):
        plt.imshow(img)
        bbox = bboxes_np[i]
        rect = patches.Rectangle(
            (bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1],
            linewidth=1, edgecolor='r', facecolor='none'
        )
        plt.gca().add_patch(rect)
        plt.title(f"Target View {i}")
        plt.axis("off")
        plt.savefig(f"{debug_dir}/debug_bbox_{i}.png")
        plt.close()


def debug_point_cloud(point_cloud, debug_dir):
    """
    Save point cloud as a 3D plot

    Args:
        point_cloud (Nx3): Point cloud
        debug_dir (str): Directory to save the images
    """
    assert len(point_cloud.shape) == 2
    assert point_cloud.shape[-1] == 3
    assert os.path.isdir(debug_dir)
    # visualize the point cloud
    plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(
        point_cloud.cpu()[:, 0], point_cloud.cpu()[:, 1],
        point_cloud.cpu()[:, 2], c='r', marker='o'
    )
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.savefig(f"{debug_dir}/point_cloud.png")
    plt.close()


def debug_matches(imgs1, imgs2, kps1, kps2, matches, debug_dir):
    assert os.path.isdir(debug_dir)
    for i, (img1, img2, kp1, kp2, match) in \
        enumerate(zip(imgs1, imgs2, kps1, kps2, matches)):
        mkp1, mkp2 = kp1[match[..., 0]], kp2[match[..., 1]]
        viz2d.plot_images([img1.cpu(), img2.cpu()])
        viz2d.plot_matches(mkp1, mkp2, color="lime", lw=0.2)
        viz2d.save_plot(f"{debug_dir}/debug_matches_{i}.png")
        plt.close()



def debug_image_pairs(imgs1, imgs2, debug_dir):
    """
    Compare two sets of RGB images side by side.

    Args:
        imgs1 (Tensor): [N, 3, H, W] rendered or predicted images
        imgs2 (Tensor): [N, 3, H, W] ground truth or captured images
        debug_dir (str): directory to save debug outputs
    """
    assert imgs1.ndim == imgs2.ndim == 4, f"Expected 4D tensors, got {imgs1.shape}, {imgs2.shape}"
    assert imgs1.shape[1] == imgs2.shape[1] == 3, f"Expected RGB channels, got {imgs1.shape[1]}, {imgs2.shape[1]}"
    assert os.path.isdir(debug_dir), f"Debug directory {debug_dir} does not exist"

    # Convert to numpy, permute to HWC
    imgs1 = imgs1.permute(0, 2, 3, 1).cpu().numpy()
    imgs2 = imgs2.permute(0, 2, 3, 1).cpu().numpy()

    # Normalize if data is in [0,1] float, otherwise clip to [0,255]
    if imgs1.max() <= 1.0: imgs1 = (imgs1 * 255).astype(np.uint8)
    if imgs2.max() <= 1.0: imgs2 = (imgs2 * 255).astype(np.uint8)

    for idx, (im1, im2) in enumerate(zip(imgs1, imgs2)):
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.imshow(im1)
        plt.title("Rendered Image")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(im2)
        plt.title("Captured Image")
        plt.axis("off")

        plt.tight_layout()
        plt.savefig(f"{debug_dir}/debug_imgpairs_{idx}.png", bbox_inches="tight")
        plt.close()

def debug_depth_pairs(depths1, depths2, debug_dir):
    """
    Compare rendered vs captured depth maps side by side.
    """

    # Remove singleton dimensions so both become [N, H, W]
    d1s = depths1.squeeze().cpu().numpy()   # [N, H, W]
    d2s = depths2.squeeze().cpu().numpy()   # [N, H, W]

    # If N==1, make sure it’s still batched
    if d1s.ndim == 2:
        d1s = d1s[None, ...]
        d2s = d2s[None, ...]

    assert d1s.shape == d2s.shape, f"Shape mismatch: {d1s.shape} vs {d2s.shape}"
    N = d1s.shape[0]

    for idx in range(N):
        d1 = d1s[idx]
        d2 = d2s[idx]

        vmax = max(d1.max(), d2.max())
        vmin = min(d1.min(), d2.min())

        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.imshow(d1, cmap="viridis", vmin=vmin, vmax=vmax)
        plt.title("Rendered Depth")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(d2, cmap="viridis", vmin=vmin, vmax=vmax)
        plt.title("Captured Depth")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(np.abs(d1 - d2), cmap="magma")
        plt.title("|Diff|")
        plt.axis("off")

        plt.tight_layout()
        plt.savefig(f"{debug_dir}/debug_depthpairs_{idx}.png", bbox_inches="tight")
        plt.close()



def debug_images(imgs, debug_dir):
    """
    Save torch images as images

    Args:
        imgs (Nx3xHxW): Images
        debug_dir (str): Directory to save the images
    """
    assert os.path.isdir(debug_dir)
    assert len(imgs.shape) == 4 and imgs.shape[1] == 3
    for i, img in enumerate(imgs):
        img = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(img).save(f"{debug_dir}/img_{i}.png")


def debug_masks(masks, debug_dir):
    """
    Save torch masks as images

    Args:
        masks (Nx1xHxW): Binary masks
        debug_dir (str): Directory to save the images
    """
    assert os.path.isdir(debug_dir)
    assert len(masks.shape) == 4 and masks.shape[1] == 1

    mask_dir = os.path.join(debug_dir, "masks")
    os.makedirs(mask_dir, exist_ok=True) 

    for i, mask in enumerate(tqdm(masks, desc="Save masks")):
        mask = (mask.squeeze() * 255).byte().cpu().numpy()
        Image.fromarray(mask).save(os.path.join(mask_dir, f"mask_{i}.png"))


def debug_depths(depths, debug_dir):
    """
    Save torch depths as images

    Args:
        depths (Nx1xHxW): Depth maps
        debug_dir (str): Directory to save the images
    """
    assert os.path.isdir(debug_dir)
    assert len(depths.shape) == 4 and depths.shape[1] == 1
    for i, depth in enumerate(tqdm(depths, desc="Save depths")):
        depth = depth / depth.max()
        depth = (depth.squeeze() * 255).byte().cpu().numpy()
        Image.fromarray(depth).save(f"{debug_dir}/depth_{i}.png")