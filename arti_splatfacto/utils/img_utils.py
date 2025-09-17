import torch
import torchvision.utils as vutils
from typing import Dict




def psnr_masked(self, image, rgb, mask):
    assert mask.dtype == torch.bool
    # mask: [1, 1, H, W], image/rgb: [1, 3, H, W]
    mask = mask.expand(-1, 3, -1, -1)   
    return self.psnr(image[mask], rgb[mask])

def crop_imgs_w_masks(images, masks, resize=(256, 256)):
    """
    Crop the images to the smallest bbox containing the masks
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


def compute_2D_bbox(points):
    """
    Compute bboxes for a batch of 2D points
    """
    assert len(points.shape) == 3
    mins, _ = torch.min(points, dim=1)
    maxs, _= torch.max(points, dim=1)
    bboxes = torch.cat((mins, maxs), dim=1)
    return bboxes       

def batch_crop_resize(
    img, rois, out_H, out_W, aligned=True, interpolation="bilinear"
):
    """
    Crop and resize images
    """
    assert len(img.shape) >= 3 and img.shape[-3] == 3, \
        "Error: Image size must be (*, 3, H, W)"
    assert rois.shape[-1] == 4, "Error: Bboxes should be Bx4"
    roi_idx = torch.arange(rois.size(0)).view(-1, 1).to(rois)
    rois = torch.cat((roi_idx, rois), dim=-1)
    # Crop and resize
    output_size = (out_H, out_W)
    from torchvision.ops import RoIAlign, RoIPool
    if interpolation == "bilinear":
        op = RoIAlign(output_size, 1.0, 0, aligned=aligned)
    elif interpolation == "nearest":
        op = RoIPool(output_size, 1.0)  #
    else:
        raise ValueError(f"Wrong interpolation type: {interpolation}")
    return op(img, rois)    