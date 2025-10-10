# Simple rendering utilities for Splatfacto
import os
import json
import torch
import numpy as np
from PIL import Image

from nerfstudio.cameras.camera_paths import get_path_from_json
from nerfstudio.cameras.cameras import Cameras



def render_cameras(pipeline, cameras, device="cpu"):
    """
    Splatfacto render at cameras

    Args:
        pipeline (SplatfactoPipeline): Splatfacto pipeline
        cameras (NeRFStudio Cameras): Cameras
        device (str): Device

    Returns:
        rgbs (Nx3xHxW Tensor): RGBs
        depths (Nx1xHxW Tensor): Depths
    """
    assert cameras.size > 0, "No cameras provided"
    with torch.no_grad():
        rgbs, depths = [], []
        for cam_idx in range(cameras.size):
            outputs = pipeline.model.get_outputs_for_camera(
                cameras[cam_idx:cam_idx + 1], obb_box=None
            )
            rgb, depth = outputs["rgb"], outputs["depth"]
            rgbs.append(rgb.cpu().numpy())
            depths.append(depth.cpu().numpy())
        rgbs = torch.tensor(
            np.array(rgbs), device=device
        ).permute(0, 3, 1, 2)
        depths = torch.tensor(
            np.array(depths), device=device
        ).permute(0, 3, 1, 2)
    return rgbs, depths


def render_during_train(
    pipeline, cam_path, step, output_dir, render_every=100, loop=False
):
    """
    Render at camera path during training
    Render every 100 steps

    Args:
        pipeline (Pipeline): Vanilla Pipeline
        cam_path (List[Camera]): Camera path
        step (int): Step
        output_dir (str): Output directory
        render_every (int): Render frequency
        loop (bool): Use all poses in the camera path and loop around,
        if False just use the first pose
    """
    assert os.path.isfile(cam_path), f"{cam_path} not a file"
    assert os.path.isdir(output_dir), f"{output_dir} not a directory"
    with open(cam_path, "r", encoding="utf-8") as f:
        camera_path = json.load(f)
    if step % render_every == 0:
        with torch.no_grad():
            cameras = get_path_from_json(camera_path)
            if loop:
                cam = cameras[step % len(cameras):step % len(cameras)+1]
            else:
                cam = cameras[0:1]
            rgb = pipeline.model.get_outputs_for_camera(
                cam, obb_box=None
            )["rgb"].cpu().numpy() * 255
            Image.fromarray((rgb).astype(np.uint8)).save(
                f"{output_dir}/{step:05g}.png"
            )


def render_3dgs_at_cam(cam, gaussians, device="cuda"):
    """
    Render 3D Gaussians at a given camera

    Args:
        cam (NeRFStudio Cameras): a single camera
        gaussians (Dict): gaussian parameters

    Returns:
        rgb (1x3xHxW Tensor): RGB image
        depth_im (1x1xHxW Tensor): Depth image
    """
    if not isinstance(cam, Cameras):
        print("Called get_outputs with not a camera")
        return {}
    assert cam.shape[0] == 1, "Only one camera at a time"
    from gsplat.project_gaussians import project_gaussians
    from gsplat.rasterize import rasterize_gaussians
    from gsplat.sh import spherical_harmonics

    background = torch.tensor([0.1490, 0.1647, 0.2157]).to(device)
    params = [
        "opacities", "means", "features_dc", "features_rest", "scales", "quats"
    ]
    if "gauss_params.means" in gaussians:
        for param in params:
            gaussians[param] = gaussians[f"gauss_params.{param}"].to(device)

    # shift the camera to center of scene looking at center
    R = cam.camera_to_worlds[0, :3, :3]  # 3 x 3
    T = cam.camera_to_worlds[0, :3, 3:4]  # 3 x 1
    # flip the z and y axes to align with gsplat conventions
    R_edit = torch.diag(
        torch.tensor([1, -1, -1], device=device, dtype=R.dtype)
    )
    R = R @ R_edit
    # analytic matrix inverse to get world2camera matrix
    R_inv = R.T
    T_inv = -R_inv @ T
    viewmat = torch.eye(4, device=R.device, dtype=R.dtype)
    viewmat[:3, :3] = R_inv
    viewmat[:3, 3:4] = T_inv
    # calculate the FOV of the camera given fx and fy, width and height
    cx = cam.cx.item()
    cy = cam.cy.item()
    W, H = int(cam.width.item()), int(cam.height.item())

    opacities_crop = gaussians["opacities"]
    means_crop = gaussians["means"]
    features_dc_crop = gaussians["features_dc"]
    features_rest_crop = gaussians["features_rest"]
    scales_crop = gaussians["scales"]
    quats_crop = gaussians["quats"]

    colors_crop = torch.cat(
        (features_dc_crop[:, None, :], features_rest_crop), dim=1
    )
    BLOCK_WIDTH = 16
    xys, depths, radii, conics, comp, num_tiles_hit, cov3d = project_gaussians(
        means_crop,
        torch.exp(scales_crop),
        1,
        quats_crop / quats_crop.norm(dim=-1, keepdim=True),
        viewmat.squeeze()[:3, :],
        cam.fx.item(),
        cam.fy.item(),
        cx,
        cy,
        H,
        W,
        BLOCK_WIDTH,
    )  # type: ignore

    assert radii.sum() != 0, "No Gaussian can be projected to the input camera"

    viewdirs = means_crop.detach() - cam.camera_to_worlds.detach()[..., :3, 3]
    viewdirs = viewdirs / viewdirs.norm(dim=-1, keepdim=True)
    rgbs = spherical_harmonics(3, viewdirs, colors_crop)
    rgbs = torch.clamp(rgbs + 0.5, min=0.0)  # type: ignore

    assert (num_tiles_hit > 0).any()  # type: ignore

    # Apply the compensation of screen space blurring to gaussians
    rasterize_mode = "antialiased"
    opacities = None
    if rasterize_mode == "antialiased":
        opacities = torch.sigmoid(opacities_crop) * comp[:, None]
    elif rasterize_mode == "classic":
        opacities = torch.sigmoid(opacities_crop)
    else:
        raise ValueError("Unknown rasterize_mode: %s", rasterize_mode)

    rgb, alpha = rasterize_gaussians(  # type: ignore
        xys, depths, radii, conics, num_tiles_hit,  # type: ignore
        rgbs, opacities, H, W, BLOCK_WIDTH, background=background,
        return_alpha=True,
    )  # type: ignore
    alpha = alpha[..., None]
    rgb = torch.clamp(rgb, max=1.0)
    rgb = rgb.unsqueeze(0).permute(0, 3, 1, 2)
    depth_im = None
    depth_im = rasterize_gaussians(  # type: ignore
        xys, depths, radii, conics, num_tiles_hit, # type: ignore
        depths[:, None].repeat(1, 3), opacities, H, W,
        BLOCK_WIDTH, background=torch.zeros(3, device=device),
    )[..., 0:1]  # type: ignore
    depth_im = torch.where(
        alpha > 0, depth_im / alpha, depth_im.detach().max()
    )
    depth_im = depth_im.unsqueeze(0).permute(0, 3, 1, 2).detach()

    return rgb, depth_im