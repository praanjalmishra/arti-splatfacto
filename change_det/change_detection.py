# 3DGS-based change detection
import argparse
from ast import Not
import json
import os
import re
from pathlib import Path
import datetime
from tabnanny import verbose

import open3d as o3d
import torchvision.utils as vutils
import torchvision.transforms.functional as TF
import torch.nn.functional as F

import cv2
import numpy as np
import torch
from lightglue import LightGlue, SuperPoint, viz2d
from matplotlib import pyplot as plt
from PIL import Image
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from nerfstudio.cameras.camera_paths import get_path_from_json
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.models.splatfacto import SplatfactoModel
from nerfstudio.utils.debug_utils import (
    debug_image_pairs, debug_images, debug_masks, debug_matches,
    debug_point_cloud, debug_point_prompts, debug_depths
)
from nerfstudio.utils.effsam_utils import (
    effsam_predict, effsam_embedding, effsam_refine_masks,
    effsam_batch_predict, compute_2D_bbox, expand_2D_bbox,
    get_effsam_embedding_in_masks
)
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.utils.gauss_utils import transform_gaussians
from nerfstudio.utils.img_utils import (
    extract_depths_at_pixels, image_align, filter_features_with_mask,
    in_image, split_masks, dilate_masks
)
from nerfstudio.utils.io import (
    load_from_json, write_to_json, read_dataset, read_imgs, read_transforms,
    save_masks, params_to_cameras, cameras_to_params, save_imgs
)
from nerfstudio.utils.misc import extract_last_number
from nerfstudio.utils.obj_3d_seg import Object3DSeg, Obj3DFeats
from nerfstudio.utils.pcd_utils import (
    compute_3D_bbox, compute_point_cloud, expand_3D_bbox,
    point_cloud_filtering, mahalanobis_filter, nn_distance, pcd_size, bbox2voxel, visualize_bbox3d_matplotlib, 
    points_to_occupancy
)
from nerfstudio.utils.proj_utils import (
    depths_to_points, proj_check_3D_points, project_points, draw_projected_bbox_on_image
)
from nerfstudio.utils.poses import to4x4
from nerfstudio.utils.render_utils import render_cameras, render_3dgs_at_cam
from nerfstudio.viewer_legacy.viser.message_api import P

def camera_clone(cameras):
    """
    Clone a Cameras object

    Args:
        cameras (Cameras): Cameras object to clone

    Returns:
        cameras_new (Cameras): Cloned Cameras object
    """
    cameras_new = Cameras(
        camera_to_worlds=cameras.camera_to_worlds.clone(),
        fx=cameras.fx.clone(), fy=cameras.fy.clone(),
        cx=cameras.cx.clone(), cy=cameras.cy.clone(),
        distortion_params=cameras.distortion_params.clone(),
        width=cameras.width, height=cameras.height
    )
    return cameras_new



class ChangeDet:
    """
    Export a 3D segmentation for a target object
    """
    #debug_dir = "/local/home/pmishra/cvg/3dgscd/debug/Mustard"

    """Directory to save debug output"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    """Device"""
    extractor = SuperPoint(max_num_keypoints=4096).eval().to(device)
    """SuperPoint extractor"""
    matcher = LightGlue(features='superpoint').eval().to(device)
    """LightGlue matcher"""

    def __init__(self, load_config: Path, output_dir: Path, debug=False):
        # Path to the config.yml file of the pretrained 3DGS
        self.load_config = load_config
        # Path to save the output 3D segmentation
        self.output_dir = output_dir

        # Path to save the debug output
        self.debug = debug
        if debug:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.debug_dir = self.output_dir / "debug" / timestamp
            os.makedirs(self.debug_dir, exist_ok=True)
            print(f"[Debug] debug outputs will be saved to: {self.debug_dir}")
        else:
            self.debug_dir = None

    def image_diff(self, capture, render, threshold=1e-2, kernel_ratio=0.0):
        """
        Image differencing for change detection
        TODO: weird that capture and render cannot be swapped

        Args:
            capture (1x3xHxW): Captured image
            render (1x3xHxW): Rendered image
            threshold (float): Threshold for mask area below which it's ignored
            kernel_ratio (float): Gaussian blur kernel fractional size
            (no blur if < 0)

        Return:
            masks (Mx1xHxW): Masks for the changed regions
            masks_all (Mx1xHxW): masks + change regions occupying small areas
        """
        H, W = capture.shape[-2:]
        device = render.device
        # Align render to capture
        render, align_mask = image_align(capture, render)
        capture = capture[0].permute(1, 2, 0).cpu().numpy()
        render = render[0].permute(1, 2, 0).cpu().numpy()
        align_mask = align_mask.squeeze().cpu().numpy().astype(np.uint8)

        # Gaussian blur to filter high-freq signal that 3DGS fails to fit
        if kernel_ratio > 0:
            kernel_size = int(W * kernel_ratio)
            kernel_size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size
            capture = cv2.GaussianBlur(capture, (kernel_size, kernel_size), 0)
            render = cv2.GaussianBlur(render, (kernel_size, kernel_size), 0)

        # # Uncomment to debug
        # viz2d.plot_images([capture, render])
        # viz2d.save_plot(f"{self.debug_dir}/debug.png")
        # plt.close()

        # Get pixel-aligned image embeddings using sam_embedding
        # emb1 = effsam_embedding(capture)
        # emb2 = effsam_embedding(render)
        emb1 = effsam_embedding(capture)
        emb2 = effsam_embedding(render)
        # Calculate cosine similarity between embeddings
        norm1 = torch.nn.functional.normalize(emb1, p=2, dim=1)
        norm2 = torch.nn.functional.normalize(emb2, p=2, dim=1)
        # # Uncomment to debug
        # save_imgs(norm1[:, :3], [f"{self.debug_dir}/feat1.png"])
        # save_imgs(norm2[:, :3], [f"{self.debug_dir}/feat2.png"])
        similarity_map = torch.nn.functional.cosine_similarity(
            norm1, norm2, dim=1
        )
        similarity_map = similarity_map.squeeze().cpu().numpy()
        similarity_map = (similarity_map * 255).astype(np.uint8)
        # Uncomment to debug
        # cv2.imwrite(f"{self.debug_dir}/similarity_map.png", similarity_map)
        # Threshold the SAM cosine similarity map
        thresh = cv2.threshold(
            similarity_map, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
        )[1]
        # remove the influence of the black region in the aligned image
        thresh = thresh * align_mask
        # Uncomment to debug
        # cv2.imwrite(f"{self.debug_dir}/thresh.png", thresh)
        # Find contours in the thresholded binary image
        contours, _ = cv2.findContours(
            thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        # Obtain masks for large enough contours
        masks, masks_all = [], []
        for contour in contours:
            mask = np.zeros((H, W))
            cv2.drawContours(
                mask, [contour], -1, (255, 255, 255), thickness=cv2.FILLED
            )
            mask = torch.from_numpy(mask).unsqueeze(0)
            masks_all.append(mask)
            if cv2.contourArea(contour) >= threshold * H * W:
                masks.append(mask)
        masks = torch.stack(masks, dim=0).to(device)
        masks_all = torch.stack(masks_all, dim=0).to(device)
        # # Uncomment to debug
        # viz2d.plot_images([capture, render])
        # viz2d.save_plot(f"{self.debug_dir}/debug.png")
        # plt.close()

        # save_imgs(norm1[:, :3], [f"{self.debug_dir}/feat1.png"])
        # save_imgs(norm2[:, :3], [f"{self.debug_dir}/feat2.png"])

        # cv2.imwrite(f"{self.debug_dir}/thresh.png", thresh)


        # for i, mask in enumerate(masks):
        #     cv2.imwrite(
        #         f"{self.debug_dir}/mask_{i}.png",
        #         mask.squeeze().cpu().numpy()
        #     )
        return masks, masks_all


    def masks_to_3D(self, masks_list, depths, Ks, cam_poses):
        """
        Convert 2D masks to 3D point clouds.

        Args:
            masks_list (List[Tensor]): List of binary masks, each (1, H, W)
            depths (Tensor): (N, 1, H, W) Depth maps rendered from 3DGS
            Ks (Tensor): (N, 3, 3) Intrinsics per view
            cam_poses (Tensor): (N, 4, 4) Extrinsics (camera-to-world) per view

        Returns:
            pcds (List[np.ndarray]): List of 3D point clouds (K_i, 3) per mask
        """
        assert len(masks_list) == depths.shape[0], \
            f"Mismatch: {len(masks_list)} masks vs {depths.shape[0]} depths"

        device = depths.device
        pcds = []

        for i in range(len(masks_list)):
            mask_tensor = masks_list[i]  # shape: (1, H, W)
            print(f"mask_tensor: {mask_tensor.shape}")
            mask = mask_tensor.squeeze().bool()  # (H, W)
            depth = depths[i, 0]  # (H, W)
            K = Ks[i]  # (3, 3)
            cam_pose = cam_poses[i]  # (4, 4)

            # Get image pixel coordinates
            y, x = torch.where(mask)
            z = depth[y, x]
            valid = z > 0
            x, y, z = x[valid], y[valid], z[valid]

            if x.numel() == 0:
                pcds.append(torch.empty((0, 3)))
                continue

            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]

            X = (x - cx) * z / fx
            Y = (y - cy) * z / fy
            Z = z

            pts_cam = torch.stack([X, Y, Z, torch.ones_like(Z)], dim=1).T  # (4, N)
            pts_world = (cam_pose @ pts_cam).T[:, :3]  # (N, 3)

            pcds.append(pts_world.cpu().numpy()) 

        return pcds

    def save_pcds(self, pcds, output_dir):
        """
        Save 3D point clouds to PCD files.

        Args:
            pcds (List[np.ndarray]): List of 3D point clouds (K_i, 3) per mask
            output_dir (Path): Directory to save the PCD files
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcds)
        o3d.io.write_point_cloud(output_dir, pcd)
        print(f"[INFO] Saved point cloud to {output_dir}")

    def project_points_to_image(self, points_3d, K, cam_pose, image_shape):
        """
        Project 3D points into 2D image space.

        Args:
            points_3d (np.ndarray): (N, 3) points in world frame
            K (np.ndarray): (3, 3) camera intrinsics
            cam_pose (np.ndarray): (4, 4) camera-to-world matrix
            image_shape (tuple): (H, W) of the image

        Returns:
            projected_pts (np.ndarray): (M, 2) projected pixel coords
        """
        # Invert the camera pose to get world-to-camera
        w2c = np.linalg.inv(cam_pose)
        N = points_3d.shape[0]

        # Convert to homogeneous
        pts_h = np.concatenate([points_3d, np.ones((N, 1))], axis=1).T  # (4, N)
        pts_cam = (w2c @ pts_h).T[:, :3]  # (N, 3)

        # Remove points behind the camera
        valid = pts_cam[:, 2] > 0
        pts_cam = pts_cam[valid]

        # Project to image plane
        pts_2d = (K @ pts_cam.T).T  # (N, 3)
        pts_2d = pts_2d[:, :2] / pts_2d[:, 2:3]

        # Filter those falling inside the image
        H, W = image_shape
        x, y = pts_2d[:, 0], pts_2d[:, 1]
        in_bounds = (x >= 0) & (x < W) & (y >= 0) & (y < H)

        return pts_2d[in_bounds].astype(np.int32)

    def visualize_projection(self, image, points_2d, output_path):
        vis = image.copy()
        for x, y in points_2d:
            cv2.circle(vis, (x, y), 2, (0, 255, 0), -1)
        cv2.imwrite(output_path, vis)
        print(f"[DEBUG] Saved projection overlay: {output_path}")

    def get_features_in_masks(self, rgbs, masks, flip=False):
        """
        Extract SuperPoint descriptors in the masked regions

        Args:
            rgbs (Nx3xHxW): RGB images
            masks (N-list of Mx1xHxW): Image masks

        Returns:
            feats (N-list of M-list of TxC): SuperPoint descriptors
        """
        assert rgbs.shape[1] == 3 and len(rgbs) == len(masks)
        if flip:
            # Rotate images by 180 degrees if flip is True
            rgbs = torch.flip(rgbs, [2, 3])
            for m in masks:
                m = torch.flip(m, [2, 3])
        feats_all = []
        for i in range(len(rgbs)):
            feat_i = []
            for j in range(len(masks[i])):
                feat = self.extractor.extract(rgbs[i])
                # Ensure keypoints are within image
                feat["keypoints"].clamp_(min=0)
                # Filter keypoints using masks
                feat = filter_features_with_mask(feat, masks[i][j:j+1])
                if flip:
                    H, W = rgbs.shape[2], rgbs.shape[3]
                    feat['keypoints'][:, 0] = W - feat['keypoints'][:, 0] - 1
                    feat['keypoints'][:, 1] = H - feat['keypoints'][:, 1] - 1
                feat_i.append(feat)
            feats_all.append(feat_i)
        return feats_all

    def match_move_out(
        self, rgbs, depths, masks, poses, Ks, pcd_filter=0.9, embed_sim_thresh=0.4
    ):
        """
        Associate and fuse 2D move-out masks across post-change views
        to obtain obj templates
        NOTE: This is only for moved or removed objects

        Args:
            rgbs (Nx3xHxW): RGB images
            depths (Nx1xHxW): Depth images
            masks (N-list of Mx1xHxW): Sampling masks
            poses (Nx4x4): Camera poses wrt world
            Ks (Nx3x3): Camera intrinsics

        Returns:
            pcds (K-list of Lx3): Object point clouds
            pcd_feats (K-list of Obj3DFeats): Object SuperPoint descriptors
        """
        assert rgbs.shape[1] == 3
        assert depths.shape[1] == 1
        assert poses.shape[1:] == (4, 4)
        assert Ks.shape[1:] == (3, 3)
        assert len(rgbs) == len(depths) == len(masks) == len(poses) == len(Ks)
        device = rgbs.device
        N = len(rgbs)
        def compute_feats_3D(feats, depth, pose, K):
            pixels = feats["keypoints"]
            if pixels.ndim == 3:
                pixels = pixels.view(-1, 2)  # (N,2)
            depths_at_kps = extract_depths_at_pixels(pixels, depth)
            pts_at_kps = depths_to_points(pixels, depths_at_kps, pose, K)
            return pts_at_kps

        # Extract SuperPoint descriptors in multi-view masked RGB images
        feats = self.get_features_in_masks(rgbs, masks)
        # Extract EfficientSAM embeddings for objects across multi-view images
        embeds = get_effsam_embedding_in_masks(rgbs, masks)
        # Initialize object pcds
        pcds, pcd_sizes, pcd_counts, pcd_feats, pcd_embeds = [], [], [], [], []
        for j in range(len(masks[0])):
            pcd = compute_point_cloud(
                depths[0:1], poses[0:1], Ks[0:1], masks[0][j:j+1]
            )
            pcd = mahalanobis_filter(pcd, pcd_filter)
            pcds.append(pcd)
            pcd_sizes.append(pcd_size(pcd))
            pcd_counts.append(1)
            # Extract 3D positions of keypoints
            pts3D = compute_feats_3D(
                feats[0][j], depths[0:1], poses[0], Ks[0]
            )
            pcd_feats.append(Obj3DFeats([feats[0][j]], [pts3D]))
            pcd_embeds.append(embeds[0][j:j+1, :])
        # Associate move-out masks with the object point clouds w/ NN matching
        for i in range(1, N):
            dist_mat = torch.tensor(pcd_sizes).reshape(-1, 1).to(device)
            dist_mat = dist_mat.repeat(1, len(masks[i]) + len(pcds))
            new_pcds = []
            for j in range(len(masks[i])):
                pcd = compute_point_cloud(
                    depths[i:i+1], poses[i:i+1], Ks[i:i+1], masks[i][j:j+1]
                )
                pcd = mahalanobis_filter(pcd, pcd_filter)
                for k in range(len(pcds)):
                    dist_mat[k, j] = nn_distance(pcds[k], pcd)
                new_pcds.append(pcd)
            # print(f"dist_mat:\n {dist_mat.cpu().numpy()}")
            row_ind, col_ind = linear_sum_assignment(dist_mat.cpu().numpy())
            # print(f"row_ind: {row_ind}, col_ind: {col_ind}")
            # Update existing object point clouds
            for r, c in zip(row_ind, col_ind):
                # check feature sim btw matched object segments
                embed_sim = torch.cosine_similarity(
                    pcd_embeds[r], embeds[i][c], dim=-1
                )
                if c < len(masks[i]):
                    if embed_sim.max() > embed_sim_thresh:
                        pcds[r] = torch.cat((pcds[r], new_pcds[c]), dim=0)
                        pcd_sizes[r] = pcd_size(pcds[r])
                        pcd_counts[r] += 1
                        pts3D = compute_feats_3D(
                            feats[i][c], depths[i:i+1], poses[i], Ks[i]
                        )
                        pcd_feats[r].add_feats(feats[i][c], pts3D)
                        pcd_embeds[r] = torch.cat(
                            (pcd_embeds[r], embeds[i][c:c+1]), dim=0
                        )
                    else:
                        pcds.append(new_pcds[c])
                        pcd_sizes.append(pcd_size(new_pcds[c]))
                        pcd_counts.append(1)
                        pts3D = compute_feats_3D(
                            feats[i][c], depths[i:i+1], poses[i], Ks[i]
                        )
                        pcd_feats.append(Obj3DFeats([feats[i][c]], [pts3D]))
                        pcd_embeds.append(embeds[i][c:c+1, :])
            # Add new object point clouds
            for k in range(len(masks[i])):
                if k not in col_ind:
                    pcds.append(new_pcds[k])
                    pcd_sizes.append(pcd_size(new_pcds[k]))
                    pcd_counts.append(1)
                    pts3D = compute_feats_3D(
                        feats[i][k], depths[i:i+1], poses[i], Ks[i]
                    )
                    pcd_feats.append(Obj3DFeats([feats[i][k]], [pts3D]))
                    pcd_embeds.append(embeds[i][k:k+1, :])
        # Filter out object point clouds that appear in <25% of images
        pcds = [p for p, ct in zip(pcds, pcd_counts) if ct > N * 0.25]
        pcd_feats = [
            e for e, ct in zip(pcd_feats, pcd_counts) if ct > N * 0.25
        ]
        if self.debug_dir is not None:

            for idx, pcd in enumerate(pcds):
                self.save_pcds(
                    pcd.cpu().numpy(),  
                    self.debug_dir / f"move_out_obj_{idx}.ply" 
                )

        return pcds, pcd_feats

    def match_move_in(self, rgbs, masks, depths, poses, Ks, pcd_filter=0.95):
        """
        Associate and fuse move-in masks across post-change views
        to obtain per-object move-in masks and fused point clouds
        NOTE: This is only for inserted objects

        Args:
            rgbs (Nx3xHxW): RGB images
            depths (Nx1xHxW): Depth images
            masks (N-list of Mx1xHxW): Sampling masks
            poses (Nx4x4): Camera poses wrt world
            Ks (Nx3x3): Camera intrinsics
            pcd_filter (float): Point cloud filtering percentile

        Returns:
            obj_masks_move_in (K-list of Lx1xHxW): Per-object move-in across views
            view_indices (K-list of L): view indices of the move-in masks
            pcds_post (K-list of Lx3): Fused post-change point clouds
        """
        # Check input shapes
        assert len(rgbs) == len(depths) == len(masks) == len(poses) == len(Ks)
        device = rgbs.device
        N = len(rgbs)

        # Extract embeddings for objects across multi-view images
        embeds = get_effsam_embedding_in_masks(rgbs, masks)
        
        # Initialize with first view's masks AND point clouds
        obj_masks_move_in = [masks[0][i:i+1] for i in range(len(masks[0]))]
        view_indices = [[0] for _ in range(len(masks[0]))]
        
        # Initialize point clouds for each object (just like match_move_out)
        pcds_post = []
        for i in range(len(masks[0])):
            pcd = compute_point_cloud(
                depths[0:1], poses[0:1], Ks[0:1], masks[0][i:i+1]
            )
            pcd = mahalanobis_filter(pcd, pcd_filter)  # Filter each view
            pcds_post.append(pcd)

        for ii, (masks_i, embeds_i) in enumerate(zip(masks[1:], embeds[1:]), start=1):
            if embeds_i.size(0) == 0:
                continue

            sim_mat = torch.cosine_similarity(
                embeds[0][:, None, :], embeds_i[None, :, :], dim=-1
            )
            row_ind, col_ind = linear_sum_assignment(-sim_mat.cpu().numpy())

            # Compute point clouds for current view's masks
            new_pcds = []
            for j in range(len(masks_i)):
                pcd = compute_point_cloud(
                    depths[ii:ii+1], poses[ii:ii+1], Ks[ii:ii+1], masks_i[j:j+1]
                )
                pcd = mahalanobis_filter(pcd, pcd_filter)  # Filter each view
                new_pcds.append(pcd)

            # Update existing objects with geometric validation
            for r, c in zip(row_ind, col_ind):
                if c >= len(masks_i):
                    continue

                # Compute point cloud for reference object (latest view it was seen)
                last_view_idx = view_indices[r][-1]
                pcd_ref = compute_point_cloud(
                    depths[last_view_idx:last_view_idx+1],
                    poses[last_view_idx:last_view_idx+1],
                    Ks[last_view_idx:last_view_idx+1],
                    obj_masks_move_in[r][-1:]
                )

                # Use the new point cloud for candidate
                pcd_cand = new_pcds[c]

                # Compute geometric similarity (e.g., Chamfer distance)
                dist = nn_distance(pcd_ref, pcd_cand)
                geom_thresh = 0.1  # in meters or normalized units

                if dist < geom_thresh:
                    # Update masks
                    obj_masks_move_in[r] = torch.cat((obj_masks_move_in[r], masks_i[c:c+1]), dim=0)
                    view_indices[r].append(ii)
                    
                    # FUSE POINT CLOUDS 
                    pcds_post[r] = torch.cat((pcds_post[r], pcd_cand), dim=0)
                else:
                    # Create new object
                    obj_masks_move_in.append(masks_i[c:c+1])
                    view_indices.append([ii])
                    pcds_post.append(pcd_cand)

            # Add new unmatched objects
            for k in range(len(masks_i)):
                if k not in col_ind:
                    obj_masks_move_in.append(masks_i[k:k+1])
                    view_indices.append([ii])
                    pcds_post.append(new_pcds[k])

        # Filter masks that appear in fewer than 25% of views
        min_views = int(N * 0.1)
        obj_masks_move_in_filtered = []
        view_indices_filtered = []
        pcds_post_filtered = []
        
        for m, v, p in zip(obj_masks_move_in, view_indices, pcds_post):
            if len(v) > min_views:
                obj_masks_move_in_filtered.append(m)
                view_indices_filtered.append(v)
                # Apply final filtering to fused point cloud
                p_filtered = mahalanobis_filter(p, pcd_filter)
                pcds_post_filtered.append(p_filtered)

        # Debugging: save masks
        if self.debug_dir is not None:
            debug_dir = self.debug_dir / "move_in_masks"
            os.makedirs(debug_dir, exist_ok=True)
            for obj_id, (masks_per_obj, views) in enumerate(zip(obj_masks_move_in_filtered, view_indices_filtered)):
                for i, (mask, v_idx) in enumerate(zip(masks_per_obj, views)):
                    mask_img = (mask.squeeze(0).cpu().numpy() * 255).astype("uint8")
                    path = debug_dir / f"obj{obj_id}_view{v_idx}_mask.png"
                    print(f"Saving mask to {path}")
                    TF.to_pil_image(mask_img).save(path)
            
            # Also save the fused point clouds for debugging
            for obj_id, pcd in enumerate(pcds_post_filtered):
                np.save(self.debug_dir / f"obj{obj_id}_post_fused_pcd.npy", pcd.cpu().numpy())
                print(f"Saved fused post-change PCD for obj {obj_id}: {len(pcd)} points")
                #save as ply
                pcd_o3d = o3d.geometry.PointCloud()
                pcd_o3d.points = o3d.utility.Vector3dVector(pcd.cpu().numpy())
                o3d.io.write_point_cloud(
                    self.debug_dir / f"obj{obj_id}_post_fused_pcd.ply", pcd_o3d
                )

        return obj_masks_move_in_filtered, view_indices_filtered, pcds_post_filtered

    # def match_move_in(self, rgbs, masks):
    #     """
    #     Associate and fuse move-in masks across post-change views
    #     to obtain per-object move-in masks
    #     NOTE: This is only for inserted objects

    #     Args:
    #         rgbs (Nx3xHxW): RGB images
    #         masks (N-list of Mx1xHxW): Sampling masks

    #     Returns:
    #         masks_move_in (K-list of Lx1xHxW): Per-object move-in across views
    #         view_indices (K-list of L): view indices of the move-in masks
    #     """
    #     assert rgbs.shape[1] == 3
    #     assert len(rgbs) == len(masks)
    #     # Extract EffSAM embeddings for objects across multi-view images
    #     embeds = get_effsam_embedding_in_masks(rgbs, masks)
    #     # Initialize move-in masks
    #     obj_masks_move_in = [masks[0][i:i+1] for i in range(len(masks[0]))]
    #     view_indices = [[0] for _ in range(len(masks[0]))]
    #     for ii, (masks_i, embeds_i) in enumerate(zip(masks[1:], embeds[1:])):
    #         if embeds_i.size(0) == 0:
    #             continue
    #         sim_mat = torch.cosine_similarity(
    #             embeds[0][:,None,:], embeds_i[None,:,:], dim=-1
    #         )
    #         row_ind, col_ind = linear_sum_assignment(-sim_mat.cpu().numpy())
    #         # Update existing objects
    #         for r, c in zip(row_ind, col_ind):
    #             if c < len(masks_i):
    #                 obj_masks_move_in[r] = torch.cat(
    #                     (obj_masks_move_in[r], masks_i[c:c+1]), dim=0
    #                 )
    #                 view_indices[r].append(ii+1)
    #         # Add new objects
    #         for k in range(len(masks_i)):
    #             if k not in col_ind:
    #                 obj_masks_move_in.append(masks_i[k:k+1])
    #                 view_indices.append([ii+1])
    #     # Filter out move-in masks that appear in <25% of images
    #     obj_masks_move_in = [
    #         m for m in obj_masks_move_in if m.size(0) > len(rgbs) * 0.1
    #     ]
    #     view_indices = [i for i in view_indices if len(i) > len(rgbs) * 0.1]
    #     return obj_masks_move_in, view_indices


    def match_move_in_depth(self, rgbs, masks, depths, poses, Ks):
        """
        Associate and fuse move-in masks across post-change views
        to obtain per-object move-in masks
        NOTE: This is only for inserted objects

        Args:
            rgbs (Nx3xHxW): RGB images
            depths (Nx1xHxW): Depth images
            masks (N-list of Mx1xHxW): Sampling masks
            poses (Nx4x4): Camera poses wrt world
            Ks (Nx3x3): Camera intrinsics

        Returns:
            obj_masks_move_in (K-list of Lx1xHxW): Per-object move-in across views
            view_indices (K-list of L): view indices of the move-in masks
        """

        assert len(rgbs) == len(depths) == len(masks) == len(poses) == len(Ks)
        device = rgbs.device
        N = len(rgbs)

        # Extract embeddings for objects across multi-view images
        embeds = get_effsam_embedding_in_masks(rgbs, masks)
        # Initialize with first view's masks
        obj_masks_move_in = [masks[0][i:i+1] for i in range(len(masks[0]))]
        view_indices = [[0] for _ in range(len(masks[0]))]

        for ii, (masks_i, embeds_i) in enumerate(zip(masks[1:], embeds[1:]), start=1):
            if embeds_i.size(0) == 0:
                continue

            sim_mat = torch.cosine_similarity(
                embeds[0][:, None, :], embeds_i[None, :, :], dim=-1
            )
            row_ind, col_ind = linear_sum_assignment(-sim_mat.cpu().numpy())

            # Update existing objects with geometric validation
            for r, c in zip(row_ind, col_ind):
                if c >= len(masks_i):
                    continue

                # Compute point cloud for reference object (latest view it was seen)
                last_view_idx = view_indices[r][-1]
                pcd_ref = compute_point_cloud(
                    depths[last_view_idx:last_view_idx+1],
                    poses[last_view_idx:last_view_idx+1],
                    Ks[last_view_idx:last_view_idx+1],
                    obj_masks_move_in[r][-1:]
                )

                # Compute point cloud for candidate match
                pcd_cand = compute_point_cloud(
                    depths[ii:ii+1], poses[ii:ii+1], Ks[ii:ii+1], masks_i[c:c+1]
                )

                # Compute geometric similarity (e.g., Chamfer distance)
                dist = nn_distance(pcd_ref, pcd_cand)
                geom_thresh = 0.05  # in meters or normalized units

                if dist < geom_thresh:
                    obj_masks_move_in[r] = torch.cat((obj_masks_move_in[r], masks_i[c:c+1]), dim=0)
                    view_indices[r].append(ii)
                else:
                    obj_masks_move_in.append(masks_i[c:c+1])
                    view_indices.append([ii])

            # Add new unmatched objects
            for k in range(len(masks_i)):
                if k not in col_ind:
                    obj_masks_move_in.append(masks_i[k:k+1])
                    view_indices.append([ii])

        # Filter masks that appear in fewer than 25% of views
        min_views = int(N * 0.25)
        obj_masks_move_in_filtered = []
        view_indices_filtered = []
        for m, v in zip(obj_masks_move_in, view_indices):
            if len(v) > min_views:
                obj_masks_move_in_filtered.append(m)
                view_indices_filtered.append(v)

        # Debugging: save masks
        if self.debug_dir is not None:
            debug_dir = self.debug_dir / "move_in_masks"
            os.makedirs(debug_dir, exist_ok=True)
            for obj_id, (masks_per_obj, views) in enumerate(zip(obj_masks_move_in_filtered, view_indices_filtered)):
                for i, (mask, v_idx) in enumerate(zip(masks_per_obj, views)):
                    mask_img = (mask.squeeze(0).cpu().numpy() * 255).astype("uint8")
                    path = debug_dir / f"obj{obj_id}_view{v_idx}_mask.png"
                    print(f"Saving mask to {path}")
                    TF.to_pil_image(mask_img).save(path)

        return obj_masks_move_in_filtered, view_indices_filtered



    def masks_to_bbox3d(
        self, masks, poses, Ks, dist_params, gauss_filter_percent=0.5,
        obj_pts_filter_percent=0.8, num_sample=1000000, proj_check_cutoff=0.99
    ):
        """
        Obj masks on multi views to rough object bbox3D
        for finer obj segmentation of *inserted* objects

        Args:
            masks (Nx1xHxW): Object move-out masks on the sparse views
            poses (Nx4x4): Camera poses wrt world
            Ks (Nx3x3): Camera intrinsics
            dist_params (Nx4): Camera distortion parameters

        Returns:
            bbox3d (2-tuple of 3-tuple of floats): MinMax xyz of the obj bbox3D
        """
        # Get the 3D bbox of all Gaussians in the 3DGS model
        gauss_means = self.pipeline_pretrain.model.gauss_params.means
        gauss_means = mahalanobis_filter(gauss_means, gauss_filter_percent)
        device = gauss_means.device
        min_xyz = gauss_means.min(dim=0)[0]
        max_xyz = gauss_means.max(dim=0)[0]
        # Sample points in the 3D bbox
        pts_sampled = torch.rand(num_sample, 3, device=device) * \
            (max_xyz - min_xyz) + min_xyz
        occupied = proj_check_3D_points(
            pts_sampled, poses, Ks, dist_params, masks,
            cutoff=proj_check_cutoff
        )
        pts_occupy = pts_sampled[occupied]
        pts_occupy = mahalanobis_filter(
            pts_occupy, obj_pts_filter_percent
        )
        bbox3d = (
            pts_occupy.min(dim=0)[0].detach().cpu().numpy(),
            pts_occupy.max(dim=0)[0].detach().cpu().numpy()
        )
        return bbox3d

    def check_visibility(
        self, pcds, masks, poses, Ks, dist_params, H, W, threshold=0.95
    ):
        """
        Check visibility of object point clouds

        Args:
            pcds (M-list of Lx3): Object point clouds
            masks (MxNx1xHxW): Object move-out masks on the sparse views
            poses (Nx4x4): Camera poses wrt world
            Ks (Nx3x3): Camera intrinsics
            dist_params (Nx4): Camera distortion parameters
            H (int): Image height
            W (int): Image width
            
        Returns:
            vis (M-list of N-list of int): Views where obj pcd is fully visible
        """
        assert len(pcds) == len(masks)
        assert masks.shape[1] == len(poses)
        vis = []
        for ii in range(len(pcds)):
            pcd_proj, _ = project_points(
                pcds[ii], poses, Ks, dist_params, H, W
            )
            # We count how many object points can project in masks
            vis_ii = []
            for jj, proj in enumerate(pcd_proj):
                proj = proj.round().long().unique(dim=0)
                proj_in = in_image(proj, H, W)
                proj_in_ratio = proj_in.sum().item() / proj.size(0)
                proj = proj[proj_in]
                in_mask_count = masks[ii, jj, 0][proj[:, 1], proj[:, 0]].sum()
                in_mask_ratio = in_mask_count / proj.size(0)
                if in_mask_ratio > threshold and proj_in_ratio > threshold:
                    vis_ii.append(jj)
            vis.append(vis_ii)
        return vis


    def preprocess_depths(self, depths_raw, target_height, target_width):
        """
        Convert depth from [N, 1, 1, H, W] to [N, 1, target_H, target_W] using bilinear upsampling
        """
        # Remove singleton dim
        depths = depths_raw.squeeze(2)  # Shape: [N, 1, H, W]
        
        # Upsample
        depths_upsampled = F.interpolate(depths, size=(target_height, target_width), mode='bilinear', align_corners=False)
        return depths_upsampled

    def pretrain_iteration(self, rgbs, masks, cameras, gaussians):
        """
        Forward pass through the transformed pre-trained 3DGS

        Args:
            rgbs (Nx3xHxW): Captured sparse view RGB images
            masks (Nx1xHxW): Sampling masks on the sparse views
            cameras (Cameras): NeRFStudio Cameras object of size N
            gaussians (dict): Transformed Gaussian parameters

        Returns:
            rgb_loss: RGB loss btw the captured and rendered pixels
        """
        rgbs = rgbs.permute(0, 2, 3, 1)
        masks = masks.permute(0, 2, 3, 1)
        batches = [
            { "image": rgb, "mask": mask, "image_idx": ii} 
            for ii, (rgb, mask) in enumerate(zip(rgbs, masks))
        ]
        loss_accumulated = torch.tensor(0.0, device=rgbs.device)
        for ii, batch in enumerate(batches):
            camera = cameras[ii:ii+1]
            # outputs = self.pipeline_pretrain.model(camera)
            color, _ = render_3dgs_at_cam(camera, gaussians)
            outputs = {
                "rgb": color.squeeze().permute(1, 2, 0), "background": None
            }
            loss_dict = self.pipeline_pretrain.model.get_loss_dict(
                outputs, batch, None
            )
            loss = sum(loss_dict.values())
            loss_accumulated += loss
        # global debug_count
        # debug_count += 1
        # Uncomment to vis the pose regression process
        # blend = rgbs[-1] * 0.2 + outputs["rgb"] * masks[-1] * 0.8
        # blend = (blend.detach().cpu().numpy() * 255).astype(np.uint8)
        # Image.fromarray(blend).save(self.debug_dir + f"blend{debug_count}.png")
        return loss_accumulated

    
    def refine_obj_pose_change(
        self, rgbs, obj_segs, cameras, lr=1e-3, epochs=100, patience=20,
        optim="obj+cam"
    ):
        """
        Refine object pose change to make the object pose pixel-perfect

        Args:
            rgbs (Nx3xHxW): Captured sparse view RGB images
            obj_segs (M-list of Obj3DSeg): Object 3D segments
            cameras (Cameras): NeRFStudio Cameras object
            batch_size (int): Batch size for training
            epochs (int): Number of epochs
            patience (int): Number of epochs to wait for plateau
            optim (str): Variables to optim, choices: "obj+cam", "obj", "cam"
        
        Returns:
            poses_refined (M-list of 4x4): Refined object pose change
            cameras (Cameras): Cameras w/ refined camera poses
        """
        from nerfstudio.cameras.lie_groups import exp_map_SO3xR3
        assert hasattr(self, "pipeline_pretrain"), \
            "Pre-training pipeline not loaded yet"
        device = rgbs.device
        torch.cuda.empty_cache()
        cameras.camera_to_worlds = cameras.camera_to_worlds.to(device)
        c2w, Ks, dist, H, W = cameras_to_params(cameras, device)
        # Project object 3D seg voxel grid points to have obj's masks        
        in_objs, poses_init, obj_masks = [], [], []
        for obj_seg in obj_segs:
            in_obj = obj_seg.query(self.pipeline_pretrain.model.means)
            pose_init = obj_seg.get_pose_change().clone().to(device)
            obj_mask = ~obj_seg.project(c2w, Ks, dist, H, W)
            in_objs.append(in_obj)
            poses_init.append(pose_init)
            obj_masks.append(obj_mask)
        obj_masks = torch.all(torch.stack(obj_masks, dim=0), dim=0)
        torch.cuda.empty_cache()
        # Uncomment to debug
        # debug_masks(obj_masks, self.debug_dir)
        # Pre-trained Gaussians
        gauss0 = {
            name: self.pipeline_pretrain.model.gauss_params[name].data.clone()
            for name in [
                "means", "scales", "quats", "features_dc", "features_rest",
                "opacities"
            ]
        }
        cam0 = camera_clone(cameras)
        # Make a pose update parameter
        poses_update = torch.nn.Parameter(
            torch.zeros((len(poses_init), 6), device=device)
        )
        cam_pose_update = torch.nn.Parameter(
            torch.zeros((len(cameras), 6), device=device)
        )
        param = []
        if "obj" in optim:
            param.append(poses_update)
        if "cam" in optim:
            param.append(cam_pose_update)
        assert len(param) > 0, "No parameters to optimize"
        optimizer = torch.optim.Adam(param, lr=lr)
        # Training loop
        best_loss, initial_loss = float("inf"), None
        plateau_count = 0
        with tqdm(total=epochs, desc="pose change opt") as pbar:
            for idx in range(epochs):
                optimizer.zero_grad()
                poses_update4x4 = to4x4(exp_map_SO3xR3(poses_update))
                poses_update4x4 = poses_update4x4.reshape(-1, 4, 4)
                cam_pose_update4x4 = to4x4(exp_map_SO3xR3(cam_pose_update))
                cam_pose_update4x4 = cam_pose_update4x4.reshape(-1, 4, 4)
                # Transform object Gaussians
                assert self.pipeline_pretrain.model.means.shape[0] > 0
                means, quats = gauss0["means"], gauss0["quats"]
                for pose_init, in_obj, pose_update4x4 in zip(
                    poses_init, in_objs, poses_update4x4
                ):
                    means_trans, quats_trans = transform_gaussians(
                        pose_init @ pose_update4x4,
                        gauss0["means"], gauss0["quats"]
                    )
                    means = torch.where(
                        in_obj.unsqueeze(-1).repeat(1, 3), means_trans, means
                    )
                    quats = torch.where(
                        in_obj.unsqueeze(-1).repeat(1, 4), quats_trans, quats
                    )
                gauss = {
                    name : gauss0[name] for name in [
                        "scales", "features_dc", "features_rest", "opacities"
                    ]
                }
                gauss["means"], gauss["quats"] = means, quats
                # Update camera pose
                cameras.camera_to_worlds = \
                    cam0.camera_to_worlds @ cam_pose_update4x4
                # Forward pass
                rgb_loss = self.pretrain_iteration(
                    rgbs, obj_masks, cameras, gauss
                )
                # Backward pass
                rgb_loss.backward()
                optimizer.step()
                pbar.set_postfix(
                    {'Epoch': idx+1, 'RGB Loss': f'{rgb_loss.item():.4f}'}
                )
                pbar.update(1)
                if initial_loss is None:
                    initial_loss = rgb_loss.item()
                if rgb_loss.item() < best_loss:
                    best_loss = rgb_loss.item()
                    plateau_count = 0
                else:
                    plateau_count += 1
                    if plateau_count > patience:
                        print(f"Early stopping at epoch {idx+1} after plateau")
                        break
        if rgb_loss.item() > initial_loss:
            print("Warning: RGB loss increased after pose change refinement")
        poses_refined = [
            pose_init @ pose_update4x4.detach() for pose_init, pose_update4x4
            in zip(poses_init, poses_update4x4)
        ]
        return poses_refined, cameras

    def mask_proj(
        self, cams, obj_segs, dilate=0.15, new=False, occlusion_check=True
    ):
        """
        Project object 3D segmentation to target cameras w/ occlusion-awareness
        
        Args:
            cams (Cameras): Target camera views
            obj_segs (M-list of Obj3DSeg): Object 3D segments
            dilate (float): Dilate the 3D segments to check if points in mask
            new (bool): Use object's new pose for mask projection
            occlusion_check (bool or bool-list): Do we check occlusion?

        Returns:
            masks (Nx1xHxW): 2D move-out or -in masks on the target views
        """
        assert occlusion_check is bool or len(occlusion_check) == len(obj_segs)
        # Render depths at target cameras
        poses, Ks, dist, H, W = cameras_to_params(cams)
        if not new:
            _, depths = render_cameras(
                self.pipeline_pretrain, cams, device=self.device
            )
        else:
            gauss0 = {
                name: self.pipeline_pretrain.model.gauss_params[name].data.clone()
                for name in [
                    "means", "scales", "quats", "features_dc", "features_rest",
                    "opacities"
                ]
            }
            means, quats = gauss0["means"], gauss0["quats"]
            for obj_seg in obj_segs:
                in_obj = obj_seg.query(self.pipeline_pretrain.model.means)
                means_trans, quats_trans = transform_gaussians(
                    obj_seg.get_pose_change(), gauss0["means"], gauss0["quats"]
                )
                means = torch.where(
                    in_obj.unsqueeze(-1).repeat(1, 3), means_trans, means
                )
                quats = torch.where(
                    in_obj.unsqueeze(-1).repeat(1, 4), quats_trans, quats
                )
            gauss = {
                name : gauss0[name] for name in [
                    "scales", "features_dc", "features_rest", "opacities"
                ]
            }
            gauss["means"], gauss["quats"] = means, quats
            depths = []
            for ii in range(len(cams)):
                _, depth = render_3dgs_at_cam(cams[ii:ii+1], gauss)
                depths.append(depth)
            depths = torch.cat(depths, dim=0)
        # debug_depths(depths, self.debug_dir)
        # Project object 3D segmentation to target
        masks_no_occl_all_obj = []
        for obj_ind, obj_seg in enumerate(obj_segs):
            if not new:
                masks = obj_seg.project(poses, Ks, dist, H, W)
            else:
                masks = obj_seg.project_new(poses, Ks, dist, H, W)
            if not occlusion_check or not occlusion_check[obj_ind]:
                masks_no_occl_all_obj.append(masks)
                continue
            # dilate the 3D segments due to noise
            voxel_dilated = obj_seg.dilate_uniform(
                int(obj_seg.voxel.size(0) * dilate)
            )
            masks_no_occl = []
            for dd, pp, kk, mm in zip(depths, poses, Ks, masks):
                pcd_in_mask = compute_point_cloud(
                    dd[None], pp[None], kk[None], mm[None]
                )
                if not new:
                    not_occluded = obj_seg.query(pcd_in_mask, voxel_dilated)
                else:
                    not_occluded = obj_seg.query_new(pcd_in_mask, voxel_dilated)
                change_inds = (mm==1).nonzero()
                change_inds_no_occl = change_inds[not_occluded]
                mm_no_occl = torch.zeros_like(mm)
                mm_no_occl[
                    0, change_inds_no_occl[:, 1], change_inds_no_occl[:, 2]
                ] = 1
                masks_no_occl.append(mm_no_occl)
            masks_no_occl = torch.stack(masks_no_occl, dim=0)
            masks_no_occl_all_obj.append(masks_no_occl)
        masks_no_occl_union = torch.any(
            torch.stack(masks_no_occl_all_obj, dim=0), dim=0
        )
        return masks_no_occl_union



    def main(
        self, transforms_json=None, configs=None, checkpoint_dir=None,
        refine_pose=True, debug=False
    ):
        """
        Estimate moved objects' 3D masks and pose changes

        Args:
            transforms_json (Path or str):
                transforms.json for the post-reconfig training dataset
            configs (Path or str): hyperparameters
            refine_pose (bool): Refine object pose change and camera poses

        Returns:
            obj_3D_seg (list of Obj3DSeg): Object 3D segmentation
        """
        if configs is None:
            configs = {
                "sam_threshold": 0.95,
                "mask_refine_sparse_view": 0.0,
                "area_threshold": 0.01,
                "pcd_filtering": 0.98,
                "pre_train_pred_bbox_expand": 0.05,
                "voxel_dim": 300,
                "bbox3d_expand": 1.8,
                "mask3d_dilate_uniform": 1,
                "mask3d_dilate_top": 0,
                "pose_change_break": None,
                "pose_refine_lr": 1e-3,
                "pose_refine_epochs": 50,
                "pose_refine_patience": 20,
                "vis_check_threshold": 0.8,
                "proj_check_cutoff": 0.95,
                "val_move_in_dilate_3d": 0.05,
                "val_move_out_dilate_3d": 0.05,
            }
        else:
            json_path = Path(configs)
            assert json_path.exists(), f"{json_path} does not exist"
            with open(json_path, "r") as f:
                configs = json.load(f)

        assert self.output_dir.exists(), f"{self.output_dir} does not exist"
        assert transforms_json is not None, "Need transforms.json for CD!"

        # Load pre-trained 3DGS
        assert os.path.isfile(self.load_config)
        _, self.pipeline_pretrain, _, _ = eval_setup(
            self.load_config, test_mode="inference",
            checkpoint_dir=checkpoint_dir, data_path=Path(transforms_json).parent
        )

        device = self.device
        # ----------------------------Load data -------------------------------
        # Load pre-training images and camera info + new images and camera info
        color_images, depth_images, img_fnames, c2w, K, dist_params, cameras = \
            read_transforms(transforms_json)
        # Undistort images
        assert dist_params.sum() < 1e-6, \
            "All images must be undistorted before change detection"
        sparse_view_file_ids, train_file_ids = [], []
        sparse_view_indices, pretrain_indices = [], []
        for ii, path in enumerate(img_fnames):
            id_int = extract_last_number(path.name)
            if "rgb_new" in path.as_posix():
                sparse_view_file_ids.append(id_int)
                sparse_view_indices.append(ii)
            else:
                pretrain_indices.append(ii)
            train_file_ids.append(id_int)

        N, _, H, W = color_images.shape
        # Get sparse-view captured images
        rgbs_captured_sparse_view = \
            color_images[sparse_view_indices].to(device)

        # Get sparse-view captured depths
        depths_captured_sparse_view = depth_images[sparse_view_indices].to(device)
        # Get sparse view camera parameters
        cameras_sparse_view = cameras[torch.tensor(sparse_view_indices, dtype=torch.long)]
        cam_poses_sparse_view = c2w[sparse_view_indices]
        Ks_sparse_view = K[sparse_view_indices]
        dist_params_sparse_view = dist_params[sparse_view_indices]
        # Get pre-training images and cameras
        color_images_pretrain_view = color_images[pretrain_indices]
        cam_poses_pretrain_view = c2w[pretrain_indices]
        Ks_pretrain_view = K[pretrain_indices]
        dist_params_pretrain_view = dist_params[pretrain_indices]
        # -------------------------------------------------------

        # Render images at the sparse viewpoints
        rgbs_render_sparse_view, depths_sparse_view = render_cameras(
            self.pipeline_pretrain, cameras_sparse_view, device=device
        )
        if debug:
            debug_image_pairs(
                rgbs_render_sparse_view, rgbs_captured_sparse_view,
                self.debug_dir
            )

        # Sec.IV.C: 2D Change detection on post-change views
        masks_changed_sparse, masks_changed_sparse_all = [], []
        points_changed_sparse = []


        for ii in range(len(sparse_view_file_ids)): 
            masks_changed, masks_changed_all = self.image_diff(
                rgbs_render_sparse_view[ii:ii+1],
                rgbs_captured_sparse_view[ii:ii+1]
            )

            masks_changed_sparse.append(masks_changed)
            masks_changed_sparse_all.append(masks_changed_all)

        if debug:
            masks_changed_tensor = torch.cat(masks_changed_sparse, dim=0)
            save_masks(
                masks_changed_tensor / 255.0, [
                    f"{self.debug_dir}/masks_changed{i}.png"
                    for i in range(len(masks_changed_tensor))
                ]
            )

                
        masks_move_out_sparse_view = []
        for ii, masks_changed in enumerate(masks_changed_sparse):
            masks_render, scores_render = effsam_refine_masks(
                rgbs_render_sparse_view[ii:ii+1], masks_changed,
                expand=configs["mask_refine_sparse_view"]
            )
            # Move-out masks have SAM prediction score > 0.95 on rendered image
            masks_out = [
                masks_render[i:i+1] for i, s in enumerate(scores_render)
                if s > configs["sam_threshold"]
            ]
            if len(masks_out) > 0:
                masks_out = torch.cat(masks_out, dim=0)
                masks_out = split_masks(masks_out, configs["area_threshold"])
            else:
                masks_out = torch.empty(0, 1, H, W, device=device)
            masks_move_out_sparse_view.append(masks_out)
        # Ignore views with overlapped move-out regions
        num_move_out = max([m.size(0) for m in masks_move_out_sparse_view])

        no_overlap_ind = []
        for i in range(len(masks_move_out_sparse_view)):
            if masks_move_out_sparse_view[i].size(0) >= num_move_out:
                no_overlap_ind.append(i)
        if debug:
            masks_to_save = torch.cat(masks_move_out_sparse_view, dim=0)
            save_masks(masks_to_save, [
                f"{self.debug_dir}/masks_move_out{i}.png"
                for i in range(len(masks_to_save))
            ])


        #########################
        masks_move_out_sparse_view = []

        for ii, masks_changed in enumerate(masks_changed_sparse):
            masks_render, scores_render = effsam_refine_masks(
                rgbs_render_sparse_view[ii:ii+1], masks_changed,
                expand=configs["mask_refine_sparse_view"]
            )
                        
            masks_out = [
                masks_render[i:i+1] for i, s in enumerate(scores_render)
                if s > configs["sam_threshold"]
            ]

            if self.debug_dir:
                save_masks(
                    masks_render,
                    [
                        f"{self.debug_dir}/masks_render_view{ii}_mask{j}.png"
                        for j in range(masks_render.shape[0])
                    ]
                )

            if len(masks_out) > 0:
                masks_out = torch.cat(masks_out, dim=0)
                masks_out = split_masks(masks_out, configs["area_threshold"])
            else:
                masks_out = torch.empty((0, 1, H, W)).to(device)
            
            masks_move_out_sparse_view.append(masks_out)
        
        num_move_out = max([m.size(0) for m in masks_move_out_sparse_view])
        print(f"[INFO] Found {num_move_out} masks for moved objects")



        # Filter out masks with too few points
        no_overlap_ind = []
        for i in range(len(masks_move_out_sparse_view)):
            if masks_move_out_sparse_view[i].size(0) >= num_move_out:
                no_overlap_ind.append(i)
                print(f"[INFO] View {i} has enough masks")
        if debug:
            masks_to_save = torch.cat(masks_move_out_sparse_view, dim=0)
            save_masks(masks_to_save, [
                f"{self.debug_dir}/masks_move_out{i}.png"
                for i in range(len(masks_to_save))
            ])
        # import pdb; pdb.set_trace()

        ## Object Association across (Prec-change) views
        pcds, pcd_feats = self.match_move_out(
            rgbs_render_sparse_view[no_overlap_ind],
            depths_sparse_view[no_overlap_ind],
            [masks_move_out_sparse_view[i] for i in no_overlap_ind],
            cam_poses_sparse_view[no_overlap_ind],
            Ks_sparse_view[no_overlap_ind],
            pcd_filter=configs["pcd_filtering"],
            embed_sim_thresh=0.9
        )

        ## Multi view move-in mask association across post-change views
        masks_move_in_sparse_view = []
        for ii, masks_changed in enumerate(masks_changed_sparse):
            masks_captured, scores_captured = effsam_refine_masks(
                rgbs_captured_sparse_view[ii:ii+1], masks_changed,
                expand=configs["mask_refine_sparse_view"]
            )
            # Move-in masks have SAM prediction score > 0.95 on captured image
            masks_in = [
                masks_captured[i:i+1] for i, s in enumerate(scores_captured)
                if s > 0.8
            ]
            if self.debug_dir:
                save_masks(
                    masks_captured,
                    [
                        f"{self.debug_dir}/masks_captured_view{ii}_mask{j}.png"
                        for j in range(masks_captured.shape[0])
                    ]
                )
            if len(masks_in) > 0:
                masks_in = torch.cat(masks_in, dim=0)
                masks_in = split_masks(masks_in, configs["area_threshold"])
            else:
                masks_in = torch.empty(0, 1, H, W, device=device)
            print(f"[View {ii}] Move-in candidates after SAM filtering: {len(masks_in)}")
            masks_move_in_sparse_view.append(masks_in)



        # Move-in masks w/ few inlier matches to obj templates are for inserted
        feats_move_in = self.get_features_in_masks(
            rgbs_captured_sparse_view, masks_move_in_sparse_view
        )

        masks_move_in_inserted = []
        for feats_i, masks_i in zip(feats_move_in, masks_move_in_sparse_view):
            masks_move_in_inserted_i = []
            for feat_i, mask_i in zip(feats_i, masks_i):
                num_inlier_max = 0
                for pcd_feat in pcd_feats:
                    _, num_inlier, _ = pcd_feat.PnP(
                        feat_i, Ks_sparse_view[0], H, W
                    )
                    num_inlier_max = max(num_inlier_max, num_inlier)
                if num_inlier_max < 5:
                    masks_move_in_inserted_i.append(mask_i[None])
            if len(masks_move_in_inserted_i) > 0:
                masks_move_in_inserted_i = torch.cat(
                    masks_move_in_inserted_i, dim=0
                )
            else:
                masks_move_in_inserted_i = torch.empty(
                    0, 1, H, W, device=device
                )
            masks_move_in_inserted.append(masks_move_in_inserted_i)
            print("###articulated masks")
            print(f"Found {len(masks_move_in_inserted_i)} matched masks")

       
        # preprocess captured depth
        H_target, W_target = rgbs_captured_sparse_view.shape[-2:]  # e.g., 480x640
        depths_captured_sparse_view = self.preprocess_depths(depths_captured_sparse_view, H_target, W_target)


        # obj_masks_move_in, obj_move_in_view_indices = self.match_move_in_depth(
        #     rgbs_captured_sparse_view, 
        #     masks_move_in_inserted,
        #     depths_captured_sparse_view,
        #     cam_poses_sparse_view,
        #     Ks_sparse_view
        # )
        # Pose change estimation for move-in objects
        obj_masks_move_in, obj_move_in_view_indices, pcds_post = self.match_move_in(
            rgbs_captured_sparse_view, 
            masks_move_in_inserted,
            depths_captured_sparse_view,
            cam_poses_sparse_view,
            Ks_sparse_view
        )


        pcd_posts = []
        print(f"[INFO] Found {len(obj_masks_move_in)} move-in objects across views")
        print(f"[INFO] Move-in masks view indices: {obj_move_in_view_indices}")

        # Obect pose change estimation
        feat_masks = [
            dilate_masks(m.any(dim=0, keepdim=True), 10)
            for m in masks_changed_sparse_all
        ]
        feats = self.get_features_in_masks(
            rgbs_captured_sparse_view, feat_masks
        )

        # Debug view 1  
        # debug_point_prompts(
        #     rgbs_captured_sparse_view[1:2], feats[1][0]["keypoints"],
        #     self.debug_dir
        # )

        # Sec.IV.F: Object pose change estimation
        pose_changes = []
        num_sparse_views = len(rgbs_captured_sparse_view)

        for obj_id, pcd_feat in enumerate(pcd_feats):
            best_pose = None
            best_inliers = 0
            best_matches = 0

            for view_idx in tqdm(range(num_sparse_views), desc=f"Pose estimation for Obj {obj_id}"):
                # Run PnP with 2D-3D correspondences
                pose_est, inliers, matches = pcd_feat.PnP(
                    feats[view_idx][0], Ks_sparse_view[view_idx], H, W, self.matcher
                )

                if pose_est is None:
                    continue

                # Transform to global coordinate frame (paper eq.)
                pose_est = cam_poses_sparse_view[view_idx] @ pose_est.inverse()

                # Use early break if specified in configs
                if configs.get("pose_change_break") and \
                configs["pose_change_break"][obj_id] is not None and \
                view_idx == configs["pose_change_break"][obj_id]:
                    best_pose = pose_est
                    best_inliers = inliers
                    best_matches = matches
                    break

                if inliers > best_inliers:
                    best_pose = pose_est
                    best_inliers = inliers
                    best_matches = matches

                # Optional debug visualization
                if debug:
                    m2d, m3d = pcd_feat.match(feats[view_idx][0], self.matcher)
                    m3d_proj, _ = project_points(
                        m3d, cam_poses_sparse_view[0:1], Ks_sparse_view[0:1],
                        dist_params_sparse_view[0:1], H, W
                    )
                    debug_matches(
                        rgbs_render_sparse_view[0:1], 
                        rgbs_captured_sparse_view[view_idx:view_idx+1],
                        m3d_proj, [m2d],
                        torch.arange(m2d.shape[0])[None, :, None].repeat(1, 1, 2),
                        self.debug_dir
                    )

            # Logging
            if best_pose is None:
                print(f"[Obj {obj_id}] ❌ Pose estimation failed (likely removed).")
            else:
                print(f"[Obj {obj_id}] ✅ pose_change:\n{best_pose.cpu().numpy()}")
            print(f"[Obj {obj_id}] inlier_ratio: {best_inliers} / {best_matches}")

            pose_changes.append(best_pose)


        # visualization
        if self.debug_dir is not None:
            for obj_id, (pcd, pose) in enumerate(zip(pcds, pose_changes)):
                np.save(self.debug_dir / f"obj{obj_id}_pre_change_pcd.npy", pcd.cpu().numpy())

                pose_np = pose.cpu().numpy() if pose is not None else None
                with open(self.debug_dir / f"obj{obj_id}_pose_change.json", "w") as f:
                    json.dump(pose_np.tolist() if pose_np is not None else None, f)

        # Summary
        num_moved = sum(pc is not None for pc in pose_changes)
        print(f"# Moved objects: {num_moved}")
        print(f"# Removed objects: {len(pose_changes) - num_moved}")
        print(f"# Inserted objects: {len(obj_masks_move_in)}")


        # Sec.IV.E: 3D object segmentation
        # Get more 2D masks for moved and removed objects
        if len(pcds) > 0:
            # Project the object pcd to pre-change views to get 2D bboxes
            bboxes2d = []
            for pcd in pcds:
                pcd_proj, is_point_in_img = project_points(
                    pcd, cam_poses_pretrain_view, Ks_pretrain_view,
                    dist_params_pretrain_view, H, W
                )
                if not is_point_in_img.all():
                    print("WARN: Some points are out of the pre-change images")
                # if debug:
                #     debug_point_prompts(
                #         color_images_pretrain_view, pcd_proj, self.debug_dir
                #     )
                bbox2d = compute_2D_bbox(pcd_proj)
                # Slightly expand 2D bboxes to improve SAM predictions
                bbox2d = expand_2D_bbox(
                    bbox2d, configs["pre_train_pred_bbox_expand"]
                )
                bboxes2d.append(bbox2d)
            bboxes2d = torch.stack(bboxes2d, dim=1) # NxMx4

            # SAM predict all move-out masks (batched for multi-object)
            masks_move_out_pretrain_view, scores = [], []
            for img, bbox2d in tqdm(
                zip(color_images_pretrain_view, bboxes2d), desc="SAM predict"
            ):
                mask, score = effsam_batch_predict(
                    img[None].to(device), bbox2d
                )
                masks_move_out_pretrain_view.append(mask)
                scores.append(score)
            masks_move_out_pretrain_view = torch.stack(
                masks_move_out_pretrain_view, dim=1
            ) # MxNx1xHxW
            scores = [list(t) for t in zip(*scores)] # M-list of N-list
            # if debug:
            #     debug_masks(
            #         masks_move_out_pretrain_view[0, ...], self.debug_dir
            #     )

            # Get high score mask indices
            high_score_inds = []
            for ss in scores:
                high_score = [i for i, x in enumerate(ss) if x > 0.95]
                if len(high_score) > 0:
                    print(f"High score masks: {len(high_score)} / {len(ss)}")
                else:
                    print("All masks look great!!")
                high_score_inds.append(high_score)
            # Check visibility of object point clouds
            visible = self.check_visibility(
                pcds, masks_move_out_pretrain_view, cam_poses_pretrain_view,
                Ks_pretrain_view, dist_params_pretrain_view, H, W,
                threshold=configs["vis_check_threshold"]
            )
            for vv in visible:
                print(
                    f"Visible views: {len(vv)} / {len(cam_poses_pretrain_view)}"
                )
            # Views having high-score masks and objects fully visible
            high_score_inds = [
                list(set(hs) & set(vis))
                for hs, vis in zip(high_score_inds, visible)
            ]
            for inds in high_score_inds:
                print(
                    f"#Views for 3D seg: {len(inds)} / {len(Ks_pretrain_view)}"
                )

        # # Multi-view mask fusion
        obj_segs = []
        # For moved and removed objects
        for ii in range(len(pcds)):
            bbox3d = compute_3D_bbox(pcds[ii])
            bbox3d = expand_3D_bbox(bbox3d, configs["bbox3d_expand"])

            voxel = points_to_occupancy(pcds[ii], bbox3d[0], bbox3d[1], (30, 30, 30))

            obj3Dseg = Object3DSeg(
                *bbox3d, voxel, pose_changes[ii], bbox3d,
                configs["mask3d_dilate_uniform"], configs["mask3d_dilate_top"]
            )
            # obj3Dseg.save(self.debug_dir / f"obj3Dseg_pre{ii}.pt")

            obj_segs.append(obj3Dseg)

        # For inserted objects
        obj_segs_inserted = []

        for ii, pcd_post in enumerate(pcds_post):
            bbox3d = compute_3D_bbox(pcd_post)
            bbox3d = expand_3D_bbox(bbox3d, configs["bbox3d_expand"])
            print(f"[Obj {ii}] bbox3d from fused PCD: {bbox3d}")

            # Create dummy voxel grid (binary mask = all occupied)
            voxel_dim = (30, 30, 30)
            voxel = points_to_occupancy(pcd_post, bbox3d[0], bbox3d[1], voxel_dim)
            occ_grid = torch.ones((1, 1, *voxel.shape[-3:]), dtype=torch.bool, device=device)

            obj3Dseg = Object3DSeg(
                bbox_min=bbox3d[0],
                bbox_max=bbox3d[1],
                voxel=voxel,
                pose_change=torch.eye(4, device=device), 
                tight_bbox=bbox3d,  
                mask_dilate_uniform=configs.get("mask3d_dilate_uniform", 1),
                mask_dilate_top=configs.get("mask3d_dilate_top", 0)
            )

            # obj3Dseg.save(self.debug_dir / f"obj3Dseg_post{ii}.pt")
            obj_segs_inserted.append(obj3Dseg)


        # Sec.IV.G: Global pose refinement
        if refine_pose and len(pcds) > 0:
            new_cameras = params_to_cameras(
                cam_poses_sparse_view, Ks_sparse_view, 
                dist_params_sparse_view, H, W
            )
            # for ii in range(len(obj_segs)):
            pose_changes, new_cameras = self.refine_obj_pose_change(
                rgbs_captured_sparse_view, obj_segs, new_cameras,
                lr=configs["pose_refine_lr"],
                epochs=5,
                patience=configs["pose_refine_patience"]
            )
            print("refined:")
            for ii, pose_change in enumerate(pose_changes):
                obj_segs[ii].set_pose_change(pose_change)
                print(pose_change)
        
        # Sec.IV.H: Occlusion-Aware Mask Projection
        # Optimize eval camera poses

        rgbs_eval, _, eval_fnames, _, _, _, cams_eval = \
            read_transforms(transforms_json, mode="val")
        if refine_pose:
            _, cams_eval = self.refine_obj_pose_change(
                rgbs_eval.to(device), obj_segs+obj_segs_inserted, cams_eval,
                lr=configs["pose_refine_lr"],
                epochs=5,
                patience=configs["pose_refine_patience"], optim="cam"
            )
            eval_file_ids = []
            for ii, path in enumerate(eval_fnames):
                id_int = extract_last_number(path.name)
                eval_file_ids.append(id_int)

        # Project 3D obj segs to eval images
        _, _, val_files, _, _, _, _ = read_transforms(
            transforms_json, read_images=False, mode="val"
        )
        # not checking occlusion for inserted object for now
        occlusion_check = [True]*len(obj_segs) + [False]*len(obj_segs_inserted)
        cams_eval = cams_eval.to(device)
        cams_eval.camera_to_worlds = cams_eval.camera_to_worlds.to(device)
        val_masks_move_out_no_occl = self.mask_proj(
            cams_eval, obj_segs+obj_segs_inserted, new=False,
            dilate=configs["val_move_out_dilate_3d"],
            occlusion_check=occlusion_check
        )
        val_masks_move_in_no_occl = self.mask_proj(
            cams_eval, obj_segs+obj_segs_inserted, new=True,
            dilate=configs["val_move_in_dilate_3d"],
            occlusion_check=occlusion_check
        )
        val_file_ids = []
        for ii, path in enumerate(val_files):
            id_int = extract_last_number(path.name)
            val_file_ids.append(id_int)
        # Save eval masks
        mask_output_dir = self.output_dir / "masks_new"
        os.makedirs(mask_output_dir, exist_ok=True)
        mask_files = [
            mask_output_dir / f"mask_{ii:05d}.png" for ii in val_file_ids
        ]
        save_masks(val_masks_move_out_no_occl, mask_files)
        mask_files = [
            mask_output_dir / f"mask_new_{ii:05d}.png" for ii in val_file_ids
        ]
        save_masks(val_masks_move_in_no_occl, mask_files)
        # Uncomment to save object 3D segmentations
        for ii, obj_seg in enumerate(obj_segs+obj_segs_inserted):
            obj_seg.save(self.output_dir / f"obj3Dseg{ii}.pt")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3DGS change detection")
    parser.add_argument(
        "--config", "-c", required=True, type=str,
        help="Path to the config.yml file of the pretrained 3DGS"
    )
    parser.add_argument(
        "--output", "-o", required=True, type=str,
        help="Path to save the output 3D segmentation"
    )
    parser.add_argument(
        "--transform", "-t", type=str,
        help="Path to transforms.json with info on both old and new images"
    )
    parser.add_argument(
        "--ckpt", "-ckpt", type=str, default=None,
        help="Path to the parent folder of 3DGS checkpoint"
    )
    parser.add_argument(
        "--debug", "-d", action="store_true",
        help="Debug mode"
    )
    args = parser.parse_args()

    # Load hyperparams
    hyperparams = f"{os.path.dirname(args.transform)}/configs.json"
    hyperparams = hyperparams if os.path.exists(hyperparams) else None
    # Detect changes
    change_det = ChangeDet(Path(args.config), Path(args.output), debug=args.debug)
    change_det.main(
        transforms_json=args.transform, configs=hyperparams, 
        checkpoint_dir=Path(args.ckpt), refine_pose=False, debug=args.debug
    )