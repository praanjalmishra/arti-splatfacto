# 3DGS-based change detection
import argparse
from ast import Not
import gc
import json
import os
import random
import re
from pathlib import Path
import datetime
from tabnanny import verbose

import open3d as o3d
import torchvision.utils as vutils
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from torchvision import transforms

import cv2
import numpy as np
import torch
from lightglue import LightGlue, SuperPoint, viz2d
from matplotlib import pyplot as plt
from PIL import Image
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
import random

from change_det.utils.sam_refine import refine_change_detection_masks
from nerfstudio.cameras.camera_paths import get_path_from_json
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.models.splatfacto import SplatfactoModel
from change_det.utils.debug_utils import (
    debug_image_pairs, debug_depth_pairs, debug_images, debug_masks, debug_matches,
    debug_point_cloud, debug_point_prompts, debug_depths
)
from mask_refine.effsam_utils import (
    effsam_predict, effsam_embedding, effsam_refine_masks,
    effsam_batch_predict, compute_2D_bbox, expand_2D_bbox,
    get_effsam_embedding_in_masks
)
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.utils.gauss_utils import transform_gaussians
from change_det.utils.img_utils import (
    extract_depths_at_pixels, image_align, filter_features_with_mask,
    in_image, split_masks, dilate_masks, overlay_mask_on_image
)
from change_det.utils.proj_utils import depths_to_points
from change_det.utils.io import (
    load_from_json, write_to_json, read_dataset, read_imgs, read_transforms,
    save_masks, params_to_cameras, cameras_to_params, save_imgs
)
from change_det.utils.misc import extract_last_number
# from nerfstudio.utils.obj_3d_seg import Object3DSeg, Obj3DFeats
from change_det.utils.pcd_utils import (
    compute_3D_bbox, compute_point_cloud, expand_3D_bbox,
    point_cloud_filtering, mahalanobis_filter, nn_distance, pcd_size, bbox2voxel, visualize_bbox3d_matplotlib, 
    points_to_occupancy
)
# from change_det.utils.proj_utils import (
#     depths_to_points, proj_check_3D_points, project_points, draw_projected_bbox_on_image
# )
from nerfstudio.utils.poses import to4x4
from change_det.utils.render_utils import render_cameras, render_3dgs_at_cam

from change_det.utils.image_diff import image_diff_dinov2, image_diff_effsam, image_diff_sam2_with_depth
from change_det.utils.obj_3d_seg import Object3DSeg, Obj3DFeats

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

        
        self.dinov2_model = None
        self.dinov2_processor = None
        self.use_dinov2 = True



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

        if len(masks[0]) == 0:
            print("[INFO] No move-out masks in the first view, skipping move-out matching.")

        else:
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
                if c >= len(masks[i]) or r >= len(pcd_embeds):
                    continue
                if pcd_embeds[r].numel() == 0 or embeds[i][c].numel() == 0:
                    continue

                embed_sim = torch.cosine_similarity(
                    pcd_embeds[r], embeds[i][c], dim=-1
                )

                if embed_sim.max() > embed_sim_thresh:
                    pcds[r] = torch.cat((pcds[r], new_pcds[c]), dim=0)
                    pcd_sizes[r] = pcd_size(pcds[r])
                    pcd_counts[r] += 1
                    pts3D = compute_feats_3D(feats[i][c], depths[i:i+1], poses[i], Ks[i])
                    pcd_feats[r].add_feats(feats[i][c], pts3D)
                    pcd_embeds[r] = torch.cat((pcd_embeds[r], embeds[i][c:c+1]), dim=0)
                else:
                    pcds.append(new_pcds[c])
                    pcd_sizes.append(pcd_size(new_pcds[c]))
                    pcd_counts.append(1)
                    pts3D = compute_feats_3D(feats[i][c], depths[i:i+1], poses[i], Ks[i])
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





    def main(
        self, transforms_json=None, configs=None,
        refine_pose=True, debug=False
    ):
        """
        Estimate moved objects' 3D masks and pose changes

        Args:
            transforms_json (Path or str):
                transforms.json for the post-reconfig training dataset
            configs (Path or str): hyperparameters

        Returns:
            obj_3D_seg (list of Obj3DSeg): Object 3D segmentation
        """
        if configs is None:
            configs = {
                "sam_threshold": 0.75,
                "mask_refine_sparse_view": 0.15,
                "area_threshold": 0.01,
                "cd_kernel_ratio": 0.1,
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
            self.load_config, test_mode="inference"
        )

        device = self.device

        # ---------------------------- Load data -------------------------------
        # Load all frames from transforms.json
        color_images, depth_images, img_fnames, c2w, K, dist_params, cameras, times = read_transforms(transforms_json)

        assert dist_params.sum() < 1e-6, "All images must be undistorted before change detection"

        # Separate pretrain (time == 0) and post-change (time == 1) frames
        pretrain_indices = [i for i, t in enumerate(times) if t == 0.0]
        postchange_indices = [i for i, t in enumerate(times) if t == 1.0]

        # sparse_view_indices = postchange_indices[3:10]  # e.g. N_sparse = 3

        # sparse_view_indices = [postchange_indices[0]]

        # Define how many frames to sample from the end
        N_SPARSE = 1  # change as needed
        FRACTION_END = 0.0  # take frames from the last 20% of the timeline, for example

        # Indices sorted by time
        sorted_indices = np.argsort(times)
        sorted_times = np.array(times)[sorted_indices]

        # Identify which frames are in the "post-change" region (near t=1)
        threshold_time = 1.0 - FRACTION_END  # e.g. if FRACTION_END=0.2, then t > 0.8
        postchange_indices = [i for i, t in zip(sorted_indices, sorted_times) if t >= threshold_time]
    
        # Sample N_SPARSE frames from the post-change portion
        num_samples = min(N_SPARSE, len(postchange_indices))
        # sparse_view_indices = random.sample(postchange_indices, num_samples)
        sparse_view_indices = [len(img_fnames) - 1]


        # print(f"Sampled {num_samples} frames from post-change region (t > {threshold_time:.2f})")
        print("Indices:", sparse_view_indices)

        # Get tensors
        N, _, H, W = color_images.shape

        # ---------------------------- Post-change -----------------------------
        rgbs_captured_sparse_view = color_images[sparse_view_indices].to(device)
        depths_captured_sparse_view = depth_images[sparse_view_indices].to(device)
        cam_poses_sparse_view = c2w[sparse_view_indices]
        Ks_sparse_view = K[sparse_view_indices]
        dist_params_sparse_view = dist_params[sparse_view_indices]
        cameras_sparse_view = cameras[torch.tensor(sparse_view_indices)]

        # ---------------------------- Pre-train -------------------------------
        color_images_pretrain_view = color_images[pretrain_indices].to(device)
        depths_pretrain_view = depth_images[pretrain_indices].to(device)
        cam_poses_pretrain_view = c2w[pretrain_indices]
        Ks_pretrain_view = K[pretrain_indices]
        dist_params_pretrain_view = dist_params[pretrain_indices]
        cameras_pretrain_view = cameras[torch.tensor(pretrain_indices)]

        # -------------------------------------------------------

        
        # ----------------------------Render Pre-Change---------------------------
        print("[INFO] Rendering pre-trained Splatfacto at post-change viewpoints...")
        rgbs_render_sparse_view, depths_render_sparse_view = render_cameras(
            self.pipeline_pretrain, cameras_sparse_view, device=device
        )

        # if self.debug_dir is not None:
        #     debug_rgb_dir = self.debug_dir / "debug_rgb"
        #     debug_depth_dir = self.debug_dir / "debug_depth"

        #     debug_rgb_dir.mkdir(parents=True, exist_ok=True)
        #     debug_depth_dir.mkdir(parents=True, exist_ok=True)

        #     debug_image_pairs(rgbs_render_sparse_view, rgbs_captured_sparse_view, debug_rgb_dir)
        #     debug_depth_pairs(depths_render_sparse_view, depths_captured_sparse_view, debug_depth_dir)



        print("[INFO] Cleaning up GPU memory...")
        del self.pipeline_pretrain
        torch.cuda.empty_cache()

        masks_changed_sparse, masks_changed_sparse_all = [], []


        for ii in tqdm(range(len(sparse_view_indices)), desc="Running CD"): 
            
            masks_changed, masks_changed_all = image_diff_sam2_with_depth(
                rgbs_render_sparse_view[ii:ii+1],  
                rgbs_captured_sparse_view[ii:ii+1],   
                depths_render_sparse_view[ii:ii+1],
                depths_captured_sparse_view[ii:ii+1],       
                debug_dir=self.debug_dir,
                threshold=configs["area_threshold"],
                kernel_ratio=configs["cd_kernel_ratio"],
                depth_weight=0.20,
            )
            
            print(f"[INFO] View {ii}: Detected {masks_changed.size(0)} changed masks.")
            print(f"[INFO] Shape of masks_changed: {masks_changed.shape}")
            
            # Check if any masks were detected
            if masks_changed.numel() == 0 or masks_changed.size(0) == 0:
                print(f"[WARNING] No masks detected for view {ii}, skipping refinement")
                masks_changed_sparse.append(masks_changed)
                masks_changed_sparse_all.append(masks_changed_all)
                continue
            
            # Refine masks with SAM2
            refined_masks, iou = refine_change_detection_masks(
                image_captured=rgbs_captured_sparse_view[ii],  # [3, H, W]
                masks_changed=masks_changed,  # [N, 1, H, W] in range [0, 255]
                device="cuda",
                num_positive_points=10,
                num_negative_points=20,
                debug=True 
            )
            
            print(f"[INFO] Refined masks shape: {refined_masks.shape}, "
                f"range: [{refined_masks.min():.3f}, {refined_masks.max():.3f}]")
            
            # Convert back to [0, 255] for consistency with your pipeline
            refined_masks_uint8 = (refined_masks * 255.0).to(torch.uint8)
            
            masks_changed_sparse.append(refined_masks_uint8)
            masks_changed_sparse_all.append(masks_changed_all)

        # Save masks (now they should be visible)
        if debug:
            masks_changed_tensor = torch.cat(masks_changed_sparse, dim=0)
            # Normalize to [0, 1] for saving
            save_masks(
                masks_changed_tensor / 255.0, 
                [f"{self.debug_dir}/masks_refined_{i}.png" 
                for i in range(len(masks_changed_tensor))]
            )
            
            # Save overlays
            for ii, masks_changed in enumerate(masks_changed_sparse):
                if masks_changed.numel() == 0:
                    continue
                for mi, mask in enumerate(masks_changed):
                    # Normalize mask for overlay
                    mask_normalized = mask.float() / 255.0
                    overlay = overlay_mask_on_image(
                        rgbs_captured_sparse_view[ii:ii+1], 
                        mask_normalized
                    )
                    cv2.imwrite(
                        f"{self.debug_dir}/overlay_refined_view{ii}_mask{mi}.png",
                        cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR) 
                    )
                    
        import pdb; pdb.set_trace()
        masks_move_out_sparse_view = []

        for ii, masks_changed in enumerate(masks_changed_sparse):
            # Refine predicted masks for this view
            masks_render, scores_render = effsam_refine_masks(
                rgbs_render_sparse_view[ii:ii + 1],
                masks_changed,
                expand=configs["mask_refine_sparse_view"]
            )

            # select only high-confidence masks
            high_conf_indices = [i for i, s in enumerate(scores_render) if s > configs["sam_threshold"]]

            if len(high_conf_indices) > 0:
                masks_out = masks_render[high_conf_indices]  
                masks_out = split_masks(masks_out, configs["area_threshold"])  # Filter small / merged regions
            else:
                masks_out = torch.empty(0, 1, H, W, device=device)

            masks_move_out_sparse_view.append(masks_out)

        # Handle empty cases robustly (no move-out masks in any view)
        if len(masks_move_out_sparse_view) == 0 or all(m.numel() == 0 for m in masks_move_out_sparse_view):
            num_move_out = 0
        else:
            num_move_out = max(m.size(0) for m in masks_move_out_sparse_view)

        print(f"[INFO] Max number of move-out masks in a view: {num_move_out}")
        print(f"[INFO] Move-out candidates per view: {[m.size(0) for m in masks_move_out_sparse_view]}")

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

            # save overlays
            for ii, masks_out in enumerate(masks_move_out_sparse_view):
                for mi, mask in enumerate(masks_out):
                    overlay = overlay_mask_on_image(
                        rgbs_render_sparse_view[ii:ii+1], mask
                    )
                    cv2.imwrite(
                        f"{self.debug_dir}/overlay_move_out_view{ii}_mask{mi}.png",
                        cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)  # ensure OpenCV format
                    )

        ##### move in #####
        masks_move_in_sparse_view = []

        for ii, masks_changed in enumerate(masks_changed_sparse):
            # Refine predicted masks for this view
            masks_render, scores_render = effsam_refine_masks(
                rgbs_captured_sparse_view[ii:ii + 1],
                masks_changed,
                expand=configs["mask_refine_sparse_view"]
            )

            # select only high-confidence masks
            high_conf_indices = [i for i, s in enumerate(scores_render) if s > configs["sam_threshold"]]

            if len(high_conf_indices) > 0:
                mask_in = masks_render[high_conf_indices]  
                mask_in = split_masks(mask_in, configs["area_threshold"])  # Filter small / merged regions
            else:
                mask_in = torch.empty(0, 1, H, W, device=device)

            masks_move_in_sparse_view.append(mask_in)

        # Handle empty cases robustly (no move-in masks in any view)
        if len(masks_move_in_sparse_view) == 0 or all(m.numel() == 0 for m in masks_move_in_sparse_view):
            num_move_in = 0
        else:
            num_move_in = max(m.size(0) for m in masks_move_in_sparse_view)

        print(f"[INFO] Max number of move-in masks in a view: {num_move_in}")
        print(f"[INFO] Move-in candidates per view: {[m.size(0) for m in masks_move_in_sparse_view]}")

        # Filter out masks with too few points
        no_overlap_ind = []
        for i in range(len(masks_move_in_sparse_view)):
            if masks_move_in_sparse_view[i].size(0) >= num_move_in:
                no_overlap_ind.append(i)
                print(f"[INFO] View {i} has enough masks")
        if debug:
            masks_to_save = torch.cat(masks_move_out_sparse_view, dim=0)
            save_masks(masks_to_save, [
                f"{self.debug_dir}/masks_move_out{i}.png"
                for i in range(len(masks_to_save))
            ])

            # save overlays
            for ii, mask_in in enumerate(masks_move_in_sparse_view):
                for mi, mask in enumerate(mask_in):
                    overlay = overlay_mask_on_image(
                        rgbs_captured_sparse_view[ii:ii+1], mask, color=(0, 255, 0)
                    )
                    cv2.imwrite(
                        f"{self.debug_dir}/overlay_move_in_view{ii}_mask{mi}.png",
                        cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)  # ensure OpenCV format
                    )

        ## Object Association across for move-out objects
        pcds, pcd_feats = self.match_move_out(
            rgbs_render_sparse_view[no_overlap_ind],
            depths_render_sparse_view[no_overlap_ind],
            [masks_move_out_sparse_view[i] for i in no_overlap_ind],
            cam_poses_sparse_view[no_overlap_ind],
            Ks_sparse_view[no_overlap_ind],
            pcd_filter=configs["pcd_filtering"],
            embed_sim_thresh=0.9
        )

        pcds, pcd_feats = self.match_move_in_2(
            rgbs_render_sparse_view[no_overlap_ind],
            depths_captured_sparse_view[no_overlap_ind],
            [masks_move_in_sparse_view[i] for i in no_overlap_ind],
            cam_poses_sparse_view[no_overlap_ind],
            Ks_sparse_view[no_overlap_ind],
            pcd_filter=configs["pcd_filtering"],
            embed_sim_thresh=0.9
        )

        import pdb; pdb.set_trace()



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
        refine_pose=False, debug=args.debug
    )