import torch
import torch.nn.functional as F
import numpy as np
import os

from change_det.utils.proj_utils import depths_to_points, project_points 

DEFAULT_DEVICE = (
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)

class CoTrackerUtils:
    def __init__(self, device=None):
        self.device = device or DEFAULT_DEVICE
        self.model = None
    
    def load_model(self):
        """Load CoTracker3 model."""
        if self.model is None:
            print("Loading CoTracker3 model...")
            torch.cuda.empty_cache()
            self.model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline")
            self.model = self.model.to(self.device)
            self.model.eval()

    def sample_points_from_masks(
        self,
        masks,
        depth,
        src_pose,
        src_intrinsics,
        src_dist,
        tgt_pose,
        tgt_intrinsics,
        tgt_dist,
        H,
        W,
        num_points_per_mask=50,
    ):
        """
        Sample points from 2D masks and reproject into video camera frame.
        If depth consistency rejects all points, fall back to in-bounds UVs.
        """

        if masks.numel() == 0:
            return torch.empty(0, 2, device=self.device)

        all_points = []
        N, _, H_mask, W_mask = masks.shape

        for i in range(N):
            mask = masks[i, 0]  # (H_mask, W_mask)
            coords = torch.nonzero(mask, as_tuple=False)  # (num_pixels, 2) [v, u]
            if coords.shape[0] == 0:
                continue

            # Random subsample
            if coords.shape[0] > num_points_per_mask:
                idx = torch.randperm(coords.shape[0], device=self.device)[:num_points_per_mask]
                coords = coords[idx]

            # Convert to [u, v]
            uv = coords[:, [1, 0]].float().to(self.device)

            if depth.ndim > 2:
                depth = depth.squeeze()
                assert depth.ndim == 2, f"Depth map wrong shape after squeeze: {depth.shape}"

            # Extract depths (M,1)
            depths = depth[coords[:, 0], coords[:, 1]].view(-1, 1).float().to(self.device)

            # 2D -> 3D (in world space from source camera)
            pts_3d = depths_to_points(uv, depths, src_pose, src_intrinsics)

            # Reproject into target (video) camera
            pts_2d, valid = project_points(
                pts_3d,
                tgt_pose[None],
                tgt_intrinsics[None],
                tgt_dist[None],
                H,
                W,
            )

            # Debug
            print(f"[DEBUG] mask {i}: sampled {uv.shape[0]} pts, "
                f"valid={valid.sum().item()} / {valid.numel()} "
                f"min_uv=({pts_2d[0,:,0].min().item():.1f},{pts_2d[0,:,1].min().item():.1f}) "
                f"max_uv=({pts_2d[0,:,0].max().item():.1f},{pts_2d[0,:,1].max().item():.1f})")

            # Keep valid points if any
            if valid.any():
                all_points.append(pts_2d[0][valid[0]])
            else:
                # fallback: accept all projected points that are inside frame
                inside = (
                    (pts_2d[0, :, 0] >= 0) & (pts_2d[0, :, 0] < W) &
                    (pts_2d[0, :, 1] >= 0) & (pts_2d[0, :, 1] < H)
                )
                if inside.any():
                    print(f"[DEBUG] mask {i}: fallback kept {inside.sum().item()} in-frame points")
                    all_points.append(pts_2d[0][inside])

        if len(all_points) == 0:
            return torch.empty(0, 2, device=self.device)

        return torch.cat(all_points, dim=0)  # (M, 2)



    
    def track_points_in_video(self, video_frames, query_points, query_frame=0):
        """
        Track points across video frames.
        
        Args:
            video_frames: (B, T, 3, H, W) or (1, T, 3, H, W) - video
            query_points: (M, 2) - points to track [u, v]
            query_frame: int - frame index to start tracking
            
        Returns:
            tracks: (B, T, M, 2) - tracked points
            visibility: (B, T, M) - visibility mask
        """
        if query_points.numel() == 0:
            B, T = video_frames.shape[:2]
            return (torch.empty(B, T, 0, 2, device=self.device),
                   torch.empty(B, T, 0, device=self.device))
        
        self.load_model()
        
        with torch.no_grad():
            # CoTracker expects query_points as (B, M, 3) where 3rd dim is [t, u, v]
            B, T = video_frames.shape[:2]
            M = query_points.shape[0]
            
            # Add time dimension (query_frame) to points
            query_frame_tensor = torch.full((M, 1), query_frame, 
                                          device=query_points.device, dtype=query_points.dtype)
            query_points_3d = torch.cat([query_frame_tensor, query_points], dim=1)  # (M, 3)
            query_points_3d = query_points_3d.unsqueeze(0)  # (1, M, 3)
            
            if B > 1:
                query_points_3d = query_points_3d.repeat(B, 1, 1)  # (B, M, 3)
            
            # Track points
            tracks, visibility = self.model(video_frames, queries=query_points_3d)
            
        return tracks, visibility
    
    def lift_tracks_to_3d(self, tracks_2d, visibility, depths, camera_intrinsics):
        """
        Lift 2D tracks to 3D using depth maps.
        
        Args:
            tracks_2d: (B, T, N, 2) - 2D pixel coordinates
            visibility: (B, T, N) - track visibility
            depths: (T, H, W) - depth maps
            camera_intrinsics: (3, 3) - camera intrinsics
            
        Returns:
            tracks_3d: (B, T, N, 3) - 3D points
            valid_3d: (B, T, N) - validity mask
        """
        if tracks_2d.numel() == 0:
            B, T = tracks_2d.shape[:2]
            return (torch.empty(B, T, 0, 3, device=self.device),
                   torch.empty(B, T, 0, device=self.device))
        
        B, T, N, _ = tracks_2d.shape
        H, W = depths.shape[1], depths.shape[2]
        
        # Sample depth values at track locations
        tracks_flat = tracks_2d.view(B * T, N, 2)  # (B*T, N, 2)
        
        # Normalize coordinates for grid_sample [-1, 1]
        u = tracks_flat[..., 0] / (W - 1) * 2 - 1
        v = tracks_flat[..., 1] / (H - 1) * 2 - 1
        grid = torch.stack([u, v], dim=-1).unsqueeze(2)  # (B*T, N, 1, 2)
        
        # Expand depths for sampling
        depths_expanded = depths.unsqueeze(1).repeat(B, 1, 1, 1)  # (B*T, 1, H, W)
        
        # Sample depths
        sampled_depths = F.grid_sample(
            depths_expanded, grid, align_corners=True, mode="bilinear"
        ).squeeze(1).squeeze(-1)  # (B*T, N)
        
        sampled_depths = sampled_depths.view(B, T, N)  # (B, T, N)
        
        # Back-project to 3D
        fx, fy = camera_intrinsics[0, 0], camera_intrinsics[1, 1]
        cx, cy = camera_intrinsics[0, 2], camera_intrinsics[1, 2]
        
        z = sampled_depths
        x = (tracks_2d[..., 0] - cx) * z / fx
        y = (tracks_2d[..., 1] - cy) * z / fy
        tracks_3d = torch.stack([x, y, z], dim=-1)
        
        # Valid mask
        valid_3d = (z > 0) & (z < 100) #& visibility.bool()
        
        return tracks_3d, valid_3d
    
    def cluster_motion_trajectories(self, tracks_3d, valid_3d, motion_threshold=0.05):
        """
        Simple clustering: static vs moving objects.
        
        Args:
            tracks_3d: (B, T, N, 3) - 3D trajectories
            valid_3d: (B, T, N) - validity mask
            motion_threshold: float - threshold in meters
            
        Returns:
            motion_labels: (B, N) - 0 for static, 1 for moving
        """
        B, T, N, _ = tracks_3d.shape
        motion_labels = torch.zeros(B, N, dtype=torch.long, device=self.device)
        
        for b in range(B):
            for n in range(N):
                valid_frames = valid_3d[b, :, n]
                if valid_frames.sum() < 2:
                    continue
                
                # Get first and last valid positions
                valid_indices = torch.where(valid_frames)[0]
                start_pos = tracks_3d[b, valid_indices[0], n]
                end_pos = tracks_3d[b, valid_indices[-1], n]
                
                displacement = torch.norm(end_pos - start_pos).item()
                
                if displacement > motion_threshold:
                    motion_labels[b, n] = 1  # Moving
                else:
                    motion_labels[b, n] = 0  # Static
        
        return motion_labels
    



    def visualize_tracks_video(self, video_np, tracks_3d, valid_3d,
                            video_intrinsics, video_pose,
                            motion_labels=None,
                            out_path="debug_tracks.mp4",
                            fps=30, max_points=100):
        """
        Overlay tracked 3D points onto video frames and save as MP4.
        """
        import imageio
        import cv2
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        # Convert intrinsics/pose
        K = video_intrinsics.cpu().numpy() if hasattr(video_intrinsics, "cpu") else video_intrinsics
        c2w = video_pose.cpu().numpy() if hasattr(video_pose, "cpu") else video_pose
        w2c = np.linalg.inv(c2w)

        # Shapes
        T, H, W, _ = video_np.shape
        tracks_3d = tracks_3d.cpu().numpy() if hasattr(tracks_3d, "cpu") else tracks_3d
        valid_3d  = valid_3d.cpu().numpy() if hasattr(valid_3d, "cpu") else valid_3d
        if motion_labels is not None and hasattr(motion_labels, "cpu"):
            motion_labels = motion_labels.cpu().numpy()

        # Defensive fix: if only one frame of 3D tracks, repeat across T
        if tracks_3d.shape[0] == 1 and T > 1:
            tracks_3d = np.repeat(tracks_3d, T, axis=0)
            valid_3d  = np.repeat(valid_3d,  T, axis=0)

        frames_out = []

        for t in range(T):
            frame = video_np[t].copy()

            pts_3d = tracks_3d[0, t]  # (N, 3)
            mask   = valid_3d[0, t]   # (N,)
            # # Homogenize → (4, M)
            # pts_3d_h = np.concatenate([pts_3d, np.ones((pts_3d.shape[0], 1))], axis=1).T  # (4, M)
            # pts_cam = (w2c @ pts_3d_h).T[:, :3]  # (M, 3)

            pts_cam = pts_3d

            zs = pts_cam[:, 2]
            uv = (K @ pts_cam.T).T
            uv = uv[:, :2] / (uv[:, 2:3] + 1e-8)  # avoid div0

            count = 0
            for i, (u, v, z, is_valid) in enumerate(zip(uv[:, 0], uv[:, 1], zs, mask)):
                if not bool(is_valid):
                    if i < 5:  # limit spam
                        print(f"[DEBUG] t={t}, point {i}: invalid mask")
                    continue
                if z <= 0:
                    if i < 5:
                        print(f"[DEBUG] t={t}, point {i}: behind camera (z={z:.3f})")
                    continue
                if not (0 <= u < W and 0 <= v < H):
                    if i < 5:
                        print(f"[DEBUG] t={t}, point {i}: outside frame (u={u:.1f}, v={v:.1f})")
                    continue

                # If we reach here, the point is valid
                #print(f"[DEBUG] t={t}, point {i}: drawing at ({u:.1f}, {v:.1f}), z={z:.3f}")
                color = (0, 255, 0)
                cv2.circle(frame, (int(u), int(v)), 3, color, -1)
                count += 1
                if count >= max_points:
                    break


            frames_out.append(frame.astype(np.uint8))

        # Save video using imageio (ffmpeg under the hood)
        imageio.mimsave(out_path, frames_out, fps=fps, quality=8, macro_block_size=None)
        print(f"[INFO] Saved track overlay video: {out_path}")