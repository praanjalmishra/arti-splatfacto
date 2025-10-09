"""
CoTracker RGB-D Processing Module

Handles the data acquisition and trajectory processing phase of the 4D RANSAC pipeline.
This includes:
- Loading and preprocessing RGB-D video sequences
- Running CoTracker for 2D trajectory extraction
- Uplifting 2D trajectories to 3D using depth data
- Filtering and validating trajectories
- Segmenting rigid parts (moving vs static)
"""
import os
import json
import torch
import numpy as np
from typing import List, Optional, Tuple
from pathlib import Path
import cv2
from tqdm import tqdm
from cotracker.utils.visualizer import Visualizer, read_video_from_path

from data_structures import (
    TrackPoint2D, Trajectory2D, Point3D, Trajectory3D,
    CameraIntrinsics, TrajectoryFilterConfig,
    trajectory_2d_to_3d
)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import DBSCAN

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# from effsam_utils import effsam_embedding, effsam_refine_masks
from dinov2_utils import load_dinov2_model, compute_dinov2_similarity

import glob
from PIL import Image

def compute_change_mask(first_frame: torch.Tensor, 
                       last_frame: torch.Tensor,
                       out_dir: str,
                       threshold: float = 1e-1,
                       use_dinov2: bool = True,
                       dinov2_model=None,
                       dinov2_processor=None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute change mask using DINOv2 or simple difference.
    """
    os.makedirs(out_dir, exist_ok=True)
    
    # Ensure batch dimension
    if first_frame.dim() == 3:
        first_frame = first_frame.unsqueeze(0)
    if last_frame.dim() == 3:
        last_frame = last_frame.unsqueeze(0)
    
    H, W = first_frame.shape[-2:]
    device = first_frame.device
    
    # Downsample for efficiency
    if max(H, W) > 480:
        scale = 480 / max(H, W)
        new_h = int(H * scale)
        new_w = int(W * scale)
        new_h = (new_h // 8) * 8
        new_w = (new_w // 8) * 8
        
        print(f"Downsampling for change detection: {H}x{W} → {new_h}x{new_w}")
        first_frame = torch.nn.functional.interpolate(
            first_frame, size=(new_h, new_w), mode='bilinear', align_corners=False
        )
        last_frame = torch.nn.functional.interpolate(
            last_frame, size=(new_h, new_w), mode='bilinear', align_corners=False
        )
        H, W = new_h, new_w
    
    # Convert to numpy
    first_np = first_frame[0].permute(1, 2, 0).cpu().numpy()
    last_np = last_frame[0].permute(1, 2, 0).cpu().numpy()
    first_np = np.clip(first_np, 0, 1)
    last_np = np.clip(last_np, 0, 1)
    
    torch.cuda.empty_cache()
    
    # Save debug images
    cv2.imwrite(f"{out_dir}/first_frame.png", 
                (first_np[..., ::-1] * 255).astype(np.uint8))
    cv2.imwrite(f"{out_dir}/last_frame.png", 
                (last_np[..., ::-1] * 255).astype(np.uint8))
    
    # === KEY CHANGE: Use DINOv2 or simple diff ===
    if use_dinov2 and dinov2_model is not None:
        print("Using DINOv2 semantic similarity...")
        sim_map = compute_dinov2_similarity(
            first_np, last_np, dinov2_model, dinov2_processor, device
        )
        # Resize to original resolution
        sim_map = cv2.resize(sim_map, (W, H), interpolation=cv2.INTER_CUBIC)
        similarity_map = (sim_map * 255).astype(np.uint8)
    else:
        print("Using simple color difference...")
        diff = np.abs(first_np - last_np)
        diff_gray = np.mean(diff, axis=2)
        similarity_map = (255 - (diff_gray * 255)).astype(np.uint8)
    
    cv2.imwrite(f"{out_dir}/similarity_map.png", similarity_map)
    
    # Threshold
    thresh = cv2.threshold(
        similarity_map, 0, 255, 
        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    )[1]
    
    cv2.imwrite(f"{out_dir}/threshold_mask.png", thresh)
    
    # Find contours
    contours, _ = cv2.findContours(
        thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    print(f"Found {len(contours)} change regions")
    
    masks = []
    masks_all = []
    
    for i, contour in enumerate(contours):
        mask = np.zeros((H, W), dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
        
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).float() / 255.0
        masks_all.append(mask_tensor)
        
        area_ratio = cv2.contourArea(contour) / (H * W)
        if area_ratio >= threshold:
            masks.append(mask_tensor)
            print(f"Region {i}: area={area_ratio:.4f} (kept)")
            cv2.imwrite(f"{out_dir}/mask_region_{i}.png", mask)
        else:
            print(f"Region {i}: area={area_ratio:.4f} (discarded)")
    
    if len(masks) == 0:
        print("Warning: No large change regions found, using all regions")
        masks = masks_all
    
    # Stack and move to device
    masks = torch.stack(masks, dim=0).to(device)
    masks_all = torch.stack(masks_all, dim=0).to(device)
    
    print(f"Final masks: {masks.shape[0]} large regions, {masks_all.shape[0]} total")
    
    return masks, masks_all


class CoTrackerRGBD:
    """
    Main class for processing RGB-D video sequences with CoTracker.
    
    This class handles the complete data acquisition pipeline from raw RGB-D
    video to filtered 3D trajectories ready for RANSAC processing.
    """
    
    def __init__(self, 
                trajectory_filter_config: TrajectoryFilterConfig,
                device: str = DEFAULT_DEVICE,
                grid_size: int = 30,
                grid_query_frame: int = 0,
                backward_tracking: bool = True,
                use_change_mask: bool = True,
                change_threshold: float = 1e-2,
                max_resolution: int = 640,  # NEW
                max_frames: int = 60):       # NEW
        """
        Args:
            max_resolution: Maximum spatial dimension (pixels)
            max_frames: Maximum number of frames to process
        """
        self.trajectory_filter_config = trajectory_filter_config
        self.device = device
        self.grid_size = grid_size
        self.grid_query_frame = grid_query_frame
        self.backward_tracking = backward_tracking
        self.use_change_mask = use_change_mask
        self.change_threshold = change_threshold
        self.max_resolution = max_resolution
        self.max_frames = max_frames
        
        self.model = None
        self.camera_intrinsics = None
        self.camera_extrinsics = None
        self.video_h = None  # Store video dimensions
        self.video_w = None

        self.use_dinov2 = True  
        self.dinov2_model = None
        self.dinov2_processor = None

        
    def load_cotracker_model(self):
        """Load the CoTracker3 model."""
        print("Loading CoTracker3 model...")
        torch.cuda.empty_cache()
        self.model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline")
        self.model = self.model.to(self.device)
        
        if torch.cuda.is_available():
            available_memory = (torch.cuda.get_device_properties(0).total_memory - 
                              torch.cuda.memory_allocated())
            print(f"Available GPU memory: {available_memory / 1e9:.2f} GB")

    def compute_change_segmentation(self, video_tensor: torch.Tensor, out_dir: str):
        print("Computing change detection mask...")
        
        if self.use_dinov2 and self.dinov2_model is None:
            print("Loading DINOv2 for change detection...")
            self.dinov2_processor, self.dinov2_model = load_dinov2_model(self.device)
        
        first_frame = video_tensor[0, 0]
        last_frame = video_tensor[0, -1]
        
        change_dir = os.path.join(out_dir, "change_detection")
        masks, masks_all = compute_change_mask(
            first_frame, last_frame, 
            out_dir=change_dir,
            threshold=self.change_threshold,
            use_dinov2=self.use_dinov2,
            dinov2_model=self.dinov2_model,
            dinov2_processor=self.dinov2_processor
        )
        
        # Combine all large masks into single segmentation
        if len(masks) > 0:
            areas = [mask.sum().item() for mask in masks]  # number of pixels in each mask
            best_idx = int(torch.tensor(areas).argmax())   # index of largest region
            segmentation = masks[best_idx].clone()         # (1, H, W)
        else:
            print("Warning: No change masks found, using full frame")
            H, W = video_tensor.shape[-2:]
            segmentation = torch.ones((1, H, W), device=self.device)

        segmentation = segmentation.unsqueeze(0)
        self.change_masks = masks
        return segmentation
    
    def process_rgbd_sequence(self, 
                            video_path: str,
                            depth_dir: str, 
                            out_dir: str,
                            camera_metadata_path: str) -> Tuple[List[Trajectory3D], CameraIntrinsics]:
        """Process RGB-D sequence with memory optimizations."""
        print("=== Starting RGB-D Processing (Memory Optimized) ===")
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"Initial GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        
        print("Step 1: Loading RGB video...")
        video_tensor, original_T, H, W = self._load_and_preprocess_video(video_path)
        self.video_h, self.video_w = H, W  # Store for depth matching
        
        # Clear cache after video load
        torch.cuda.empty_cache()
        
        print("Step 2: Loading depth sequence...")
        depth_sequence = self._load_depth_sequence(depth_dir, video_tensor.shape[1])
        
        print("Step 3: Loading camera parameters...")
        camera_intrinsics = self._load_camera_parameters(camera_metadata_path)
        
        # Scale camera intrinsics to match downsampled resolution
        scale_x = W / camera_intrinsics.w
        scale_y = H / camera_intrinsics.h
        camera_intrinsics.fx *= scale_x
        camera_intrinsics.fy *= scale_y
        camera_intrinsics.cx *= scale_x
        camera_intrinsics.cy *= scale_y
        camera_intrinsics.w = W
        camera_intrinsics.h = H
        print(f"Scaled camera intrinsics: fx={camera_intrinsics.fx:.1f}, fy={camera_intrinsics.fy:.1f}")
        
        segmentation_mask = None
        if self.use_change_mask:
            print("Step 3.5: Computing change-based segmentation...")
            torch.cuda.empty_cache() 
            segmentation_mask = self.compute_change_segmentation(video_tensor, out_dir)
            torch.cuda.empty_cache() 

        print("Step 4: Extracting 2D trajectories...")
        trajectories_2d = self._extract_2d_trajectories(
            video_tensor, 
            segmentation_mask=segmentation_mask
        )
        print(f"Extracted {len(trajectories_2d)} 2D trajectories")
        
        # Clear video from GPU after tracking
        del video_tensor
        if segmentation_mask is not None:
            del segmentation_mask
        torch.cuda.empty_cache()
        
        print("Step 5: Converting to 3D trajectories...")
        trajectories_3d = self._convert_to_3d_trajectories(
            trajectories_2d, depth_sequence, camera_intrinsics
        )
        print(f"Converted {len(trajectories_3d)} trajectories to 3D")
        
        print("Step 6: Filtering trajectories...")
        filtered_trajectories = self._filter_trajectories(trajectories_3d)
        print(f"After filtering: {len(filtered_trajectories)} trajectories")
        
        print("Step 7: Segmenting rigid parts...")
        segmented_trajectories = self._segment_rigid_parts(filtered_trajectories)
        
        moving_trajectories = [t for t in segmented_trajectories if t.rigid_part == 1]
        print(f"Moving trajectories: {len(moving_trajectories)}/{len(segmented_trajectories)}")
        
        self.visualize_result(out_dir=out_dir)
        plot_trajectories_3d(moving_trajectories, out_path=f"{out_dir}/trajectories_3d.png")
        
        print("=== RGB-D Processing Complete ===")
        return moving_trajectories, camera_intrinsics


    def _load_and_preprocess_video(self, frames_dir: str) -> Tuple[torch.Tensor, int, int, int]:
        """Load and preprocess RGB frames from directory with memory optimization."""
        import glob
        from PIL import Image
        
        frame_files = sorted(glob.glob(os.path.join(frames_dir, "frame_*.png")))
        
        if len(frame_files) == 0:
            raise ValueError(f"No frames found in {frames_dir}")
        
        print(f"Loading {len(frame_files)} frames from {frames_dir}")
        
        # Read first frame to get dimensions
        first_img = Image.open(frame_files[0])
        original_w, original_h = first_img.size
        
        # AGGRESSIVE DOWNSAMPLING: Target max dimension of 640px
        max_dim = 640
        if max(original_h, original_w) > max_dim:
            scale = max_dim / max(original_h, original_w)
            new_h = int(original_h * scale)
            new_w = int(original_w * scale)
            # Make divisible by 8 for neural networks
            new_h = (new_h // 8) * 8
            new_w = (new_w // 8) * 8
            print(f"Downsampling spatial resolution: {original_h}x{original_w} → {new_h}x{new_w}")
        else:
            new_h, new_w = original_h, original_w
        
        # Temporal downsampling: keep max 60 frames
        original_T = len(frame_files)
        if original_T > 60:
            stride = max(1, original_T // 60)
            frame_files = frame_files[::stride]
            print(f"Temporal downsampling: {original_T} → {len(frame_files)} frames (stride={stride})")
        
        # Load and resize frames
        frames = []
        for idx, frame_path in enumerate(frame_files):
            if idx % 10 == 0:
                print(f"Loading frame {idx}/{len(frame_files)}")
            
            img = Image.open(frame_path).convert('RGB')
            
            # Resize if needed
            if (img.size[1], img.size[0]) != (new_h, new_w):
                img = img.resize((new_w, new_h), Image.BILINEAR)
            
            img_array = np.array(img, dtype=np.float32) / 255.0  # Normalize to [0,1]
            frames.append(img_array)
        
        print(f"Loaded {len(frames)} frames at {new_h}x{new_w}")
        
        # Stack and convert to torch: (1, T, C, H, W)
        video_np = np.stack(frames, axis=0)
        video = torch.from_numpy(video_np).permute(0, 3, 1, 2)[None].float()
        
        return video.to(self.device), original_T, new_h, new_w
    
    
    def _load_depth_sequence(self, depth_dir: str, num_frames: int) -> torch.Tensor:
        """Load depth sequence with matching downsampling."""
        depth_dir = Path(depth_dir)
        depth_files = sorted(list(depth_dir.glob("*.npy")) + list(depth_dir.glob("*.png")))

        if len(depth_files) == 0:
            raise FileNotFoundError(f"No depth files found in {depth_dir}")

        # Match temporal downsampling from video
        original_count = len(depth_files)
        if original_count > num_frames:
            stride = max(1, original_count // num_frames)
            depth_files = depth_files[::stride][:num_frames]
            print(f"Depth temporal downsampling: {original_count} → {len(depth_files)} (stride={stride})")

        depths = []
        for depth_file in tqdm(depth_files[:num_frames], desc="Loading depth"):
            if depth_file.suffix == '.npy':
                depth = np.load(depth_file)
            else:
                depth = cv2.imread(str(depth_file), cv2.IMREAD_ANYDEPTH)
                if depth is None:
                    raise ValueError(f"Could not load: {depth_file}")
                depth = depth.astype(np.float32) / 1000.0
            
            depths.append(depth)

        depth_tensor = torch.from_numpy(np.array(depths))

        # Spatial downsampling to match video resolution
        if depth_tensor.shape[1:] != (self.video_h, self.video_w):
            print(f"Resizing depth: {depth_tensor.shape[1:]} → ({self.video_h}, {self.video_w})")
            depth_tensor = torch.nn.functional.interpolate(
                depth_tensor.unsqueeze(1),
                size=(self.video_h, self.video_w),
                mode='nearest'
            ).squeeze(1)

        return depth_tensor.to(self.device)
    
    def _load_camera_parameters(self, camera_metadata_path: str) -> CameraIntrinsics:
        """Load camera intrinsics from metadata file."""
        with open(camera_metadata_path, 'r') as f:
            metadata = json.load(f)
        
        fx = metadata["fl_x"]
        fy = metadata["fl_y"]
        cx = metadata["cx"] 
        cy = metadata["cy"]
        w = metadata["w"]
        h = metadata["h"]
        
        camera_intrinsics = CameraIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy, w=w, h=h)
        
        # Store extrinsics for potential future use
        if "frames" in metadata and len(metadata["frames"]) > 0:
            first_frame = metadata["frames"][0]
            self.camera_extrinsics = torch.tensor(
                first_frame["transform_matrix"], dtype=torch.float32
            ).to(self.device)
        
        print(f"Camera intrinsics: fx={fx}, fy={fy}, cx={cx}, cy={cy}")
        return camera_intrinsics
    
    def _extract_2d_trajectories(self, video_tensor: torch.Tensor,
                                 segmentation_mask: Optional[torch.Tensor] = None) -> List[Trajectory2D]:
        """Extract 2D trajectories using CoTracker."""
        if self.model is None:
            self.load_cotracker_model()
        
        print("Computing 2D trajectories...")
        with torch.no_grad():
            pred_tracks, pred_visibility, _ = self.model(
                video_tensor,
                grid_size=self.grid_size,
                grid_query_frame=self.grid_query_frame,
                backward_tracking=self.backward_tracking,
                segm_mask=segmentation_mask
            )

        self._last_pred_tracks = pred_tracks
        self._last_pred_visibility = pred_visibility
        self._last_video_tensor = video_tensor
        
        print(f"2D tracks shape: {pred_tracks.shape}")
        print(f"Visibility shape: {pred_visibility.shape}")
        
        # Convert CoTracker output to our data structures
        return self._cotracker_to_trajectories_2d(pred_tracks, pred_visibility)
    
    def _cotracker_to_trajectories_2d(self, 
                                    pred_tracks: torch.Tensor, 
                                    pred_visibility: torch.Tensor) -> List[Trajectory2D]:
        """
        Convert CoTracker output to our Trajectory2D data structures.
        
        Args:
            pred_tracks: Shape (B, T, N, 2) - batch, time, num_points, coordinates
            pred_visibility: Shape (B, T, N) - visibility mask
            
        Returns:
            List of Trajectory2D objects
        """
        # Move to CPU for processing
        tracks = pred_tracks[0].cpu().numpy()  # Remove batch dimension: (T, N, 2)
        visibility = pred_visibility[0].cpu().numpy()  # (T, N)
        
        T, N, _ = tracks.shape
        trajectories_2d = []
        
        for track_id in range(N):
            points = []
            for frame in range(T):
                if visibility[frame, track_id]:  #  include visible points
                    u, v = tracks[frame, track_id]
                    confidence = float(visibility[frame, track_id])
                    
                    point = TrackPoint2D(
                        frame=frame,
                        u=float(u),
                        v=float(v),
                        confidence=confidence
                    )
                    points.append(point)
            
            # Only keep trajectories with sufficient points
            if len(points) >= self.trajectory_filter_config.min_length:
                trajectory = Trajectory2D(track_id=track_id, points=points)
                trajectories_2d.append(trajectory)
        
        return trajectories_2d
    
    def _convert_to_3d_trajectories(self, 
                                  trajectories_2d: List[Trajectory2D],
                                  depth_sequence: torch.Tensor,
                                  camera_intrinsics: CameraIntrinsics) -> List[Trajectory3D]:
        """Convert 2D trajectories to 3D using depth data."""
        depth_np = depth_sequence.cpu().numpy()
        trajectories_3d = []
        
        for traj_2d in tqdm(trajectories_2d, desc="Converting to 3D"):
            traj_3d = trajectory_2d_to_3d(traj_2d, depth_np, camera_intrinsics)
            
            # Only keep trajectories with sufficient 3D points
            if len(traj_3d) >= self.trajectory_filter_config.min_length:
                trajectories_3d.append(traj_3d)
        
        return trajectories_3d
    
    def _filter_trajectories(self, trajectories_3d: List[Trajectory3D]) -> List[Trajectory3D]:
        """Filter trajectories based on quality metrics."""
        filtered_trajectories = []
        
        for traj in trajectories_3d:
            if self._is_trajectory_valid(traj):
                # Apply smoothing if configured
                if self.trajectory_filter_config.smoothing_window > 1:
                    traj = self._smooth_trajectory(traj)
                filtered_trajectories.append(traj)
        
        return filtered_trajectories
    
    def _is_trajectory_valid(self, trajectory: Trajectory3D) -> bool:
        """Check if a trajectory meets quality criteria."""
        # Length check
        if len(trajectory) < self.trajectory_filter_config.min_length:
            return False
        
        # Velocity jump check
        positions = trajectory.get_all_positions()
        if len(positions) < 2:
            return False
        
        velocities = np.diff(positions, axis=0)
        velocity_magnitudes = np.linalg.norm(velocities, axis=1)
        
        # Check for unrealistic velocity jumps
        if len(velocity_magnitudes) > 1:
            velocity_changes = np.diff(velocity_magnitudes)
            max_velocity_change = np.max(np.abs(velocity_changes))
            
            if max_velocity_change > self.trajectory_filter_config.max_velocity_jump:
                return False
        
        # Check for NaN or infinite values
        if np.any(~np.isfinite(positions)):
            return False
        
        return True
    
    def _smooth_trajectory(self, trajectory: Trajectory3D) -> Trajectory3D:
        """Apply smoothing to a 3D trajectory."""
        if len(trajectory.points) < 3:
            return trajectory
        
        positions = trajectory.get_all_positions()
        window = self.trajectory_filter_config.smoothing_window
        
        # Simple moving average smoothing
        smoothed_positions = []
        for i in range(len(positions)):
            start_idx = max(0, i - window // 2)
            end_idx = min(len(positions), i + window // 2 + 1)
            smoothed_pos = np.mean(positions[start_idx:end_idx], axis=0)
            smoothed_positions.append(smoothed_pos)
        
        smoothed_positions = np.array(smoothed_positions)
        
        # Create new trajectory with smoothed positions
        smoothed_points = []
        for i, point in enumerate(trajectory.points):
            smoothed_point = Point3D(
                frame=point.frame,
                x=smoothed_positions[i, 0],
                y=smoothed_positions[i, 1],
                z=smoothed_positions[i, 2],
                confidence=point.confidence
            )
            smoothed_points.append(smoothed_point)
        
        return Trajectory3D(
            track_id=trajectory.track_id,
            points=smoothed_points,
            rigid_part=trajectory.rigid_part
        )

    def _segment_rigid_parts(self, trajectories_3d: List[Trajectory3D]) -> List[Trajectory3D]:
        """
        Segment trajectories into rigid parts (moving vs static).
        More robust version using displacement, velocity variance, and clustering.
        """
        from sklearn.cluster import KMeans
        
        if not trajectories_3d:
            return trajectories_3d

        features = []
        for traj in trajectories_3d:
            positions = traj.get_all_positions()
            if len(positions) < 2:
                features.append([0, 0, 0])
                continue

            # Total displacement normalized by length
            displacement = np.linalg.norm(positions[-1] - positions[0]) / len(positions)

            # Frame-to-frame velocities
            velocities = np.diff(positions, axis=0)
            vel_mags = np.linalg.norm(velocities, axis=1)

            avg_velocity = np.mean(vel_mags)
            var_velocity = np.var(vel_mags)

            # Feature vector: [displacement, avg velocity, variance]
            features.append([displacement, avg_velocity, var_velocity])

        features = np.array(features)

        # Cluster into 2 groups (moving vs static)
        kmeans = KMeans(n_clusters=2, random_state=42, n_init="auto")
        labels = kmeans.fit_predict(features)

        # Decide which cluster is "moving" (the one with higher avg displacement)
        cluster_motion = [features[labels == k, 0].mean() for k in range(2)]
        moving_cluster = int(np.argmax(cluster_motion))

        segmented_trajectories = []
        for i, traj in enumerate(trajectories_3d):
            segmented_traj = Trajectory3D(
                track_id=traj.track_id,
                points=traj.points,
                rigid_part=1 if labels[i] == moving_cluster else 0
            )
            segmented_trajectories.append(segmented_traj)

        moving_count = sum(1 for traj in segmented_trajectories if traj.rigid_part == 1)
        static_count = len(segmented_trajectories) - moving_count
        print(f"Rigid part segmentation: {moving_count} moving, {static_count} static trajectories")

        return segmented_trajectories


    def _cluster_trajectories(
        self,
        trajectories: List[Trajectory3D],
        eps: float = 0.05,
        min_samples: int = 5
    ) -> List[Trajectory3D]:
        """
        Cluster 3D trajectories to reduce redundancy/noise (simplified version).
        
        Args:
            trajectories: List of Trajectory3D objects
            eps: DBSCAN neighborhood size
            min_samples: Minimum samples for a cluster
        
        Returns:
            Representative trajectories (cluster medoids)
        """

        if not trajectories:
            return trajectories

        # Represent each trajectory by mean displacement vector
        features = []
        for traj in trajectories:
            pts = traj.get_all_positions()
            if len(pts) < 2:
                features.append([0, 0, 0])
            else:
                disp = pts[-1] - pts[0]
                features.append(disp / (np.linalg.norm(disp) + 1e-8))
        features = np.array(features)

        # Cluster with DBSCAN
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(features)
        labels = clustering.labels_

        clustered = []
        for lbl in set(labels):
            if lbl == -1:
                # noise points
                continue
            cluster_trajs = [t for t, l in zip(trajectories, labels) if l == lbl]
            if not cluster_trajs:
                continue

            cluster_features = [f for f, l in zip(features, labels) if l == lbl]
            cluster_mean = np.mean(cluster_features, axis=0)

            distances = [np.linalg.norm(f - cluster_mean) for f in cluster_features]
            medoid_idx = np.argmin(distances)

            clustered.append(cluster_trajs[medoid_idx])

        print(f"[Clustering] Reduced {len(trajectories)} → {len(clustered)} trajectories")
        return clustered
  

    def visualize_result(self, out_dir="./saved_videos", pad_value=50, linewidth=3):
        if not hasattr(self, "_last_pred_tracks"):
            print("No prediction available to visualize.")
            return

        video_for_vis = (self._last_video_tensor * 255.0).byte()
        
        vis = Visualizer(save_dir=out_dir, pad_value=pad_value, linewidth=linewidth)
        vis.visualize(
            video_for_vis,  
            self._last_pred_tracks,
            self._last_pred_visibility,
            query_frame=0 if self.backward_tracking else self.grid_query_frame,
        )
    

# Utility functions for external use
def process_rgbd_video(video_path: str,
                      depth_dir: str, 
                      out_dir: str,
                      camera_metadata_path: str,
                      trajectory_filter_config: TrajectoryFilterConfig,
                      **kwargs) -> Tuple[List[Trajectory3D], CameraIntrinsics]:
    """
    Convenience function to process an RGB-D video sequence.
    
    Args:
        video_path: Path to RGB video
        depth_dir: Directory with depth images
        camera_metadata_path: Path to camera metadata JSON
        trajectory_filter_config: Configuration for trajectory filtering
        **kwargs: Additional parameters for CoTrackerRGBD
        
    Returns:
        Tuple of (3D trajectories, camera intrinsics)
    """
    processor = CoTrackerRGBD(trajectory_filter_config, **kwargs)
    return processor.process_rgbd_sequence(video_path, depth_dir, out_dir, camera_metadata_path)


def plot_trajectories_3d(trajectories_3d, out_path="trajectories_3d.png"):
    """Visualize 3D trajectories with equal axis scaling (red = moving, blue = static)."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    all_points = []

    for traj in trajectories_3d:
        pts = traj.get_all_positions()  # Nx3 numpy array
        if len(pts) < 2:
            continue
        all_points.append(pts)
        if traj.rigid_part == 1:  # moving
            ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], c="red", alpha=0.7)
        else:  # static
            ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], c="blue", alpha=0.3)

    # --- Equal aspect ratio ---
    if all_points:
        all_points = np.vstack(all_points)
        x_limits = [all_points[:, 0].min(), all_points[:, 0].max()]
        y_limits = [all_points[:, 1].min(), all_points[:, 1].max()]
        z_limits = [all_points[:, 2].min(), all_points[:, 2].max()]

        max_range = max(
            x_limits[1] - x_limits[0],
            y_limits[1] - y_limits[0],
            z_limits[1] - z_limits[0],
        ) / 2.0

        mid_x = np.mean(x_limits)
        mid_y = np.mean(y_limits)
        mid_z = np.mean(z_limits)

        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.title("3D Trajectories (Red = moving, Blue = static)")
    # plt.savefig(out_path, dpi=300)
    # print(f"3D trajectories plot saved to {out_path}")
    # plt.show()



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test CoTracker RGB-D Processing")
    parser.add_argument("--video", type=str, required=True, help="Path to RGB video")
    parser.add_argument("--depth", type=str, required=True, help="Path to depth images folder")
    parser.add_argument("--camera", type=str, required=True, help="Path to camera metadata JSON")
    parser.add_argument("--min_length", type=int, default=5, help="Minimum trajectory length")
    parser.add_argument("--max_velocity_jump", type=float, default=0.5, help="Max frame-to-frame velocity change (m/s)")
    parser.add_argument("--smoothing", type=int, default=3, help="Smoothing window size")
    args = parser.parse_args()

    # Build trajectory filter config
    filter_config = TrajectoryFilterConfig(
        min_length=args.min_length,
        max_velocity_jump=args.max_velocity_jump,
        smoothing_window=args.smoothing,
    )

    # Run pipeline
    trajectories_3d, camera_intrinsics = process_rgbd_video(
        video_path=args.video,
        depth_dir=args.depth,
        out_dir=args.out_dir if hasattr(args, 'out_dir') else "./saved_videos",
        camera_metadata_path=args.camera,
        trajectory_filter_config=filter_config,
    )

    # Print summary
    print(f"Final: {len(trajectories_3d)} 3D trajectories extracted")
    print(f"Camera intrinsics: {camera_intrinsics.to_matrix()}")
