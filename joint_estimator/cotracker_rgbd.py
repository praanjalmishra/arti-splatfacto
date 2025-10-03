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
from typing import List, Tuple
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
                 backward_tracking: bool = True):
        """
        Initialize the CoTracker RGB-D processor.
        
        Args:
            trajectory_filter_config: Configuration for trajectory filtering
            device: Device to run computations on
            grid_size: Grid size for CoTracker point sampling
            grid_query_frame: Frame to use for grid initialization
            backward_tracking: Whether to track backwards in time
        """
        self.trajectory_filter_config = trajectory_filter_config
        self.device = device
        self.grid_size = grid_size
        self.grid_query_frame = grid_query_frame
        self.backward_tracking = backward_tracking
        
        self.model = None
        self.camera_intrinsics = None
        self.camera_extrinsics = None
        
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
    
    def process_rgbd_sequence(self, 
                            video_path: str,
                            depth_dir: str, 
                            out_dir: str,
                            camera_metadata_path: str) -> Tuple[List[Trajectory3D], CameraIntrinsics]:
        """
        Process a complete RGB-D sequence to extract 3D trajectories.
        
        Args:
            video_path: Path to RGB video file
            depth_dir: Directory containing depth images
            camera_metadata_path: Path to camera metadata JSON
            
        Returns:
            Tuple of (list of 3D trajectories, camera intrinsics)
        """
        print("=== Starting RGB-D Processing ===")
        
        print("Step 1: Loading RGB video...")
        video_tensor, original_T, H, W = self._load_and_preprocess_video(video_path)
        
        print("Step 2: Loading depth sequence...")
        depth_sequence = self._load_depth_sequence(depth_dir, video_tensor.shape[1])
        
        print("Step 3: Loading camera parameters...")
        camera_intrinsics = self._load_camera_parameters(camera_metadata_path)
        
        print("Step 4: Extracting 2D trajectories...")
        trajectories_2d = self._extract_2d_trajectories(video_tensor)
        print(f"Extracted {len(trajectories_2d)} 2D trajectories")
        
        print("Step 5: Converting to 3D trajectories...")
        trajectories_3d = self._convert_to_3d_trajectories(
            trajectories_2d, depth_sequence, camera_intrinsics
        )
        print(f"Successfully converted {len(trajectories_3d)} trajectories to 3D")
        
        print("Step 6: Filtering trajectories...")
        filtered_trajectories = self._filter_trajectories(trajectories_3d)
        print(f"After filtering: {len(filtered_trajectories)} trajectories")
        
        print("Step 7: Segmenting rigid parts...")
        segmented_trajectories = self._segment_rigid_parts(filtered_trajectories)

        moving_trajectories = [traj for traj in segmented_trajectories if traj.rigid_part == 1]
        print(f"Using only moving trajectories: {len(moving_trajectories)} out of {len(segmented_trajectories)}")

        # # Step 8: Cluster trajectories (only moving ones)
        # print("Step 8: Clustering trajectories...")
        # clustered_trajectories = self._cluster_trajectories(moving_trajectories)

        # visualize the trajectories

        self.visualize_result(out_dir=out_dir)

        plot_trajectories_3d(moving_trajectories, out_path="./saved_videos/trajectories_3d.png")

        print("=== RGB-D Processing Complete ===")
        return moving_trajectories, camera_intrinsics


    def _load_and_preprocess_video(self, frames_dir: str) -> Tuple[torch.Tensor, int, int, int]:
        """Load and preprocess RGB frames from directory with memory optimization."""
        import glob
        from PIL import Image
        
        # Get all frame files sorted by name
        frame_files = sorted(glob.glob(os.path.join(frames_dir, "frame_*.png")))
        
        if len(frame_files) == 0:
            raise ValueError(f"No frames found in {frames_dir}")
        
        print(f"Loading {len(frame_files)} frames from {frames_dir}")
        
        frames = []
        for idx, frame_path in enumerate(frame_files):
            if idx % 10 == 0:  # Progress update every 10 frames
                print(f"Loading frame {idx}/{len(frame_files)}")
            
            img = Image.open(frame_path).convert('RGB')
            img_array = np.array(img)
            frames.append(img_array)
        
        print(f"Loaded all {len(frames)} frames")
        
        # Stack into numpy array: (T, H, W, C)
        video_np = np.stack(frames, axis=0)
        
        # Convert to torch and rearrange to (1, T, C, H, W)
        video = torch.from_numpy(video_np).permute(0, 3, 1, 2)[None].float()
        
        original_T, H, W = video.shape[1], video.shape[3], video.shape[4]
        
        # Memory optimization: downsample if needed
        if original_T > 60:
            print(f"Downsampling video from {original_T} frames...")
            stride = original_T // 60  # Keep ~60 frames max
            video = video[:, ::stride]
            print(f"New video shape: {video.shape}")
        elif H > 480:
            print(f"Downsampling spatial resolution from {H}x{W}...")
            video = torch.nn.functional.interpolate(
                video.squeeze(0), 
                size=(480, 640), 
                mode='bilinear', 
                align_corners=False
            ).unsqueeze(0)
            print(f"New video shape: {video.shape}")
        
        return video.to(self.device), original_T, H, W
    
    
    def _load_depth_sequence(self, depth_dir: str, num_frames: int) -> torch.Tensor:
        """Load sequence of depth images."""
        depth_dir = Path(depth_dir)
        depth_files = sorted(list(depth_dir.glob("*.png")) + list(depth_dir.glob("*.npy")))
        
        if len(depth_files) == 0:
            raise FileNotFoundError(f"No depth files found in {depth_dir}")
        
        # Take every other frame if video was downsampled
        if len(depth_files) > num_frames:
            depth_files = depth_files[::2][:num_frames]
        
        depths = []
        for depth_file in tqdm(depth_files[:num_frames], desc="Loading depth frames"):
            if depth_file.suffix == '.npy':
                depth = np.load(depth_file)
            else:
                depth = cv2.imread(str(depth_file), cv2.IMREAD_ANYDEPTH)
                if depth is None:
                    raise ValueError(f"Could not load depth image: {depth_file}")

                depth = depth.astype(np.float32)/1000.0
            depths.append(depth)
        
        return torch.from_numpy(np.array(depths)).to(self.device)
    
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
    
    def _extract_2d_trajectories(self, video_tensor: torch.Tensor) -> List[Trajectory2D]:
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

            # Pick medoid trajectory (closest to cluster mean)
            cluster_features = [f for f, l in zip(features, labels) if l == lbl]
            cluster_mean = np.mean(cluster_features, axis=0)

            distances = [np.linalg.norm(f - cluster_mean) for f in cluster_features]
            medoid_idx = np.argmin(distances)

            clustered.append(cluster_trajs[medoid_idx])

        print(f"[Clustering] Reduced {len(trajectories)} → {len(clustered)} trajectories")
        return clustered
  

    def visualize_result(self, out_dir="./saved_videos", pad_value=120, linewidth=3):
        """Visualize last predicted tracks on the original video."""
        if not hasattr(self, "_last_pred_tracks"):
            print("No prediction available to visualize.")
            return

        vis = Visualizer(save_dir=out_dir, pad_value=pad_value, linewidth=linewidth)
        vis.visualize(
            self._last_video_tensor,
            self._last_pred_tracks,
            self._last_pred_visibility,
            query_frame=0 if self.backward_tracking else self.grid_query_frame,
        )
        print(f"Visualization saved to {out_dir}")

    

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
