"""
TAPIP3D Trajectory Loader for 4D RANSAC Pipeline

Converts TAPIP3D .result.npz files into Trajectory3D objects compatible
with the existing RANSAC pipeline.
"""

import numpy as np
from typing import List, Tuple
from pathlib import Path
import os

from joint_estimator.data_structures import (
    Trajectory3D, Point3D, TrajectoryFilterConfig
)


def load_tapip3d_trajectories(
    npz_path: str,
    filter_config: TrajectoryFilterConfig,
    visibility_threshold: float = 0.5,
    subsample_frames: int = 1
) -> Tuple[List[Trajectory3D], dict]:
    """
    Load and convert TAPIP3D results to Trajectory3D format.
    
    Args:
        npz_path: Path to .result.npz file
        filter_config: Trajectory filtering configuration
        visibility_threshold: Minimum visibility score (0-1) to include point
        subsample_frames: Use every Nth frame (1 = all frames)
    
    Returns:
        (trajectories_3d, metadata)
        - trajectories_3d: List of filtered Trajectory3D objects
        - metadata: Dict with intrinsics, extrinsics, query_points
    """
    
    if os.path.isdir(npz_path):
        npz_files = sorted(
            [f for f in os.listdir(npz_path) if f.endswith(".npz")],
            reverse=True
        )
        if not npz_files:
            raise FileNotFoundError(f"No .result.npz file found in directory: {npz_path}")
        npz_path = os.path.join(npz_path, npz_files[0])

    # Load TAPIP3D output
    data = np.load(npz_path, allow_pickle=True)
    
    coords = data['coords']      # (T, N, 3)
    visibs = data['visibs']      # (T, N)
    intrinsics = data['intrinsics']  # (T, 3, 3)
    extrinsics = data['extrinsics']  # (T, 4, 4)
    query_points = data['query_points']  # (N, 4)
    
    T, N, _ = coords.shape
    
    print(f"[TAPIP3D Loader] Processing {N} tracks across {T} frames")
    print(f"Visibility threshold: {visibility_threshold}")
    print(f"Frame subsampling: 1/{subsample_frames}")
    
    trajectories_3d = []
    
    for track_idx in range(N):
        points_3d = []
        
        for frame_idx in range(0, T, subsample_frames):
            vis_score = visibs[frame_idx, track_idx]
            
            # Skip low-visibility points
            if vis_score < visibility_threshold:
                continue
            
            x, y, z = coords[frame_idx, track_idx]
            
            # Skip invalid coordinates (NaN, inf)
            if not np.all(np.isfinite([x, y, z])):
                continue
            
            points_3d.append(Point3D(
                frame=frame_idx,
                x=float(x),
                y=float(y),
                z=float(z),
                confidence=float(vis_score)
            ))
        
        # Apply trajectory length filter
        if len(points_3d) >= filter_config.min_length:
            traj = Trajectory3D(
                track_id=track_idx,
                points=points_3d
            )
            
            # Apply velocity filtering
            if _check_velocity_valid(traj, filter_config.max_velocity_jump):
                trajectories_3d.append(traj)
    
    print(f"[TAPIP3D Loader] Loaded {len(trajectories_3d)}/{N} valid trajectories")
    
    metadata = {
        'intrinsics': intrinsics,
        'extrinsics': extrinsics,
        'query_points': query_points,
        'total_frames': T,
        'total_tracks': N
    }
    
    return trajectories_3d, metadata


def _check_velocity_valid(traj: Trajectory3D, max_velocity_jump: float) -> bool:
    """Check if trajectory has reasonable frame-to-frame velocities."""
    if len(traj.points) < 2:
        return True
    
    positions = traj.get_all_positions()
    
    for i in range(1, len(positions)):
        frame_diff = traj.points[i].frame - traj.points[i-1].frame
        if frame_diff == 0:
            continue
        
        displacement = np.linalg.norm(positions[i] - positions[i-1])
        velocity = displacement / frame_diff
        
        # Filter out unrealistic jumps
        if velocity > max_velocity_jump:
            return False
    
    return True