"""
Trajectory preprocessing utilities for robust joint estimation.
Handles smoothing, filtering, and quality checks for noisy 3D trajectories.
"""

import numpy as np
from typing import List
from scipy.signal import savgol_filter
from scipy.ndimage import uniform_filter1d

from data_structures import Trajectory3D, Point3D



def preprocess_trajectories(
    trajectories: List[Trajectory3D],
    smooth_window: int = 5,
    min_length: int = 5,
    min_displacement: float = 0.01,
    max_acceleration_percentile: float = 95,
    accel_threshold: float = 0.5,
    use_savgol: bool = True,
    sample_stride: int = 5,
) -> List[Trajectory3D]:
    """
    Joint-agnostic trajectory preprocessing.

    Simplified temporal sub-sampling and smoothing pipeline
    for stabilizing noisy RGB-D or depth-tracked trajectories.
    No assumptions about motion type (hinge, slider, etc.).

    Args:
        trajectories: List of input 3D trajectories
        smooth_window: Window size for smoothing
        min_length: Minimum number of points to keep trajectory
        min_displacement: Minimum total motion magnitude
        max_acceleration_percentile: Percentile for jerk filtering
        accel_threshold: Max acceleration magnitude at that percentile
        use_savgol: Whether to apply Savitzky–Golay smoothing
        sample_stride: Frame stride for uniform sub-sampling
    """

    processed = []
    print(f"Preprocessing {len(trajectories)} trajectories...")

    for traj in trajectories:
        pts = traj.get_all_positions()
        if len(pts) < min_length:
            continue

        # --- Simple uniform temporal subsampling ---

        pts = pts[::sample_stride]
        if len(pts) < min_length // 2:
            continue


        # --- Minimum displacement check ---
        total_displacement = np.linalg.norm(pts[-1] - pts[0])
        if total_displacement < min_displacement:
            continue

        # --- Smoothness check before smoothing ---
        if len(pts) >= 3:
            vel = np.diff(pts, axis=0)
            acc = np.diff(vel, axis=0)
            accel_mag = np.linalg.norm(acc, axis=1)
            accel_at_percentile = np.percentile(accel_mag, max_acceleration_percentile)
            if accel_at_percentile > accel_threshold:
                continue

        # --- Apply smoothing (Savitzky–Golay or moving average) ---
        try:
            if use_savgol and len(pts) >= smooth_window:
                # Ensure odd window length for Savitzky–Golay
                if smooth_window % 2 == 0:
                    smooth_window += 1
                smoothed = savgol_filter(
                    pts,
                    window_length=min(smooth_window, len(pts)//2*2+1),
                    polyorder=2,
                    axis=0,
                )
            else:
                smoothed = uniform_filter1d(pts, size=smooth_window, axis=0, mode="nearest")
        except Exception:
            smoothed = pts

        # --- Post-smoothing displacement sanity check ---
        smoothed_disp = np.linalg.norm(smoothed[-1] - smoothed[0])
        if smoothed_disp < 0.5 * min_displacement:
            continue

        # --- Velocity consistency check ---
        vel = np.diff(smoothed, axis=0)
        vel_norm = np.linalg.norm(vel, axis=1)
        if np.std(vel_norm) > 3 * np.mean(vel_norm + 1e-6):
            continue

        # --- Rebuild smoothed trajectory ---
        new_points = [
            Point3D(frame=i, x=smoothed[i, 0], y=smoothed[i, 1], z=smoothed[i, 2])
            for i in range(len(smoothed))
        ]
        processed.append(
            Trajectory3D(track_id=traj.track_id, points=new_points, rigid_part=traj.rigid_part)
        )

    print(f"✓ Preprocessing: {len(trajectories)} → {len(processed)} trajectories")
    return processed
