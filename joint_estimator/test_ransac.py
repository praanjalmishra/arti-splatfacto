"""
Improved test script for RANSAC core with better debugging and more robust test cases.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from data_structures import Point3D, Trajectory3D, RANSACConfig, JointType
from ransac_core import estimate_joint_from_trajectories


def make_hinge_trajectories(axis, pivot, angles, n_points=30, noise=0.01):
    """Generate synthetic hinge trajectories with better parameters."""
    trajectories = []
    axis = axis / np.linalg.norm(axis)
    
    print(f"Generating hinge data:")
    print(f"  True axis: {axis}")
    print(f"  True pivot: {pivot}")
    print(f"  Angles: {angles[0]:.3f} to {angles[-1]:.3f} rad")
    print(f"  Noise level: {noise}")
    
    for i in range(n_points):
        # Create points at different distances from axis
        distance_from_axis = 1.5 + 0.5 * np.random.rand()  # 0.5 to 1.0 meters
        
        # Create a random point perpendicular to the axis
        random_perp = np.random.randn(3)
        random_perp = random_perp - np.dot(random_perp, axis) * axis  # Remove component along axis
        if np.linalg.norm(random_perp) > 0:
            random_perp = random_perp / np.linalg.norm(random_perp)
        else:
            random_perp = np.array([1, 0, 0]) if abs(axis[0]) < 0.9 else np.array([0, 1, 0])
        
        base_point = pivot + distance_from_axis * random_perp
        
        points = []
        for frame_idx, angle in enumerate(angles):
            # Rodrigues rotation formula
            K = np.array([
                [0, -axis[2], axis[1]],
                [axis[2], 0, -axis[0]],
                [-axis[1], axis[0], 0]
            ])
            R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
            
            # Rotate around pivot
            rotated_point = R @ (base_point - pivot) + pivot
            
            # Add noise
            noisy_point = rotated_point + noise * np.random.randn(3)
            
            points.append(Point3D(
                frame=frame_idx, 
                x=noisy_point[0], 
                y=noisy_point[1], 
                z=noisy_point[2]
            ))
            
        trajectory = Trajectory3D(track_id=i, points=points, rigid_part=1)  # Mark as moving
        trajectories.append(trajectory)
    
    # Add some static trajectories (not moving)
    for i in range(5):
        static_point = pivot + 2 * np.random.randn(3)  # Random static point
        points = []
        for frame_idx in range(len(angles)):
            # Static point with just noise
            noisy_point = static_point + noise * np.random.randn(3)
            points.append(Point3D(
                frame=frame_idx,
                x=noisy_point[0],
                y=noisy_point[1], 
                z=noisy_point[2]
            ))
        trajectory = Trajectory3D(track_id=n_points + i, points=points, rigid_part=0)  # Mark as static
        trajectories.append(trajectory)
    
    return trajectories


def make_slider_trajectories(direction, start_point, steps, n_points=10, noise=0.01):
    """Generate synthetic slider trajectories."""
    trajectories = []
    direction = direction / np.linalg.norm(direction)
    
    print(f"Generating slider data:")
    print(f"  True direction: {direction}")
    print(f"  Start point: {start_point}")
    print(f"  Steps: {steps[0]:.3f} to {steps[-1]:.3f}")
    print(f"  Noise level: {noise}")
    
    for i in range(n_points):
        # Random offset perpendicular to slide direction
        offset = np.random.randn(3) * 0.2
        offset = offset - np.dot(offset, direction) * direction  # Remove component along slide direction
        base_point = start_point + offset
        
        points = []
        for frame_idx, step in enumerate(steps):
            translated_point = base_point + step * direction
            noisy_point = translated_point + noise * np.random.randn(3)
            
            points.append(Point3D(
                frame=frame_idx,
                x=noisy_point[0],
                y=noisy_point[1],
                z=noisy_point[2]
            ))
            
        trajectory = Trajectory3D(track_id=i, points=points, rigid_part=1)  # Mark as moving
        trajectories.append(trajectory)
    
    return trajectories


def test_joint_estimation():
    """Test joint estimation with improved parameters."""
    
    # More permissive RANSAC config for testing
    config = RANSACConfig(
        max_iterations=2000,
        error_threshold=0.1,  # 10cm tolerance (more permissive)
        min_inliers=5,        # Lower requirement
        min_trajectory_length=3,  # Shorter minimum length
        early_termination_threshold=0.7  # 70% consensus
    )
    
    # Choose test case
    test_case = "hinge"  # Change to "slider" to test slider
    
    if test_case == "hinge":
        print("\n=== Testing Hinge Joint with RANSAC ===")
        
        # Create a clearer test case
        true_axis = np.array([0, 0, 1])  # Z-axis rotation
        true_pivot = np.array([0, 0, 0])  # Origin
        angles = np.linspace(0, np.pi/3, 15)  # 0 to 60 degrees, more frames
        
        trajectories = make_hinge_trajectories(
            true_axis, true_pivot, angles, 
            n_points=20, noise=0.02  # Moderate noise
        )
        
    else:
        print("\n=== Testing Slider Joint with RANSAC ===")
        
        true_dir = np.array([1, 0, 0])  # X-direction
        start = np.array([0, 0, 0])
        steps = np.linspace(0, 1.5, 15)  # 1.5 meter translation
        
        trajectories = make_slider_trajectories(
            true_dir, start, steps,
            n_points=15, noise=0.02
        )
    
    print(f"\nGenerated {len(trajectories)} trajectories")
    
    # Analyze trajectory properties
    moving_trajs = [t for t in trajectories if t.rigid_part == 1]
    static_trajs = [t for t in trajectories if t.rigid_part == 0]
    print(f"Moving trajectories: {len(moving_trajs)}")
    print(f"Static trajectories: {len(static_trajs)}")
    
    # Check trajectory lengths
    traj_lengths = [len(t.points) for t in trajectories]
    print(f"Trajectory lengths: min={min(traj_lengths)}, max={max(traj_lengths)}, avg={np.mean(traj_lengths):.1f}")
    
    # Check motion magnitudes
    motion_mags = []
    for traj in trajectories:
        if len(traj.points) >= 2:
            start_pos = np.array([traj.points[0].x, traj.points[0].y, traj.points[0].z])
            end_pos = np.array([traj.points[-1].x, traj.points[-1].y, traj.points[-1].z])
            motion_mag = np.linalg.norm(end_pos - start_pos)
            motion_mags.append(motion_mag)
    
    print(f"Motion magnitudes: min={min(motion_mags):.3f}, max={max(motion_mags):.3f}, avg={np.mean(motion_mags):.3f}")
    
    # Run RANSAC
    print("\n" + "="*50)
    result = estimate_joint_from_trajectories(trajectories, config)
    print("="*50)
    
    if result.success:
        print(f"\n✅ RANSAC Success: {result.joint_type.value}")
        print(f"Confidence: {result.confidence:.3f}")
        print(f"Processing time: {result.processing_time:.2f}s")
        print(f"Inliers: {len(result.inlier_trajectories)} / {len(trajectories)}")
        
        if result.joint_type == JointType.HINGE:
            hinge_params = result.get_hinge_params()
            print(f"Estimated axis: {hinge_params.axis}")
            print(f"Estimated pivot: {hinge_params.pivot}")
            if test_case == "hinge":
                print(f"True axis: {true_axis}")
                print(f"True pivot: {true_pivot}")
                axis_error = np.linalg.norm(hinge_params.axis - true_axis)
                pivot_error = np.linalg.norm(hinge_params.pivot - true_pivot)
                print(f"Axis error: {axis_error:.4f}")
                print(f"Pivot error: {pivot_error:.4f}")
                
        elif result.joint_type == JointType.SLIDER:
            slider_params = result.get_slider_params()
            print(f"Estimated direction: {slider_params.direction}")
            if test_case == "slider":
                print(f"True direction: {true_dir}")
                dir_error = np.linalg.norm(slider_params.direction - true_dir)
                print(f"Direction error: {dir_error:.4f}")
        
    else:
        print(f"\n❌ RANSAC Failed: {result.error_message}")
        return
    
    # Visualization
    print("\nGenerating visualization...")
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot trajectories
    for traj in trajectories:
        pts = traj.get_all_positions()
        if traj in result.inlier_trajectories:
            color = "green"
            alpha = 0.8
            linewidth = 2
        else:
            color = "red" 
            alpha = 0.4
            linewidth = 1
            
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], 
               color=color, alpha=alpha, linewidth=linewidth)
        
        # Mark start and end points
        ax.scatter(pts[0, 0], pts[0, 1], pts[0, 2], 
                  color=color, s=30, alpha=0.8, marker='o')
        ax.scatter(pts[-1, 0], pts[-1, 1], pts[-1, 2], 
                  color=color, s=30, alpha=0.8, marker='s')
    
    # Plot estimated joint
    if result.success:
        if result.joint_type == JointType.HINGE:
            pivot = result.parameters.pivot
            axis = result.parameters.axis
            line_length = 2.0
            line_pts = np.array([pivot - line_length * axis, pivot + line_length * axis])
            ax.plot(line_pts[:, 0], line_pts[:, 1], line_pts[:, 2], 
                   "blue", linewidth=3, label="Estimated axis")
            ax.scatter(pivot[0], pivot[1], pivot[2], 
                      color="blue", s=100, marker='*', label="Estimated pivot")
            
        elif result.joint_type == JointType.SLIDER:
            ref_pt = result.parameters.reference_point
            dir_vec = result.parameters.direction
            line_length = 2.0
            line_pts = np.array([ref_pt - line_length * dir_vec, ref_pt + line_length * dir_vec])
            ax.plot(line_pts[:, 0], line_pts[:, 1], line_pts[:, 2], 
                   "blue", linewidth=3, label="Estimated direction")
    
    # Formatting
    ax.legend()
    ax.set_xlabel('X')
    ax.set_ylabel('Y') 
    ax.set_zlabel('Z')
    ax.set_title(f"RANSAC Result: {result.joint_type.value if result.success else 'FAILED'}\n"
                f"Inliers: {len(result.inlier_trajectories)}/{len(trajectories)}")
    
    # Equal aspect ratio
    max_range = 0
    for traj in trajectories:
        pts = traj.get_all_positions()
        max_range = max(max_range, np.max(np.abs(pts)))
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_zlim(-max_range, max_range)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    test_joint_estimation()