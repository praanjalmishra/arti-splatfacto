#!/usr/bin/env python3
"""
Complete Pipeline Test

Test the full 4D RANSAC Joint Estimation pipeline with synthetic data.
"""

import numpy as np
import sys
from pathlib import Path

# Add the joint_estimator directory to path
sys.path.append(str(Path(__file__).parent))

from data_structures import Point3D, Trajectory3D, create_default_config
from main import JointEstimator


def generate_realistic_hinge_data():
    """Generate realistic hinge data similar to a door opening."""
    print("Generating realistic door hinge data...")
    
    # Door parameters
    true_axis = np.array([0, 0, 1])  # Vertical hinge (Z-axis)
    true_pivot = np.array([0, 0, 0])  # Hinge at origin
    
    # Door opening sequence: 0 to 90 degrees over 20 frames
    angles = np.linspace(0, np.pi/2, 20)
    
    trajectories = []
    
    # Generate trajectories for points on the door at different distances from hinge
    door_points = [
        np.array([0.4, 0, 0.5]),   # Door handle area
        np.array([0.8, 0, 0.5]),   # Door edge
        np.array([0.4, 0, 1.0]),   # Upper door handle area  
        np.array([0.8, 0, 1.0]),   # Upper door edge
        np.array([0.4, 0, 0.0]),   # Lower door handle area
        np.array([0.8, 0, 0.0]),   # Lower door edge
        np.array([0.6, 0, 0.5]),   # Middle of door
        np.array([0.2, 0, 0.5]),   # Near hinge
    ]
    
    for track_id, base_point in enumerate(door_points):
        points = []
        
        for frame, angle in enumerate(angles):
            # Rotate point around hinge axis
            # Rodrigues' rotation formula
            K = np.array([
                [0, -true_axis[2], true_axis[1]],
                [true_axis[2], 0, -true_axis[0]], 
                [-true_axis[1], true_axis[0], 0]
            ])
            R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
            
            # Apply rotation around pivot
            rotated_point = R @ (base_point - true_pivot) + true_pivot
            
            # Add realistic noise (1-2mm)
            noise = np.random.normal(0, 0.002, 3)
            noisy_point = rotated_point + noise
            
            point = Point3D(
                frame=frame,
                x=noisy_point[0],
                y=noisy_point[1], 
                z=noisy_point[2]
            )
            points.append(point)
        
        trajectory = Trajectory3D(track_id=track_id, points=points, rigid_part=1)
        trajectories.append(trajectory)
    
    # Add some static trajectories (door frame)
    static_points = [
        np.array([-0.1, -0.05, 0.5]),  # Door frame left
        np.array([-0.1, -0.05, 1.0]),  # Door frame left top
        np.array([-0.1, -0.05, 0.0]),  # Door frame left bottom
    ]
    
    for track_id, static_point in enumerate(static_points):
        points = []
        for frame in range(len(angles)):
            # Static point with minimal noise
            noise = np.random.normal(0, 0.001, 3)
            noisy_point = static_point + noise
            
            point = Point3D(
                frame=frame,
                x=noisy_point[0],
                y=noisy_point[1],
                z=noisy_point[2]
            )
            points.append(point)
        
        trajectory = Trajectory3D(track_id=len(door_points) + track_id, points=points, rigid_part=0)
        trajectories.append(trajectory)
    
    print(f"Generated {len(trajectories)} trajectories ({len(door_points)} moving, {len(static_points)} static)")
    print(f"True hinge: axis={true_axis}, pivot={true_pivot}")
    print(f"True range: 0° to 90°")
    
    return trajectories, true_axis, true_pivot


def generate_realistic_slider_data():
    """Generate realistic slider data similar to a drawer opening."""
    print("Generating realistic drawer slider data...")
    
    # Drawer parameters
    true_direction = np.array([1, 0, 0])  # Slide along X-axis
    slide_distance = 0.3  # 30cm drawer travel
    
    # Drawer opening sequence
    distances = np.linspace(0, slide_distance, 15)
    
    trajectories = []
    
    # Generate trajectories for points on the drawer
    drawer_points = [
        np.array([0, 0.2, 0.1]),   # Drawer front left
        np.array([0, -0.2, 0.1]),  # Drawer front right  
        np.array([0, 0.2, -0.1]),  # Drawer front left bottom
        np.array([0, -0.2, -0.1]), # Drawer front right bottom
        np.array([0, 0, 0]),       # Drawer center
        np.array([0, 0.1, 0.05]),  # Drawer handle area
    ]
    
    for track_id, base_point in enumerate(drawer_points):
        points = []
        
        for frame, distance in enumerate(distances):
            # Translate point along slide direction
            translated_point = base_point + distance * true_direction
            
            # Add realistic noise (1-2mm)
            noise = np.random.normal(0, 0.002, 3)
            noisy_point = translated_point + noise
            
            point = Point3D(
                frame=frame,
                x=noisy_point[0],
                y=noisy_point[1],
                z=noisy_point[2]
            )
            points.append(point)
        
        trajectory = Trajectory3D(track_id=track_id, points=points, rigid_part=1)
        trajectories.append(trajectory)
    
    # Add some static trajectories (cabinet frame)
    static_points = [
        np.array([-0.05, 0.25, 0.15]),  # Cabinet frame
        np.array([-0.05, -0.25, 0.15]), # Cabinet frame
        np.array([-0.05, 0, 0.15]),     # Cabinet frame center
    ]
    
    for track_id, static_point in enumerate(static_points):
        points = []
        for frame in range(len(distances)):
            # Static point with minimal noise
            noise = np.random.normal(0, 0.001, 3)
            noisy_point = static_point + noise
            
            point = Point3D(
                frame=frame,
                x=noisy_point[0],
                y=noisy_point[1],
                z=noisy_point[2]
            )
            points.append(point)
        
        trajectory = Trajectory3D(track_id=len(drawer_points) + track_id, points=points, rigid_part=0)
        trajectories.append(trajectory)
    
    print(f"Generated {len(trajectories)} trajectories ({len(drawer_points)} moving, {len(static_points)} static)")
    print(f"True slider: direction={true_direction}")
    print(f"True range: 0m to {slide_distance}m")
    
    return trajectories, true_direction


def test_hinge_estimation():
    """Test complete pipeline with hinge data."""
    print("\n" + "="*70)
    print("TESTING HINGE JOINT ESTIMATION")
    print("="*70)
    
    # Generate test data
    trajectories, true_axis, true_pivot = generate_realistic_hinge_data()
    
    # Create configuration
    config = create_default_config(fx=525, fy=525, cx=320, cy=240, w=640, h=480)
    config.ransac_config.max_iterations = 300
    config.ransac_config.error_threshold = 0.01  # 1cm tolerance
    config.ransac_config.min_inliers = 5
    config.ransac_config.min_trajectory_length = 10
    
    # Initialize estimator
    estimator = JointEstimator(config)
    
    # Run estimation
    result = estimator.estimate_joint_from_trajectories(trajectories, visualize=True)
    
    if result and result.success:
        hinge_params = result.get_hinge_params()
        
        # Calculate accuracy
        axis_error = np.linalg.norm(hinge_params.axis - true_axis)
        pivot_error = np.linalg.norm(hinge_params.pivot - true_pivot)
        
        print(f"\nACCURACY ANALYSIS:")
        print(f"Axis error: {axis_error:.4f} (true: {true_axis}, estimated: {hinge_params.axis})")
        print(f"Pivot error: {pivot_error:.4f}m (true: {true_pivot}, estimated: {hinge_params.pivot})")
        
        if hinge_params.angle_min is not None and hinge_params.angle_max is not None:
            estimated_range = np.degrees(hinge_params.angle_max - hinge_params.angle_min)
            print(f"Estimated range: {estimated_range:.1f}° (expected ~90°)")
        
        return True
    else:
        print("HINGE TEST FAILED")
        return False


def test_slider_estimation():
    """Test complete pipeline with slider data."""
    print("\n" + "="*70)
    print("TESTING SLIDER JOINT ESTIMATION") 
    print("="*70)
    
    # Generate test data
    trajectories, true_direction = generate_realistic_slider_data()
    
    # Create configuration
    config = create_default_config(fx=525, fy=525, cx=320, cy=240, w=640, h=480)
    config.ransac_config.max_iterations = 300
    config.ransac_config.error_threshold = 0.01  # 1cm tolerance
    config.ransac_config.min_inliers = 4
    config.ransac_config.min_trajectory_length = 8
    
    # Initialize estimator
    estimator = JointEstimator(config)
    
    # Run estimation
    result = estimator.estimate_joint_from_trajectories(trajectories, visualize=True)
    
    if result and result.success:
        slider_params = result.get_slider_params()
        
        # Calculate accuracy
        direction_error = np.linalg.norm(slider_params.direction - true_direction)
        
        print(f"\nACCURACY ANALYSIS:")
        print(f"Direction error: {direction_error:.4f} (true: {true_direction}, estimated: {slider_params.direction})")
        
        if slider_params.translation_min is not None and slider_params.translation_max is not None:
            estimated_range = slider_params.translation_max - slider_params.translation_min
            print(f"Estimated range: {estimated_range:.3f}m (expected ~0.300m)")
        
        return True
    else:
        print("SLIDER TEST FAILED") 
        return False


def run_complete_tests():
    """Run complete test suite."""
    print("4D RANSAC JOINT ESTIMATION - COMPLETE PIPELINE TEST")
    print("="*70)
    
    results = []
    
    # Test hinge estimation
    try:
        hinge_success = test_hinge_estimation()
        results.append(("Hinge", hinge_success))
    except Exception as e:
        print(f"Hinge test failed with exception: {e}")
        results.append(("Hinge", False))
    
    # Test slider estimation
    try:
        slider_success = test_slider_estimation() 
        results.append(("Slider", slider_success))
    except Exception as e:
        print(f"Slider test failed with exception: {e}")
        results.append(("Slider", False))
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for test_name, success in results:
        status = "PASS" if success else "FAIL"
        print(f"{test_name:20s}: {status}")
    
    total_passed = sum(1 for _, success in results if success)
    print(f"\nOverall: {total_passed}/{len(results)} tests passed")
    
    return total_passed == len(results)


if __name__ == "__main__":
    success = run_complete_tests()
    sys.exit(0 if success else 1)