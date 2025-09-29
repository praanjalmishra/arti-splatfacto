"""
Main file for 4D RANSAC Joint Estimation Pipeline

python implementation of the Li & Wan 2016 "Mobility Fitting using 4D RANSAC" approach.
This module coordinates the entire pipeline from RGB-D video to final joint parameters.

Pipeline stages:
1. Data Acquisition (RGB-D video + CoTracker)
2. 3D Trajectory Generation 
3. RANSAC Joint Fitting
4. Post-Processing (range of motion)
5. Visualization and output
"""
import numpy as np
import argparse
import json
import time
import torch
from pathlib import Path
from typing import Optional

from data_structures import (
    PipelineConfig, CameraIntrinsics, RANSACConfig, TrajectoryFilterConfig,
    create_default_config
)
from cotracker_rgbd import process_rgbd_video
from ransac_core import estimate_joint_from_trajectories
from post_processing import process_joint_result, visualize_joint_result
from utils import export_joint_and_inliers


class JointEstimator:
    """
    Main class for 4D RANSAC Joint Estimation.
    
    Coordinates the complete pipeline from RGB-D input to joint parameters.
    """
    
    def __init__(self, config: PipelineConfig):
        """
        Initialize the joint estimator.
        
        Args:
            config: Complete pipeline configuration
        """
        self.config = config
        
        print("4D RANSAC Joint Estimator Initialized")
        print(f"Camera: fx={config.camera_intrinsics.fx}, fy={config.camera_intrinsics.fy}")
        print(f"RANSAC: {config.ransac_config.max_iterations} iterations, ε={config.ransac_config.error_threshold}")
        print(f"Joint types: {[jt.value for jt in config.joint_types_to_test]}")
    
    def estimate_joint_from_rgbd(self, 
                                video_path: str,
                                depth_dir: str,
                                out_dir: str,
                                camera_metadata_path: str,
                                visualize: bool = True,
                                extrinsics: Optional[torch.Tensor] = None) -> Optional[object]:
        """
        Complete pipeline: RGB-D video → Joint parameters.
        
        Args:
            video_path: Path to RGB video file
            depth_dir: Directory containing depth images  
            camera_metadata_path: JSON file with camera parameters
            visualize: Whether to show 3D visualization
            
        Returns:
            Joint estimation result with parameters and range of motion
        """
        print("\n" + "="*60)
        print("4D RANSAC JOINT ESTIMATION PIPELINE")
        print("="*60)
        
        total_start_time = time.time()
        
        try:
            # Phase 1: Data Acquisition & 3D Trajectory Generation
            print("\n PHASE 1: RGB-D Processing & Trajectory Generation")
            trajectories_3d, camera_intrinsics = process_rgbd_video(
                video_path=video_path,
                depth_dir=depth_dir,
                out_dir=out_dir,
                camera_metadata_path=camera_metadata_path,
                trajectory_filter_config=self.config.trajectory_filter_config,
                grid_size=30,  # CoTracker grid size
                backward_tracking=True
            )
            
            if len(trajectories_3d) == 0:
                print("[ERROR] No valid 3D trajectories generated")
                return None

            print(f"[CoTracker] Generated {len(trajectories_3d)} valid 3D trajectories")

            # import pdb; pdb.set_trace()
            # Phase 2: RANSAC Joint Fitting
            print("\n PHASE 2: RANSAC Joint Fitting")
            ransac_result = estimate_joint_from_trajectories(
                trajectories_3d, self.config.ransac_config
            )
            
            if not ransac_result.success:
                print(f"[ERROR] RANSAC failed: {ransac_result.error_message}")
                if visualize:
                    visualize_joint_result(ransac_result, trajectories_3d, "FAILED Joint Estimation")
                return ransac_result
            
            print(f"✓ RANSAC succeeded: {ransac_result.joint_type.value} joint found")
            
            # Phase 3: Post-Processing
            print("\n PHASE 3: Post-Processing & Range Calculation")
            final_result = process_joint_result(ransac_result)
            
            total_time = time.time() - total_start_time
            print(f"🏁 PIPELINE COMPLETE in {total_time:.2f}s")
            
            # Phase 4: Visualization
            if visualize:
                print("\n PHASE 4: Visualization")
                visualize_joint_result(final_result, trajectories_3d, "4D RANSAC Joint Estimation")
            
            return final_result
            
        except Exception as e:
            print(f"[ERROR] Pipeline failed with error: {e}")
            if self.config.debug_mode:
                import traceback
                traceback.print_exc()
            return None
    
    def estimate_joint_from_trajectories(self, trajectories_3d, visualize: bool = True):
        """
        Estimate joint from pre-computed 3D trajectories (for testing).
        
        Args:
            trajectories_3d: List of 3D trajectories
            visualize: Whether to show visualization
            
        Returns:
            Joint estimation result
        """
        print("\n" + "="*60)
        print("4D RANSAC JOINT ESTIMATION (TRAJECTORIES INPUT)")
        print("="*60)
        
        total_start_time = time.time()
        
        try:
            # Phase 1: RANSAC Joint Fitting
            print("\n PHASE 1: RANSAC Joint Fitting")
            ransac_result = estimate_joint_from_trajectories(
                trajectories_3d, self.config.ransac_config
            )
            
            if not ransac_result.success:
                print(f"[ERROR] RANSAC failed: {ransac_result.error_message}")
                if visualize:
                    visualize_joint_result(ransac_result, trajectories_3d, "FAILED Joint Estimation")
                return ransac_result
            
            # Phase 2: Post-Processing
            print("\n PHASE 2: Post-Processing & Range Calculation")
            final_result = process_joint_result(ransac_result)
            
            total_time = time.time() - total_start_time
            print(f"PIPELINE COMPLETE in {total_time:.2f}s")
            
            # Phase 3: Visualization
            if visualize:
                print("\n PHASE 3: Visualization")
                visualize_joint_result(final_result, trajectories_3d, "4D RANSAC Joint Estimation")
            
            return final_result
            
        except Exception as e:
            print(f"[ERROR] Pipeline failed with error: {e}")
            if self.config.debug_mode:
                import traceback
                traceback.print_exc()
            return None


def create_pipeline_config_from_args(args) -> PipelineConfig:
    """Create pipeline configuration from command line arguments."""
    
    # Camera intrinsics
    camera_intrinsics = CameraIntrinsics(
        fx=args.fx, fy=args.fy, cx=args.cx, cy=args.cy , w=args.w, h=args.h
    )
    
    # RANSAC configuration
    ransac_config = RANSACConfig(
        max_iterations=args.max_iterations,
        error_threshold=args.error_threshold,
        min_inliers=args.min_inliers,
        min_trajectory_length=args.min_trajectory_length,
        early_termination_threshold=args.early_termination_threshold
    )
    
    # Trajectory filtering
    trajectory_filter_config = TrajectoryFilterConfig(
        min_length=args.min_trajectory_length,
        max_velocity_jump=args.max_velocity_jump,
        smoothing_window=args.smoothing_window
    )
    
    return PipelineConfig(
        camera_intrinsics=camera_intrinsics,
        ransac_config=ransac_config,
        trajectory_filter_config=trajectory_filter_config,
        debug_mode=args.debug
    )


def main():
    """Main entry point for joint estimation pipeline."""
    parser = argparse.ArgumentParser(
        description="4D RANSAC Joint Estimation Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input data arguments
    parser.add_argument("--video_path", type=str, required=True,
                       help="Path to RGB video file")
    parser.add_argument("--depth_dir", type=str, required=True,
                       help="Directory containing depth images")
    parser.add_argument("--out_dir", type=str, default="./output",
                       help="Directory to save intermediate outputs")
    parser.add_argument("--camera_metadata", type=str, required=True,
                       help="JSON file with camera parameters")
    
    # Camera parameters (can override metadata file)
    parser.add_argument("--fx", type=float, default=None,
                       help="Focal length X (override metadata)")
    parser.add_argument("--fy", type=float, default=None,
                       help="Focal length Y (override metadata)")
    parser.add_argument("--cx", type=float, default=None,
                       help="Principal point X (override metadata)")
    parser.add_argument("--cy", type=float, default=None,
                       help="Principal point Y (override metadata)")
    
    # RANSAC parameters
    parser.add_argument("--max_iterations", type=int, default=2000,
                       help="Maximum RANSAC iterations")
    parser.add_argument("--error_threshold", type=float, default=0.03,
                       help="Error threshold for inliers (meters)")
    parser.add_argument("--min_inliers", type=int, default=5,
                       help="Minimum inliers to accept model")
    parser.add_argument("--min_trajectory_length", type=int, default=20,
                       help="Minimum trajectory length")
    parser.add_argument("--early_termination_threshold", type=float, default=0.8,
                       help="Early termination consensus threshold")
    
    # Trajectory filtering parameters
    parser.add_argument("--max_velocity_jump", type=float, default=0.5,
                       help="Maximum velocity jump (m/s)")
    parser.add_argument("--smoothing_window", type=int, default=3,
                       help="Trajectory smoothing window size")
    
    # Output and visualization
    parser.add_argument("--no_viz", action="store_true",
                       help="Disable 3D visualization")

    parser.add_argument("--debug", action="store_true",
                       help="Enable debug mode")
    
    args = parser.parse_args()
    
    # Validate input files
    if not Path(args.video_path).exists():
        print(f"[ERROR] Video file not found: {args.video_path}")
        return 1
    
    if not Path(args.depth_dir).exists():
        print(f"[ERROR] Depth directory not found: {args.depth_dir}")
        return 1
    
    if not Path(args.camera_metadata).exists():
        print(f"[ERROR] Camera metadata file not found: {args.camera_metadata}")
        return 1
    
    if any(param is None for param in [args.fx, args.fy, args.cx, args.cy]):
        try:
            with open(args.camera_metadata, 'r') as f:
                metadata = json.load(f)
            
            args.fx = args.fx or metadata["fl_x"]
            args.fy = args.fy or metadata["fl_y"]
            args.cx = args.cx or metadata["cx"]
            args.cy = args.cy or metadata["cy"]
            args.w = metadata.get("w", None)
            args.h = metadata.get("h", None)

            if "frames" in metadata and len(metadata["frames"]) > 0:
                args.extrinsics = torch.tensor(
                    metadata["frames"][0]["transform_matrix"],
                    dtype=torch.float32
                )
            
        except Exception as e:
            print(f"ERROR: Failed to load camera parameters: {e}")
            return 1
    
    # Create pipeline config
    config = create_pipeline_config_from_args(args)
    
    # Initialize estimator
    estimator = JointEstimator(config)
    
    # Run the pipeline
    result = estimator.estimate_joint_from_rgbd(
        video_path=args.video_path,
        depth_dir=args.depth_dir,
        out_dir=args.out_dir, 
        camera_metadata_path=args.camera_metadata,
        visualize=not args.no_viz,
        extrinsics=args.extrinsics
    )
    
    if result is None:
        return 1
    
    if args.out_dir and result.success:
        save_result_to_json(result, Path(args.out_dir) / "joint_schemas.json")
        print(f"Results saved to: {Path(args.out_dir) / 'joint_schemas.json'}")


    if args.out_dir:
        export_joint_and_inliers(
            result=result,
            inlier_trajectories=result.inlier_trajectories,
            extrinsics=args.extrinsics,
            output_dir=args.out_dir,
            filename_prefix="prismatic" if result.joint_type.value == "slider" else "revolute"
        )

    return 0 if result.success else 1


def save_result_to_json(result, output_path: str):
    """Save joint estimation result to JSON file (Joint_schema.json format)."""

    joints_out = []

    if result.success:
        if result.joint_type.value == "hinge":  # revolute
            hinge_params = result.get_hinge_params()
            joint_data = {
                "joint_type": "revolute",
                "joint_axis": hinge_params.axis.tolist(),
                "joint_pivot": hinge_params.pivot.tolist(),
                "joint_limits": [
                    float(np.degrees(hinge_params.angle_min)) if hinge_params.angle_min is not None else None,
                    float(np.degrees(hinge_params.angle_max)) if hinge_params.angle_max is not None else None
                ]
            }
            joints_out.append(joint_data)

        elif result.joint_type.value == "slider":  # prismatic
            slider_params = result.get_slider_params()
            joint_data = {
                "joint_type": "prismatic",
                "joint_axis": slider_params.direction.tolist(),
                "joint_pivot": (
                    slider_params.reference_point.tolist()
                    if slider_params.reference_point is not None
                    else [0, 0, 0]
                ),
                "joint_limits": [
                    slider_params.translation_min,
                    slider_params.translation_max
                ]
            }
            joints_out.append(joint_data)

    with open(output_path, 'w') as f:
        json.dump(joints_out, f, indent=2)



if __name__ == "__main__":
    import sys
    sys.exit(main())