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

from re import T
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
from joint_estimator.preprocessing_traj import preprocess_trajectories
from ransac_core import estimate_joint_from_trajectories
from post_processing import process_joint_result, visualize_joint_result
from utils import export_joint_and_inliers, export_joint_and_inliers_tapip3d, save_result_to_json


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
                grid_size=40, 
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
            final_result, per_frame_values = process_joint_result(ransac_result)
            
            total_time = time.time() - total_start_time
            print(f"🏁 PIPELINE COMPLETE in {total_time:.2f}s")
            
            # Phase 4: Visualization
            if visualize:
                print("\n PHASE 4: Visualization")
                visualize_joint_result(final_result, trajectories_3d, "4D RANSAC Joint Estimation")

            return final_result, per_frame_values

        except Exception as e:
            print(f"[ERROR] Pipeline failed with error: {e}")
            if self.config.debug_mode:
                import traceback
                traceback.print_exc()
            return None
    
    def estimate_joint_from_tapip3d(self,
                                    tapip3d_result_path: str,
                                    out_dir: str,
                                    visualize: bool = True,
                                    visibility_threshold: float = 0.5) -> Optional[object]:
        """
        Complete pipeline: TAPIP3D results → Joint parameters.
        
        Args:
            tapip3d_result_path: Path to .result.npz file
            out_dir: Output directory for results
            visualize: Whether to show 3D visualization
            visibility_threshold: Minimum visibility score for points
            
        Returns:
            Joint estimation result with parameters and range of motion
        """
        print("\n" + "="*60)
        print("4D RANSAC JOINT ESTIMATION (TAPIP3D INPUT)")
        print("="*60)
        
        total_start_time = time.time()
        
        # Phase 1: Load TAPIP3D Trajectories
        print("\n📦 PHASE 1: Loading TAPIP3D Trajectories")
        from tapip3d_loader import load_tapip3d_trajectories
        
        trajectories_3d, metadata = load_tapip3d_trajectories(
            npz_path=tapip3d_result_path,
            filter_config=self.config.trajectory_filter_config,
            visibility_threshold=visibility_threshold
        )
        
        if len(trajectories_3d) == 0:
            print("[ERROR] No valid 3D trajectories loaded from TAPIP3D")
            return None
        
        print(f"✓ Loaded {len(trajectories_3d)} valid trajectories")

        print("\n🔧 PHASE 1.5: Preprocessing Trajectories")
        trajectories_3d = preprocess_trajectories(
            trajectories_3d,
            smooth_window=5,
            min_length=15,
            min_displacement=0.01,
            max_acceleration_percentile=95,
            accel_threshold=0.5
        )
        
        # Phase 2: RANSAC Joint Fitting
        print("\nPHASE 2: RANSAC Joint Fitting")
        ransac_result = estimate_joint_from_trajectories(
            trajectories_3d, self.config.ransac_config
        )
        
        if not ransac_result.success:
            print(f"[ERROR] RANSAC failed: {ransac_result.error_message}")
            if visualize:
                visualize_joint_result(ransac_result, trajectories_3d, 
                                    "FAILED Joint Estimation (TAPIP3D)")
            return ransac_result
        
        print(f"✓ RANSAC succeeded: {ransac_result.joint_type.value} joint found")
        
        # Phase 3: Post-Processing
        print("\n PHASE 3: Post-Processing & Range Calculation")
        final_result, per_frame_values = process_joint_result(ransac_result)
        
        total_time = time.time() - total_start_time
        print(f"4D RANSAC complete in {total_time:.2f}s")

        # Phase 4: Visualization
        if visualize:
            print("\nPHASE 4: Visualization")
            visualize_joint_result(final_result, trajectories_3d, 
                                "4D RANSAC Joint Estimation (TAPIP3D)")
        
        return final_result, per_frame_values
        


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
        early_termination_threshold=args.early_termination_threshold,
        use_tms_preclustering=args.use_tms_preclustering
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
    
    # Input data arguments (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--video_path", type=str,
                            help="Path to RGB video file (CoTracker mode)")
    input_group.add_argument("--tapip3d_result", type=str,
                            help="Path to TAPIP3D .result.npz file")

    # Additional inputs for CoTracker mode
    parser.add_argument("--depth_dir", type=str,
                    help="Directory containing depth images (required for video_path)")
    parser.add_argument("--camera_metadata", type=str,
                    help="JSON file with camera parameters (required for video_path)")

    parser.add_argument("--out_dir", type=str, default="./output",
                    help="Directory to save intermediate outputs")

    # TAPIP3D-specific parameters
    parser.add_argument("--visibility_threshold", type=float, default=0.8,
                    help="Minimum visibility score for TAPIP3D points (0-1)")
    
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
    parser.add_argument("--max_iterations", type=int, default=200,
                       help="Maximum RANSAC iterations")
    parser.add_argument("--error_threshold", type=float, default=0.05,
                       help="Error threshold for inliers (meters)")
    parser.add_argument("--min_inliers", type=int, default=30,
                       help="Minimum inliers to accept model")
    parser.add_argument("--min_trajectory_length", type=int, default=5,
                       help="Minimum trajectory length")
    parser.add_argument("--early_termination_threshold", type=float, default=0.95,
                       help="Early termination consensus threshold")
    parser.add_argument("--use_tms_preclustering", action="store_true", default=False,
                       help="Use TMS pre-clustering to speed up RANSAC")
    
    # Trajectory filtering parameters
    parser.add_argument("--max_velocity_jump", type=float, default=0.5,
                       help="Maximum velocity jump (m/s)")
    parser.add_argument("--smoothing_window", type=int, default=5,
                       help="Trajectory smoothing window size")
    
    # Output and visualization
    parser.add_argument("--no_viz", action="store_true",
                       help="Disable 3D visualization")

    parser.add_argument("--debug", action="store_true",
                       help="Enable debug mode")
    
    args = parser.parse_args()
    

    # Validate inputs based on mode
    if args.video_path:
        # CoTracker mode validation
        if not args.depth_dir or not args.camera_metadata:
            print("[ERROR] --depth_dir and --camera_metadata required when using --video_path")
            return 1
        
        if not Path(args.video_path).exists():
            print(f"[ERROR] Video file not found: {args.video_path}")
            return 1
        
        if not Path(args.depth_dir).exists():
            print(f"[ERROR] Depth directory not found: {args.depth_dir}")
            return 1
        
        if not Path(args.camera_metadata).exists():
            print(f"[ERROR] Camera metadata file not found: {args.camera_metadata}")
            return 1
        


    elif args.tapip3d_result:
        # TAPIP3D mode validation
        if not Path(args.tapip3d_result).exists():
            print(f"[ERROR] TAPIP3D result file not found: {args.tapip3d_result}")
            return 1


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

    config = create_pipeline_config_from_args(args)

    estimator = JointEstimator(config)

    if args.video_path:
        print("Running CoTracker RGBD mode...")
        result, per_frame_values = estimator.estimate_joint_from_rgbd(
            video_path=args.video_path,
            depth_dir=args.depth_dir,
            out_dir=args.out_dir,
            camera_metadata_path=args.camera_metadata,
            visualize=not args.no_viz,
            extrinsics=args.extrinsics
        )
    else:  
        print("Running TAPIP3D mode...")
        result, per_frame_values = estimator.estimate_joint_from_tapip3d(
            tapip3d_result_path=args.tapip3d_result,
            out_dir=args.out_dir,
            visualize=not args.no_viz,
            visibility_threshold=args.visibility_threshold
        )
    
    if result is None:
        return 1
    

    if args.out_dir and args.video_path :
        export_joint_and_inliers(
            result=result,
            inlier_trajectories=result.inlier_trajectories,
            extrinsics=args.extrinsics,
            output_dir=args.out_dir,
            filename_prefix="prismatic" if result.joint_type.value == "slider" else "revolute"
        )
        save_result_to_json(result, per_frame_values, Path(args.out_dir) / "joint_schemas.json", coordinate_system="world")
        print(f"Results saved to: {Path(args.out_dir) / 'joint_schemas.json'}")


    if args.out_dir and not args.video_path :
        export_joint_and_inliers_tapip3d(
            result=result,
            inlier_trajectories=result.inlier_trajectories,
            output_dir=args.out_dir,
            filename_prefix="prismatic" if result.joint_type.value == "slider" else "revolute",
            voxel_resolution=16
        )
        save_result_to_json(result, per_frame_values, Path(args.out_dir) / "joint_schemas.json", coordinate_system="world")
        print(f"Results saved to: {Path(args.out_dir) / 'joint_schemas.json'}")

    return 0 if result.success else 1




if __name__ == "__main__":
    import sys
    sys.exit(main())