"""
Post-Processing Module for Joint Estimation Pipeline

This module handles the final phase of the 4D RANSAC pipeline:
- Range of motion calculation (angle_min/max for hinges, translation_min/max for sliders)
- Parameter refinement and validation
- Quality metrics and confidence scores
- Output formatting and visualization support
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from data_structures import (
    Trajectory3D, HingeParameters, SliderParameters, JointEstimationResult,
    JointType
)


class PostProcessor:
    """
    Post-processing for joint estimation results.
    
    Handles range of motion calculation, parameter validation, and result refinement.
    """
    
    def __init__(self):
        pass
    
    def process_result(self, result: JointEstimationResult) -> JointEstimationResult:
        """
        Complete post-processing of joint estimation result.
        
        Args:
            result: Raw result from RANSAC core
            
        Returns:
            Enhanced result with range of motion and refined parameters
        """
        if not result.success:
            return result
        
        print(f"=== Post-Processing {result.joint_type.value.upper()} Joint ===")
        
        # Calculate range of motion
        if result.joint_type == JointType.HINGE:
            refined_params = self._calculate_hinge_range_of_motion(
                result.get_hinge_params(), result.inlier_trajectories
            )
        elif result.joint_type == JointType.SLIDER:
            refined_params = self._calculate_slider_range_of_motion(
                result.get_slider_params(), result.inlier_trajectories
            )
        else:
            refined_params = result.parameters
        
        per_frame_articulation = self._calculate_per_frame_articulation(
            refined_params, result.inlier_trajectories, result.joint_type
        )
        # Validate refined parameters
        validation_score = self._validate_parameters(refined_params, result.inlier_trajectories)
        
        # Update result with refined parameters
        enhanced_result = JointEstimationResult(
            success=result.success,
            joint_type=result.joint_type,
            parameters=refined_params,
            confidence=min(result.confidence + validation_score * 0.1, 1.0),  # Boost confidence slightly
            inlier_trajectories=result.inlier_trajectories,
            total_trajectories=result.total_trajectories,
            processing_time=result.processing_time,
            error_message=result.error_message,
            per_frame_articulation=per_frame_articulation
        )
        
        # Print summary
        self._print_result_summary(enhanced_result)
        
        return enhanced_result
    
    def _calculate_hinge_range_of_motion(self, 
                                       hinge_params: HingeParameters,
                                       inlier_trajectories: List[Trajectory3D]) -> HingeParameters:
        """
        Calculate angle_min and angle_max for hinge joint.
        
        Following Li & Wan 2016: "For each frame, calculate the angle of the door 
        relative to its starting position. The min and max of these angles are your range."
        """
        print("Calculating hinge range of motion...")
        
        all_angles = []
        
        for traj in inlier_trajectories:
            if len(traj.points) < 2:
                continue
            
            # Use first point as reference
            reference_point = np.array([traj.points[0].x, traj.points[0].y, traj.points[0].z])
            
            for point in traj.points[1:]:
                current_point = np.array([point.x, point.y, point.z])
                
                # Calculate rotation angle relative to reference
                angle = self._calculate_rotation_angle(
                    reference_point, current_point, 
                    hinge_params.axis, hinge_params.pivot
                )
                
                all_angles.append(angle)
        
        if len(all_angles) == 0:
            print("Warning: No angles calculated, using default range")
            return HingeParameters(
                axis=hinge_params.axis,
                pivot=hinge_params.pivot,
                angle_min=0.0,
                angle_max=0.0
            )

        all_angles = np.unwrap(np.array(all_angles))  # ensure continuity
        angle_min = np.percentile(all_angles, 5)   # 5th percentile
        angle_max = np.percentile(all_angles, 98)  # 98th percentile
        
        print(f"Hinge range: {np.degrees(angle_min):.1f}° to {np.degrees(angle_max):.1f}°")
        
        return HingeParameters(
            axis=hinge_params.axis,
            pivot=hinge_params.pivot,
            angle_min=angle_min,
            angle_max=angle_max
        )
    
    def _calculate_rotation_angle(self, 
                                reference_point: np.ndarray,
                                current_point: np.ndarray,
                                axis: np.ndarray,
                                pivot: np.ndarray) -> float:
        """Calculate rotation angle between two points around a hinge axis."""
        # Translate so pivot is at origin
        ref_centered = reference_point - pivot
        cur_centered = current_point - pivot
        
        # Project onto plane perpendicular to axis
        ref_proj = ref_centered - np.dot(ref_centered, axis) * axis
        cur_proj = cur_centered - np.dot(cur_centered, axis) * axis
        
        # Handle points on the axis
        ref_norm = np.linalg.norm(ref_proj)
        cur_norm = np.linalg.norm(cur_proj)
        
        if ref_norm < 1e-6 or cur_norm < 1e-6:
            return 0.0  # Point is on the axis
        
        # Calculate angle between projected vectors
        cos_angle = np.dot(ref_proj, cur_proj) / (ref_norm * cur_norm)
        cos_angle = np.clip(cos_angle, -1, 1)
        angle = np.arccos(cos_angle)
        
        # Determine sign using cross product
        cross = np.cross(ref_proj, cur_proj)
        if np.dot(cross, axis) < 0:
            angle = -angle
        
        return angle
    
    def _calculate_slider_range_of_motion(self,
                                        slider_params: SliderParameters,
                                        inlier_trajectories: List[Trajectory3D]) -> SliderParameters:
        """
        Calculate translation_min and translation_max for slider joint.
        
        Following Li & Wan 2016: "For each frame, calculate the distance the drawer 
        has traveled along its direction vector v."
        """
        print("Calculating slider range of motion...")
        
        all_distances = []
        
        for traj in inlier_trajectories:
            if len(traj.points) < 2:
                continue
            
            # Use first point as reference
            reference_point = np.array([traj.points[0].x, traj.points[0].y, traj.points[0].z])
            
            for point in traj.points[1:]:
                current_point = np.array([point.x, point.y, point.z])
                
                # Calculate translation distance along direction
                displacement = current_point - reference_point
                distance = np.dot(displacement, slider_params.direction)
                
                all_distances.append(distance)
        
        if len(all_distances) == 0:
            print("Warning: No distances calculated, using default range")
            return SliderParameters(
                direction=slider_params.direction,
                reference_point=slider_params.reference_point,
                translation_min=0.0,
                translation_max=0.0
            )
        
        all_distances = np.array(all_distances)
        translation_min = np.min(all_distances)
        translation_max = np.max(all_distances)
        
        print(f"Slider range: {translation_min:.3f}m to {translation_max:.3f}m")
        
        return SliderParameters(
            direction=slider_params.direction,
            reference_point=slider_params.reference_point,
            translation_min=translation_min,
            translation_max=translation_max
        )
    

    def _calculate_per_frame_articulation(self,
                                        parameters: object,
                                        inlier_trajectories: List[Trajectory3D],
                                        joint_type: JointType) -> Dict:
        """
        Calculate per-frame articulation values for mapping to multi-view data.

        Returns:
            Dict with frame_idx → articulation_value mapping
        """
        # Group points by frame
        frame_articulation = {}
        
        if joint_type == JointType.HINGE:
            hinge_params = parameters
            
            # For each inlier trajectory
            for traj in inlier_trajectories:
                reference_point = np.array([traj.points[0].x, traj.points[0].y, traj.points[0].z])
                
                for frame_idx, point in enumerate(traj.points):
                    current_point = np.array([point.x, point.y, point.z])
                    
                    angle = self._calculate_rotation_angle(
                        reference_point, current_point,
                        hinge_params.axis, hinge_params.pivot
                    )
                    
                    if frame_idx not in frame_articulation:
                        frame_articulation[frame_idx] = []
                    frame_articulation[frame_idx].append(angle)
            
            # Average angles per frame
            per_frame_values = {}
            for frame_idx, angles in frame_articulation.items():
                per_frame_values[frame_idx] = float(np.median(angles))
        
        elif joint_type == JointType.SLIDER:
            slider_params = parameters
            
            for traj in inlier_trajectories:
                reference_point = np.array([traj.points[0].x, traj.points[0].y, traj.points[0].z])
                
                for frame_idx, point in enumerate(traj.points):
                    current_point = np.array([point.x, point.y, point.z])
                    displacement = current_point - reference_point
                    distance = np.dot(displacement, slider_params.direction)
                    
                    if frame_idx not in frame_articulation:
                        frame_articulation[frame_idx] = []
                    frame_articulation[frame_idx].append(distance)
            
            per_frame_values = {}
            for frame_idx, distances in frame_articulation.items():
                per_frame_values[frame_idx] = float(np.median(distances))
        
        # Normalize to [0, 1]
        values = np.array(list(per_frame_values.values()))
        min_val, max_val = values.min(), values.max()
        
        normalized = {}
        for frame_idx, val in per_frame_values.items():
            if max_val > min_val:
                normalized[frame_idx] = (val - min_val) / (max_val - min_val)
            else:
                normalized[frame_idx] = 0.0
        
        return {
            'raw_values': per_frame_values,
            'normalized': normalized,
            'min': float(min_val),
            'max': float(max_val)
        }


    def _validate_parameters(self, 
                           parameters: object,
                           inlier_trajectories: List[Trajectory3D]) -> float:
        """
        Validate joint parameters and return quality score (0-1).
        
        Args:
            parameters: Joint parameters to validate
            inlier_trajectories: Supporting trajectories
            
        Returns:
            Quality score between 0-1 (higher is better)
        """
        if not inlier_trajectories:
            return 0.0
        
        quality_scores = []
        
        # Check trajectory coverage (more trajectories = better)
        coverage_score = min(len(inlier_trajectories) / 20.0, 1.0)  # Cap at 20 trajectories
        quality_scores.append(coverage_score)
        
        # Check temporal consistency (longer trajectories = better)
        avg_traj_length = np.mean([len(traj.points) for traj in inlier_trajectories])
        temporal_score = min(avg_traj_length / 10.0, 1.0)  # Cap at 10 frames
        quality_scores.append(temporal_score)
        
        # Check motion magnitude (significant motion = better)
        motion_magnitudes = []
        for traj in inlier_trajectories:
            if len(traj.points) >= 2:
                start = np.array([traj.points[0].x, traj.points[0].y, traj.points[0].z])
                end = np.array([traj.points[-1].x, traj.points[-1].y, traj.points[-1].z])
                magnitude = np.linalg.norm(end - start)
                motion_magnitudes.append(magnitude)
        
        if motion_magnitudes:
            avg_motion = np.mean(motion_magnitudes)
            motion_score = min(avg_motion / 0.5, 1.0)  # Cap at 0.5m movement
            quality_scores.append(motion_score)
        
        return np.mean(quality_scores)
    
    def _print_result_summary(self, result: JointEstimationResult):
        """Print detailed summary of joint estimation result."""
        print(f"\n{'='*50}")
        print(f"JOINT ESTIMATION SUMMARY")
        print(f"{'='*50}")
        print(f"Joint Type: {result.joint_type.value.upper()}")
        print(f"Success: {'✓' if result.success else '✗'}")
        print(f"Confidence: {result.confidence:.3f}")
        print(f"Processing Time: {result.processing_time:.2f}s")
        print(f"Inliers: {len(result.inlier_trajectories)} / {result.total_trajectories}")
        
        if result.success:
            if result.joint_type == JointType.HINGE:
                hinge_params = result.get_hinge_params()
                print(f"\nHINGE PARAMETERS:")
                print(f"  Axis: [{hinge_params.axis[0]:.3f}, {hinge_params.axis[1]:.3f}, {hinge_params.axis[2]:.3f}]")
                print(f"  Pivot: [{hinge_params.pivot[0]:.3f}, {hinge_params.pivot[1]:.3f}, {hinge_params.pivot[2]:.3f}]")
                if hinge_params.angle_min is not None and hinge_params.angle_max is not None:
                    print(f"  Range: {np.degrees(hinge_params.angle_min):.1f}° to {np.degrees(hinge_params.angle_max):.1f}°")
                    print(f"  Total Range: {np.degrees(hinge_params.angle_max - hinge_params.angle_min):.1f}°")
                    
            elif result.joint_type == JointType.SLIDER:
                slider_params = result.get_slider_params()
                print(f"\nSLIDER PARAMETERS:")
                print(f"  Direction: [{slider_params.direction[0]:.3f}, {slider_params.direction[1]:.3f}, {slider_params.direction[2]:.3f}]")
                if slider_params.reference_point is not None:
                    print(f"  Reference: [{slider_params.reference_point[0]:.3f}, {slider_params.reference_point[1]:.3f}, {slider_params.reference_point[2]:.3f}]")
                if slider_params.translation_min is not None and slider_params.translation_max is not None:
                    print(f"  Range: {slider_params.translation_min:.3f}m to {slider_params.translation_max:.3f}m")
                    print(f"  Total Range: {slider_params.translation_max - slider_params.translation_min:.3f}m")
        
        print(f"{'='*50}\n")
    
    def visualize_result(self, 
                        result: JointEstimationResult,
                        trajectories_3d: List[Trajectory3D],
                        title: str = "Joint Estimation Result") -> None:
        """
        Create 3D visualization of joint estimation result.
        
        Args:
            result: Joint estimation result
            trajectories_3d: All trajectories (for context)
            title: Plot title
        """
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot all trajectories
        for traj in trajectories_3d:
            pts = traj.get_all_positions()
            
            if result.success and traj in result.inlier_trajectories:
                color = "green"
                alpha = 0.8
                linewidth = 2
                label = "Inliers" if traj == result.inlier_trajectories[0] else ""
            else:
                color = "red"
                alpha = 0.4
                linewidth = 1
                label = "Outliers" if traj == trajectories_3d[0] and traj not in result.inlier_trajectories else ""
            
            ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], 
                   color=color, alpha=alpha, linewidth=linewidth, label=label)
            
            # Mark start and end points
            ax.scatter(pts[0, 0], pts[0, 1], pts[0, 2], 
                      color=color, s=30, alpha=0.8, marker='o')
            ax.scatter(pts[-1, 0], pts[-1, 1], pts[-1, 2], 
                      color=color, s=30, alpha=0.8, marker='s')
        
        # Plot estimated joint
        if result.success:
            if result.joint_type == JointType.HINGE:
                hinge_params = result.get_hinge_params()
                
                # Draw rotation axis
                axis_length = 2.0
                axis_start = hinge_params.pivot - axis_length * hinge_params.axis
                axis_end = hinge_params.pivot + axis_length * hinge_params.axis
                
                ax.plot([axis_start[0], axis_end[0]], 
                       [axis_start[1], axis_end[1]], 
                       [axis_start[2], axis_end[2]], 
                       'blue', linewidth=4, label='Hinge Axis')
                
                # Mark pivot point
                ax.scatter(hinge_params.pivot[0], hinge_params.pivot[1], hinge_params.pivot[2],
                          color='blue', s=200, marker='*', label='Pivot')
                
            elif result.joint_type == JointType.SLIDER:
                slider_params = result.get_slider_params()
                
                # Draw slider direction
                if slider_params.reference_point is not None:
                    ref_point = slider_params.reference_point
                else:
                    # Use centroid of inlier trajectories as reference
                    all_points = []
                    for traj in result.inlier_trajectories:
                        all_points.extend(traj.get_all_positions())
                    ref_point = np.mean(all_points, axis=0)
                
                direction_length = 2.0
                dir_start = ref_point - direction_length * slider_params.direction
                dir_end = ref_point + direction_length * slider_params.direction
                
                ax.plot([dir_start[0], dir_end[0]], 
                       [dir_start[1], dir_end[1]], 
                       [dir_start[2], dir_end[2]], 
                       'blue', linewidth=4, label='Slide Direction')
                
                # Mark reference point
                ax.scatter(ref_point[0], ref_point[1], ref_point[2],
                          color='blue', s=200, marker='*', label='Pivot')
        
        # Formatting
        ax.legend()
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        
        # Create comprehensive title
        if result.success:
            confidence_str = f"(Confidence: {result.confidence:.2f})"
            inlier_str = f"{len(result.inlier_trajectories)}/{result.total_trajectories} inliers"
            full_title = f"{title}\n{result.joint_type.value.upper()} Joint {confidence_str} - {inlier_str}"
        else:
            full_title = f"{title}\nFAILED: {result.error_message}"
        
        ax.set_title(full_title)
        
        # Set equal aspect ratio
        all_points = []
        for traj in trajectories_3d:
            all_points.extend(traj.get_all_positions())
        
        if all_points:
            all_points = np.array(all_points)
            max_range = np.max(np.ptp(all_points, axis=0)) / 2
            mid_point = np.mean(all_points, axis=0)
            
            ax.set_xlim(mid_point[0] - max_range, mid_point[0] + max_range)
            ax.set_ylim(mid_point[1] - max_range, mid_point[1] + max_range)
            ax.set_zlim(mid_point[2] - max_range, mid_point[2] + max_range)
        
        plt.tight_layout()
        plt.show()


# Convenience functions
def process_joint_result(result: JointEstimationResult) -> JointEstimationResult:
    """Convenience function for post-processing joint estimation results."""
    processor = PostProcessor()
    return processor.process_result(result)


def visualize_joint_result(result: JointEstimationResult, 
                          trajectories_3d: List[Trajectory3D],
                          title: str = "Joint Estimation Result") -> None:
    """Convenience function for visualizing joint estimation results."""
    processor = PostProcessor()
    processor.visualize_result(result, trajectories_3d, title)