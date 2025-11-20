"""
Post-Processing Module for Joint Estimation Pipeline 

Key improvements over original:
1. Zero-referenced motion (angle_min=0, translation_min=0)
2. Proper frame indexing from Point3D.frame attribute
3. Handles sparse temporal data from preprocessing
4. Enhanced outlier filtering with IQR method
5. Trajectory quality weighting
6. Temporal smoothing for noisy data
7. Confidence bounds on estimates
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import gaussian_filter1d

from joint_estimator.data_structures import (
    Trajectory3D, HingeParameters, SliderParameters, JointEstimationResult,
    JointType
)

class PostProcessor:
    """
    Post-processing for joint estimation results.
    
    Handles range of motion calculation, parameter validation, and result refinement.
    """
    
    def __init__(self, smoothing_sigma: float = 1.0, outlier_threshold: float = 1.5):
        """
        Initialize post-processor with configuration.
        
        Args:
            smoothing_sigma: Gaussian smoothing sigma for temporal filtering
            outlier_threshold: IQR multiplier for outlier detection (1.5 is standard)
        """
        self.smoothing_sigma = smoothing_sigma
        self.outlier_threshold = outlier_threshold
    
    def process_result(self, result: JointEstimationResult) -> Tuple[JointEstimationResult, dict]:
        """
        Complete post-processing of joint estimation result.
        
        Args:
            result: Raw result from RANSAC core
            
        Returns:
            Enhanced result with range of motion and refined parameters
        """
        if not result.success:
            return result, {}
        
        print(f"=== Post-Processing {result.joint_type.value.upper()} Joint ===")
        
        # Calculate range of motion
        if result.joint_type == JointType.HINGE:
            refined_params, per_frame_values = self._calculate_hinge_range_of_motion(
                result.get_hinge_params(), result.inlier_trajectories
            )
        elif result.joint_type == JointType.SLIDER:
            refined_params, per_frame_values = self._calculate_slider_range_of_motion(
                result.get_slider_params(), result.inlier_trajectories
            )
        else:
            refined_params = result.parameters
            per_frame_values = {}
        
        # Validate refined parameters
        validation_score = self._validate_parameters(refined_params, result.inlier_trajectories)
        
        # Update result with refined parameters
        enhanced_result = JointEstimationResult(
            success=result.success,
            joint_type=result.joint_type,
            parameters=refined_params,
            confidence=min(result.confidence + validation_score * 0.1, 1.0),
            inlier_trajectories=result.inlier_trajectories,
            total_trajectories=result.total_trajectories,
            processing_time=result.processing_time,
            error_message=result.error_message 
        )

        # Print summary
        self._print_result_summary(enhanced_result)
        
        return enhanced_result, per_frame_values
    
    def _calculate_trajectory_weights(self, 
                                     trajectories: List[Trajectory3D],
                                     joint_params: object) -> np.ndarray:
        """
        Calculate quality weights for each trajectory based on multiple factors.
        
        Args:
            trajectories: List of trajectories to weight
            joint_params: Joint parameters for motion calculation
            
        Returns:
            Array of weights (one per trajectory)
        """
        weights = []
        
        for traj in trajectories:
            if len(traj.points) < 2:
                weights.append(0.0)
                continue
            
            # Factor 1: Length (longer trajectories are more reliable)
            length_score = min(len(traj.points) / 10.0, 1.0)
            
            # Factor 2: Motion magnitude (more motion = better signal)
            positions = traj.get_all_positions()
            motion_magnitude = np.linalg.norm(positions[-1] - positions[0])
            motion_score = min(motion_magnitude / 0.5, 1.0)
            
            # Factor 3: Consistency (less jitter = better)
            if len(traj.points) >= 3:
                # Calculate velocity variations
                velocities = np.diff(positions, axis=0)
                velocity_magnitudes = np.linalg.norm(velocities, axis=1)
                consistency_score = 1.0 - min(np.std(velocity_magnitudes) / (np.mean(velocity_magnitudes) + 1e-6), 1.0)
            else:
                consistency_score = 0.5
            
            # Combine scores (weighted average)
            weight = 0.4 * length_score + 0.4 * motion_score + 0.2 * consistency_score
            weights.append(max(weight, 0.1))  # Minimum weight of 0.1
        
        return np.array(weights)
    
    def _remove_angle_outliers(self, 
                              angles: np.ndarray, 
                              weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Remove outlier angles using Interquartile Range (IQR) method.
        
        Args:
            angles: Array of angle measurements
            weights: Corresponding weights
            
        Returns:
            (filtered_angles, filtered_weights)
        """
        if len(angles) <= 2:
            return angles, weights
        
        # Calculate quartiles
        q1 = np.percentile(angles, 25)
        q3 = np.percentile(angles, 75)
        iqr = q3 - q1
        
        # Define outlier bounds
        lower_bound = q1 - self.outlier_threshold * iqr
        upper_bound = q3 + self.outlier_threshold * iqr
        
        # Filter outliers
        mask = (angles >= lower_bound) & (angles <= upper_bound)
        
        return angles[mask], weights[mask]
    
    def _apply_temporal_smoothing(self, 
                                 per_frame_values: Dict[int, float],
                                 method: str = 'gaussian') -> Dict[int, float]:
        """
        Apply temporal smoothing to reduce noise in per-frame values.
        
        Args:
            per_frame_values: Dictionary mapping frame indices to values
            method: Smoothing method ('gaussian', 'moving_average', or 'none')
            
        Returns:
            Smoothed per-frame values
        """
        if len(per_frame_values) < 3 or method == 'none':
            return per_frame_values
        
        # Sort by frame index
        sorted_frames = sorted(per_frame_values.keys())
        values = np.array([per_frame_values[f] for f in sorted_frames])
        
        # Apply smoothing
        if method == 'gaussian':
            # Gaussian smoothing with sigma parameter
            smoothed_values = gaussian_filter1d(values, sigma=self.smoothing_sigma)
        elif method == 'moving_average':
            # Simple moving average with window size 3
            window_size = 3
            smoothed_values = np.convolve(values, np.ones(window_size)/window_size, mode='same')
        else:
            smoothed_values = values
        
        # Reconstruct dictionary
        return {frame: smoothed_values[i] for i, frame in enumerate(sorted_frames)}
    
    # def _calculate_hinge_range_of_motion(self, 
    #                                 hinge_params: HingeParameters,
    #                                 inlier_trajectories: List[Trajectory3D]) -> Tuple[HingeParameters, dict]:
    #     """
    #     Calculate angle_min, angle_max, and per-frame angles for hinge joint.
    #     Zero-referenced: angle_min = 0, angles represent opening from closed state.
        
    #     Improvements:
    #     - Outlier rejection using IQR method
    #     - Temporal smoothing for noisy data
    #     - Weighted averaging by trajectory quality
    #     - Better handling of angle wrapping
        
    #     Returns:
    #         (HingeParameters, per_frame_angles) where per_frame_angles is dict {frame_idx: angle}
    #     """
    #     print("Calculating hinge range of motion...")
        
    #     # Store angles per frame with trajectory quality weights
    #     frame_angles = {}  # {frame_idx: [(angle, weight), ...]}
        
    #     # Calculate quality weight for each trajectory based on:
    #     # 1. Length (longer = better)
    #     # 2. Motion magnitude (more motion = better)
    #     # 3. Consistency (less jitter = better)
    #     trajectory_weights = self._calculate_trajectory_weights(inlier_trajectories, hinge_params)
        
    #     # First pass: calculate all angles relative to their trajectory's first point
    #     for traj_idx, traj in enumerate(inlier_trajectories):
    #         if len(traj.points) < 2:
    #             continue
            
    #         traj_weight = trajectory_weights[traj_idx]
            
    #         # Use first point as reference (closed state) for THIS trajectory
    #         reference_point = np.array([traj.points[0].x, traj.points[0].y, traj.points[0].z])
            
    #         for point in traj.points:
    #             current_point = np.array([point.x, point.y, point.z])
                
    #             # Calculate rotation angle relative to reference
    #             angle = self._calculate_rotation_angle(
    #                 reference_point, current_point, 
    #                 hinge_params.axis, hinge_params.pivot
    #             )
                
    #             frame_idx = point.frame
    #             if frame_idx not in frame_angles:
    #                 frame_angles[frame_idx] = []
    #             frame_angles[frame_idx].append((angle, traj_weight))
        
    #     if len(frame_angles) == 0:
    #         print("Warning: No angles calculated, using default range")
    #         return HingeParameters(
    #             axis=hinge_params.axis,
    #             pivot=hinge_params.pivot,
    #             angle_min=0.0,
    #             angle_max=0.0
    #         ), {}
        
    #     # Aggregate angles per frame with outlier rejection and weighting
    #     per_frame_angles_raw = {}
    #     per_frame_stds = {}
        
    #     for frame_idx, angle_weight_pairs in frame_angles.items():
    #         angles = np.array([aw[0] for aw in angle_weight_pairs])
    #         weights = np.array([aw[1] for aw in angle_weight_pairs])
            
    #         # Remove outliers using IQR method
    #         angles_clean, weights_clean = self._remove_angle_outliers(angles, weights)
            
    #         if len(angles_clean) == 0:
    #             continue
            
    #         # Weighted average
    #         per_frame_angles_raw[frame_idx] = np.average(angles_clean, weights=weights_clean)
    #         per_frame_stds[frame_idx] = np.std(angles_clean)
        
    #     if len(per_frame_angles_raw) == 0:
    #         print("Warning: All angles filtered as outliers, using default range")
    #         return HingeParameters(
    #             axis=hinge_params.axis,
    #             pivot=hinge_params.pivot,
    #             angle_min=0.0,
    #             angle_max=0.0
    #         ), {}
        
    #     # Apply temporal smoothing to reduce noise
    #     per_frame_angles_smoothed = self._apply_temporal_smoothing(per_frame_angles_raw)
        
    #     # Zero-reference to minimum angle (closed position)
    #     angle_offset = min(per_frame_angles_smoothed.values())
        
    #     per_frame_angles = {
    #         frame_idx: angle - angle_offset
    #         for frame_idx, angle in per_frame_angles_smoothed.items()
    #     }
        
    #     # Calculate range with confidence bounds
    #     angle_min = 0.0  # Always start at 0 (closed position)
    #     angle_max = max(per_frame_angles.values())
        
    #     # Calculate average uncertainty
    #     avg_std = np.mean(list(per_frame_stds.values())) if per_frame_stds else 0.0
        
    #     print(f"Hinge range: {np.degrees(angle_min):.1f}° to {np.degrees(angle_max):.1f}°")
    #     print(f"Total range of motion: {np.degrees(angle_max):.1f}°")
    #     print(f"Average angle uncertainty: ±{np.degrees(avg_std):.1f}°")
    #     print(f"Per-frame angles computed for {len(per_frame_angles)} frames (sparse sampling)")
    #     print(f"Applied trajectory weighting and IQR outlier filtering")
        
    #     return HingeParameters(
    #         axis=hinge_params.axis,
    #         pivot=hinge_params.pivot,
    #         angle_min=angle_min,
    #         angle_max=angle_max
    #     ), per_frame_angles
        

    def _calculate_hinge_range_of_motion(
        self, hinge_params: HingeParameters,
        inlier_trajectories: List[Trajectory3D]
    ) -> Tuple[HingeParameters, dict]:
        """
        Calculate angle_min, angle_max, and per-frame angles for hinge joint.
        Zero-referenced: angle_min = 0, angles represent opening from closed state.

        Fixes:
        - Use global reference (first visible frame) instead of per-trajectory reference
        - Enforce zero angle at closed (minimum) configuration
        - Preserve outlier filtering, smoothing, and weighting
        """

        print("Calculating hinge range of motion (global reference fix)...")

        if not inlier_trajectories:
            print("No inlier trajectories — returning default hinge parameters.")
            return hinge_params, {}

        # ------------------------------------------------------------
        # 1. Determine the global reference frame (earliest frame index)
        # ------------------------------------------------------------
        all_frames = [p.frame for traj in inlier_trajectories for p in traj.points]
        min_global_frame = min(all_frames)
        max_global_frame = max(all_frames)

        # Gather all 3D points that exist at the earliest frame
        global_ref_points = []
        for traj in inlier_trajectories:
            for p in traj.points:
                if p.frame == min_global_frame:
                    global_ref_points.append(np.array([p.x, p.y, p.z]))

        if len(global_ref_points) == 0:
            print("⚠️ No trajectories contain the first frame — using first trajectory start as reference.")
            first_traj = inlier_trajectories[0]
            global_ref_point = np.array([first_traj.points[0].x,
                                        first_traj.points[0].y,
                                        first_traj.points[0].z])
        else:
            global_ref_point = np.mean(global_ref_points, axis=0)

        # ------------------------------------------------------------
        # 2. Calculate trajectory weights for quality-aware averaging
        # ------------------------------------------------------------
        trajectory_weights = self._calculate_trajectory_weights(inlier_trajectories, hinge_params)

        # ------------------------------------------------------------
        # Align hinge axis direction with actual motion
        # ------------------------------------------------------------
        # Compute approximate net motion direction
        net_motion_sum = 0.0
        for traj in inlier_trajectories:
            if len(traj.points) < 2:
                continue
            p_start = np.array([traj.points[0].x, traj.points[0].y, traj.points[0].z])
            p_end = np.array([traj.points[-1].x, traj.points[-1].y, traj.points[-1].z])
            # Vector from pivot to end points
            v_start = p_start - hinge_params.pivot
            v_end = p_end - hinge_params.pivot
            # Rotation direction measure (signed)
            cross_dir = np.cross(v_start, v_end)
            net_motion_sum += np.dot(cross_dir, hinge_params.axis)

        # If the average motion is opposite to the axis direction, flip it
        if net_motion_sum < 0:
            print("↻ Flipping hinge axis orientation to match motion direction.")
            hinge_params.axis = -hinge_params.axis


        # ------------------------------------------------------------
        # 3. Compute angles per frame (relative to global reference)
        # ------------------------------------------------------------
        frame_angles = {}  # {frame_idx: [(angle, weight), ...]}

        for traj_idx, traj in enumerate(inlier_trajectories):
            traj_weight = trajectory_weights[traj_idx]
            for point in traj.points:
                current_point = np.array([point.x, point.y, point.z])
                angle = self._calculate_rotation_angle(
                    global_ref_point, current_point,
                    hinge_params.axis, hinge_params.pivot
                )

                frame_idx = point.frame
                if frame_idx not in frame_angles:
                    frame_angles[frame_idx] = []
                frame_angles[frame_idx].append((angle, traj_weight))

        if len(frame_angles) == 0:
            print("Warning: No hinge angles calculated — using default range.")
            return HingeParameters(
                axis=hinge_params.axis,
                pivot=hinge_params.pivot,
                angle_min=0.0,
                angle_max=0.0
            ), {}

        # ------------------------------------------------------------
        # 4. Aggregate per-frame angles (IQR filtering + weighting)
        # ------------------------------------------------------------
        per_frame_angles_raw = {}
        per_frame_stds = {}

        for frame_idx, angle_weight_pairs in frame_angles.items():
            angles = np.array([aw[0] for aw in angle_weight_pairs])
            weights = np.array([aw[1] for aw in angle_weight_pairs])

            # Outlier removal (IQR)
            angles_clean, weights_clean = self._remove_angle_outliers(angles, weights)
            if len(angles_clean) == 0:
                continue

            # Weighted mean
            per_frame_angles_raw[frame_idx] = np.average(angles_clean, weights=weights_clean)
            per_frame_stds[frame_idx] = np.std(angles_clean)

        if len(per_frame_angles_raw) == 0:
            print("Warning: All angles removed by filtering — using defaults.")
            return HingeParameters(
                axis=hinge_params.axis,
                pivot=hinge_params.pivot,
                angle_min=0.0,
                angle_max=0.0
            ), {}

        # ------------------------------------------------------------
        # 5. Temporal smoothing
        # ------------------------------------------------------------
        per_frame_angles_smoothed = self._apply_temporal_smoothing(per_frame_angles_raw)

        # ------------------------------------------------------------
        # 6. Re-zero based on the minimum observed angle (closed state)
        # ------------------------------------------------------------
        angle_offset = min(per_frame_angles_smoothed.values())
        per_frame_angles = {
            f: angle - angle_offset for f, angle in per_frame_angles_smoothed.items()
        }

        # ------------------------------------------------------------
        # 7. Final statistics and summary
        # ------------------------------------------------------------
        angle_min = 0.0
        angle_max = max(per_frame_angles.values())
        avg_std = np.mean(list(per_frame_stds.values())) if per_frame_stds else 0.0

        print(f"Hinge range: {np.degrees(angle_min):.1f}° → {np.degrees(angle_max):.1f}°")
        print(f"Total motion: {np.degrees(angle_max):.1f}° over {len(per_frame_angles)} frames")
        print(f"Avg uncertainty: ±{np.degrees(avg_std):.2f}°")
        print(f"Global reference frame: {min_global_frame}")
        print(f"Applied global reference alignment + IQR filtering")

        # ------------------------------------------------------------
        # 8. Return refined parameters and per-frame angle map
        # ------------------------------------------------------------
        refined_params = HingeParameters(
            axis=hinge_params.axis,
            pivot=hinge_params.pivot,
            angle_min=angle_min,
            angle_max=angle_max
        )

        return refined_params, per_frame_angles

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
                                        inlier_trajectories: List[Trajectory3D]) -> Tuple[SliderParameters, dict]:
        """
        Calculate translation_min, translation_max, and per-frame translations.
        Zero-referenced: translation_min = 0, values represent extension from closed state.
        
        Improvements:
        - Outlier rejection using IQR method
        - Temporal smoothing for noisy data
        - Weighted averaging by trajectory quality
        
        Returns:
            (SliderParameters, per_frame_translations) where per_frame_translations is dict {frame_idx: translation}
        """
        print("Calculating slider range of motion...")
        
        # Store translations per frame with trajectory quality weights
        frame_translations = {}  # {frame_idx: [(trans, weight), ...]}
        
        # Calculate trajectory weights
        trajectory_weights = self._calculate_trajectory_weights(inlier_trajectories, slider_params)
        
        for traj_idx, traj in enumerate(inlier_trajectories):
            if len(traj.points) < 2:
                continue
            
            traj_weight = trajectory_weights[traj_idx]
            
            # Use first point as reference (closed state)
            reference_point = np.array([traj.points[0].x, traj.points[0].y, traj.points[0].z])
            
            for point in traj.points:
                current_point = np.array([point.x, point.y, point.z])
                
                # Calculate translation distance along direction
                displacement = current_point - reference_point
                distance = np.dot(displacement, slider_params.direction)
                
                # Use the actual frame number from Point3D.frame
                frame_idx = point.frame
                if frame_idx not in frame_translations:
                    frame_translations[frame_idx] = []
                frame_translations[frame_idx].append((distance, traj_weight))
        
        if len(frame_translations) == 0:
            print("Warning: No distances calculated, using default range")
            return SliderParameters(
                direction=slider_params.direction,
                reference_point=slider_params.reference_point,
                translation_min=0.0,
                translation_max=0.0
            ), {}
        
        # Aggregate translations per frame with outlier rejection and weighting
        per_frame_translations_raw = {}
        per_frame_stds = {}
        
        for frame_idx, trans_weight_pairs in frame_translations.items():
            translations = np.array([tw[0] for tw in trans_weight_pairs])
            weights = np.array([tw[1] for tw in trans_weight_pairs])
            
            # Remove outliers using IQR method
            translations_clean, weights_clean = self._remove_angle_outliers(translations, weights)
            
            if len(translations_clean) == 0:
                continue
            
            # Weighted average
            per_frame_translations_raw[frame_idx] = np.average(translations_clean, weights=weights_clean)
            per_frame_stds[frame_idx] = np.std(translations_clean)
        
        if len(per_frame_translations_raw) == 0:
            print("Warning: All translations filtered as outliers, using default range")
            return SliderParameters(
                direction=slider_params.direction,
                reference_point=slider_params.reference_point,
                translation_min=0.0,
                translation_max=0.0
            ), {}
        
        # Apply temporal smoothing
        per_frame_translations_smoothed = self._apply_temporal_smoothing(per_frame_translations_raw)
        
        # Zero-reference: find the minimum translation (closed state) and shift
        translation_offset = min(per_frame_translations_smoothed.values())
        
        # Shift all translations so minimum is at 0
        per_frame_translations = {
            frame_idx: trans - translation_offset
            for frame_idx, trans in per_frame_translations_smoothed.items()
        }
        
        # Calculate range
        translation_min = 0.0
        translation_max = max(per_frame_translations.values())
        
        # Calculate average uncertainty
        avg_std = np.mean(list(per_frame_stds.values())) if per_frame_stds else 0.0
        
        print(f"Slider range: {translation_min:.3f}m to {translation_max:.3f}m")
        print(f"Total range of motion: {translation_max:.3f}m")
        print(f"Average translation uncertainty: ±{avg_std:.3f}m")
        print(f"Per-frame translations computed for {len(per_frame_translations)} frames (sparse sampling)")
        print(f"Applied trajectory weighting and IQR outlier filtering")
        
        return SliderParameters(
            direction=slider_params.direction,
            reference_point=slider_params.reference_point,
            translation_min=translation_min,
            translation_max=translation_max
        ), per_frame_translations

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
        coverage_score = min(len(inlier_trajectories) / 20.0, 1.0)
        quality_scores.append(coverage_score)
        
        # Check temporal consistency (longer trajectories = better)
        avg_traj_length = np.mean([len(traj.points) for traj in inlier_trajectories])
        temporal_score = min(avg_traj_length / 10.0, 1.0)
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
            motion_score = min(avg_motion / 0.5, 1.0)
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
                          color='blue', s=200, marker='*', label='Reference')
        
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
    
    def plot_motion_over_time(self,
                             per_frame_values: Dict[int, float],
                             joint_type: JointType,
                             title: str = "Joint Motion Over Time") -> None:
        """
        Plot joint motion (angle or translation) over time.
        
        Args:
            per_frame_values: Dictionary mapping frame indices to motion values
            joint_type: Type of joint (for axis labeling)
            title: Plot title
        """
        if not per_frame_values:
            print("No per-frame values to plot")
            return
        
        # Sort by frame index
        frames = sorted(per_frame_values.keys())
        values = [per_frame_values[f] for f in frames]
        
        plt.figure(figsize=(10, 6))
        plt.plot(frames, values, 'b-', linewidth=2, marker='o', markersize=4)
        plt.grid(True, alpha=0.3)
        plt.xlabel('Frame Index')
        
        if joint_type == JointType.HINGE:
            # Convert to degrees for display
            values_deg = [np.degrees(v) for v in values]
            plt.plot(frames, values_deg, 'b-', linewidth=2, marker='o', markersize=4)
            plt.ylabel('Angle (degrees)')
            plt.title(f"{title}\nHinge Joint Rotation")
        elif joint_type == JointType.SLIDER:
            plt.ylabel('Translation (m)')
            plt.title(f"{title}\nSlider Joint Translation")
        
        plt.tight_layout()
        plt.show()


def process_joint_result(result: JointEstimationResult, 
                        smoothing_sigma: float = 1.0,
                        outlier_threshold: float = 1.5) -> Tuple[JointEstimationResult, dict]:
    """
    Convenience function for post-processing joint estimation results.
    
    Args:
        result: Raw estimation result
        smoothing_sigma: Gaussian smoothing parameter (default: 1.0)
        outlier_threshold: IQR multiplier for outlier detection (default: 1.5)
    """
    processor = PostProcessor(smoothing_sigma=smoothing_sigma, 
                            outlier_threshold=outlier_threshold)
    return processor.process_result(result)


def visualize_joint_result(result: JointEstimationResult, 
                          trajectories_3d: List[Trajectory3D],
                          title: str = "Joint Estimation Result") -> None:
    """Convenience function for visualizing joint estimation results."""
    processor = PostProcessor()
    processor.visualize_result(result, trajectories_3d, title)


def plot_joint_motion(per_frame_values: Dict[int, float],
                     joint_type: JointType,
                     title: str = "Joint Motion Over Time") -> None:
    """Convenience function for plotting motion over time."""
    processor = PostProcessor()
    processor.plot_motion_over_time(per_frame_values, joint_type, title)