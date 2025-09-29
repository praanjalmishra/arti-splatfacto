"""
RANSAC Core Implementation for Joint Estimation

This is the heart of the 4D RANSAC pipeline. It robustly fits joint models
to 3D trajectory data by iteratively:
1. Sampling minimal trajectory sets
2. Fitting joint hypotheses 
3. Counting consensus (inliers)
4. Selecting the best model
"""

import numpy as np
import random
from typing import List, Optional, Tuple
from tqdm import tqdm
import time

from data_structures import (
    Trajectory3D, RANSACConfig, JointType, ModelFitResult, 
    JointEstimationResult
)
from joint_model import JointModelBase, create_joint_models


class RANSACCore:
    """
    Core RANSAC implementation for joint estimation.
    
    This class implements the main RANSAC loop that robustly fits
    joint models to 3D trajectory data.
    """
    
    def __init__(self, config: RANSACConfig):
        """
        Initialize RANSAC core with configuration.
        
        Args:
            config: RANSAC configuration parameters
        """
        self.config = config
        self.joint_models = create_joint_models()
        
        # Set random seed for reproducible results (optional)
        random.seed(42)
        np.random.seed(42)
    
    def estimate_joint(self, trajectories_3d: List[Trajectory3D]) -> JointEstimationResult:
        """
        Main entry point - estimate joint parameters from 3D trajectories.
        
        Args:
            trajectories_3d: List of 3D trajectories from CoTracker processing
            
        Returns:
            JointEstimationResult with best joint model and parameters
        """
        print(f"=== Starting RANSAC Joint Estimation ===")
        print(f"Input: {len(trajectories_3d)} trajectories")
        
        start_time = time.time()
        
        # # Filter trajectories by minimum length
        valid_trajectories = self._filter_trajectories(trajectories_3d)

        # # Step 0: Pre-cluster trajectories with TMS
        # clusters = self._build_tms_clusters(trajectories_3d)
        # representatives = [self._choose_representative(c) for c in clusters]

        # # Step 1: Filter representatives
        # valid_trajectories = self._filter_trajectories(representatives)

        print(f"After filtering: {len(valid_trajectories)} valid trajectories")
        
        if len(valid_trajectories) < self.config.min_inliers:
            return JointEstimationResult(
                success=False,
                joint_type=JointType.UNKNOWN,
                parameters=None,
                confidence=0.0,
                inlier_trajectories=[],
                total_trajectories=len(trajectories_3d),
                processing_time=time.time() - start_time,
                error_message="Insufficient valid trajectories"
            )
        
        # Try each joint type
        best_result = None
        best_consensus_count = 0
        
        for joint_type, joint_model in self.joint_models.items():
            print(f"\n--- Testing {joint_type.value.upper()} model ---")
            
            result = self._fit_joint_model(joint_model, valid_trajectories)
            
            if result and result.inlier_count > best_consensus_count:
                best_result = result
                best_consensus_count = result.inlier_count
                print(f"New best model: {joint_type.value} with {result.inlier_count} inliers")
        
        processing_time = time.time() - start_time
        
        if best_result is None or best_result.inlier_count < self.config.min_inliers:
            return JointEstimationResult(
                success=False,
                joint_type=JointType.UNKNOWN,
                parameters=None,
                confidence=0.0,
                inlier_trajectories=[],
                total_trajectories=len(trajectories_3d),
                processing_time=processing_time,
                error_message="No suitable joint model found"
            )
        
        print(f"\n=== RANSAC Complete ===")
        print(f"Best model: {best_result.joint_type.value}")
        print(f"Inliers: {best_result.inlier_count}/{len(valid_trajectories)}")
        print(f"Confidence: {best_result.confidence:.3f}")
        print(f"Processing time: {processing_time:.2f}s")
        
        return JointEstimationResult(
            success=True,
            joint_type=best_result.joint_type,
            parameters=best_result.parameters,
            confidence=best_result.confidence,
            inlier_trajectories=best_result.inlier_trajectories,
            total_trajectories=len(trajectories_3d),
            processing_time=processing_time
        )

    # def _build_tms_clusters(self,
    #                         trajectories: List[Trajectory3D],
    #                         n_rtm: int = 100,
    #                         eps_r: float = 0.01,
    #                         jaccard_thresh: float = 0.7):
    #     """
    #     Build trajectory clusters using Trajectory-Model Signatures (TMS).
    #     Each trajectory gets a binary signature vector describing which
    #     rigid trajectory models (RTMs) it supports. Then cluster trajectories
    #     based on Jaccard similarity of signatures.

    #     Args:
    #         trajectories: List of trajectories
    #         n_rtm: Number of RTM candidates to generate
    #         eps_r: Residual error threshold for agreement
    #         jaccard_thresh: Jaccard distance threshold for clustering

    #     Returns:
    #         List of clusters, each a list of trajectories
    #     """
    #     if len(trajectories) < 2:
    #         return [[t] for t in trajectories]

    #     # --- Step 1: Generate RTM candidates ---
    #     rtms = []
    #     for _ in range(min(n_rtm, len(trajectories) // 2)):
    #         t1, t2 = random.sample(trajectories, 2)
    #         p1a = np.array([t1.points[0].x, t1.points[0].y, t1.points[0].z])
    #         p1b = np.array([t1.points[-1].x, t1.points[-1].y, t1.points[-1].z])

    #         p2a = np.array([t2.points[0].x, t2.points[0].y, t2.points[0].z])
    #         p2b = np.array([t2.points[-1].x, t2.points[-1].y, t2.points[-1].z])
    #         try:
    #             R, t = self.joint_models[JointType.HINGE]._estimate_rigid_transform_kabsch(
    #                 np.array([p1a, p2a]), np.array([p1b, p2b])
    #             )
    #             rtms.append((R, t))
    #         except Exception:
    #             continue

    #     if not rtms:
    #         return [[t] for t in trajectories]

    #     # --- Step 2: Compute TMS vectors ---
    #     signatures = {}
    #     for traj in trajectories:
    #         pts = traj.get_all_positions()
    #         sig = []
    #         for (R, t) in rtms:
    #             if len(pts) < 2:
    #                 sig.append(0)
    #                 continue
    #             pred = (R @ pts[:-1].T).T + t
    #             res = np.mean(np.linalg.norm(pred - pts[1:], axis=1))
    #             sig.append(1 if res < eps_r else 0)
    #         signatures[traj.track_id] = np.array(sig, dtype=np.uint8)

    #     # --- Step 3: Cluster using Jaccard similarity ---
    #     unvisited = set(signatures.keys())
    #     clusters = []

    #     while unvisited:
    #         seed = unvisited.pop()
    #         cluster = [seed]
    #         sig_seed = signatures[seed]

    #         to_check = list(unvisited)
    #         for other in to_check:
    #             sig_other = signatures[other]
    #             inter = np.sum(np.logical_and(sig_seed, sig_other))
    #             union = np.sum(np.logical_or(sig_seed, sig_other))
    #             d_j = 1.0 if union == 0 else 1 - inter / union
    #             if d_j < jaccard_thresh:
    #                 cluster.append(other)
    #                 unvisited.remove(other)

    #         clusters.append([traj for traj in trajectories if traj.track_id in cluster])

    #     return clusters

    # def _choose_representative(self, cluster: List[Trajectory3D]) -> Trajectory3D:
    #     """
    #     Pick representative trajectory from a cluster.
    #     Strategy: choose the longest trajectory (most frames).
    #     """
    #     return max(cluster, key=lambda t: len(t.points))
    
    def _filter_trajectories(self, trajectories: List[Trajectory3D]) -> List[Trajectory3D]:
        """Filter trajectories based on minimum length"""
        return [
            traj for traj in trajectories 
            if len(traj.points) >= self.config.min_trajectory_length
        ]
    

    def _fit_joint_model(self, 
                        joint_model: JointModelBase, 
                        trajectories: List[Trajectory3D]) -> Optional[ModelFitResult]:
        """
        Fit a specific joint model using RANSAC.
        
        Args:
            joint_model: The joint model to fit (hinge, slider, etc.)
            trajectories: Valid 3D trajectories
            
        Returns:
            ModelFitResult with best parameters and inliers, or None if failed
        """
        if len(trajectories) < joint_model.minimal_sample_size():
            print(f"Not enough trajectories for {joint_model.get_joint_type().value} model")
            return None
        
        best_inliers = []
        best_params = None
        best_inlier_count = 0
        
        iterations_without_improvement = 0
        max_no_improvement = 50  # Early termination
        
        print(f"Running RANSAC for {self.config.max_iterations} iterations...")
        
        for iteration in tqdm(range(self.config.max_iterations)):
            # Step 1: Sample minimal set
            sample = self._sample_trajectories(trajectories, joint_model.minimal_sample_size())
            
            # Step 2: Fit model to sample
            try:
                params = joint_model.fit_from_sample(sample)
                if params is None:
                    continue
            except Exception as e:
                # Fitting failed - continue to next iteration
                continue
            
            # Step 3: Count consensus (inliers)
            inliers = []
            errors = []
            for traj in trajectories:
                try:
                    error = joint_model.calculate_trajectory_error(traj, params)
                    errors.append(error)
                    if error <= self.config.error_threshold:
                        inliers.append(traj)
                except Exception as e:
                    # Error calculation failed
                    errors.append(float('inf'))
                    continue
            
            inlier_count = len(inliers)
            
            # Debug: Print some statistics every 50 iterations
            if iteration % 50 == 0:
                valid_errors = [e for e in errors if np.isfinite(e)]
                if valid_errors:
                    min_error = min(valid_errors)
                    avg_error = np.mean(valid_errors)
                    print(f"  Iter {iteration}: {inlier_count} inliers, "
                          f"min_error={min_error:.4f}, avg_error={avg_error:.4f}")
                else:
                    print(f"  Iter {iteration}: No valid error calculations")
            
            # Step 4: Check if this is the best model so far
            if inlier_count > best_inlier_count:
                best_inlier_count = inlier_count
                best_inliers = inliers
                best_params = params
                iterations_without_improvement = 0
                
                # Early termination if we have very good consensus
                consensus_ratio = inlier_count / len(trajectories)
                if consensus_ratio >= self.config.early_termination_threshold:
                    print(f"Early termination at iteration {iteration} with {consensus_ratio:.3f} consensus")
                    break
            else:
                iterations_without_improvement += 1
            
            # Early termination if no improvement for a while
            if iterations_without_improvement >= max_no_improvement:
                print(f"Early termination: no improvement for {max_no_improvement} iterations")
                break
        
        if best_inlier_count < self.config.min_inliers:
            print(f"Insufficient inliers: {best_inlier_count} < {self.config.min_inliers}")
            return None
        
        # Refine parameters using all inliers
        print(f"Refining parameters with {best_inlier_count} inliers...")
        refined_params = joint_model.refine_parameters(best_inliers, best_params)
        
        # Calculate final fit error
        errors = []
        for traj in best_inliers:
            error = joint_model.calculate_trajectory_error(traj, refined_params)
            errors.append(error)
        
        avg_error = np.mean(errors)
        consensus_score = best_inlier_count / len(trajectories)
        
        print(f"Final result: {best_inlier_count} inliers, avg error: {avg_error:.4f}m")
        
        return ModelFitResult(
            joint_type=joint_model.get_joint_type(),
            parameters=refined_params,
            inlier_trajectories=best_inliers,
            inlier_count=best_inlier_count,
            total_trajectories=len(trajectories),
            fit_error=avg_error,
            consensus_score=consensus_score
        )
    
    def _sample_trajectories(self, 
                           trajectories: List[Trajectory3D], 
                           sample_size: int) -> List[Trajectory3D]:
        """
        Randomly sample trajectories for hypothesis generation.
        
        Args:
            trajectories: All available trajectories
            sample_size: Number of trajectories to sample
            
        Returns:
            Random sample of trajectories
        """
        if sample_size >= len(trajectories):
            return trajectories.copy()
        
        # Prefer trajectories from different rigid parts if available
        moving_trajectories = [t for t in trajectories if t.rigid_part == 1]
        static_trajectories = [t for t in trajectories if t.rigid_part == 0]
        
        sample = []
        
        # Try to get at least one from each rigid part for joint models
        if len(moving_trajectories) > 0 and len(static_trajectories) > 0 and sample_size >= 2:
            # Include at least one moving and one static trajectory
            sample.append(random.choice(moving_trajectories))
            remaining_sample_size = sample_size - 1
            
            # Fill remaining slots from all trajectories
            remaining_trajectories = [t for t in trajectories if t not in sample]
            sample.extend(random.sample(remaining_trajectories, 
                                      min(remaining_sample_size, len(remaining_trajectories))))
        else:
            # Random sampling from all trajectories
            sample = random.sample(trajectories, sample_size)
        
        return sample


# Convenience function for external use
def estimate_joint_from_trajectories(trajectories_3d: List[Trajectory3D],
                                   config: RANSACConfig) -> JointEstimationResult:
    """
    Convenience function to estimate joint parameters from 3D trajectories.
    
    Args:
        trajectories_3d: List of 3D trajectories
        config: RANSAC configuration
        
    Returns:
        JointEstimationResult with best joint model
    """
    ransac_core = RANSACCore(config)
    return ransac_core.estimate_joint(trajectories_3d)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from data_structures import Point3D, Trajectory3D, RANSACConfig, JointType

    # --- Helpers to generate synthetic motion ---
    def make_hinge_trajectories(axis, pivot, angles, n_points=30, noise=0.01):
        trajectories = []
        axis = axis / np.linalg.norm(axis)
        for i in range(n_points):
            base_point = pivot + np.random.randn(3)  # random offset
            points = []
            for angle in angles:
                # Rodrigues rotation
                K = np.array([
                    [0, -axis[2], axis[1]],
                    [axis[2], 0, -axis[0]],
                    [-axis[1], axis[0], 0]
                ])
                R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
                pt = R @ (base_point - pivot) + pivot
                pt += noise * np.random.randn(3)
                points.append(Point3D(frame=len(points), x=pt[0], y=pt[1], z=pt[2]))
            trajectories.append(Trajectory3D(track_id=i, points=points))
        return trajectories

    def make_slider_trajectories(direction, start_point, steps, n_points=10, noise=0.01):
        trajectories = []
        direction = direction / np.linalg.norm(direction)
        for i in range(n_points):
            offset = np.random.randn(3) * 0.2
            base_point = start_point + offset
            points = []
            for step in steps:
                pt = base_point + step * direction
                pt += noise * np.random.randn(3)
                points.append(Point3D(frame=len(points), x=pt[0], y=pt[1], z=pt[2]))
            trajectories.append(Trajectory3D(track_id=i, points=points))
        return trajectories

    # --- Config for RANSAC ---
    config = RANSACConfig(
        max_iterations=200,
        error_threshold=0.2,
        min_inliers=3,
        min_trajectory_length=5,
        early_termination_threshold=0.9
    )

    # --- Choose test case ---
    test_case = "hinge"  # "slider" also possible

    if test_case == "hinge":
        print("\n=== Testing Hinge Joint with RANSAC ===")
        true_axis = np.array([0, 0, 1])
        true_pivot = np.array([0, 0, 0])
        angles = np.linspace(0, np.pi/4, 12)
        trajectories = make_hinge_trajectories(true_axis, true_pivot, angles)
    else:
        print("\n=== Testing Slider Joint with RANSAC ===")
        true_dir = np.array([1, 0, 0])
        start = np.array([0, 0, 0])
        steps = np.linspace(0, 1, 12)
        trajectories = make_slider_trajectories(true_dir, start, steps)

    # --- Run RANSAC ---
    result = estimate_joint_from_trajectories(trajectories, config)

    if result.success:
        print(f"\n RANSAC Success: {result.joint_type.value}")
        print("Parameters:", result.parameters)
        print(f"Inliers: {len(result.inlier_trajectories)} / {len(trajectories)}")
    else:
        print(f"\nRANSAC Failed: {result.error_message}")

    # --- Visualization ---
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for traj in trajectories:
        pts = traj.get_all_positions()
        color = "g" if traj in result.inlier_trajectories else "r"
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=color, alpha=0.6)

    if result.success:
        if result.joint_type == JointType.HINGE:
            pivot = result.parameters.pivot
            axis = result.parameters.axis
            line_pts = np.array([pivot - 2 * axis, pivot + 2 * axis])
            ax.plot(line_pts[:, 0], line_pts[:, 1], line_pts[:, 2], "b--", lw=2, label="Estimated axis")
        elif result.joint_type == JointType.SLIDER:
            ref_pt = result.parameters.reference_point
            dir_vec = result.parameters.direction
            line_pts = np.array([ref_pt - 2 * dir_vec, ref_pt + 2 * dir_vec])
            ax.plot(line_pts[:, 0], line_pts[:, 1], line_pts[:, 2], "b--", lw=2, label="Estimated direction")

    ax.legend()
    ax.set_title(f"RANSAC {result.joint_type.value if result.success else 'FAILED'}")
    plt.show()
