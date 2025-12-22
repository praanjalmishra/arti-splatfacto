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

from joint_estimator.data_structures import (
    Trajectory3D, RANSACConfig, JointType, ModelFitResult, 
    JointEstimationResult
)
from joint_estimator.joint_model import JointModelBase, create_joint_models



class RANSACCore:
    """
    Core RANSAC implementation for joint estimation.
    
    This class implements the main RANSAC loop that robustly fits
    joint models to 3D trajectory data, with optional TMS pre-clustering.
    """
    
    def __init__(self, config: RANSACConfig):
        """
        Initialize RANSAC core with configuration.
        
        Args:
            config: RANSAC configuration parameters
        """
        self.config = config
        self.joint_models = create_joint_models()
        
        # Set random seed for reproducible results
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
        print(f"TMS pre-clustering: {'ENABLED' if self.config.use_tms_preclustering else 'DISABLED'}")
        
        start_time = time.time()
        
        # Step 1: Filter trajectories by minimum length
        valid_trajectories = self._filter_trajectories(trajectories_3d)
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
        
        # Step 2: Optional TMS clustering
        if self.config.use_tms_preclustering:
            clusters = self._build_tms_clusters(valid_trajectories)
            representatives = [self._choose_representative(c) for c in clusters]
            print(f"Using {len(representatives)} cluster representatives for RANSAC")
            ransac_input = representatives
        else:
            clusters = None
            ransac_input = valid_trajectories
        
        # Step 3: Try each joint type
        best_result = None
        best_consensus_count = 0
        
        for joint_type, joint_model in self.joint_models.items():
            print(f"\n--- Testing {joint_type.value.upper()} model ---")
            
            result = self._fit_joint_model(joint_model, ransac_input)
            
            if result:
                # If using TMS, expand inliers to full clusters
                if self.config.use_tms_preclustering and clusters:
                    result = self._expand_to_full_clusters(result, clusters, joint_model)
                
                if (best_result is None or
                    result.inlier_count > best_result.inlier_count or
                    (result.inlier_count == best_result.inlier_count and 
                    result.fit_error < best_result.fit_error)):
                    
                    best_result = result
                    print(f"New best model: {joint_type.value} with {result.inlier_count} inliers, "
                        f"avg error={result.fit_error:.4f}")

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

    def _build_tms_clusters(self,
                            trajectories: List[Trajectory3D],
                            n_rtm: int = 40,
                            eps_r: float = 0.03,
                            jaccard_thresh: float = 0.4) -> List[List[Trajectory3D]]:
        """
        Build trajectory clusters using Trajectory-Model Signatures (TMS).
        
        Algorithm:
        1. Generate N random Rigid Trajectory Models (RTMs) from trajectory pairs
        2. Build binary signature for each trajectory (which RTMs it supports)
        3. Cluster trajectories by Jaccard similarity of signatures
        
        Args:
            trajectories: List of 3D trajectories
            n_rtm: Number of RTM candidates to generate
            eps_r: Residual error threshold for RTM agreement (meters)
            jaccard_thresh: Jaccard distance threshold for clustering
        
        Returns:
            List of clusters, each containing trajectories with similar rigid motion
        """
        if len(trajectories) < 2:
            return [[t] for t in trajectories]
        
        print(f"[TMS] Building clusters from {len(trajectories)} trajectories...")
        
        # --- Step 1: Generate RTM candidates (rigid transformations) ---
        rtms = []
        attempts = 0
        max_attempts = n_rtm * 3
        
        while len(rtms) < n_rtm and attempts < max_attempts:
            attempts += 1
            
            # Sample two trajectories with sufficient length
            candidates = [t for t in trajectories if len(t.points) >= 3]
            if len(candidates) < 2:
                break
                
            t1, t2 = random.sample(candidates, 2)
            
            # Get start and end positions for each trajectory
            try:
                p1_start = np.array([t1.points[0].x, t1.points[0].y, t1.points[0].z])
                p1_end = np.array([t1.points[-1].x, t1.points[-1].y, t1.points[-1].z])
                
                p2_start = np.array([t2.points[0].x, t2.points[0].y, t2.points[0].z])
                p2_end = np.array([t2.points[-1].x, t2.points[-1].y, t2.points[-1].z])
                
                # Estimate rigid transformation from start to end frames
                points_start = np.array([p1_start, p2_start])
                points_end = np.array([p1_end, p2_end])
                
                # Use Kabsch algorithm (from HingeJointModel)
                R, t = self.joint_models[JointType.HINGE]._estimate_rigid_transform_kabsch(
                    points_start, points_end
                )
                
                # Validate: check if transformation is reasonable
                # Reject if rotation is too extreme or translation too large
                rot_angle = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
                trans_dist = np.linalg.norm(t)
                
                if rot_angle < np.pi and trans_dist < 1.0:  # Reasonable motion
                    rtms.append((R, t))
                    
            except Exception:
                continue
        
        if len(rtms) == 0:
            print("[TMS] Failed to generate RTMs, returning single cluster")
            return [trajectories]
        
        print(f"[TMS] Generated {len(rtms)} valid RTMs")
        
        # --- Step 2: Compute TMS signature vectors ---
        signatures = {}
        
        for traj in trajectories:
            pts = traj.get_all_positions()
            if len(pts) < 2:
                signatures[traj.track_id] = np.zeros(len(rtms), dtype=np.uint8)
                continue
            
            sig = []
            for (R, t) in rtms:
                # Test if this RTM explains the trajectory motion
                # Predict positions using RTM and compare to actual
                predicted = (R @ pts[:-1].T).T + t
                residuals = np.linalg.norm(predicted - pts[1:], axis=1)
                avg_residual = np.mean(residuals)
                
                # Mark as 1 if RTM explains this trajectory well
                sig.append(1 if avg_residual < eps_r else 0)
            
            signatures[traj.track_id] = np.array(sig, dtype=np.uint8)
        
        # --- Step 3: Cluster using Jaccard similarity ---
        unvisited = set(signatures.keys())
        clusters = []
        
        while unvisited:
            seed_id = unvisited.pop()
            cluster = [seed_id]
            sig_seed = signatures[seed_id]
            
            to_check = list(unvisited)
            for other_id in to_check:
                sig_other = signatures[other_id]
                
                # Compute Jaccard distance: 1 - (intersection / union)
                inter = np.sum(np.logical_and(sig_seed, sig_other))
                union = np.sum(np.logical_or(sig_seed, sig_other))
                
                jaccard_dist = 1.0 if union == 0 else 1.0 - (inter / union)
                
                if jaccard_dist < jaccard_thresh:
                    cluster.append(other_id)
                    unvisited.remove(other_id)
            
            # Convert track IDs back to trajectory objects
            cluster_trajs = [t for t in trajectories if t.track_id in cluster]
            if cluster_trajs:
                clusters.append(cluster_trajs)
        
        print(f"[TMS] Found {len(clusters)} rigid motion clusters")
        for i, cluster in enumerate(clusters):
            print(f"  Cluster {i}: {len(cluster)} trajectories")
        
        return clusters

    def _choose_representative(self, cluster: List[Trajectory3D]) -> Trajectory3D:
        """
        Pick representative trajectory from a cluster.
        Strategy: choose the longest trajectory (most temporal coverage).
        """
        return max(cluster, key=lambda t: len(t.points))

    def _expand_to_full_clusters(self, 
                                 result: ModelFitResult,
                                 clusters: List[List[Trajectory3D]],
                                 joint_model: JointModelBase) -> ModelFitResult:
        """
        Expand inliers from cluster representatives to all cluster members.
        
        After RANSAC finds good representatives, include all trajectories
        from the same clusters if they also fit the model well.
        """
        # Find which clusters are represented in inliers
        inlier_track_ids = {t.track_id for t in result.inlier_trajectories}
        active_clusters = []
        
        for cluster in clusters:
            # Check if any representative from this cluster is an inlier
            cluster_track_ids = {t.track_id for t in cluster}
            if cluster_track_ids & inlier_track_ids:  # Intersection
                active_clusters.append(cluster)
        
        # Test all trajectories from active clusters
        expanded_inliers = []
        errors = []
        
        for cluster in active_clusters:
            for traj in cluster:
                try:
                    error = joint_model.calculate_trajectory_error(traj, result.parameters)
                    if error <= self.config.error_threshold:
                        expanded_inliers.append(traj)
                        errors.append(error)
                except Exception:
                    continue
        
        if not expanded_inliers:
            return result  # Fallback to original
        
        avg_error = np.mean(errors)
        
        print(f"[Expansion] {len(result.inlier_trajectories)} representatives → {len(expanded_inliers)} full trajectories")
        
        return ModelFitResult(
            joint_type=result.joint_type,
            parameters=result.parameters,
            inlier_trajectories=expanded_inliers,
            inlier_count=len(expanded_inliers),
            total_trajectories=result.total_trajectories,
            fit_error=avg_error,
            consensus_score=len(expanded_inliers) / result.total_trajectories
        )
    
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
        
        # --- Refinement Phase ---
        print(f"Refining parameters with {best_inlier_count} inliers...")

        try:
            refined_params = joint_model.refine_parameters(best_inliers, best_params)
        except Exception as e:
            print(f"[WARN] Refinement threw an exception: {e}")
            refined_params = best_params

        # Validate refined parameters
        def params_valid(p):
            if p is None:
                return False
            arrs = []
            for val in vars(p).values():
                if isinstance(val, (list, tuple, np.ndarray)):
                    arrs.append(np.asarray(val, dtype=float))
                elif isinstance(val, (float, int)):
                    arrs.append(np.array([val], dtype=float))
            return all(np.all(np.isfinite(a)) for a in arrs)

        if not params_valid(refined_params):
            print("[WARN] Refinement produced invalid (NaN/Inf) parameters; reverting to pre-refinement params.")
            refined_params = best_params

        # --- Compute final error robustly ---
        errors = []
        for traj in best_inliers:
            try:
                e = joint_model.calculate_trajectory_error(traj, refined_params)
                if np.isfinite(e):
                    errors.append(e)
            except Exception:
                continue

        if len(errors) == 0:
            avg_error = float('inf')
        else:
            avg_error = np.mean(errors)

        if not np.isfinite(avg_error):
            print("[WARN] Final avg error non-finite; rejecting this model.")
            return None

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
