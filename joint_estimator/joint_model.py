"""
Joint Model Implementations for 4D RANSAC Pipeline

This module defines the geometric models for different joint types (hinge, slider, etc.)
and provides methods for:
- Fitting joint parameters from trajectory samples
- Calculating how well trajectories match joint models
- Refining joint parameters using all inlier trajectories
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from data_structures import (
    Trajectory3D, Point3D, HingeParameters, SliderParameters,
    JointType, ModelFitResult
)


class JointModelBase(ABC):
    """
    Abstract base class for all joint models.
    
    Each joint type (hinge, slider, etc.) implements this interface
    to provide RANSAC with consistent fitting and error calculation methods.
    """
    
    @abstractmethod
    def get_joint_type(self) -> JointType:
        """Return the type of joint this model represents."""
        pass
    
    @abstractmethod
    def minimal_sample_size(self) -> int:
        """Return minimum number of trajectory points needed to fit this model."""
        pass
    
    @abstractmethod
    def fit_from_sample(self, trajectory_sample: List[Trajectory3D]) -> Optional[object]:
        """
        Fit joint parameters from a minimal sample of trajectories.
        
        Args:
            trajectory_sample: Minimal set of trajectories to fit model
            
        Returns:
            Joint parameters object (HingeParameters, SliderParameters, etc.)
            or None if fitting failed
        """
        pass
    
    @abstractmethod
    def calculate_trajectory_error(self, trajectory: Trajectory3D, joint_params: object) -> float:
        """
        Calculate how well a trajectory fits the joint model.
        
        Args:
            trajectory: 3D trajectory to evaluate
            joint_params: Joint parameters from fit_from_sample
            
        Returns:
            RMS error in meters (lower is better fit)
        """
        pass
    
    @abstractmethod
    def refine_parameters(self, 
                         inlier_trajectories: List[Trajectory3D], 
                         initial_params: object) -> object:
        """
        Refine joint parameters using all inlier trajectories.
        
        Args:
            inlier_trajectories: All trajectories that fit the model
            initial_params: Initial parameters from minimal sample
            
        Returns:
            Refined joint parameters
        """
        pass
    
    def validate_sample(self, trajectory_sample: List[Trajectory3D]) -> bool:
        """
        Check if trajectory sample is valid for fitting.
        
        Args:
            trajectory_sample: Trajectories to validate
            
        Returns:
            True if sample can be used for fitting
        """
        if len(trajectory_sample) < self.minimal_sample_size():
            return False
        
        # Check that trajectories have sufficient length and temporal overlap
        for traj in trajectory_sample:
            if len(traj.points) < 2:
                return False
        
        return True



class HingeJointModel(JointModelBase):
    """
    Model for hinge/revolute joints.
    
    A hinge joint rotates around a fixed axis. Points on the moving part
    trace circular arcs around this axis.
    """
    
    def get_joint_type(self) -> JointType:
        return JointType.HINGE
    
    def minimal_sample_size(self) -> int:
        # Need at least 2 trajectories with 2 time points each
        # This gives us 4 point correspondences to solve for rotation + translation
        return 2
    
    def fit_from_sample(self, trajectory_sample: List[Trajectory3D]) -> Optional[HingeParameters]:
        """
        Fit hinge parameters using the Mobility Fitting approach from Li & Wan 2016.
        
        Algorithm (following paper exactly):
        1. Extract rigid transformation M=(R,t) between two frames
        2. Hinge axis = eigenvector of R with eigenvalue 1
        3. Hinge pivot c from: (I-R)c = t
        """
        if not self.validate_sample(trajectory_sample):
            return None
        
        try:
            # Use first trajectory with sufficient length
            traj = None
            for t in trajectory_sample:
                if len(t.points) >= 2:
                    traj = t
                    break
            
            if traj is None:
                return None
            
            # Get two frames from trajectory (first and last for maximum motion)
            points_i = np.array([traj.points[0].x, traj.points[0].y, traj.points[0].z])
            points_j = np.array([traj.points[-1].x, traj.points[-1].y, traj.points[-1].z])
            
            # For multiple trajectories, collect point correspondences
            if len(trajectory_sample) > 1:
                points_t1 = []
                points_t2 = []
                
                for t in trajectory_sample:
                    if len(t.points) >= 2:
                        p1 = np.array([t.points[0].x, t.points[0].y, t.points[0].z])
                        p2 = np.array([t.points[-1].x, t.points[-1].y, t.points[-1].z])
                        points_t1.append(p1)
                        points_t2.append(p2)
                
                if len(points_t1) < 2:
                    return None
                    
                points_t1 = np.array(points_t1)
                points_t2 = np.array(points_t2)
            else:
                # Single trajectory - create artificial correspondence
                points_t1 = points_i.reshape(1, -1)
                points_t2 = points_j.reshape(1, -1)
            
            # Estimate rigid transformation M = (R, t) using Procrustes/Kabsch
            R, t = self._estimate_rigid_transform_kabsch(points_t1, points_t2)
            
            # Extract hinge axis (eigenvector of R with eigenvalue 1)
            axis = self._extract_rotation_axis(R)
            if axis is None:
                return None
            
            # Find pivot point: (I - R)c = t
            pivot = self._solve_pivot_point(R, t)
            
            return HingeParameters(axis=axis, pivot=pivot)
            
        except Exception as e:
            return None
    
    def _estimate_rigid_transform_kabsch(self, points_t1: np.ndarray, points_t2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate rigid transformation using Kabsch algorithm (SVD-based Procrustes).
        Following Li & Wan 2016 equation: M = argmin_{R,t} Σ ||p_i^b - (Rp_i^a + t)||^2
        
        Returns:
            R: 3x3 rotation matrix
            t: 3D translation vector
        """
        # Center the point sets
        centroid_1 = np.mean(points_t1, axis=0)
        centroid_2 = np.mean(points_t2, axis=0)
        
        centered_1 = points_t1 - centroid_1
        centered_2 = points_t2 - centroid_2
        
        # Compute cross-covariance matrix H
        H = centered_1.T @ centered_2
        
        # SVD decomposition
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        
        # Ensure proper rotation matrix (det(R) = 1)
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        
        # Compute translation
        t = centroid_2 - R @ centroid_1
        
        return R, t
    
    def _extract_rotation_axis(self, R: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract rotation axis from rotation matrix.
        Following Li & Wan 2016: "hinge direction a is the eigenvector of R with eigenvalue 1"
        """
        try:
            # Find eigenvector corresponding to eigenvalue 1
            eigenvalues, eigenvectors = np.linalg.eig(R)
            
            # Find eigenvalue closest to 1
            closest_to_1 = np.argmin(np.abs(eigenvalues - 1.0))
            
            if np.abs(eigenvalues[closest_to_1] - 1.0) > 0.1:
                # Not a good rotation matrix for hinge
                return None
            
            axis = eigenvectors[:, closest_to_1].real
            
            # Ensure it's a unit vector
            axis = axis / np.linalg.norm(axis)
            
            return axis
            
        except Exception:
            return None
    
    def _solve_pivot_point(self, R: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        Solve for pivot point using Li & Wan 2016 equation: (I - R)c = t
        """
        I = np.eye(3)
        A = I - R
        
        # Solve linear system (I - R)c = t
        try:
            # Use least squares since the system may be under-constrained
            pivot, residuals, rank, s = np.linalg.lstsq(A, t, rcond=None)
            return pivot
        except np.linalg.LinAlgError:
            # If direct solution fails, find minimum norm solution
            try:
                pivot = np.linalg.pinv(A) @ t
                return pivot
            except:
                # Fallback: use translation centroid
                return t / 2
    
    def _rotation_matrix_to_axis_angle(self, R: np.ndarray) -> Tuple[np.ndarray, float]:
        """Convert rotation matrix to axis-angle representation."""
        # Use scipy for robust conversion
        rot = Rotation.from_matrix(R)
        axis_angle = rot.as_rotvec()
        
        if np.linalg.norm(axis_angle) < 1e-6:
            # No significant rotation - return arbitrary axis
            return np.array([0, 0, 1]), 0.0
        
        angle = np.linalg.norm(axis_angle)
        axis = axis_angle / angle
        
        return axis, angle
    
    def _find_pivot_point(self, 
                         p1: np.ndarray, 
                         p2: np.ndarray, 
                         axis: np.ndarray, 
                         R: np.ndarray) -> np.ndarray:
        """
        Find pivot point on rotation axis - simplified approach.
        
        For hinge motion, the pivot is approximately the midpoint between
        the centroids of the two point sets, projected onto the rotation axis.
        """
        # Simple approach: use midpoint between centroids
        midpoint = (p1 + p2) / 2
        
        # Project midpoint onto rotation axis
        # Find point on axis closest to midpoint
        # If axis passes through origin: pivot = (midpoint · axis) * axis
        
        # More general: assume axis passes through the centroid of all points
        axis_origin = midpoint  # Simple assumption
        
        # The pivot is just the centroid for this simplified version
        return axis_origin
    
    def _rotate_point_around_axis(self, 
                                 point: np.ndarray, 
                                 pivot: np.ndarray, 
                                 axis: np.ndarray, 
                                 angle: float) -> np.ndarray:
        """Rotate a point around an axis by given angle."""
        # Translate so pivot is at origin
        translated = point - pivot
        
        # Create rotation matrix using Rodrigues' formula
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
        
        # Rotate and translate back
        rotated = R @ translated + pivot
        return rotated
    
    def calculate_trajectory_error(self, trajectory: Trajectory3D, joint_params: HingeParameters) -> float:
        """
        Calculate fitting error following Li & Wan 2016 Equation 11:
        D_h(f_i, f_k) = min_θ ||M_{h,θ} p_i - p_k||
        
        Key insight: For hinge motion, we test if 80% of trajectory lifespan 
        can be explained by rotation around the hinge axis.
        """
        if len(trajectory.points) < 2:
            return float('inf')
        
        points = trajectory.get_all_positions()
        reference_point = points[0]  # Use first frame as reference
        
        valid_fits = 0
        total_frames = len(points) - 1  # Exclude reference frame
        errors = []
        
        for i, current_point in enumerate(points[1:], 1):
            # Find optimal rotation angle θ that minimizes ||M_{h,θ} p_i - p_k||
            min_error = self._find_optimal_hinge_angle(
                reference_point, current_point, 
                joint_params.axis, joint_params.pivot
            )
            
            errors.append(min_error)
            
            # Following paper: point supports hinge if error < threshold
            if min_error <= 0.05:  # εh = 0.05 from paper
                valid_fits += 1
        
        # Following paper: "discard hinge if supporting points < 80% of lifespan"
        support_ratio = valid_fits / total_frames if total_frames > 0 else 0
        
        if support_ratio < 0.8:
            return float('inf')  # Reject this hinge
        
        # Return average error for valid trajectory
        return np.mean(errors)
    
    def _find_optimal_hinge_angle(self, 
                                 p_reference: np.ndarray,
                                 p_current: np.ndarray, 
                                 axis: np.ndarray, 
                                 pivot: np.ndarray) -> float:
        """
        Find rotation angle θ that minimizes ||M_{h,θ} p_reference - p_current||
        This implements the min_θ part of Equation 11.
        """
        # Translate points so pivot is at origin
        p_ref_centered = p_reference - pivot
        p_cur_centered = p_current - pivot
        
        # Project points onto plane perpendicular to rotation axis
        p_ref_proj = p_ref_centered - np.dot(p_ref_centered, axis) * axis
        p_cur_proj = p_cur_centered - np.dot(p_cur_centered, axis) * axis
        
        # Check if both points are essentially on the axis (no rotation possible)
        if np.linalg.norm(p_ref_proj) < 1e-6 or np.linalg.norm(p_cur_proj) < 1e-6:
            # Points are on the rotation axis - just check distance consistency
            return np.linalg.norm(p_cur_centered - p_ref_centered)
        
        # Find optimal rotation angle between projected vectors
        cos_angle = np.dot(p_ref_proj, p_cur_proj) / (
            np.linalg.norm(p_ref_proj) * np.linalg.norm(p_cur_proj)
        )
        cos_angle = np.clip(cos_angle, -1, 1)
        optimal_angle = np.arccos(cos_angle)
        
        # Check both positive and negative angles
        angles_to_test = [optimal_angle, -optimal_angle]
        min_error = float('inf')
        
        for angle in angles_to_test:
            # Apply rotation M_{h,θ}
            rotated_point = self._rotate_point_around_axis(
                p_reference, pivot, axis, angle
            )
            error = np.linalg.norm(rotated_point - p_current)
            min_error = min(min_error, error)
        
        return min_error
    
    def _estimate_rotation_angle(self, 
                               point1: np.ndarray, 
                               point2: np.ndarray, 
                               axis: np.ndarray, 
                               pivot: np.ndarray) -> float:
        """Estimate rotation angle between two points around an axis."""
        # Project points onto plane perpendicular to axis
        v1 = point1 - pivot
        v2 = point2 - pivot
        
        # Remove component along axis
        v1_proj = v1 - np.dot(v1, axis) * axis
        v2_proj = v2 - np.dot(v2, axis) * axis
        
        # Calculate angle between projected vectors
        cos_angle = np.dot(v1_proj, v2_proj) / (np.linalg.norm(v1_proj) * np.linalg.norm(v2_proj) + 1e-8)
        cos_angle = np.clip(cos_angle, -1, 1)
        
        angle = np.arccos(cos_angle)
        
        # Determine sign using cross product
        cross = np.cross(v1_proj, v2_proj)
        if np.dot(cross, axis) < 0:
            angle = -angle
        
        return angle
    
    def refine_parameters(self,
                          inlier_trajectories: List[Trajectory3D],
                          initial_params: HingeParameters) -> HingeParameters:
        """
        Simple circular-aware refinement for hinge joints.
        
        For each trajectory with enough points:
        1. Fit circle to first, middle, last points
        2. Use circle normal as axis candidate
        3. Average all valid axes and find pivot from centers
        """
        if not inlier_trajectories or len(inlier_trajectories) < 2:
            return initial_params
        
        valid_axes = []
        valid_centers = []
        
        # Try to fit circles to longer trajectories
        for traj in inlier_trajectories:
            if len(traj.points) < 5:
                continue
                
            # Get 3 spread-out points from trajectory
            points = np.array([[p.x, p.y, p.z] for p in traj.points])
            p1 = points[0]                    # first
            p2 = points[len(points)//2]       # middle  
            p3 = points[-1]                   # last
            
            try:
                # Simple 3-point circle fit
                center, axis = self._fit_simple_circle(p1, p2, p3)
                
                # Basic validation
                radius = np.linalg.norm(p1 - center)
                if 0.01 < radius < 1.0:  # reasonable radius
                    valid_axes.append(axis)
                    valid_centers.append(center)
                    
            except:
                continue
        
        if len(valid_axes) < 2:
            # Fallback to simple refinement
            all_points = []
            for traj in inlier_trajectories:
                all_points.extend([[p.x, p.y, p.z] for p in traj.points])
            
            refined_pivot = np.mean(all_points, axis=0) if all_points else initial_params.pivot
            return HingeParameters(axis=initial_params.axis, pivot=refined_pivot)
        
        # Average the valid axes (handle direction consistency)
        reference_axis = valid_axes[0]
        for i in range(1, len(valid_axes)):
            if np.dot(valid_axes[i], reference_axis) < 0:
                valid_axes[i] = -valid_axes[i]
        
        refined_axis = np.mean(valid_axes, axis=0)
        refined_axis = refined_axis / np.linalg.norm(refined_axis)
        
        # Pivot = average of circle centers
        refined_pivot = np.mean(valid_centers, axis=0)
        
        print(f"Circle refinement: used {len(valid_axes)} trajectories")
        return HingeParameters(axis=refined_axis, pivot=refined_pivot)

    def _fit_simple_circle(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> tuple:
        """
        Fit circle through 3 points. Returns (center, normal).
        """
        # Find plane normal from cross product
        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        
        if np.linalg.norm(normal) < 1e-8:
            raise ValueError("Collinear points")
        
        normal = normal / np.linalg.norm(normal)
        
        # Find circle center using perpendicular bisectors
        # Midpoints of two chords
        mid12 = (p1 + p2) / 2
        mid23 = (p2 + p3) / 2
        
        # Directions perpendicular to chords (in the plane)
        dir12 = np.cross(normal, p2 - p1)
        dir23 = np.cross(normal, p3 - p2)
        
        # Solve intersection of two lines: mid12 + t*dir12 = mid23 + s*dir23
        # This is a simple 3D line intersection problem
        
        # Use simplified approach: center is equidistant from all 3 points
        # Solve system: |center - p1|² = |center - p2|² = |center - p3|²
        
        A = 2 * np.array([p2 - p1, p3 - p1])
        b = np.array([
            np.dot(p2, p2) - np.dot(p1, p1),
            np.dot(p3, p3) - np.dot(p1, p1)
        ])
        
        # Project to 2D for stable solving
        u = v1 / np.linalg.norm(v1)
        v = np.cross(normal, u)
        
        # Convert to 2D coordinates
        p1_2d = np.array([0, 0])
        p2_2d = np.array([np.dot(p2 - p1, u), np.dot(p2 - p1, v)])
        p3_2d = np.array([np.dot(p3 - p1, u), np.dot(p3 - p1, v)])
        
        # Solve 2D circle center
        A_2d = 2 * np.array([
            [p2_2d[0], p2_2d[1]],
            [p3_2d[0], p3_2d[1]]
        ])
        b_2d = np.array([
            p2_2d[0]**2 + p2_2d[1]**2,
            p3_2d[0]**2 + p3_2d[1]**2
        ])
        
        center_2d = np.linalg.solve(A_2d, b_2d)
        
        # Convert back to 3D
        center_3d = p1 + center_2d[0] * u + center_2d[1] * v
        
        return center_3d, normal



class SliderJointModel(JointModelBase):
    """
    Model for slider/prismatic joints.
    
    A slider joint translates along a fixed direction. All points on the
    moving part move parallel to each other along this direction.
    """
    
    def get_joint_type(self) -> JointType:
        return JointType.SLIDER
    
    def minimal_sample_size(self) -> int:
        # Need at least 1 trajectory with 2 time points
        return 1
    
    def fit_from_sample(self, trajectory_sample: List[Trajectory3D]) -> Optional[SliderParameters]:
        """
        Fit slider parameters from trajectory sample.
        
        Algorithm:
        1. Calculate displacement vectors for each trajectory
        2. Average them to get slide direction
        3. Normalize to get unit direction vector
        """
        if not self.validate_sample(trajectory_sample):
            return None
        
        try:
            displacement_vectors = []
            
            for traj in trajectory_sample:
                if len(traj.points) < 2:
                    continue
                
                # Calculate displacement from first to last point
                p1 = traj.points[0]
                p2 = traj.points[-1]
                
                displacement = np.array([p2.x - p1.x, p2.y - p1.y, p2.z - p1.z])
                
                if np.linalg.norm(displacement) > 1e-6:  # Avoid zero displacements
                    displacement_vectors.append(displacement)
            
            if len(displacement_vectors) == 0:
                return None
            
            # Average displacement vectors and normalize
            avg_displacement = np.mean(displacement_vectors, axis=0)
            direction = avg_displacement / np.linalg.norm(avg_displacement)
            
            # Use first trajectory's first point as reference
            reference_point = np.array([
                trajectory_sample[0].points[0].x,
                trajectory_sample[0].points[0].y,
                trajectory_sample[0].points[0].z
            ])
            
            return SliderParameters(
                direction=direction,
                reference_point=reference_point
            )
            
        except Exception as e:
            return None
    
    def calculate_trajectory_error(self, trajectory: Trajectory3D, joint_params: SliderParameters) -> float:
        """
        Calculate fitting error following Li & Wan 2016 Equation 12:
        D_v(f_i, f_k) = min_τ ||p_i + τv - p_k||
        
        For slider joint, motion should be pure translation along direction v.
        """
        if len(trajectory.points) < 2:
            return float('inf')
        
        points = trajectory.get_all_positions()
        reference_point = points[0]  # Use first frame as reference
        
        valid_fits = 0
        total_frames = len(points) - 1
        errors = []
        
        for current_point in points[1:]:
            # Find optimal translation distance τ that minimizes ||p_i + τv - p_k||
            min_error = self._find_optimal_translation_distance(
                reference_point, current_point, joint_params.direction
            )
            
            errors.append(min_error)
            
            # Following paper: point supports slider if error < threshold
            if min_error <= 0.05:  # εv = 0.05 from paper
                valid_fits += 1
        
        # Following paper: "discard slider if supporting points < 80% of lifespan"
        support_ratio = valid_fits / total_frames if total_frames > 0 else 0
        
        if support_ratio < 0.8:
            return float('inf')  # Reject this slider
        
        # Return average error for valid trajectory
        return np.mean(errors)
    
    def _find_optimal_translation_distance(self, 
                                         p_reference: np.ndarray,
                                         p_current: np.ndarray,
                                         direction: np.ndarray) -> float:
        """
        Find translation distance τ that minimizes ||p_reference + τ*direction - p_current||
        This implements the min_τ part of Equation 12.
        """
        # Displacement from reference to current
        displacement = p_current - p_reference
        
        # Optimal translation distance is projection of displacement onto direction
        optimal_tau = np.dot(displacement, direction)
        
        # Predicted position after optimal translation
        predicted_point = p_reference + optimal_tau * direction
        
        # Error is distance from predicted to actual position
        error = np.linalg.norm(predicted_point - p_current)
        
        return error
    
    def refine_parameters(self, 
                         inlier_trajectories: List[Trajectory3D], 
                         initial_params: SliderParameters) -> SliderParameters:
        """Refine slider parameters using all inlier trajectories."""
        if not inlier_trajectories:
            return initial_params
        
        # Collect all displacement vectors
        all_displacements = []
        
        for traj in inlier_trajectories:
            if len(traj.points) >= 2:
                p1 = traj.points[0]
                p2 = traj.points[-1]
                
                displacement = np.array([p2.x - p1.x, p2.y - p1.y, p2.z - p1.z])
                
                if np.linalg.norm(displacement) > 1e-6:
                    all_displacements.append(displacement)
        
        if len(all_displacements) == 0:
            return initial_params
        
        # Refine direction using all displacements
        avg_displacement = np.mean(all_displacements, axis=0)
        refined_direction = avg_displacement / np.linalg.norm(avg_displacement)
        
        return SliderParameters(
            direction=refined_direction,
            reference_point=initial_params.reference_point,
            translation_min=initial_params.translation_min,
            translation_max=initial_params.translation_max
        )


# Factory function for creating joint models
def create_joint_models() -> Dict[JointType, JointModelBase]:
    """Create instances of all available joint models."""
    return {
        JointType.HINGE: HingeJointModel(),
        JointType.SLIDER: SliderJointModel(),
    }


def get_joint_model(joint_type: JointType) -> Optional[JointModelBase]:
    """Get a specific joint model by type."""
    models = create_joint_models()
    return models.get(joint_type)




if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F

    # --- Helper to make synthetic 3D trajectories ---
    def make_hinge_trajectories(axis, pivot, angles, n_points=5, noise=0.01):
        """Generate synthetic hinge trajectories."""
        trajectories = []
        axis = axis / np.linalg.norm(axis)

        # random initial points not on axis
        for i in range(n_points):
            base_point = pivot + np.random.randn(3)  # random offset
            points = []
            for angle in angles:
                # Rodrigues rotation formula
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

    def make_slider_trajectories(direction, start_point, steps, n_points=5, noise=0.01):
        """Generate synthetic slider trajectories."""
        trajectories = []
        direction = direction / np.linalg.norm(direction)
        for i in range(n_points):
            offset = np.random.randn(3) * 0.5 # random offset perpendicular
            base_point = start_point + offset
            points = []
            for step in steps:
                pt = base_point + step * direction
                pt += noise * np.random.randn(3)
                points.append(Point3D(frame=len(points), x=pt[0], y=pt[1], z=pt[2]))
            trajectories.append(Trajectory3D(track_id=i, points=points))
        return trajectories


    # --- Test Hinge Model ---
    hinge_model = HingeJointModel()
    true_axis = np.array([0, 0, 1])
    true_pivot = np.array([0, 0, 0])
    angles = np.linspace(0, np.pi/4, 10)
    hinge_trajs = make_hinge_trajectories(true_axis, true_pivot, angles)

    hinge_params = hinge_model.fit_from_sample(hinge_trajs)
    print("\n=== Hinge Joint Test ===")
    print("True axis:", true_axis, " | Estimated axis:", hinge_params.axis)
    print("True pivot:", true_pivot, " | Estimated pivot:", hinge_params.pivot)

    errors = [hinge_model.calculate_trajectory_error(t, hinge_params) for t in hinge_trajs]
    print("Mean trajectory error:", np.mean(errors))

    # --- Visualization for Hinge ---
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(121, projection='3d')
    for traj in hinge_trajs:
        pts = traj.get_all_positions()
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], marker='o', alpha=0.7)

    # Plot estimated axis as a line through pivot
    pivot = hinge_params.pivot
    axis = hinge_params.axis
    line_pts = np.array([pivot - 2 * axis, pivot + 2 * axis])
    ax.plot(line_pts[:, 0], line_pts[:, 1], line_pts[:, 2], 'r--', lw=2, label="Estimated axis")
    ax.set_title("Hinge trajectories & estimated axis")
    ax.legend()

    # --- Test Slider Model ---
    slider_model = SliderJointModel()
    true_dir = np.array([1, 0, 0])
    start = np.array([0, 0, 0])
    steps = np.linspace(0, 1, 10)
    slider_trajs = make_slider_trajectories(true_dir, start, steps)

    slider_params = slider_model.fit_from_sample(slider_trajs)
    print("\n=== Slider Joint Test ===")
    print("True direction:", true_dir, " | Estimated direction:", slider_params.direction)
    print("Reference point:", slider_params.reference_point)

    errors = [slider_model.calculate_trajectory_error(t, slider_params) for t in slider_trajs]
    print("Mean trajectory error:", np.mean(errors))

    # --- Visualization for Slider ---
    ax2 = fig.add_subplot(122, projection='3d')
    for traj in slider_trajs:
        pts = traj.get_all_positions()
        ax2.plot(pts[:, 0], pts[:, 1], pts[:, 2], marker='o', alpha=0.7)

    # Plot estimated slide direction as a line from reference point
    ref_pt = slider_params.reference_point
    dir_vec = slider_params.direction
    line_pts = np.array([ref_pt - 2 * dir_vec, ref_pt + 2 * dir_vec])
    ax2.plot(line_pts[:, 0], line_pts[:, 1], line_pts[:, 2], 'g--', lw=2, label="Estimated direction")
    ax2.set_title("Slider trajectories & estimated direction")
    ax2.legend()

    plt.tight_layout()
    plt.show()