"""
Core data structures for the 4D RANSAC Joint Estimation Pipeline.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Union, Tuple
import numpy as np
from enum import Enum


# Basic Data Types

@dataclass
class TrackPoint2D:
    """A single 2D tracking point at a specific frame."""
    frame: int
    u: float  # Pixel x
    v: float  # Pixel y-
    confidence: float = 1.0  # Tracking confidence 


@dataclass
class Point3D:
    """A single 3D point at a specific frame."""
    frame: int
    x: float  
    y: float
    z: float
    confidence: float = 1.0  # Reconstruction confidence


@dataclass
class Trajectory2D:
    """A complete 2D trajectory for a tracked point."""
    track_id: int
    points: List[TrackPoint2D]
    
    def __len__(self) -> int:
        return len(self.points)
    
    def get_frame_range(self) -> Tuple[int, int]:
        """Returns (min_frame, max_frame)."""
        if not self.points:
            return (0, 0)
        frames = [p.frame for p in self.points]
        return (min(frames), max(frames))


@dataclass
class Trajectory3D:
    """A complete 3D trajectory for a tracked point."""
    track_id: int
    points: List[Point3D]
    rigid_part: Optional[int] = None  # Which rigid part this belongs to (0=static, 1=moving)
    
    def __len__(self) -> int:
        return len(self.points)
    
    def get_frame_range(self) -> Tuple[int, int]:
        """Returns (min_frame, max_frame)."""
        if not self.points:
            return (0, 0)
        frames = [p.frame for p in self.points]
        return (min(frames), max(frames))
    
    def get_positions_at_frame(self, frame: int) -> Optional[np.ndarray]:
        """Get 3D position at specific frame."""
        for point in self.points:
            if point.frame == frame:
                return np.array([point.x, point.y, point.z])
        return None
    
    def get_all_positions(self) -> np.ndarray:
        """Get all 3D positions as (N, 3) array."""
        return np.array([[p.x, p.y, p.z] for p in self.points])


# Camera Parameters 

@dataclass
class CameraIntrinsics:
    """Camera intrinsic parameters for RGB-D projection."""
    fx: float  # Focal length x
    fy: float  # Focal length y
    cx: float  # Principal point x
    cy: float  # Principal point y
    w: int     # width
    h: int     # Height     

    
    def to_matrix(self) -> np.ndarray:
        """Convert to 3x3 intrinsic matrix."""
        return np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ])
    
    def project_3d_to_2d(self, point_3d: np.ndarray) -> np.ndarray:
        """Project 3D point to 2D pixel coordinates."""
        x, y, z = point_3d
        u = (x * self.fx / z) + self.cx
        v = (y * self.fy / z) + self.cy
        return np.array([u, v])
    
    def unproject_2d_to_3d(self, u: float, v: float, depth: float) -> np.ndarray:
        """Unproject 2D pixel + depth to 3D point."""
        x = (u - self.cx) * depth / self.fx
        y = (v - self.cy) * depth / self.fy
        z = depth
        return np.array([x, y, z])


# Joint Models and Parameters

class JointType(Enum):
    """Types of joints that can be estimated."""
    HINGE = "hinge"
    SLIDER = "slider"
    UNKNOWN = "unknown"


@dataclass
class HingeParameters:
    """Parameters for a hinge joint (revolute joint)."""
    axis: np.ndarray  # 3D unit vector defining rotation axis
    pivot: np.ndarray  # 3D point on the rotation axis
    angle_min: Optional[float] = None  # Minimum rotation angle (radians)
    angle_max: Optional[float] = None  # Maximum rotation angle (radians)
    
    def __post_init__(self):
        """Ensure axis is normalized."""
        self.axis = self.axis / np.linalg.norm(self.axis)
    
    def get_rotation_matrix(self, angle: float) -> np.ndarray:
        """Get rotation matrix for given angle around the axis."""
        # Rodrigues' rotation formula
        K = np.array([
            [0, -self.axis[2], self.axis[1]],
            [self.axis[2], 0, -self.axis[0]],
            [-self.axis[1], self.axis[0], 0]
        ])
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
        return R
    
    def transform_point(self, point: np.ndarray, angle: float) -> np.ndarray:
        """Transform a 3D point by the hinge motion."""
        # Translate to origin, rotate, translate back
        centered = point - self.pivot
        R = self.get_rotation_matrix(angle)
        rotated = R @ centered
        return rotated + self.pivot


@dataclass
class SliderParameters:
    """Parameters for a slider joint (prismatic joint)."""
    direction: np.ndarray  # 3D unit vector defining slide direction
    translation_min: Optional[float] = None  # Minimum translation distance
    translation_max: Optional[float] = None  # Maximum translation distance
    reference_point: Optional[np.ndarray] = None  # Reference point on the sliding part
    
    def __post_init__(self):
        """Ensure direction is normalized."""
        self.direction = self.direction / np.linalg.norm(self.direction)
    
    def transform_point(self, point: np.ndarray, distance: float) -> np.ndarray:
        """Transform a 3D point by the slider motion."""
        return point + distance * self.direction


# RANSAC Configuration

@dataclass
class RANSACConfig:
    """Configuration parameters for RANSAC algorithm."""
    max_iterations: int = 1000
    error_threshold: float = 0.05  # Maximum error for inlier (meters)
    min_inliers: int = 10  # Minimum inliers to accept model
    confidence: float = 0.99  # Desired confidence level
    min_trajectory_length: int = 5  # Minimum points per trajectory
    early_termination_threshold: float = 0.8  # Stop if this fraction are inliers
    use_tms_preclustering: bool = True


# Pipeline Configuration

@dataclass
class TrajectoryFilterConfig:
    """Configuration for trajectory filtering."""
    min_length: int = 5  # Minimum trajectory length
    max_velocity_jump: float = 0.5  # Maximum frame-to-frame velocity change (m/s)
    smoothing_window: int = 3  # Window size for trajectory smoothing
    max_tracks: Optional[int] = None


@dataclass
class PipelineConfig:
    """Overall pipeline configuration."""
    camera_intrinsics: CameraIntrinsics
    ransac_config: RANSACConfig
    trajectory_filter_config: TrajectoryFilterConfig
    joint_types_to_test: List[JointType] = field(
        default_factory=lambda: [JointType.HINGE, JointType.SLIDER]
    )
    debug_mode: bool = False


# ============================================================================
# Results and Outputs
# ============================================================================

@dataclass
class ModelFitResult:
    """Result of fitting a joint model to trajectories."""
    joint_type: JointType
    parameters: Union[HingeParameters, SliderParameters]
    inlier_trajectories: List[Trajectory3D]
    inlier_count: int
    total_trajectories: int
    fit_error: float  # Average error of inlier trajectories
    consensus_score: float  # Fraction of trajectories that are inliers
    
    @property
    def confidence(self) -> float:
        """Confidence score based on consensus."""
        return self.consensus_score


@dataclass
class JointEstimationResult:
    """Final result of the joint estimation pipeline."""
    success: bool
    joint_type: JointType
    parameters: Optional[Union[HingeParameters, SliderParameters]]
    confidence: float
    inlier_trajectories: List[Trajectory3D]
    total_trajectories: int
    processing_time: float  # Seconds
    error_message: Optional[str] = None
    
    def is_hinge(self) -> bool:
        return self.joint_type == JointType.HINGE
    
    def is_slider(self) -> bool:
        return self.joint_type == JointType.SLIDER
    
    def get_hinge_params(self) -> Optional[HingeParameters]:
        if self.is_hinge():
            return self.parameters
        return None
    
    def get_slider_params(self) -> Optional[SliderParameters]:
        if self.is_slider():
            return self.parameters
        return None


# ============================================================================
# Utility Functions
# ============================================================================

def trajectory_2d_to_3d(traj_2d: Trajectory2D, 
                       depth_sequence: List[np.ndarray], 
                       camera_intrinsics: CameraIntrinsics) -> Trajectory3D:
    """Convert 2D trajectory to 3D using depth data."""
    points_3d = []
    
    for point_2d in traj_2d.points:
        frame_idx = point_2d.frame
        if frame_idx >= len(depth_sequence):
            continue
            
        depth_img = depth_sequence[frame_idx]
        u, v = int(point_2d.u), int(point_2d.v)
        
        # Check bounds
        if (0 <= u < depth_img.shape[1] and 
            0 <= v < depth_img.shape[0]):
            
            depth = depth_img[v, u]
            if depth > 0:  # Valid depth
                point_3d_coords = camera_intrinsics.unproject_2d_to_3d(
                    point_2d.u, point_2d.v, depth
                )
                point_3d = Point3D(
                    frame=point_2d.frame,
                    x=point_3d_coords[0],
                    y=point_3d_coords[1], 
                    z=point_3d_coords[2],
                    confidence=point_2d.confidence
                )
                points_3d.append(point_3d)
    
    return Trajectory3D(
        track_id=traj_2d.track_id,
        points=points_3d
    )


def create_default_config(fx: float, fy: float, cx: float, cy: float, w: int, h: int) -> PipelineConfig:
    """Create a default pipeline configuration."""
    camera_intrinsics = CameraIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy, w=w, h=h)
    ransac_config = RANSACConfig()
    trajectory_filter_config = TrajectoryFilterConfig()
    
    return PipelineConfig(
        camera_intrinsics=camera_intrinsics,
        ransac_config=ransac_config,
        trajectory_filter_config=trajectory_filter_config
    )