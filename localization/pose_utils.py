"""
Pose and coordinate system utilities.
Handles conversions between ARKit, COLMAP, and OpenGL conventions.
"""

import numpy as np
from typing import Tuple
import pycolmap
from pyquaternion import Quaternion


def arkit_to_colmap_pose(c2w_arkit: np.ndarray) -> np.ndarray:
    """
    Convert ARKit camera-to-world to COLMAP world-to-camera.
    
    ARKit uses Y-up, Z-back (right-handed)
    COLMAP uses Y-down, Z-forward (right-handed)
    
    Args:
        c2w_arkit: 4x4 camera-to-world matrix in ARKit convention
    
    Returns:
        4x4 world-to-camera matrix in COLMAP convention
    """
    # Flip Y and Z axes to convert coordinate systems
    flip_yz = np.eye(4)
    flip_yz[1, 1] = -1
    flip_yz[2, 2] = -1
    
    c2w_cv = c2w_arkit @ flip_yz
    w2c_cv = np.linalg.inv(c2w_cv)
    
    return w2c_cv


def colmap_to_arkit_pose(w2c_colmap: np.ndarray) -> np.ndarray:
    """
    Convert COLMAP world-to-camera to ARKit camera-to-world.
    
    Args:
        w2c_colmap: 4x4 world-to-camera matrix in COLMAP convention
    
    Returns:
        4x4 camera-to-world matrix in ARKit convention
    """
    # Invert to get camera-to-world
    c2w_cv = np.linalg.inv(w2c_colmap)
    
    # Flip Y and Z axes
    flip_yz = np.eye(4)
    flip_yz[1, 1] = -1
    flip_yz[2, 2] = -1
    
    c2w_arkit = c2w_cv @ flip_yz
    
    return c2w_arkit


def colmap_to_opengl_pose(w2c_colmap: np.ndarray) -> np.ndarray:
    """
    Convert COLMAP world-to-camera to OpenGL camera-to-world.
    
    COLMAP: Y-down, Z-forward
    OpenGL: Y-up, Z-back
    
    Args:
        w2c_colmap: 4x4 world-to-camera matrix in COLMAP convention
    
    Returns:
        4x4 camera-to-world matrix in OpenGL convention
    """
    c2w = np.linalg.inv(w2c_colmap)
    
    # Flip Y and Z to convert COLMAP to OpenGL
    c2w[0:3, 1:3] *= -1
    
    return c2w


def qvec_tvec_to_matrix(qvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """
    Convert COLMAP quaternion + translation to 4x4 matrix.
    
    Args:
        qvec: Quaternion [w, x, y, z]
        tvec: Translation [x, y, z]
    
    Returns:
        4x4 transformation matrix
    """
    w2c = np.eye(4)
    w2c[:3, :3] = pycolmap.qvec_to_rotmat(qvec)
    w2c[:3, 3] = tvec
    return w2c


def matrix_to_qvec_tvec(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert 4x4 matrix to COLMAP quaternion + translation.
    
    Args:
        matrix: 4x4 transformation matrix (world-to-camera)
    
    Returns:
        Tuple of (qvec [w,x,y,z], tvec [x,y,z])
    """
    R = matrix[:3, :3]
    tvec = matrix[:3, 3]
    
    q = Quaternion(matrix=R, atol=1e-06)
    qvec = np.array([q.w, q.x, q.y, q.z])
    
    return qvec, tvec


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """
    Transform 3D points by a 4x4 transformation matrix.
    
    Args:
        points: Nx3 array of 3D points
        transform: 4x4 transformation matrix
    
    Returns:
        Nx3 array of transformed points
    """
    points_h = np.hstack([points, np.ones((points.shape[0], 1))])
    points_transformed = (transform @ points_h.T).T
    
    return points_transformed[:, :3]


def invert_pose(pose: np.ndarray) -> np.ndarray:
    """
    Invert a 4x4 pose matrix.
    
    Args:
        pose: 4x4 transformation matrix
    
    Returns:
        4x4 inverted transformation matrix
    """
    return np.linalg.inv(pose)


def compose_poses(pose1: np.ndarray, pose2: np.ndarray) -> np.ndarray:
    """
    Compose two 4x4 pose matrices.
    
    Args:
        pose1: First 4x4 transformation matrix
        pose2: Second 4x4 transformation matrix
    
    Returns:
        4x4 composed transformation matrix (pose1 @ pose2)
    """
    return pose1 @ pose2


def normalize_quaternion(qvec: np.ndarray) -> np.ndarray:
    """
    Normalize a quaternion to unit length.
    
    Args:
        qvec: Quaternion [w, x, y, z]
    
    Returns:
        Normalized quaternion
    """
    return qvec / np.linalg.norm(qvec)