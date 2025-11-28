"""
I/O utilities for transforms, COLMAP models, and data formats.
"""

import json
import numpy as np
import pycolmap
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from hloc.utils.read_write_model import Camera, Image, Point3D, write_model, read_model
from .pose_utils import arkit_to_colmap_pose, colmap_to_arkit_pose, matrix_to_qvec_tvec

def load_arkit_transforms(transforms_path: Path) -> Dict:
    """
    Load ARKit transforms.json file.
    
    Args:
        transforms_path: Path to transforms.json or transforms_arkit.json
    
    Returns:
        Dictionary with camera intrinsics and frame data
    """
    with open(transforms_path, 'r') as f:
        data = json.load(f)
    
    print(f"Loaded ARKit transforms: {len(data['frames'])} frames")
    return data


def save_transforms(
    frames: List[Dict],
    camera_params: Dict,
    output_path: Path,
    additional_fields: Optional[Dict] = None
):
    """
    Save transforms.json file.
    
    Args:
        frames: List of frame dictionaries with 'file_path' and 'transform_matrix'
        camera_params: Dict with 'fl_x', 'fl_y', 'cx', 'cy', 'w', 'h'
        output_path: Output path for transforms.json
        additional_fields: Optional additional fields to include (e.g., 'ply_file_path')
    """
    transforms = {
        'camera_model': 'PINHOLE',
        'fl_x': camera_params['fl_x'],
        'fl_y': camera_params['fl_y'],
        'cx': camera_params['cx'],
        'cy': camera_params['cy'],
        'w': camera_params['w'],
        'h': camera_params['h'],
        'camera_angle_x': 2 * np.arctan(camera_params['w'] / (2 * camera_params['fl_x'])),
        'camera_angle_y': 2 * np.arctan(camera_params['h'] / (2 * camera_params['fl_y'])),
        'ply_file_path': "fused_pc.ply",
        'frames': frames
    }
    
    # Add any additional fields
    if additional_fields:
        transforms.update(additional_fields)
    
    with open(output_path, 'w') as f:
        json.dump(transforms, f, indent=2)
    
    print(f"✓ Saved transforms: {output_path}")


def arkit_to_colmap_model(
    arkit_data: Dict,
    output_dir: Path,
    camera_id: int = 1
) -> Path:
    """
    Convert ARKit transforms to COLMAP model format.
    
    Args:
        arkit_data: ARKit transforms dictionary
        output_dir: Output directory for COLMAP model
        camera_id: Camera ID to use (default: 1)
    
    Returns:
        Path to the created COLMAP model directory
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Extract camera intrinsics
    fx = arkit_data['fl_x']
    fy = arkit_data['fl_y']
    cx = arkit_data['cx']
    cy = arkit_data['cy']
    w = arkit_data['w']
    h = arkit_data['h']
    
    # Create COLMAP camera
    camera = Camera(
        id=camera_id,
        model='PINHOLE',
        width=int(w),
        height=int(h),
        params=[fx, fy, cx, cy]
    )
    cameras = {camera_id: camera}
    
    # Create COLMAP images
    images = {}
    for idx, frame in enumerate(arkit_data['frames']):
        image_name = Path(frame['file_path']).name
        
        # Convert pose
        c2w_arkit = np.array(frame['transform_matrix'])
        w2c_colmap = arkit_to_colmap_pose(c2w_arkit)
        
        # Extract quaternion and translation
        qvec, tvec = matrix_to_qvec_tvec(w2c_colmap)
        
        # Create COLMAP image
        image_id = idx + 1
        image = Image(
            id=image_id,
            qvec=qvec,
            tvec=tvec,
            camera_id=camera_id,
            name=image_name,
            xys=[],
            point3D_ids=[]
        )
        images[image_id] = image
    
    # Write COLMAP model (no 3D points initially)
    points3D = {}
    write_model(
        images=images,
        cameras=cameras,
        points3D=points3D,
        path=str(output_dir),
        ext='.bin'
    )
    
    print(f"✓ Created COLMAP model: {output_dir}")
    print(f"  - {len(images)} images")
    print(f"  - {len(cameras)} cameras")
    
    return output_dir


def load_colmap_model(model_path: Path) -> pycolmap.Reconstruction:
    """
    Load COLMAP reconstruction model.
    
    Args:
        model_path: Path to COLMAP model directory
    
    Returns:
        pycolmap.Reconstruction object
    """
    reconstruction = pycolmap.Reconstruction(str(model_path))
    
    print(f"✓ Loaded COLMAP model: {model_path}")
    print(f"  - {len(reconstruction.images)} images")
    print(f"  - {len(reconstruction.points3D)} 3D points")
    
    return reconstruction


def colmap_model_to_transforms(
    reconstruction: pycolmap.Reconstruction,
    output_path: Path,
    camera_id: int = 1,
    depth_dir: Optional[str] = None
) -> Path:
    """
    Convert COLMAP reconstruction to transforms.json format.
    
    Args:
        reconstruction: pycolmap.Reconstruction object
        output_path: Output path for transforms.json
        camera_id: Camera ID to extract intrinsics from
        depth_dir: Optional depth directory path to include in frames
    
    Returns:
        Path to the saved transforms.json
    """
    # Get camera intrinsics
    camera = reconstruction.cameras[camera_id]
    fx, fy, cx, cy = camera.params
    w, h = camera.width, camera.height
    
    camera_params = {
        'fl_x': fx,
        'fl_y': fy,
        'cx': cx,
        'cy': cy,
        'w': w,
        'h': h
    }
    
    # Convert images to frames
    frames = []
    for image_id in sorted(reconstruction.images.keys()):
        image = reconstruction.images[image_id]
        
        # Convert COLMAP pose to ARKit format
        w2c_colmap = np.eye(4)
        w2c_colmap[:3, :3] = pycolmap.qvec_to_rotmat(image.qvec)
        w2c_colmap[:3, 3] = image.tvec
        
        c2w_arkit = colmap_to_arkit_pose(w2c_colmap)
        
        frame = {
            'file_path': f'frames/{image.name}',
            'transform_matrix': c2w_arkit.tolist()
        }
        
        # Add depth path if specified
        if depth_dir:
            depth_name = Path(image.name).stem + '.npy'
            frame['depth_file_path'] = f'{depth_dir}/{depth_name}'
        
        frames.append(frame)
    
    # Save transforms
    save_transforms(frames, camera_params, output_path)
    
    return output_path

def get_camera_intrinsics(arkit_data: Dict) -> Tuple[float, float, float, float, int, int]:
    """
    Extract camera intrinsics from ARKit transforms.
    
    Args:
        arkit_data: ARKit transforms dictionary
    
    Returns:
        Tuple of (fx, fy, cx, cy, width, height)
    """
    return (
        arkit_data['fl_x'],
        arkit_data['fl_y'],
        arkit_data['cx'],
        arkit_data['cy'],
        arkit_data['w'],
        arkit_data['h']
    )


def create_pycolmap_camera(
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    width: int,
    height: int,
    camera_id: int = 1
) -> pycolmap.Camera:
    """
    Create a pycolmap.Camera object.
    
    Args:
        fx, fy: Focal lengths
        cx, cy: Principal point
        width, height: Image dimensions
        camera_id: Camera ID
    
    Returns:
        pycolmap.Camera object
    """
    return pycolmap.Camera(
        model='PINHOLE',
        width=int(width),
        height=int(height),
        params=[fx, fy, cx, cy]
    )


def get_image_name_to_id(reconstruction: pycolmap.Reconstruction) -> Dict[str, int]:
    """
    Create mapping from image name to image ID.
    
    Args:
        reconstruction: pycolmap.Reconstruction object
    
    Returns:
        Dictionary mapping image names to image IDs
    """
    return {img.name: img_id for img_id, img in reconstruction.images.items()}


def count_observations(reconstruction: pycolmap.Reconstruction) -> Dict[int, int]:
    """
    Count number of 3D point observations per image.
    
    Args:
        reconstruction: pycolmap.Reconstruction object
    
    Returns:
        Dictionary mapping image IDs to observation counts
    """
    counts = {}
    for img_id, image in reconstruction.images.items():
        counts[img_id] = image.num_points3D()
    return counts