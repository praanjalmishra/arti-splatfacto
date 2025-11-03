"""
Temporal RGB-D Voxelization for Articulated Region Masking

Creates dense 3D voxel masks from temporal RGB-D sequences with per-frame 2D masks.
Replaces sparse trajectory supervision with volumetric articulation-aware occupancy.
"""

import numpy as np
import torch
import json
from pathlib import Path
from PIL import Image
from typing import Dict, List, Tuple, Optional
import cv2
from tqdm import tqdm


class TemporalVoxelMaskGenerator:
    """
    Generate volumetric 3D masks from temporal RGB-D + 2D mask sequences.
    
    Fuses multi-frame depth observations into a unified voxel grid representing
    the articulated region (moving part + revealed canonical geometry).
    """
    
    def __init__(
        self,
        voxel_resolution: int = 32,
        depth_scale: float = 1000.0,  # 16-bit depth: 1 unit = 1mm
        depth_max: float = 5.0,  # meters
        depth_min: float = 0.1,  # meters
        tsdf_truncation: float = 0.05,  # meters
        use_tsdf: bool = True,
        padding: float = 0.05  # bbox padding in meters
    ):
        """
        Args:
            voxel_resolution: Grid resolution (e.g., 32³)
            depth_scale: Depth units to meters (1000 for mm)
            depth_max/min: Valid depth range in meters
            tsdf_truncation: TSDF truncation distance
            use_tsdf: Use TSDF fusion vs binary occupancy
            padding: Extra space around point cloud bbox
        """
        self.voxel_resolution = voxel_resolution
        self.depth_scale = depth_scale
        self.depth_max = depth_max
        self.depth_min = depth_min
        self.tsdf_truncation = tsdf_truncation
        self.use_tsdf = use_tsdf
        self.padding = padding
        
    def load_nerf_metadata(self, metadata_path: str) -> Dict:
        """Load NeRF format camera metadata."""
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        return metadata
    
    def load_depth_image(self, depth_path: str) -> np.ndarray:
        """Load 16-bit depth image and convert to meters."""
        depth_img = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
        if depth_img is None:
            raise FileNotFoundError(f"Depth image not found: {depth_path}")
        
        # Convert to meters
        depth_meters = depth_img.astype(np.float32) / self.depth_scale
        
        # Clamp to valid range
        depth_meters[depth_meters < self.depth_min] = 0
        depth_meters[depth_meters > self.depth_max] = 0
        
        return depth_meters
    
    def load_mask_image(self, mask_path: str) -> np.ndarray:
        """Load 2D binary mask."""
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Mask not found: {mask_path}")
        
        # Binarize (assume 0 = background, >0 = object)
        binary_mask = (mask > 127).astype(np.uint8)
        return binary_mask
    
    def backproject_depth_to_3d(
        self,
        depth: np.ndarray,
        mask: np.ndarray,
        intrinsics: Dict[str, float],
        extrinsics: np.ndarray
    ) -> np.ndarray:
        """
        Backproject masked depth pixels to 3D world coordinates.
        
        Args:
            depth: (H, W) depth map in meters
            mask: (H, W) binary mask
            intrinsics: {fx, fy, cx, cy}
            extrinsics: 4x4 camera-to-world transform (NeRF format)
            
        Returns:
            points_3d: (N, 3) world coordinates of masked valid depth pixels
        """
        H, W = depth.shape
        fx, fy = intrinsics['fx'], intrinsics['fy']
        cx, cy = intrinsics['cx'], intrinsics['cy']
        
        # Create pixel grid
        u, v = np.meshgrid(np.arange(W), np.arange(H))
        
        # Filter by mask and valid depth
        valid_mask = (mask > 0) & (depth > 0)
        u_valid = u[valid_mask]
        v_valid = v[valid_mask]
        z_valid = depth[valid_mask]
        
        if len(z_valid) == 0:
            return np.zeros((0, 3), dtype=np.float32)
        
        # Backproject to camera coordinates (OpenCV convention: +Z forward)
        x_cam = (u_valid - cx) * z_valid / fx
        y_cam = (v_valid - cy) * z_valid / fy
        z_cam = z_valid  # +Z forward

        # Convert OpenCV → OpenGL convention for NeRF poses
        # (X, Y, Z) → (X, -Y, -Z)
        points_cam = np.stack([x_cam, -y_cam, -z_cam, np.ones_like(x_cam)], axis=1)

        
        # Transform to world coordinates
        points_world = (extrinsics @ points_cam.T).T[:, :3]
        
        return points_world.astype(np.float32)
    
    def fuse_temporal_points(
        self,
        data_dir: str,
        metadata_path: str,
        frame_indices: Optional[List[int]] = None,
        subsample_rate: int = 1
    ) -> Tuple[np.ndarray, Dict]:
        """
        Fuse multi-frame depth observations into unified point cloud.
        
        Args:
            data_dir: Root directory with frames/, depth/, masks/
            metadata_path: Path to transforms.json
            frame_indices: Specific frames to use (None = all frames)
            subsample_rate: Process every Nth frame
            
        Returns:
            all_points: (N, 3) fused 3D points in world coordinates
            metadata: Camera intrinsics and frame info
        """
        data_dir = Path(data_dir)
        metadata = self.load_nerf_metadata(metadata_path)
        
        # Extract intrinsics
        intrinsics = {
            'fx': metadata['fl_x'],
            'fy': metadata['fl_y'],
            'cx': metadata['cx'],
            'cy': metadata['cy'],
            'w': metadata['w'],
            'h': metadata['h']
        }
        
        # Determine which frames to process
        frames = metadata['frames']
        if frame_indices is not None:
            frames = [frames[i] for i in frame_indices if i < len(frames)]
        else:
            frames = frames[::subsample_rate]
        
        print(f"Processing {len(frames)} frames...")
        
        all_points = []
        for frame_data in tqdm(frames, desc="Backprojecting frames"):
            # Load frame data
            depth_path = data_dir / frame_data['depth_file_path']
            
            # Construct mask path (assume same naming convention)
            frame_name = Path(frame_data['file_path']).name
            mask_path = data_dir / 'mask' / frame_name.replace('.jpg', '.png')
            
            # Load depth and mask
            try:
                depth = self.load_depth_image(str(depth_path))
                mask = self.load_mask_image(str(mask_path))
            except FileNotFoundError as e:
                print(f"Skipping frame: {e}")
                continue
            
            # Get extrinsics (camera-to-world)
            extrinsics = np.array(frame_data['transform_matrix'], dtype=np.float32)
            
            # Backproject to 3D
            points_3d = self.backproject_depth_to_3d(
                depth, mask, intrinsics, extrinsics
            )
            
            if len(points_3d) > 0:
                all_points.append(points_3d)
        
        if not all_points:
            raise ValueError("No valid points extracted from any frame!")
        
        all_points = np.vstack(all_points)
        print(f"Fused {len(all_points)} points from {len(frames)} frames")
        
        return all_points, intrinsics
    
    def compute_robust_bbox(
        self,
        points: np.ndarray,
        quantile: float = 0.02
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute robust bounding box using quantiles to filter outliers.
        
        Args:
            points: (N, 3) point cloud
            quantile: Quantile for min/max (e.g., 0.02 = 2nd/98th percentile)
            
        Returns:
            bbox_min: (3,) minimum corner
            bbox_max: (3,) maximum corner
        """
        bbox_min = np.quantile(points, quantile, axis=0)
        bbox_max = np.quantile(points, 1 - quantile, axis=0)
        
        # Add padding
        extent = bbox_max - bbox_min
        bbox_min -= self.padding
        bbox_max += self.padding
        
        return bbox_min.astype(np.float32), bbox_max.astype(np.float32)
    
    def voxelize_binary(
        self,
        points: np.ndarray,
        bbox_min: np.ndarray,
        bbox_max: np.ndarray
    ) -> torch.Tensor:
        """
        Convert point cloud to binary voxel occupancy grid.
        
        Args:
            points: (N, 3) point cloud
            bbox_min/max: Bounding box
            
        Returns:
            voxel_grid: (R, R, R) binary occupancy
        """
        R = self.voxel_resolution
        
        # Normalize points to [0, R-1]
        normalized = (points - bbox_min) / (bbox_max - bbox_min)
        indices = (normalized * (R - 1)).astype(np.int32)
        
        # Clamp to valid range
        indices = np.clip(indices, 0, R - 1)
        
        # Fill voxel grid
        voxel_grid = np.zeros((R, R, R), dtype=bool)
        voxel_grid[indices[:, 0], indices[:, 1], indices[:, 2]] = True
        
        return torch.from_numpy(voxel_grid)
    
    def voxelize_tsdf(
        self,
        points: np.ndarray,
        bbox_min: np.ndarray,
        bbox_max: np.ndarray
    ) -> torch.Tensor:
        """
        Create TSDF (Truncated Signed Distance Field) voxel grid.
        
        Provides smoother boundaries and better gradient information.
        
        Args:
            points: (N, 3) point cloud
            bbox_min/max: Bounding box
            
        Returns:
            tsdf_grid: (R, R, R) TSDF values
        """
        R = self.voxel_resolution
        
        # Create voxel grid centers
        x = np.linspace(bbox_min[0], bbox_max[0], R)
        y = np.linspace(bbox_min[1], bbox_max[1], R)
        z = np.linspace(bbox_min[2], bbox_max[2], R)
        
        xv, yv, zv = np.meshgrid(x, y, z, indexing='ij')
        voxel_centers = np.stack([xv, yv, zv], axis=-1).reshape(-1, 3)
        
        # Compute distance to nearest point (vectorized for speed)
        print("Computing TSDF distances...")
        from scipy.spatial import cKDTree
        tree = cKDTree(points)
        distances, _ = tree.query(voxel_centers, k=1)
        
        # Apply TSDF truncation
        tsdf_values = np.clip(
            distances,
            -self.tsdf_truncation,
            self.tsdf_truncation
        )
        
        # Reshape to grid
        tsdf_grid = tsdf_values.reshape(R, R, R).astype(np.float32)
        
        return torch.from_numpy(tsdf_grid)
    
    def dilate_voxel_grid(
        self,
        voxel_grid: torch.Tensor,
        iterations: int = 1
    ) -> torch.Tensor:
        """
        Morphological dilation to slightly expand the mask.
        Helps ensure coverage of thin structures.
        """
        grid_np = voxel_grid.cpu().numpy().astype(np.uint8)
        
        # 3D dilation using scipy
        from scipy.ndimage import binary_dilation
        struct = np.ones((3, 3, 3), dtype=bool)
        
        for _ in range(iterations):
            grid_np = binary_dilation(grid_np, structure=struct).astype(np.uint8)
        
        return torch.from_numpy(grid_np.astype(bool))
    
    def generate_voxel_mask(
        self,
        data_dir: str,
        metadata_path: str,
        frame_indices: Optional[List[int]] = None,
        subsample_rate: int = 1,
        dilate_iterations: int = 1
    ) -> Dict[str, torch.Tensor]:
        """
        Complete pipeline: RGB-D sequence → voxelized 3D mask.
        
        Args:
            data_dir: Root directory with frames/, depth/, masks/
            metadata_path: Path to transforms.json
            frame_indices: Specific frames to use
            subsample_rate: Process every Nth frame
            dilate_iterations: Voxel dilation iterations
            
        Returns:
            Dictionary with:
                - voxel_grid: (R, R, R) occupancy/TSDF
                - bbox_min: (3,) bounding box minimum
                - bbox_max: (3,) bounding box maximum
                - intrinsics: Camera intrinsics dict
        """
        print("\n" + "="*60)
        print("TEMPORAL VOXEL MASK GENERATION")
        print("="*60)
        
        # Step 1: Fuse temporal point cloud
        all_points, intrinsics = self.fuse_temporal_points(
            data_dir, metadata_path, frame_indices, subsample_rate
        )
        
        # Step 2: Compute bounding box
        bbox_min, bbox_max = self.compute_robust_bbox(all_points)
        print(f"Bounding box: min={bbox_min}, max={bbox_max}")
        print(f"Extent: {bbox_max - bbox_min}")
        
        # Step 3: Voxelize
        if self.use_tsdf:
            print("Generating TSDF voxel grid...")
            voxel_grid = self.voxelize_tsdf(all_points, bbox_min, bbox_max)
        else:
            print("Generating binary occupancy grid...")
            voxel_grid = self.voxelize_binary(all_points, bbox_min, bbox_max)
        
        # Step 4: Optional dilation
        if dilate_iterations > 0:
            print(f"Applying {dilate_iterations} dilation iterations...")
            voxel_grid = self.dilate_voxel_grid(voxel_grid, dilate_iterations)
        
        occupancy = voxel_grid.sum().item()
        total = voxel_grid.numel()
        print(f"Voxel occupancy: {occupancy}/{total} ({100*occupancy/total:.2f}%)")
        
        return {
            'voxel_grid': voxel_grid,
            'bbox_min': torch.from_numpy(bbox_min),
            'bbox_max': torch.from_numpy(bbox_max),
            'intrinsics': intrinsics,
            'points': torch.from_numpy(all_points)  # For visualization
        }


def integrate_with_joint_estimation(
    voxel_mask_data: Dict[str, torch.Tensor],
    joint_result,
    output_dir: str,
    filename_prefix: str = "articulated"
):
    """
    Integrate temporal voxel mask with joint estimation results.
    
    Replaces the sparse trajectory-based voxelization in export_joint_and_inliers_tapip3d.
    
    Args:
        voxel_mask_data: Output from TemporalVoxelMaskGenerator
        joint_result: Joint estimation result
        output_dir: Output directory
        filename_prefix: Filename prefix
    """
    import os
    
    os.makedirs(output_dir, exist_ok=True)
    
    if not joint_result.success:
        print("Cannot export: joint estimation failed")
        return
    
    # Extract joint parameters
    if joint_result.joint_type.value == "hinge":
        hinge_params = joint_result.get_hinge_params()
        joint_type_str = "revolute"
        joint_axis_tensor = torch.tensor(hinge_params.axis, dtype=torch.float32)
        joint_pivot_tensor = torch.tensor(hinge_params.pivot, dtype=torch.float32)
        joint_limits = [
            float(hinge_params.angle_min) if hinge_params.angle_min is not None else 0.0,
            float(hinge_params.angle_max) if hinge_params.angle_max is not None else 0.0
        ]
    elif joint_result.joint_type.value == "slider":
        slider_params = joint_result.get_slider_params()
        if slider_params.reference_point is not None:
            joint_pivot_tensor = torch.tensor(slider_params.reference_point, dtype=torch.float32)
        else:
            joint_pivot_tensor = torch.zeros(3)
        joint_type_str = "prismatic"
        joint_axis_tensor = torch.tensor(slider_params.direction, dtype=torch.float32)
        joint_limits = [
            float(slider_params.translation_min),
            float(slider_params.translation_max)
        ]
    
    # Normalize axis
    joint_axis_tensor = joint_axis_tensor / (joint_axis_tensor.norm() + 1e-8)
    
    # Use temporal voxel mask instead of sparse trajectories
    voxel_grid = voxel_mask_data['voxel_grid']
    bbox_min = voxel_mask_data['bbox_min']
    bbox_max = voxel_mask_data['bbox_max']
    
    print(f"\n✅ Using temporal voxel mask:")
    print(f"   Resolution: {voxel_grid.shape}")
    print(f"   Occupancy: {voxel_grid.sum().item()}/{voxel_grid.numel()}")
    
    # Create Object3DSeg-compatible data
    obj3dseg_data = {
        'bbox_min': bbox_min.cpu(),
        'bbox_max': bbox_max.cpu(),
        'voxel': voxel_grid.cpu(),
        'tight_bbox': None,
        'mask_dilate_uniform': 0,
        'mask_dilate_top': 0,
        'joint_type': joint_type_str,
        'joint_axis': joint_axis_tensor.cpu(),
        'joint_pivot': joint_pivot_tensor.cpu(),
        'joint_limits': joint_limits,
        'joint_limit_min': torch.tensor(joint_limits[0]),
        'joint_limit_max': torch.tensor(joint_limits[1])
    }
    
    # Save outputs
    output_dir_masks = os.path.join(output_dir, "obj_masks")
    os.makedirs(output_dir_masks, exist_ok=True)
    
    voxel_path = os.path.join(output_dir_masks, f"obj_{filename_prefix}.pt")
    torch.save(obj3dseg_data, voxel_path)
    
    print(f"\nSaved Object3DSeg voxel data: {voxel_path}")
    print(f"   Joint type: {joint_type_str}")
    print(f"   Joint limits: {joint_limits}")
    
    # Also save raw points for debugging
    if 'points' in voxel_mask_data:
        points_path = os.path.join(output_dir_masks, f"obj_{filename_prefix}_points.npy")
        np.save(points_path, voxel_mask_data['points'].cpu().numpy())
        print(f"   Saved {len(voxel_mask_data['points'])} fused points: {points_path}")


def visualize_voxel_mask(voxel_data: Dict[str, torch.Tensor], title: str = "Temporal Voxel Mask"):
    """
    Visualize voxelized mask in 3D.
    
    Args:
        voxel_data: Output from TemporalVoxelMaskGenerator
        title: Plot title
    """
    import matplotlib.pyplot as plt

    
    voxel_grid = voxel_data['voxel_grid'].cpu().numpy()
    bbox_min = voxel_data['bbox_min'].cpu().numpy()
    bbox_max = voxel_data['bbox_max'].cpu().numpy()
    
    # Extract occupied voxel centers
    occupied = np.argwhere(voxel_grid)
    R = voxel_grid.shape[0]
    
    voxel_coords = occupied.astype(np.float32) / (R - 1)
    world_coords = voxel_coords * (bbox_max - bbox_min) + bbox_min
    
    # Plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(world_coords[:, 0], world_coords[:, 1], world_coords[:, 2],
              c='blue', marker='s', s=1, alpha=0.5)
    
    if 'points' in voxel_data:
        points = voxel_data['points'].cpu().numpy()
        ax.scatter(points[::10, 0], points[::10, 1], points[::10, 2],
                  c='red', marker='.', s=0.1, alpha=0.3, label='Input points')
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(title)
    ax.legend()
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Example usage
    generator = TemporalVoxelMaskGenerator(
        voxel_resolution=16,
        use_tsdf=False,  # Binary occupancy is usually sufficient
        padding=0.0
    )
    
    # Generate voxel mask from data directory
    voxel_data = generator.generate_voxel_mask(
        data_dir="/local/home/pmishra/cvg/arti-splatfacto/data_real/day5/multi",
        metadata_path="/local/home/pmishra/cvg/arti-splatfacto/data_real/day5/multi/transforms_aligned.json",
        subsample_rate=20,  
        dilate_iterations=1
    )
    
    # Visualize
    visualize_voxel_mask(voxel_data)
    
    print("\n✅ Temporal voxel mask generation complete!")