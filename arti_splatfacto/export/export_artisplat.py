#!/usr/bin/env python3
"""
URDF-Ready Articulated Object Export Script

Exports a trained articulated Gaussian Splatting model to URDF-compatible format:
- Background mesh (static environment)
- Per-joint canonical meshes (undeformed part geometry)
- Per-joint object meshes (with articulation applied)
- Joint metadata JSON (pivot points, axes, limits, types)

Usage:
python arti_splatfacto/export/export_artisplat.py --load-config data_real/day8/post_3/joint_0_recovery/arti_splatfacto_recovery/2025-11-24_190813/config.yml --load-config-splatfacto data_real/day8/pre/splatfacto/qed-splatter/2025-11-14_131705/config.yml --output_dir ./urdf_tsdf
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import open3d as o3d
import torch
import tyro
from tqdm import tqdm

from arti_splatfacto.model import splatfacto
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.scene_box import OrientedBox
from nerfstudio.models.splatfacto import SplatfactoModel
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.utils.rich_utils import CONSOLE

try:
    from gsplat.rendering import rasterization
except ImportError:
    print("Please install gsplat>=1.0.0")

from nerfstudio.utils.misc import torch_compile


@torch_compile()
def get_viewmat(optimized_camera_to_world):
    """
    function that converts c2w to gsplat world2camera matrix, using compile for some speed
    """
    R = optimized_camera_to_world[:, :3, :3]  # 3 x 3
    T = optimized_camera_to_world[:, :3, 3:4]  # 3 x 1
    # flip the z and y axes to align with gsplat conventions
    R = R * torch.tensor([[[1, -1, -1]]], device=R.device, dtype=R.dtype)
    # analytic matrix inverse to get world2camera matrix
    R_inv = R.transpose(1, 2)
    T_inv = -torch.bmm(R_inv, T)
    viewmat = torch.zeros(R.shape[0], 4, 4, device=R.device, dtype=R.dtype)
    viewmat[:, 3, 3] = 1.0  # homogenous
    viewmat[:, :3, :3] = R_inv
    viewmat[:, :3, 3:4] = T_inv
    return viewmat


@dataclass
class URDFExporter:
    """Export articulated Gaussian Splatting models to URDF-ready format"""

    load_config: Path
    """Path to the trained config YAML file."""

    load_config_splatfacto: Path
    """Path to the trained Splatfacto config YAML file."""
    
    output_dir: Path = Path("./export_data/")
    """Path to the output directory."""
    
    voxel_size: float = 0.01
    """TSDF voxel size for mesh reconstruction."""
    
    sdf_trunc: float = 0.06
    """TSDF truncation distance."""
    
    depth_trunc: float = 20.0
    """Maximum depth for TSDF integration."""
    
    num_cameras: int = 200
    """Number of cameras to use for mesh reconstruction."""
    
    background_only: bool = False
    """Export only the background mesh (no articulated parts)."""
    
    canonical_angle: Optional[float] = None
    """Joint angle for canonical pose. If None, uses middle of joint range."""
    
    object_angle: Optional[float] = None
    """Joint angle for object mesh. If None, uses max of joint range."""
    
    remove_small_clusters: bool = True
    """Remove small disconnected mesh clusters."""
    
    min_cluster_triangles: int = 50
    """Minimum number of triangles for a cluster to be kept."""

    background_crop_radius: float = 2.5
    """Radius (meters) around camera center to keep background geometry."""


    def get_outputs_subset(
        self,
        model,
        camera,
        subset: str,
        joint_id: str = None,
        joint_angle: Optional[float] = None,
    ):
        """
        Render specific subset (background / canonical / object) directly from model parameters.
        
        Args:
            model: ArtiSplatfacto model instance
            camera: Cameras object (1 view)
            subset: 'background', 'canonical', or 'object'
            joint_id: Which joint to use (for canonical/object)
            joint_angle: Override articulation angle (for 'object')
        """

        # 1. Select Gaussian subset
        if subset == "background":
            gaussians = model.gauss_params_fixed

        elif subset == "canonical":
            if joint_id is None:
                joint_id = model.config.active_joint_id
            gaussians = model.all_gauss_params_canon[joint_id]

        elif subset == "object":
            if joint_id is None:
                joint_id = model.config.active_joint_id
            base_gauss = model.all_gauss_params_obj[joint_id]
            if joint_angle is None:
                joint_angle = model.get_joint_angle_for_camera(camera, joint_id)
            # Apply articulation manually
            gaussians = model._apply_articulation_to_joint(base_gauss, joint_id, torch.tensor([joint_angle], device=model.device))
        else:
            raise ValueError(f"Unknown subset type '{subset}'")

        # 2. Prepare camera matrices
        viewmat = get_viewmat(camera.camera_to_worlds)
        K = camera.get_intrinsics_matrices().to(model.device)
        W, H = int(camera.width.item()), int(camera.height.item())

        # 3. Rasterize to get RGB + depth
        render, alpha, _ = rasterization(
            means=gaussians["means"],
            quats=gaussians["quats"],
            scales=torch.exp(gaussians["scales"]),
            opacities=torch.sigmoid(gaussians["opacities"]).squeeze(-1),
            colors=torch.cat(
                (gaussians["features_dc"][:, None, :], gaussians["features_rest"]), dim=1
            ),
            viewmats=viewmat,
            Ks=K,
            width=W,
            height=H,
            packed=True,
            near_plane=0.01,
            far_plane=20.0,
            render_mode="RGB+ED",  # enables depth output
            sh_degree=model.config.sh_degree,
        )

        # 4. Compose background
        background = model._get_background_color()
        rgb = torch.clamp(render[..., :3] + (1 - alpha) * background, 0.0, 1.0)
        depth = render[..., 3:4]

        return {
            "rgb": rgb.squeeze(0),
            "depth": depth.squeeze(0),
            "accumulation": alpha.squeeze(0),
        }


    def create_tsdf_volume(self) -> o3d.pipelines.integration.ScalableTSDFVolume:
        """Create an Open3D TSDF volume for mesh reconstruction."""
        return o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=self.voxel_size,
            sdf_trunc=self.sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
        )

    def integrate_depth_to_volume(
        self,
        volume: o3d.pipelines.integration.ScalableTSDFVolume,
        model: SplatfactoModel,
        cameras: Cameras,
        camera_indices: List[int],
        joint_angle: Optional[float] = None,
        joint_mask: Optional[torch.Tensor] = None,
    ):
        """
        Integrate depth maps from cameras into TSDF volume.
        
        Args:
            volume: Open3D TSDF volume to integrate into
            model: Trained Gaussian Splatting model
            cameras: Camera dataset
            camera_indices: Indices of cameras to use
            joint_angle: Optional joint angle override
            joint_mask: Optional mask for which Gaussians to include (True = include)
        """
        with torch.no_grad():
            for cam_idx in tqdm(camera_indices, desc="Integrating depth maps"):
                camera = cameras[cam_idx : cam_idx + 1].to(model.device)
                
                # Override joint angle if specified
                if joint_angle is not None:
                    if not hasattr(camera, "metadata") or camera.metadata is None:
                        camera.metadata = {}
                    camera.metadata["joint_angles"] = torch.tensor(
                        [joint_angle], device=model.device, dtype=torch.float32
                    )

                
                # Get model outputs
                outputs = self.get_outputs_subset(model, camera, subset=self.current_subset, joint_id=self.current_joint_id, joint_angle=self.current_joint_angle)
                
                if "depth" not in outputs:
                    raise KeyError("Model does not output depth maps!")
                
                depth_map = outputs["depth"]
                rgb_map = outputs["rgb"]
                
                # Apply joint mask if provided
                if joint_mask is not None:

                    pass
                
                # Prepare camera intrinsics
                H, W = camera.height.item(), camera.width.item()
                intrinsic = o3d.camera.PinholeCameraIntrinsic(
                    width=W,
                    height=H,
                    fx=camera.fx.item(),
                    fy=camera.fy.item(),
                    cx=camera.cx.item(),
                    cy=camera.cy.item(),
                )
                
                # Prepare camera extrinsics (c2w)
                c2w = torch.eye(4, dtype=torch.float, device=depth_map.device)
                c2w[:3, :4] = camera.camera_to_worlds.squeeze(0)
                # Convert from OpenGL to OpenCV convention
                c2w = c2w @ torch.diag(
                    torch.tensor([1, -1, -1, 1], device=c2w.device, dtype=torch.float)
                )
                
                # Create RGBD image
                rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    o3d.geometry.Image(
                        np.asarray(
                            rgb_map.cpu().numpy() * 255,
                            order="C",
                            dtype=np.uint8,
                        )
                    ),
                    o3d.geometry.Image(
                        np.asarray(depth_map.squeeze(-1).cpu().numpy(), order="C")
                    ),
                    depth_trunc=self.depth_trunc,
                    convert_rgb_to_intensity=False,
                    depth_scale=1.0,
                )
                
                # Integrate into volume
                volume.integrate(
                    rgbd,
                    intrinsic=intrinsic,
                    extrinsic=np.linalg.inv(c2w.cpu().numpy()),
                )

    def clean_mesh(self, mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
        """Remove small clusters and degenerate triangles from mesh."""
        if not self.remove_small_clusters:
            return mesh
        
        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug):
            (
                triangle_clusters,
                cluster_n_triangles,
                cluster_area,
            ) = mesh.cluster_connected_triangles()
        
        triangle_clusters = np.asarray(triangle_clusters)
        cluster_n_triangles = np.asarray(cluster_n_triangles)
        
        # Determine threshold (max of: 50th largest cluster or min_cluster_triangles)
        if len(cluster_n_triangles) > 50:
            n_cluster = np.sort(cluster_n_triangles)[-50]
        else:
            n_cluster = 0
        n_cluster = max(n_cluster, self.min_cluster_triangles)
        
        # Remove small clusters
        triangles_to_remove = cluster_n_triangles[triangle_clusters] < n_cluster
        mesh.remove_triangles_by_mask(triangles_to_remove)
        mesh.remove_unreferenced_vertices()
        mesh.remove_degenerate_triangles()
        
        return mesh


    def crop_mesh_by_camera_distance(
        self,
        mesh: o3d.geometry.TriangleMesh,
        cameras: Cameras,
        max_distance: float
    ) -> o3d.geometry.TriangleMesh:
        """
        Crop mesh to keep only geometry within distance from camera centers.
        
        Args:
            mesh: Input mesh
            cameras: Camera dataset (using poses as reference)
            max_distance: Maximum distance to keep (meters)
        
        Returns:
            Cropped mesh
        """
        # Extract camera positions from poses
        cam_positions = cameras.camera_to_worlds[:, :3, 3].cpu().numpy()  # [N, 3]
        cam_center = cam_positions.mean(axis=0)  # Average position
        
        # Get mesh vertices
        vertices = np.asarray(mesh.vertices)
        
        # Compute distance from camera center
        distances = np.linalg.norm(vertices - cam_center, axis=1)
        
        # Create mask for vertices to keep
        keep_mask = distances <= max_distance
        
        # Crop mesh
        mesh_cropped = mesh.select_by_index(np.where(keep_mask)[0])
        mesh_cropped.remove_unreferenced_vertices()
        mesh_cropped.remove_degenerate_triangles()
        
        CONSOLE.print(f"  Camera center: [{cam_center[0]:.3f}, {cam_center[1]:.3f}, {cam_center[2]:.3f}]")
        CONSOLE.print(f"  Crop radius: {max_distance}m")
        CONSOLE.print(f"  Cropped: {len(vertices)} → {len(mesh_cropped.vertices)} vertices")
        CONSOLE.print(f"  Removed: {(~keep_mask).sum()} vertices beyond {max_distance}m")
        
        return mesh_cropped


    def export_background_mesh(
        self,
        model: SplatfactoModel,
        cameras: Cameras,
        camera_indices: List[int],
        cameras_pre: Cameras,  # ADD THIS PARAMETER
    ) -> o3d.geometry.TriangleMesh:
        """Export background mesh with cropping based on cameras_pre."""
        CONSOLE.print("\n[bold green]Exporting background mesh...[/bold green]")

        self.current_subset = "background"
        self.current_joint_id = None
        self.current_joint_angle = None

        volume = self.create_tsdf_volume()
        self.integrate_depth_to_volume(volume, model, cameras, camera_indices)
        
        mesh = volume.extract_triangle_mesh()
        mesh = self.clean_mesh(mesh)
        
        # Crop background mesh using cameras_pre
        CONSOLE.print("\n[bold cyan]Cropping background mesh...[/bold cyan]")
        mesh = self.crop_mesh_by_camera_distance(
            mesh,
            cameras=cameras_pre,  
            max_distance=self.background_crop_radius
        )
        
        # Save mesh
        mesh_path = self.output_dir / "meshes" / "background.ply"
        mesh_path.parent.mkdir(parents=True, exist_ok=True)
        o3d.io.write_triangle_mesh(str(mesh_path), mesh)
        
        CONSOLE.print(f"✓ Background mesh saved: {mesh_path}")
        CONSOLE.print(f"  Vertices: {len(mesh.vertices)}, Triangles: {len(mesh.triangles)}")
        
        return mesh

    def export_joint_meshes(
        self,
        model: SplatfactoModel,
        cameras: Cameras,
        camera_indices: List[int],
        joint_id: int,
    ) -> Tuple[o3d.geometry.TriangleMesh, o3d.geometry.TriangleMesh]:
        """
        Export canonical and object meshes for a specific joint.
        
        Returns:
            canonical_mesh: Mesh at canonical pose
            object_mesh: Mesh at object pose (articulated)
        """
        CONSOLE.print(f"\n[bold green]Exporting meshes for joint {joint_id}...[/bold green]")
        
        # Get joint angles
        canonical_angle = self.get_canonical_angle(model)
        object_angle = self.get_object_angle(model)
        
        # Create mask for this joint's Gaussians
        # TODO: Implement proper joint masking from model
        # For now, we'll render at different poses
        
        # Export canonical mesh
        CONSOLE.print(f"\n[bold green]Exporting canonical mesh for joint {joint_id}...[/bold green]")

        self.current_subset = "canonical"
        self.current_joint_id = f"joint_{joint_id}"
        self.current_joint_angle = self.get_object_angle(model)  # max → fully open

        canonical_volume = self.create_tsdf_volume()
        self.integrate_depth_to_volume(canonical_volume, model, cameras, camera_indices)
        canonical_mesh = canonical_volume.extract_triangle_mesh()
        canonical_mesh = self.clean_mesh(canonical_mesh)
        
        CONSOLE.print(f"\n[bold green]Exporting object mesh for joint {joint_id}...[/bold green]")

        self.current_subset = "object"
        self.current_joint_id = f"joint_{joint_id}"
        self.current_joint_angle = self.get_canonical_angle(model)  # min or mid → closed

        object_volume = self.create_tsdf_volume()
        self.integrate_depth_to_volume(object_volume, model, cameras, camera_indices)

        object_mesh = object_volume.extract_triangle_mesh()
        object_mesh = self.clean_mesh(object_mesh)
        
        # Save meshes
        meshes_dir = self.output_dir / "meshes"
        meshes_dir.mkdir(parents=True, exist_ok=True)
        
        canonical_path = meshes_dir / f"joint_{joint_id}_canonical.ply"
        object_path = meshes_dir / f"joint_{joint_id}_obj.ply"
        
        o3d.io.write_triangle_mesh(str(canonical_path), canonical_mesh)
        o3d.io.write_triangle_mesh(str(object_path), object_mesh)
        
        CONSOLE.print(f"✓ Canonical mesh saved: {canonical_path}")
        CONSOLE.print(f"  Vertices: {len(canonical_mesh.vertices)}, Triangles: {len(canonical_mesh.triangles)}")
        CONSOLE.print(f"✓ Object mesh saved: {object_path}")
        CONSOLE.print(f"  Vertices: {len(object_mesh.vertices)}, Triangles: {len(object_mesh.triangles)}")
        
        return canonical_mesh, object_mesh

    def get_canonical_angle(self, model: SplatfactoModel) -> float:
        """Get the canonical joint angle (middle of range by default)."""
        if self.canonical_angle is not None:
            return self.canonical_angle
        
        joint_min, joint_max = model.joint_limits[0].item(), model.joint_limits[1].item()
        return (joint_min + joint_max) / 3.0

    def get_object_angle(self, model: SplatfactoModel) -> float:
        """Get the object mesh joint angle (max of range by default)."""
        if self.object_angle is not None:
            return self.object_angle
        
        joint_min, joint_max = model.joint_limits[0].item(), model.joint_limits[1].item()
        return joint_max

    def extract_joint_metadata(self, model: SplatfactoModel) -> Dict:
        """
        Extract joint metadata from the model.
        
        Returns a dictionary containing:
        - joint_type: 'revolute' or 'prismatic'
        - joint_limits: [min, max] in radians or meters
        - pivot: [x, y, z] (for revolute joints)
        - joint_axis: [x, y, z] unit vector
        - num_joints: Number of articulated joints
        """
        metadata = {
            "joint_type": model.joint_type if hasattr(model, "joint_type") else "unknown",
            "num_joints": 1, 
            "joints": []
        }
        
        # Extract joint limits
        if hasattr(model, "joint_limits"):
            joint_limits = model.joint_limits.cpu().numpy().tolist()
            joint_min, joint_max = joint_limits
        else:
            joint_min, joint_max = 0.0, 1.0
        

        # Extract pivot point and axis (for revolute joints)
        if hasattr(model, "joint_pivot"):
            pivot = model.joint_pivot.detach().cpu().numpy().tolist()
        else:
            pivot = [0.0, 0.0, 0.0]
        
        if hasattr(model, "joint_axis"):
            joint_axis = model.joint_axis.detach().cpu().numpy().tolist()
            # Normalize axis
            axis_norm = np.linalg.norm(joint_axis)
            if axis_norm > 0:
                joint_axis = (np.array(joint_axis) / axis_norm).tolist()
        else:
            joint_axis = [0.0, 0.0, 1.0]  # Default Z-axis
        
        # Add joint information
        joint_info = {
            "joint_id": 0,
            "type": metadata["joint_type"],
            "limits": {
                "min": joint_min,
                "max": joint_max,
                "unit": "radians" if metadata["joint_type"] == "revolute" else "meters"
            },
            "pivot_point": pivot,
            "axis": joint_axis,
            "canonical_angle": self.get_canonical_angle(model),
            "object_angle": self.get_object_angle(model),
        }
        
        metadata["joints"].append(joint_info)
        
        return metadata

    def save_metadata(self, metadata: Dict):
        """Save joint metadata to JSON file."""
        metadata_path = self.output_dir / "joint_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        CONSOLE.print(f"\n[bold green]✓ Metadata saved: {metadata_path}[/bold green]")

    def generate_urdf_template(self, metadata: Dict) -> str:
        """Generate a basic URDF template from metadata."""
        joint_info = metadata["joints"][0]
        joint_type = joint_info["type"]
        
        if joint_type == "revolute":
            axis = " ".join(map(str, joint_info["axis"]))
            origin_xyz = " ".join(map(str, joint_info["pivot"]))
            
            urdf = f"""<?xml version="1.0"?>
<robot name="articulated_object">
  
  <!-- Base link (background/static environment) -->
  <link name="base_link">
    <visual>
      <geometry>
        <mesh filename="meshes/background.ply" scale="1 1 1"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <mesh filename="meshes/background.ply" scale="1 1 1"/>
      </geometry>
    </collision>
  </link>
  
  <!-- Articulated link (movable part) -->
  <link name="joint_0_link">
    <visual>
      <geometry>
        <mesh filename="meshes/joint_0_obj.ply" scale="1 1 1"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <mesh filename="meshes/joint_0_obj.ply" scale="1 1 1"/>
      </geometry>
    </collision>
  </link>
  
  <!-- Joint connecting base to articulated part -->
  <joint name="joint_0" type="{joint_type}">
    <parent link="base_link"/>
    <child link="joint_0_link"/>
    <origin xyz="{origin_xyz}" rpy="0 0 0"/>
    <axis xyz="{axis}"/>
    <limit lower="{joint_info['limits']['min']}" 
           upper="{joint_info['limits']['max']}" 
           effort="100" 
           velocity="1.0"/>
  </joint>
  
</robot>
"""
        else:  # prismatic
            axis = " ".join(map(str, joint_info["axis"]))
            origin_xyz = " ".join(map(str, joint_info["pivot"]))
            
            urdf = f"""<?xml version="1.0"?>
<robot name="articulated_object">
  
  <link name="base_link">
    <visual>
      <geometry>
        <mesh filename="meshes/background.ply" scale="1 1 1"/>
      </geometry>
    </visual>
  </link>
  
  <link name="joint_0_link">
    <visual>
      <geometry>
        <mesh filename="meshes/joint_0_obj.ply" scale="1 1 1"/>
      </geometry>
    </visual>
  </link>
  
  <joint name="joint_0" type="prismatic">
    <parent link="base_link"/>
    <child link="joint_0_link"/>
    <origin xyz="{origin_xyz}" rpy="0 0 0"/>
    <axis xyz="{axis}"/>
    <limit lower="{joint_info['limits']['min']}" 
           upper="{joint_info['limits']['max']}" 
           effort="100" 
           velocity="1.0"/>
  </joint>
  
</robot>
"""
        
        return urdf

    def save_urdf(self, metadata: Dict):
        """Save URDF template file."""
        urdf_content = self.generate_urdf_template(metadata)
        urdf_path = self.output_dir / "robot.urdf"
        
        with open(urdf_path, 'w') as f:
            f.write(urdf_content)
        
        CONSOLE.print(f"[bold green]✓ URDF template saved: {urdf_path}[/bold green]")
        CONSOLE.print("[yellow]⚠ This is a template URDF. You may need to adjust:")
        CONSOLE.print("  - Mesh scales and origins")
        CONSOLE.print("  - Mass and inertia properties")
        CONSOLE.print("  - Joint effort and velocity limits[/yellow]")

    def main(self):
        """Main export pipeline."""
        CONSOLE.print("[bold blue]═══════════════════════════════════════════[/bold blue]")
        CONSOLE.print("[bold blue]   URDF-Ready Articulated Object Export   [/bold blue]")
        CONSOLE.print("[bold blue]═══════════════════════════════════════════[/bold blue]")
        
        # Create output directory
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)
        
        # Load model
        CONSOLE.print(f"\n[1/5] Loading model from: {self.load_config}")
        _, pipeline, _, _ = eval_setup(self.load_config)
        _, splatfacto_pipeline, _, _ = eval_setup(self.load_config_splatfacto)
        
        # if not isinstance(pipeline.model, SplatfactoModel):
        #     raise TypeError(f"Expected SplatfactoModel, got {type(pipeline.model)}")
        
        model: SplatfactoModel = pipeline.model
        
        # Verify joint-conditioned model
        if not hasattr(model, 'joint_angle_deltas'):
            CONSOLE.print("[yellow]⚠ Warning: Model does not have joint_angle_deltas.[/yellow]")
            CONSOLE.print("[yellow]  This may not be a joint-conditioned model![/yellow]")
        



        # Assume both camera sets are from Nerfstudio and share intrinsics
        cameras_pre: Cameras = splatfacto_pipeline.datamanager.train_dataset.cameras
        cameras_post: Cameras = pipeline.datamanager.train_dataset.cameras

        # Concatenate poses
        camera_to_worlds_all = torch.cat(
            [cameras_pre.camera_to_worlds, cameras_post.camera_to_worlds], dim=0
        )

        # Get number of total cameras
        num_all = camera_to_worlds_all.shape[0]

        # Create new Cameras instance reusing shared intrinsics and types
        cameras_all = Cameras(
            camera_to_worlds=camera_to_worlds_all,
            fx=cameras_pre.fx[0],  
            fy=cameras_pre.fy[0],
            cx=cameras_pre.cx[0],
            cy=cameras_pre.cy[0],
            width=cameras_pre.width[0].item(),
            height=cameras_pre.height[0].item(),
            distortion_params=(
                cameras_pre.distortion_params[0]
                if cameras_pre.distortion_params is not None
                else None
            ),
            camera_type=cameras_pre.camera_type[0].item(),
        )


        num_cameras_all = len(cameras_all)
        self.num_cameras = min(self.num_cameras, num_cameras_all)
        camera_indices_all = np.linspace(0, num_cameras_all - 1, self.num_cameras, dtype=int).tolist()

        CONSOLE.print(f"\n[bold cyan]Camera Selection:[/bold cyan]")
        CONSOLE.print(f"  Combined (pre + post) poses: {len(camera_indices_all)} / {num_cameras_all} cameras")




        # Extract joint metadata
        CONSOLE.print("\n[2/5] Extracting joint metadata...")
        metadata = self.extract_joint_metadata(model)
        self.save_metadata(metadata)
        
        CONSOLE.print(f"  Joint type: {metadata['joint_type']}")
        CONSOLE.print(f"  Number of joints: {metadata['num_joints']}")
        for joint in metadata["joints"]:
            CONSOLE.print(f"  Joint {joint['joint_id']}: "
                         f"limits=[{joint['limits']['min']:.3f}, {joint['limits']['max']:.3f}] "
                         f"{joint['limits']['unit']}")
        
        # Export background mesh
        CONSOLE.print("\n[3/5] Exporting background mesh...")


        if not self.background_only:
            background_mesh = self.export_background_mesh(
                model, 
                cameras_all, 
                camera_indices_all,
                cameras_pre=cameras_pre  # Pass cameras_pre for cropping
            )
            
            del background_mesh
            torch.cuda.empty_cache()
        # if not self.background_only:
        #     background_mesh = self.export_background_mesh(model, cameras_all, camera_indices_all)
        

        #     del background_mesh
        #     torch.cuda.empty_cache()
        # Export joint meshes
        if not self.background_only:
            CONSOLE.print("\n[4/5] Exporting articulated joint meshes...")
            num_joints = metadata["num_joints"]
            
            for joint_id in range(num_joints):
                canonical_mesh, object_mesh = self.export_joint_meshes(
                    model, cameras_all, camera_indices_all, joint_id
                )
        
        # Generate URDF
        CONSOLE.print("\n[5/5] Generating URDF template...")
        # self.save_urdf(metadata)
        
        # Summary
        CONSOLE.print("\n[bold green]═══════════════════════════════════════════[/bold green]")
        CONSOLE.print("[bold green]         Export Complete! ✓                [/bold green]")
        CONSOLE.print("[bold green]═══════════════════════════════════════════[/bold green]")
        CONSOLE.print(f"\nExported to: {self.output_dir}")
        CONSOLE.print("\nDirectory structure:")
        CONSOLE.print("  urdf_export/")
        CONSOLE.print("  ├── meshes/")
        CONSOLE.print("  │   ├── background.ply")
        for joint_id in range(metadata["num_joints"]):
            CONSOLE.print(f"  │   ├── joint_{joint_id}_canonical.ply")
            CONSOLE.print(f"  │   └── joint_{joint_id}_obj.ply")
        CONSOLE.print("  ├── joint_metadata.json")
        CONSOLE.print("  └── robot.urdf")


if __name__ == "__main__":
    tyro.cli(URDFExporter).main()