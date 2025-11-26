#!/usr/bin/env python3
"""
Isaac Sim Articulated Object Preparation Script

Takes URDF-exported meshes and prepares them for Isaac Sim:
- Re-centers meshes to proper link-local origins
- Generates simplified collision meshes
- Creates physics-ready URDF with proper inertial properties
- Validates and imports into Isaac Sim (optional)

Usage:
    python isaac_sim_prepare.py --input_dir ./urdf_export --output_dir ./isaac_sim_ready
"""

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import open3d as o3d
import tyro
from tqdm import tqdm

try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    print("Warning: trimesh not available. Install with: pip install trimesh")
    TRIMESH_AVAILABLE = False


@dataclass
class IsaacSimPreparer:
    """Prepare articulated meshes for Isaac Sim"""

    input_dir: Path = Path("./urdf_export/")
    """Path to the URDF export directory."""

    output_dir: Path = Path("./isaac_sim_ready/")
    """Path to the output directory for Isaac Sim."""

    collision_simplification: str = "convex_hull"
    """Collision mesh simplification method: 'convex_hull', 'vhacd', or 'none'."""

    vhacd_resolution: int = 100000
    """VHACD resolution (higher = more accurate but slower)."""

    vhacd_max_hulls: int = 32
    """Maximum number of convex hulls for VHACD."""

    mesh_scale: float = 1.0
    """Scale factor for all meshes."""

    default_density: float = 1000.0
    """Default density in kg/m³ for inertia calculations."""

    min_inertia: float = 1e-6
    """Minimum inertia value to avoid numerical issues."""

    export_format: str = "obj"
    """Mesh export format: 'obj', 'stl', or 'ply'."""

    recenter_meshes: bool = True
    """Recenter meshes to their centroids for better numerical stability."""

    validate_meshes: bool = True
    """Validate meshes for watertightness and manifoldness."""

    def load_metadata(self) -> Dict:
        """Load joint metadata from the export directory."""
        metadata_path = self.input_dir / "joint_metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        print(f"✓ Loaded metadata: {metadata['num_joints']} joint(s)")
        return metadata

    def load_mesh(self, mesh_path: Path) -> o3d.geometry.TriangleMesh:
        """Load a mesh file."""
        if not mesh_path.exists():
            raise FileNotFoundError(f"Mesh not found: {mesh_path}")

        mesh = o3d.io.read_triangle_mesh(str(mesh_path))
        if not mesh.has_vertices():
            raise ValueError(f"Mesh has no vertices: {mesh_path}")

        print(f"  Loaded: {mesh_path.name} ({len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles)")
        return mesh

    def validate_mesh(self, mesh: o3d.geometry.TriangleMesh, name: str) -> Dict[str, bool]:
        """Validate mesh properties."""
        if not self.validate_meshes:
            return {}

        results = {
            "is_watertight": mesh.is_watertight(),
            "is_orientable": mesh.is_orientable(),
            "is_edge_manifold": mesh.is_edge_manifold(),
            "is_vertex_manifold": mesh.is_vertex_manifold(),
        }

        print(f"  Validation for {name}:")
        for key, value in results.items():
            status = "✓" if value else "✗"
            print(f"    {status} {key}: {value}")

        return results

    def compute_mesh_centroid(self, mesh: o3d.geometry.TriangleMesh) -> np.ndarray:
        """Compute the centroid of a mesh."""
        vertices = np.asarray(mesh.vertices)
        return vertices.mean(axis=0)

    def recenter_mesh(self, mesh: o3d.geometry.TriangleMesh, offset: np.ndarray) -> o3d.geometry.TriangleMesh:
        """Recenter mesh by subtracting offset from all vertices."""
        mesh_centered = o3d.geometry.TriangleMesh(mesh)
        vertices = np.asarray(mesh_centered.vertices)
        vertices -= offset
        mesh_centered.vertices = o3d.utility.Vector3dVector(vertices)
        return mesh_centered

    def compute_mesh_inertia(
        self,
        mesh: o3d.geometry.TriangleMesh,
        density: float = None
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Compute mass, center of mass, and inertia tensor for a mesh.

        Returns:
            mass: Total mass in kg
            com: Center of mass [x, y, z]
            inertia: 3x3 inertia tensor
        """
        if density is None:
            density = self.default_density

        if not TRIMESH_AVAILABLE:
            # Fallback: simple box approximation
            vertices = np.asarray(mesh.vertices)
            bbox_min = vertices.min(axis=0)
            bbox_max = vertices.max(axis=0)
            dims = bbox_max - bbox_min
            volume = np.prod(dims)
            mass = volume * density

            # Box inertia
            inertia = np.diag([
                (dims[1]**2 + dims[2]**2) / 12.0 * mass,
                (dims[0]**2 + dims[2]**2) / 12.0 * mass,
                (dims[0]**2 + dims[1]**2) / 12.0 * mass,
            ])
            com = (bbox_min + bbox_max) / 2.0

            print(f"    Warning: Using box approximation for inertia (install trimesh for accurate computation)")
        else:
            # Use trimesh for accurate computation
            vertices = np.asarray(mesh.vertices)
            triangles = np.asarray(mesh.triangles)
            tmesh = trimesh.Trimesh(vertices=vertices, faces=triangles)

            if not tmesh.is_watertight:
                print(f"    Warning: Mesh is not watertight, inertia may be inaccurate")

            tmesh.density = density
            mass = tmesh.mass
            com = tmesh.center_mass
            inertia = tmesh.moment_inertia

        # Ensure minimum inertia
        inertia = np.maximum(inertia, np.eye(3) * self.min_inertia)
        mass = abs(mass)
        inertia = np.abs(inertia)
        return mass, com, inertia

    def create_convex_hull_collision(
        self,
        mesh: o3d.geometry.TriangleMesh
    ) -> o3d.geometry.TriangleMesh:
        """Create a convex hull collision mesh."""
        hull, _ = mesh.compute_convex_hull()
        print(f"    Created convex hull: {len(hull.vertices)} vertices, {len(hull.triangles)} triangles")
        return hull

    def create_vhacd_collision(
        self,
        mesh: o3d.geometry.TriangleMesh
    ) -> List[o3d.geometry.TriangleMesh]:
        """Create VHACD (Volumetric Hierarchical Approximate Convex Decomposition) collision meshes."""
        if not TRIMESH_AVAILABLE:
            print("    Warning: trimesh not available, falling back to convex hull")
            return [self.create_convex_hull_collision(mesh)]

        try:
            import pyvhacd
        except ImportError:
            print("    Warning: pyvhacd not available (pip install pyvhacd), falling back to convex hull")
            return [self.create_convex_hull_collision(mesh)]

        # Convert to trimesh
        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)
        tmesh = trimesh.Trimesh(vertices=vertices, faces=triangles)

        # Run VHACD
        print(f"    Running VHACD (this may take a while)...")
        hulls = []
        result = pyvhacd.compute_vhacd(
            vertices=vertices,
            triangles=triangles,
            resolution=self.vhacd_resolution,
            max_num_vertices_per_ch=64,
            max_convex_hulls=self.vhacd_max_hulls,
        )

        # Convert back to Open3D meshes
        for i, (verts, tris) in enumerate(zip(result[0], result[1])):
            hull = o3d.geometry.TriangleMesh()
            hull.vertices = o3d.utility.Vector3dVector(verts)
            hull.triangles = o3d.utility.Vector3iVector(tris)
            hulls.append(hull)

        print(f"    Created {len(hulls)} convex hulls")
        return hulls

    def save_mesh(
        self,
        mesh: o3d.geometry.TriangleMesh,
        output_path: Path,
        format: str = None
    ):
        """Save mesh to file."""
        if format is None:
            format = self.export_format

        output_path = output_path.with_suffix(f".{format}")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        o3d.io.write_triangle_mesh(str(output_path), mesh)
        print(f"  Saved: {output_path}")

    def process_background_mesh(self, metadata: Dict) -> Tuple[np.ndarray, Dict]:
        """
        Process the background (static) mesh.

        Returns:
            centroid: Mesh centroid (for coordinate system)
            inertial: Inertial properties
        """
        print("\n[1/3] Processing background mesh...")

        mesh_path = self.input_dir / "meshes" / "background.ply"
        mesh = self.load_mesh(mesh_path)

        # Validate
        self.validate_mesh(mesh, "background")

        # Compute centroid (will be used as world origin)
        centroid = self.compute_mesh_centroid(mesh)
        print(f"  Centroid: [{centroid[0]:.4f}, {centroid[1]:.4f}, {centroid[2]:.4f}]")

        # Recenter if requested
        if self.recenter_meshes:
            mesh = self.recenter_mesh(mesh, centroid)
            print(f"  Recentered mesh to origin")

        # Compute inertial properties
        mass, com, inertia = self.compute_mesh_inertia(mesh)
        inertial = {
            "mass": float(mass),
            "com": com.tolist(),
            "inertia": inertia.tolist(),
        }
        print("  Inertial properties:")
        print("  Inertia Tensor:")
        print(inertia)
        print(f"  Mass: {mass:.4f} kg")
        print(f"  COM: [{com[0]:.4f}, {com[1]:.4f}, {com[2]:.4f}]")

        # Save visual mesh
        visual_path = self.output_dir / "meshes" / "visual" / "background"
        self.save_mesh(mesh, visual_path)

        # Create collision mesh
        print("  Creating collision mesh...")
        if self.collision_simplification == "convex_hull":
            collision_mesh = self.create_convex_hull_collision(mesh)
            collision_path = self.output_dir / "meshes" / "collision" / "background"
            self.save_mesh(collision_mesh, collision_path)

        elif self.collision_simplification == "vhacd":
            collision_meshes = self.create_vhacd_collision(mesh)
            for i, collision_mesh in enumerate(collision_meshes):
                collision_path = self.output_dir / "meshes" / "collision" / f"background_{i}"
                self.save_mesh(collision_mesh, collision_path)

        elif self.collision_simplification == "none":
            collision_path = self.output_dir / "meshes" / "collision" / "background"
            self.save_mesh(mesh, collision_path)

        return centroid, inertial

    def process_joint_meshes(
        self,
        joint_id: int,
        metadata: Dict,
        world_origin: np.ndarray
    ) -> Tuple[Dict, Dict]:
        """
        Process canonical and object meshes for a joint.

        Returns:
            canonical_inertial: Inertial properties for canonical mesh
            object_inertial: Inertial properties for object mesh
        """
        print(f"\n[2/3] Processing joint {joint_id} meshes...")

        joint_info = metadata["joints"][joint_id]

        # Load meshes
        canonical_path = self.input_dir / "meshes" / f"joint_{joint_id}_canonical.ply"
        object_path = self.input_dir / "meshes" / f"joint_{joint_id}_obj.ply"

        canonical_mesh = self.load_mesh(canonical_path)
        object_mesh = self.load_mesh(object_path)

        # Validate
        self.validate_mesh(canonical_mesh, f"joint_{joint_id}_canonical")
        self.validate_mesh(object_mesh, f"joint_{joint_id}_obj")

        # Get pivot point in world coordinates
        pivot_point = np.array(joint_info["pivot_point"])
        print(f"  Pivot point (world): [{pivot_point[0]:.4f}, {pivot_point[1]:.4f}, {pivot_point[2]:.4f}]")

        # Recenter meshes relative to pivot point
        if self.recenter_meshes:
            canonical_mesh = self.recenter_mesh(canonical_mesh, pivot_point)
            object_mesh = self.recenter_mesh(object_mesh, pivot_point)
            print(f"  Recentered meshes to pivot point")

            # Update pivot point to be relative to world origin
            pivot_point_relative = pivot_point - world_origin
            joint_info["pivot_point_relative"] = pivot_point_relative.tolist()
            print(f"  Pivot point (relative to origin): [{pivot_point_relative[0]:.4f}, {pivot_point_relative[1]:.4f}, {pivot_point_relative[2]:.4f}]")

        # Compute inertial properties
        print("  Computing inertial properties...")
        
        canonical_mass, canonical_com, canonical_inertia = self.compute_mesh_inertia(canonical_mesh)
        canonical_inertial = {
            "mass": float(canonical_mass),
            "com": canonical_com.tolist(),
            "inertia": canonical_inertia.tolist(),
        }
        print(f"    Canonical - Mass: {canonical_mass:.4f} kg, COM: [{canonical_com[0]:.4f}, {canonical_com[1]:.4f}, {canonical_com[2]:.4f}]")

        object_mass, object_com, object_inertia = self.compute_mesh_inertia(object_mesh)
        object_inertial = {
            "mass": float(object_mass),
            "com": object_com.tolist(),
            "inertia": object_inertia.tolist(),
        }
        print(f"    Object - Mass: {object_mass:.4f} kg, COM: [{object_com[0]:.4f}, {object_com[1]:.4f}, {object_com[2]:.4f}]")

        # Save visual meshes
        canonical_visual_path = self.output_dir / "meshes" / "visual" / f"joint_{joint_id}_canonical"
        object_visual_path = self.output_dir / "meshes" / "visual" / f"joint_{joint_id}_obj"
        
        self.save_mesh(canonical_mesh, canonical_visual_path)
        self.save_mesh(object_mesh, object_visual_path)

        # Create collision meshes
        print("  Creating collision meshes...")
        
        # Canonical collision
        if self.collision_simplification == "convex_hull":
            canonical_collision = self.create_convex_hull_collision(canonical_mesh)
            canonical_collision_path = self.output_dir / "meshes" / "collision" / f"joint_{joint_id}_canonical"
            self.save_mesh(canonical_collision, canonical_collision_path)

        elif self.collision_simplification == "vhacd":
            canonical_collisions = self.create_vhacd_collision(canonical_mesh)
            for i, collision_mesh in enumerate(canonical_collisions):
                collision_path = self.output_dir / "meshes" / "collision" / f"joint_{joint_id}_canonical_{i}"
                self.save_mesh(collision_mesh, collision_path)

        elif self.collision_simplification == "none":
            canonical_collision_path = self.output_dir / "meshes" / "collision" / f"joint_{joint_id}_canonical"
            self.save_mesh(canonical_mesh, canonical_collision_path)

        # Object collision
        if self.collision_simplification == "convex_hull":
            object_collision = self.create_convex_hull_collision(object_mesh)
            object_collision_path = self.output_dir / "meshes" / "collision" / f"joint_{joint_id}_obj"
            self.save_mesh(object_collision, object_collision_path)

        elif self.collision_simplification == "vhacd":
            object_collisions = self.create_vhacd_collision(object_mesh)
            for i, collision_mesh in enumerate(object_collisions):
                collision_path = self.output_dir / "meshes" / "collision" / f"joint_{joint_id}_obj_{i}"
                self.save_mesh(collision_mesh, collision_path)

        elif self.collision_simplification == "none":
            object_collision_path = self.output_dir / "meshes" / "collision" / f"joint_{joint_id}_obj"
            self.save_mesh(object_mesh, object_collision_path)
            

        return canonical_inertial, object_inertial

    def generate_urdf(
        self,
        metadata: Dict,
        background_inertial: Dict,
        joint_inertials: List[Tuple[Dict, Dict]]
    ) -> str:
        """Generate a complete URDF with inertial properties."""
        print("\n[3/3] Generating URDF...")

        urdf_lines = ['<?xml version="1.0"?>']
        urdf_lines.append('<robot name="articulated_object">')
        urdf_lines.append('')

        # Helper function to format inertia matrix
        def format_inertia(inertia_matrix):
            i = inertia_matrix
            return (f'ixx="{i[0][0]:.6e}" ixy="{i[0][1]:.6e}" ixz="{i[0][2]:.6e}" '
                    f'iyy="{i[1][1]:.6e}" iyz="{i[1][2]:.6e}" izz="{i[2][2]:.6e}"')

        # Background link (base_link)
        urdf_lines.append('  <!-- Base link (static environment) -->')
        urdf_lines.append('  <link name="base_link">')
        urdf_lines.append('    <inertial>')
        urdf_lines.append(f'      <origin xyz="{background_inertial["com"][0]:.6f} {background_inertial["com"][1]:.6f} {background_inertial["com"][2]:.6f}" rpy="0 0 0"/>')
        urdf_lines.append(f'      <mass value="{background_inertial["mass"]:.6f}"/>')
        urdf_lines.append(f'      <inertia {format_inertia(background_inertial["inertia"])}/>')
        urdf_lines.append('    </inertial>')
        urdf_lines.append('    <visual>')
        urdf_lines.append('      <origin xyz="0 0 0" rpy="0 0 0"/>')
        urdf_lines.append('      <geometry>')
        urdf_lines.append(f'        <mesh filename="meshes/visual/background.{self.export_format}" scale="{self.mesh_scale} {self.mesh_scale} {self.mesh_scale}"/>')
        urdf_lines.append('      </geometry>')
        urdf_lines.append('    </visual>')
        urdf_lines.append('    <collision>')
        urdf_lines.append('      <origin xyz="0 0 0" rpy="0 0 0"/>')
        urdf_lines.append('      <geometry>')
        urdf_lines.append(f'        <mesh filename="meshes/collision/background.{self.export_format}" scale="{self.mesh_scale} {self.mesh_scale} {self.mesh_scale}"/>')
        urdf_lines.append('      </geometry>')
        urdf_lines.append('    </collision>')
        urdf_lines.append('  </link>')
        urdf_lines.append('')

        # Articulated links and joints
        for joint_id, (canonical_inertial, object_inertial) in enumerate(joint_inertials):
            joint_info = metadata["joints"][joint_id]
            joint_type = joint_info["type"]
            axis = joint_info["axis"]

            # Use relative pivot point if available, otherwise use original
            if "pivot_point_relative" in joint_info:
                pivot = joint_info["pivot_point_relative"]
            else:
                pivot = joint_info["pivot_point"]

            # Articulated link
            urdf_lines.append(f'  <!-- Articulated link {joint_id} -->')
            urdf_lines.append(f'  <link name="joint_{joint_id}_link">')
            urdf_lines.append('    <inertial>')
            urdf_lines.append(f'      <origin xyz="{object_inertial["com"][0]:.6f} {object_inertial["com"][1]:.6f} {object_inertial["com"][2]:.6f}" rpy="0 0 0"/>')
            urdf_lines.append(f'      <mass value="{object_inertial["mass"]:.6f}"/>')
            urdf_lines.append(f'      <inertia {format_inertia(object_inertial["inertia"])}/>')
            urdf_lines.append('    </inertial>')
            urdf_lines.append('    <visual>')
            urdf_lines.append('      <origin xyz="0 0 0" rpy="0 0 0"/>')
            urdf_lines.append('      <geometry>')
            urdf_lines.append(f'        <mesh filename="meshes/visual/joint_{joint_id}_obj.{self.export_format}" scale="{self.mesh_scale} {self.mesh_scale} {self.mesh_scale}"/>')
            urdf_lines.append('      </geometry>')
            urdf_lines.append('    </visual>')
            urdf_lines.append('    <collision>')
            urdf_lines.append('      <origin xyz="0 0 0" rpy="0 0 0"/>')
            urdf_lines.append('      <geometry>')
            urdf_lines.append(f'        <mesh filename="meshes/collision/joint_{joint_id}_obj.{self.export_format}" scale="{self.mesh_scale} {self.mesh_scale} {self.mesh_scale}"/>')
            urdf_lines.append('      </geometry>')
            urdf_lines.append('    </collision>')
            urdf_lines.append('  </link>')
            urdf_lines.append('')

            # Joint
            urdf_lines.append(f'  <!-- Joint {joint_id} -->')
            urdf_lines.append(f'  <joint name="joint_{joint_id}" type="{joint_type}">')
            urdf_lines.append('    <parent link="base_link"/>')
            urdf_lines.append(f'    <child link="joint_{joint_id}_link"/>')
            urdf_lines.append(f'    <origin xyz="{pivot[0]:.6f} {pivot[1]:.6f} {pivot[2]:.6f}" rpy="0 0 0"/>')
            urdf_lines.append(f'    <axis xyz="{axis[0]:.6f} {axis[1]:.6f} {axis[2]:.6f}"/>')

            limits = joint_info["limits"]
            urdf_lines.append(f'    <limit lower="{limits["min"]:.6f}" upper="{limits["max"]:.6f}" effort="100.0" velocity="1.0"/>')

            # Add dynamics for better simulation
            urdf_lines.append('    <dynamics damping="0.1" friction="0.1"/>')
            urdf_lines.append('  </joint>')
            urdf_lines.append('')

        urdf_lines.append('</robot>')

        urdf_content = '\n'.join(urdf_lines)
        return urdf_content

    def save_urdf(self, urdf_content: str):
        """Save URDF file."""
        urdf_path = self.output_dir / "robot.urdf"
        urdf_path.parent.mkdir(parents=True, exist_ok=True)

        with open(urdf_path, 'w') as f:
            f.write(urdf_content)

        print(f"✓ URDF saved: {urdf_path}")

    def save_isaac_sim_metadata(self, metadata: Dict):
        """Save Isaac Sim-specific metadata."""
        isaac_metadata = {
            "source": "articulated_gaussian_splatting",
            "mesh_scale": self.mesh_scale,
            "collision_method": self.collision_simplification,
            "original_metadata": metadata,
        }

        metadata_path = self.output_dir / "isaac_sim_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(isaac_metadata, f, indent=2)

        print(f"✓ Isaac Sim metadata saved: {metadata_path}")



    def main(self):
        """Main preparation pipeline."""
        print("═" * 60)
        print("   Isaac Sim Articulated Object Preparation")
        print("═" * 60)

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load metadata
        metadata = self.load_metadata()

        # Process background mesh
        world_origin, background_inertial = self.process_background_mesh(metadata)

        # Process joint meshes
        joint_inertials = []
        for joint_id in range(metadata["num_joints"]):
            canonical_inertial, object_inertial = self.process_joint_meshes(
                joint_id, metadata, world_origin
            )
            joint_inertials.append((canonical_inertial, object_inertial))

        # Generate URDF
        urdf_content = self.generate_urdf(metadata, background_inertial, joint_inertials)
        self.save_urdf(urdf_content)

        # Save metadata
        self.save_isaac_sim_metadata(metadata)



        # Summary
        print("\n" + "═" * 60)
        print("   Preparation Complete! ✓")
        print("═" * 60)
        print(f"\nOutput directory: {self.output_dir}")
        print("\nNext steps:")
        print("1. Review the generated URDF and meshes")
        print("2. Import into Isaac Sim (see README.md)")
        print("3. Test articulation and physics interactions")
        print("4. Fine-tune joint properties if needed")


if __name__ == "__main__":
    tyro.cli(IsaacSimPreparer).main()