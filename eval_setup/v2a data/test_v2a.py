from pathlib import Path
import pyvista as pv

partnet_dir = Path("assets/StorageFurniture/35059")
mesh = pv.read(partnet_dir / "textured_objs" / "part_0.obj")
print(f"Vertices: {len(mesh.points)}")