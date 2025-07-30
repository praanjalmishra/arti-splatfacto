import torch
import re
from typing import Dict, Optional, Union
from pathlib import Path
from arti_splatfacto.obj_3d_seg import Object3DSeg
from arti_splatfacto.gauss_utils import rot2quat, quaternion_multiply



class Scene3D:
    """
    Manages multiple Object3DSeg instances and their associated Gaussian assignments.
    """
    def __init__(self, device: str = "cuda"):
        self.objects: Dict[int, Object3DSeg] = {}
        self.gaussian_assignments: Dict[int, torch.Tensor] = {}
        self.metadata: Dict[int, Dict] = {}
        self.device = device

    def add_object(
        self,
        obj_id: int,
        obj_seg: Object3DSeg,
        gauss_mask: Optional[torch.Tensor] = None,
        meta: Optional[Dict] = None
    ):
        """
        Register an object with optional Gaussian mask and metadata.

        Args:
            obj_id (int): Unique identifier.
            obj_seg (Object3DSeg): Object 3D segmentation.
            gauss_mask (Tensor, optional): Boolean mask (N,) of assigned Gaussians.
            meta (Dict, optional): Any extra metadata.
        """
        self.objects[obj_id] = obj_seg
        self.gaussian_assignments[obj_id] = (
            gauss_mask.bool() if gauss_mask is not None else torch.zeros(0, dtype=torch.bool, device=self.device)
        )
        self.metadata[obj_id] = meta if meta is not None else {}

    def assign_gaussian_mask(self, obj_id: int, mask: torch.Tensor):
        """
        Assign a Gaussian mask for a specific object.
        """
        self.gaussian_assignments[obj_id] = mask.bool().to(self.device)

    def get_object(self, obj_id: int) -> Object3DSeg:
        return self.objects[obj_id]

    def get_metadata(self, obj_id: int) -> Dict:
        return self.metadata.get(obj_id, {})

    def apply_articulations(
        self,
        pre_means: torch.Tensor,
        pre_quats: torch.Tensor,
        joint_angles: Optional[Dict[int, float]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Apply transformations to Gaussian parameters using each object's articulation data.
        """
        new_means = pre_means.clone()
        new_quats = pre_quats.clone()

        for obj_id, obj in self.objects.items():
            mask = self.gaussian_assignments.get(obj_id, None)
            if mask is None or mask.sum() == 0:
                continue

            means_obj = pre_means[mask]
            quats_obj = pre_quats[mask]

            if obj.joint_axis is None or obj.joint_pivot is None:
                continue

            theta = (
                joint_angles[obj_id]
                if joint_angles and obj_id in joint_angles
                else obj.joint_angle
            )

            R = self.axis_angle_to_matrix(obj.joint_axis.to(self.device), theta)
            pivot = obj.joint_pivot.to(self.device)

            rotated_means = (R @ (means_obj - pivot).T).T + pivot
            R_quat = rot2quat(R.unsqueeze(0))  # shape (1, 4)
            new_quats_obj = quaternion_multiply(R_quat.expand_as(quats_obj), quats_obj)

            new_means[mask] = rotated_means
            new_quats[mask] = new_quats_obj

        return {"means": new_means, "quats": new_quats}

    @staticmethod
    def axis_angle_to_matrix(axis: torch.Tensor, angle: float) -> torch.Tensor:
        """
        Rodrigues' formula for converting axis-angle to rotation matrix.
        """
        axis = axis / torch.norm(axis)
        K = torch.tensor([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ], dtype=torch.float32, device=axis.device)

        I = torch.eye(3, dtype=torch.float32, device=axis.device)
        R = I + torch.sin(angle) * K + (1 - torch.cos(angle)) * K @ K
        return R

    @classmethod
    def from_directory(cls, seg_dir: Union[str, Path], device: str = "cuda") -> "Scene3D":
        """
        Load multiple Object3DSeg instances from directory and return Scene3D instance.
        """
        scene = cls(device=device)
        seg_dir = Path(seg_dir)

        for file in sorted(seg_dir.glob("*.pt")):
            match = re.search(r"\d+$", file.stem)
            if not match:
                raise ValueError(f"Cannot extract object ID from filename: {file.name}")
            obj_id = int(match.group())
            obj = Object3DSeg.read_from_file(file, device=device)
            scene.add_object(obj_id, obj)
            print(f"Loaded object {obj_id}")

        return scene