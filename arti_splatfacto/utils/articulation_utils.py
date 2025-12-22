import torch
from typing import Dict
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_quaternion, quaternion_multiply
from torch.nn import functional as F

def apply_joint_transform(means, quats, joint_pivot, joint_axis, joint_angle):
    """
    Differentiable revolute joint transform.
    Keeps gradient flow through joint_angle, joint_axis, and joint_pivot.
    """
    joint_pivot = joint_pivot.to(means.device)
    joint_axis = F.normalize(joint_axis.to(means.device), dim=0)

    means_local = means - joint_pivot.unsqueeze(0)

    axis_angle = joint_axis * joint_angle  # [3]
    R = axis_angle_to_matrix(axis_angle.unsqueeze(0)).squeeze(0)  # [3, 3]
    
    means_rotated = torch.matmul(means_local, R.T) + joint_pivot.unsqueeze(0)

    joint_quat = matrix_to_quaternion(R.unsqueeze(0)).squeeze(0)
    quats_rotated = quaternion_multiply(
        joint_quat.unsqueeze(0).expand_as(quats),
        quats
    )

    return means_rotated, quats_rotated


def apply_joint_transform_prismatic(means, quats, joint_pivot, joint_axis, joint_disp):
    """
    Differentiable prismatic joint transform (linear translation).
    """
    joint_axis = F.normalize(joint_axis.to(means.device), dim=0)
    joint_disp = joint_disp.to(means.device)
    
    means_translated = means + joint_axis.unsqueeze(0) * joint_disp
    
    quats_translated = quats
    return means_translated, quats_translated



def apply_articulation_to_optimizer_params(trainer, joint_angle: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Apply articulation transform (revolute or prismatic) to object Gaussians."""
    if not isinstance(joint_angle, torch.Tensor):
        raise TypeError(f"joint_angle must be torch.Tensor, got {type(joint_angle)}")

    joint_angle = joint_angle.to(trainer.device)

    if not joint_angle.requires_grad and trainer.training:
        print(f"[Warning] joint_angle has requires_grad=False (ok in eval).")

    articulated_params = {}

    obj_means = trainer.gauss_params["means"]
    obj_quats = trainer.gauss_params["quats"]


    print(f"[DEBUG] joint_angle value: {joint_angle.item():.4f}, type={type(joint_angle)}")

    if torch.all(torch.abs(joint_angle) < 1e-8):
        print(f"[Info] joint_angle ≈ 0.0, skipping articulation.")
    else:
        if trainer.joint_type == "revolute":
            obj_means, obj_quats = apply_joint_transform(
                means=obj_means, quats=obj_quats,
                joint_pivot=trainer.joint_pivot,
                joint_axis=trainer.joint_axis,
                joint_angle=joint_angle.squeeze()
            )
        elif trainer.joint_type == "prismatic":
            obj_means, obj_quats = apply_joint_transform_prismatic(
                means=obj_means, quats=obj_quats,
                joint_pivot=trainer.joint_pivot,
                joint_axis=trainer.joint_axis,
                joint_disp=joint_angle.squeeze()
            )
        else:
            raise ValueError(f"Unknown joint_type: {trainer.joint_type}")

    for name, param in trainer.gauss_params.items():
        if name == "means":
            articulated_params[name] = obj_means
        elif name == "quats":
            articulated_params[name] = obj_quats
        else:
            articulated_params[name] = param

    return articulated_params


def apply_articulation_to_params(
    params: Dict[str, torch.Tensor],
    joint_angle: torch.Tensor,
    joint_pivot: torch.Tensor,
    joint_axis: torch.Tensor,
    joint_type: str
) -> Dict[str, torch.Tensor]:
    """
    Apply articulation transform to a slice of Gaussian parameters.
    
    Args:
        params: Dict of Gaussian parameters (means, quats, scales, etc.) for ONE joint
        joint_angle: Angle/displacement for this joint [scalar tensor]
        joint_pivot: Pivot point [3]
        joint_axis: Rotation/translation axis [3]
        joint_type: "revolute" or "prismatic"
    
    Returns:
        Dict with articulated means and quats, other params unchanged
    """
    if not isinstance(joint_angle, torch.Tensor):
        joint_angle = torch.tensor([joint_angle], device=params["means"].device)
    
    # Skip articulation if angle is zero (optimization)
    if torch.abs(joint_angle).item() < 1e-8:
        return params
    
    articulated = {}
    
    # Apply transform based on joint type
    if joint_type == "revolute":
        means_art, quats_art = apply_joint_transform(
            means=params["means"],
            quats=params["quats"],
            joint_pivot=joint_pivot,
            joint_axis=joint_axis,
            joint_angle=joint_angle.squeeze()
        )
    elif joint_type == "prismatic":
        means_art, quats_art = apply_joint_transform_prismatic(
            means=params["means"],
            quats=params["quats"],
            joint_pivot=joint_pivot,
            joint_axis=joint_axis,
            joint_disp=joint_angle.squeeze()
        )
    else:
        raise ValueError(f"Unknown joint_type: {joint_type}")
    
    # Build output dict
    articulated["means"] = means_art
    articulated["quats"] = quats_art
    
    # Copy other parameters unchanged
    for key in params.keys():
        if key not in ["means", "quats"]:
            articulated[key] = params[key]
    
    return articulated