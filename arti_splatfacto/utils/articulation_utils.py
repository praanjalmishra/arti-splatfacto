import torch
from typing import Dict
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_quaternion, quaternion_multiply


def apply_joint_transform(means, quats, joint_pivot, joint_axis, joint_angle):
    """
    Apply revolute joint transformation while preserving gradients.
    """
    
    if not joint_pivot.requires_grad:
        joint_pivot = joint_pivot.detach()  
    if not joint_axis.requires_grad:
        joint_axis = joint_axis.detach()    
    means_local = means - joint_pivot.unsqueeze(0)
    axis_angle = joint_axis * (-joint_angle)
    R = axis_angle_to_matrix(axis_angle.unsqueeze(0)).squeeze(0)  # [3, 3]
    
    means_rotated = torch.matmul(means_local, R.T) + joint_pivot.unsqueeze(0)
    
    joint_quat = matrix_to_quaternion(R.unsqueeze(0)).squeeze(0)  # [4]
    quats_rotated = quaternion_multiply(
        joint_quat.unsqueeze(0).expand_as(quats), 
        quats
    )
    
    return means_rotated, quats_rotated



def apply_joint_transform_prismatic(means, quats, joint_pivot, joint_axis, joint_disp):
    """
    Apply prismatic (sliding) joint transformation.
    - means: Gaussian centers
    - quats: Gaussian orientations (unchanged)
    - joint_axis: direction of translation (normalized)
    - joint_disp: displacement (scalar)
    """
    if not joint_axis.requires_grad:
        joint_axis = joint_axis.detach()
    
    # Translate means along the axis
    translation = joint_axis * joint_disp
    means_translated = means + translation.unsqueeze(0)

    # Keep orientations unchanged
    quats_translated = quats  

    return means_translated, quats_translated


def apply_articulation_to_optimizer_params(trainer, joint_angle: float) -> Dict[str, torch.Tensor]:
    articulated_params = {}

    # Only apply articulation to object Gaussians
    obj_means = trainer.gauss_params["means"]
    obj_quats = trainer.gauss_params["quats"]

    if joint_angle != 0.0:
        if trainer.joint_type == "revolute":
            obj_means, obj_quats = apply_joint_transform(
                means=obj_means, quats=obj_quats,
                joint_pivot=trainer.joint_pivot,
                joint_axis=trainer.joint_axis,
                joint_angle=joint_angle
            )
        elif trainer.joint_type == "prismatic":
            obj_means, obj_quats = apply_joint_transform_prismatic(
                means=obj_means, quats=obj_quats,
                joint_pivot=trainer.joint_pivot,
                joint_axis=trainer.joint_axis,
                joint_disp=joint_angle
            )

    # Rebuild params: only object Gaussians articulated
    for name, param in trainer.gauss_params.items():
        if name == "means":
            articulated_params[name] = obj_means
        elif name == "quats":
            articulated_params[name] = obj_quats
        else:
            articulated_params[name] = param

    return articulated_params