from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Type, Optional, Union
from pathlib import Path

try:
    from gsplat.rendering import rasterization
except ImportError:
    print("Please install gsplat>=1.0.0")

import torch
import torch.nn.functional as F
from gsplat.strategy import DefaultStrategy, MCMCStrategy
from nerfstudio.models.splatfacto import SplatfactoModelConfig, SplatfactoModel
from nerfstudio.engine.optimizers import Optimizers
from nerfstudio.utils.spherical_harmonics import RGB2SH, SH2RGB, num_sh_bases
from nerfstudio.model_components.lib_bilagrid import BilateralGrid, color_correct, slice, total_variation_loss
from arti_splatfacto.obj_3d_seg import Object3DSeg
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.utils.misc import torch_compile
from arti_splatfacto.gauss_utils import transform_gaussians, sample_gaussians, fit_gaussian_batch, rot2quat

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
class ArtiSplatfactoModelConfig(SplatfactoModelConfig):


    _target: Type = field(default_factory=lambda: ArtiSplatfactoModel)
    obj_mask_file: Optional[Path] = None

    # Joint configuration
    joint_pivot: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    joint_axis: List[float] = field(default_factory=lambda: [0.0, 0.0, 1.0])



class ArtiSplatfactoModel(SplatfactoModel):    

    config: ArtiSplatfactoModelConfig

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Joint configuration
        self.joint_pivot = torch.tensor(self.config.joint_pivot, dtype=torch.float32)
        self.joint_axis = torch.tensor(self.config.joint_axis, dtype=torch.float32)
        self.joint_axis = F.normalize(self.joint_axis, dim=0)
        
        # Get joint angles from metadata if available
        self.metadata = kwargs.get("metadata", {})
        self.joint_angles = self.metadata.get("joint_angles", None)

    def populate_modules(self):
        """Populates the modules of the model."""
        super().populate_modules()

        def make_param(shape, requires_grad=True):
            return torch.nn.Parameter(torch.zeros(shape).float().cuda(), requires_grad=requires_grad)

        dim_sh = num_sh_bases(self.config.sh_degree)

        # Post transformation Gaussians (trainable) - THESE ARE OPTIMIZED
        self.gauss_params = torch.nn.ParameterDict({
            "means":         make_param((1, 3)),
            "scales":        make_param((1, 3)),
            "quats":         make_param((1, 4)),
            "features_dc":   make_param((1, 3)),
            "features_rest": make_param((1, dim_sh - 1, 3)),
            "opacities":     make_param((1, 1)),
        })

        # Fixed (non-trainable) Gaussians 
        self.gauss_params_fixed = torch.nn.ParameterDict({
            "means":         make_param((0, 3), requires_grad=False),
            "scales":        make_param((0, 3), requires_grad=False),
            "quats":         make_param((0, 4), requires_grad=False),
            "features_dc":   make_param((0, 3), requires_grad=False),
            "features_rest": make_param((0, dim_sh - 1, 3), requires_grad=False),
            "opacities":     make_param((0, 1), requires_grad=False),
        })

        # CANONICAL object state (for articulation reference)
        self.gauss_params_canonical = torch.nn.ParameterDict({
            "means":         make_param((0, 3), requires_grad=False),
            "scales":        make_param((0, 3), requires_grad=False),
            "quats":         make_param((0, 4), requires_grad=False),
            "features_dc":   make_param((0, 3), requires_grad=False),
            "features_rest": make_param((0, dim_sh - 1, 3), requires_grad=False),
            "opacities":     make_param((0, 1), requires_grad=False),
        })


    def state_dict(self, *args, **kwargs):
        state = super().state_dict(*args, **kwargs)
        if hasattr(self, "gauss_params_fixed"):
            for name, param in self.gauss_params_fixed.items():
                state[f"gauss_params_fixed.{name}"] = param.data

        if hasattr(self, "gauss_params_pre"):
            for name, param in self.gauss_params_pre.items():
                state[f"gauss_params_pre.{name}"] = param.data
        return state
    
    def _initialize_and_partition(self, state_dict: Dict[str, torch.Tensor]):
        """
        Initialize from full scene and partition into trainable object + fixed background.
        NO initial pose transformation - we'll apply articulation per-frame instead.
        """
        print("🚀 Initializing from full scene: partitioning Gaussians...")

        self.obj_3d_seg = Object3DSeg.read_from_file(self.config.obj_mask_file, device=self.device)
        self.obj_3d_seg.refine_mask(dilate_k=2, erode_k=1)
        
        # Identify object vs background Gaussians
        all_means = state_dict["gauss_params.means"]
        obj_mask = self.obj_3d_seg.query(all_means.to(self.device), dilate=True).cpu()
        non_obj_mask = ~obj_mask

        GAUSSIAN_PARAM_NAMES = ["means", "scales", "quats", "features_dc", "features_rest", "opacities"]
        
        # Partition data
        obj_data = {name: state_dict[f"gauss_params.{name}"][obj_mask] for name in GAUSSIAN_PARAM_NAMES}
        bg_data = {name: state_dict[f"gauss_params.{name}"][non_obj_mask] for name in GAUSSIAN_PARAM_NAMES}

        # 1. Set trainable object parameters directly (NO initial transformation)
        # These represent the object in its CANONICAL/REST pose
        for name, data in obj_data.items():
            self.gauss_params[name].data = data.to(self.device)

        # 2. Set fixed background parameters
        for name, data in bg_data.items():
            self.gauss_params_fixed[name] = torch.nn.Parameter(data.to(self.device), requires_grad=False)

        # 3. Update joint config from segmentation if available
        if hasattr(self.obj_3d_seg, 'joint_axis') and self.obj_3d_seg.joint_axis is not None:
            self.joint_axis = self.obj_3d_seg.joint_axis.to(self.device)
            print(f"📐 Updated joint axis from mask: {self.joint_axis}")
            
        if hasattr(self.obj_3d_seg, 'joint_pivot') and self.obj_3d_seg.joint_pivot is not None:
            self.joint_pivot = self.obj_3d_seg.joint_pivot.to(self.device)
            print(f"📍 Updated joint pivot from mask: {self.joint_pivot}")

        print(f"✅ Partitioning complete. Trainable: {obj_data['means'].shape[0]}, Fixed: {bg_data['means'].shape[0]}")
        print("🎯 Object parameters represent CANONICAL pose - articulation applied per-frame")


    def load_state_dict(self, state_dict, **kwargs):
        print(f"--- Loading state_dict (Training mode: {self.training}) ---")
        print(f"obj_mask_file: {self.config.obj_mask_file}")
        assert self.config.obj_mask_file is not None and self.config.obj_mask_file.exists()

        # ✅ Always load articulation info from mask
        if self.config.obj_mask_file is not None and self.config.obj_mask_file.exists():
            self.obj_3d_seg = Object3DSeg.read_from_file(self.config.obj_mask_file, device=self.device)
            if hasattr(self.obj_3d_seg, 'joint_axis') and self.obj_3d_seg.joint_axis is not None:
                self.joint_axis = self.obj_3d_seg.joint_axis.to(self.device)
                print(f"📐 Updated joint axis from mask: {self.joint_axis}")
            if hasattr(self.obj_3d_seg, 'joint_pivot') and self.obj_3d_seg.joint_pivot is not None:
                self.joint_pivot = self.obj_3d_seg.joint_pivot.to(self.device)
                print(f"📍 Updated joint pivot from mask: {self.joint_pivot}")

        # === Checkpoint logic ===
        is_partitioned_checkpoint = "gauss_params_fixed.means" in state_dict
        GAUSSIAN_PARAM_NAMES: List[str] = ["means", "scales", "quats", "features_dc", "features_rest", "opacities"]

        if is_partitioned_checkpoint:
            print("Resuming from a partitioned checkpoint...")
            super_state_dict = state_dict.copy()
            for name in GAUSSIAN_PARAM_NAMES:
                for param_group_name in ["gauss_params_fixed", "gauss_params_pre"]:
                    key = f"{param_group_name}.{name}"
                    if key in state_dict:
                        if not hasattr(self, param_group_name):
                            setattr(self, param_group_name, torch.nn.ParameterDict())
                        getattr(self, param_group_name)[name] = torch.nn.Parameter(
                            state_dict[key].to(self.device), requires_grad=False
                        )
                        if key in super_state_dict:
                            del super_state_dict[key]
            super().load_state_dict(super_state_dict, **kwargs)
            print("✅ State restored successfully into separate groups.")
        
        elif self.config.obj_mask_file is not None:
            print("⚡️ Loading full scene for inference.")
            super().load_state_dict(state_dict, **kwargs)
        
        else:
            print("Normal checkpoint loading.")
            super().load_state_dict(state_dict, **kwargs)

        self.step = state_dict.get("step", 0)
        print("--- Loading complete. Gaussians are kept in separate groups. ---")


    
    def clear_optimizer_state(self, optimizers):
        """Clear optimizer state after parameter resizing"""
        # print("!!! Clearing optimizer state after parameter resizing...")
        
        for name, optimizer in optimizers.optimizers.items():
            if name in ["means", "scales", "quats", "features_dc", "features_rest", "opacities"]:
                # Clear the optimizer state for this parameter group
                for param_group in optimizer.param_groups:
                    for param in param_group['params']:
                        if param in optimizer.state:
                            print(f"Clearing state for {name}")
                            optimizer.state[param].clear()
                            # Re-initialize the state
                            optimizer.state[param] = {}

    def step_cb(self, optimizers: Optimizers, step):
        # print(f"!!!Step callback: {step}")  
        if step == 20000:  
            self.clear_optimizer_state(optimizers)
        self.step = step
        self.optimizers = optimizers.optimizers
        self.schedulers = optimizers.schedulers


    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        gt_img = self.composite_with_background(self.get_gt_img(batch["image"]), outputs["background"])
        pred_img = outputs["rgb"]

        # === Losses ===
        Ll1 = torch.abs(gt_img - pred_img).mean()
        simloss = 1 - self.ssim(gt_img.permute(2, 0, 1)[None, ...], pred_img.permute(2, 0, 1)[None, ...])

        if self.config.use_scale_regularization and self.step % 10 == 0:
            scale_exp = torch.exp(self.scales)
            scale_reg = (
                torch.maximum(
                    scale_exp.amax(dim=-1) / scale_exp.amin(dim=-1),
                    torch.tensor(self.config.max_gauss_ratio),
                )
                - self.config.max_gauss_ratio
            )
            scale_reg = 0.1 * scale_reg.mean()
        else:
            scale_reg = torch.tensor(0.0).to(self.device)

        loss_dict = {
            "main_loss": (1 - self.config.ssim_lambda) * Ll1 + self.config.ssim_lambda * simloss,
            "scale_reg": scale_reg,
        }

        # MCMC extras
        if self.config.strategy == "mcmc":
            if self.config.mcmc_opacity_reg > 0.0:
                loss_dict["mcmc_opacity_reg"] = (
                    self.config.mcmc_opacity_reg * torch.abs(torch.sigmoid(self.gauss_params["opacities"])).mean()
                )
            if self.config.mcmc_scale_reg > 0.0:
                loss_dict["mcmc_scale_reg"] = (
                    self.config.mcmc_scale_reg * torch.abs(torch.exp(self.gauss_params["scales"])).mean()
                )

        # Camera + bilateral grid
        if self.training:
            self.camera_optimizer.get_loss_dict(loss_dict)
            if self.config.use_bilateral_grid:
                loss_dict["tv_loss"] = 10 * total_variation_loss(self.bil_grids.grids)

        return loss_dict
    

    def _get_joint_angle_for_camera(self, camera: Cameras) -> float:
        """Extract joint angle from camera.times (interpolated) or metadata (fixed)"""

        # 🎯 Method 1: Use interpolated time slider
        if hasattr(camera, 'times') and camera.times is not None and self.joint_angles is not None:
            time_val = float(camera.times.flatten()[0])
            num_frames = len(self.joint_angles)
            frame_idx = int(time_val * (num_frames - 1))
            frame_idx = max(0, min(frame_idx, num_frames - 1))
            joint_angle = self.joint_angles[frame_idx].item()
            print(f"⏰ Time {time_val:.3f} → Joint angle {joint_angle:.3f} rad")
            return joint_angle

        # ⛑️ Fallback: static camera with metadata
        if hasattr(camera, 'metadata') and camera.metadata is not None:
            joint_angle = camera.metadata.get("joint_angle", 0.0)
            print(f"📦 Metadata fallback → Joint angle {joint_angle:.3f} rad")
            return joint_angle

        return 0.0
    

    def _apply_articulation_to_canonical_params(self, joint_angle: float) -> Dict[str, torch.Tensor]:
        """
        Apply per-frame articulation to the canonical object parameters.
        
        Key insight: self.gauss_params represents the object in CANONICAL pose.
        We apply the joint transformation to get the current frame's pose.
        Gradients flow: rendered_image -> articulated_params -> canonical_params (self.gauss_params)
        """
        if joint_angle == 0.0:
            # No articulation needed - return canonical parameters directly
            return {name: param for name, param in self.gauss_params.items()}
        
        # Apply joint transformation to canonical object parameters
        means_articulated, quats_articulated = apply_joint_transform(
            means=self.gauss_params["means"],  
            quats=self.gauss_params["quats"],  
            joint_pivot=self.joint_pivot.to(self.device),
            joint_axis=self.joint_axis.to(self.device),
            joint_angle=joint_angle
        )
        
        # Return articulated parameters (gradients intact)
        articulated_params = {}
        for name, param in self.gauss_params.items():
            if name == "means":
                articulated_params[name] = means_articulated
            elif name == "quats":
                articulated_params[name] = quats_articulated
            else:
                # Colors, scales, opacities don't change with articulation
                articulated_params[name] = param
        
        return articulated_params

    def _get_gaussians_for_render(self, camera: Cameras) -> Dict[str, torch.Tensor]:
        """
        Prepare Gaussians for rendering with per-frame articulation.
        Maintains gradient flow to canonical trainable parameters.
        """
        # 1. Get joint angle for this specific frame
        joint_angle = self._get_joint_angle_for_camera(camera)
        
        # 2. Apply articulation to canonical object parameters (preserving gradients)
        articulated_obj_params = self._apply_articulation_to_canonical_params(joint_angle)
        
        # 3. Combine with fixed background parameters
        if hasattr(self, "gauss_params_fixed") and self.gauss_params_fixed["means"].shape[0] > 0:
            full_scene_params = {}
            for name in articulated_obj_params.keys():
                # Concatenate articulated object + fixed background
                full_scene_params[name] = torch.cat(
                    [articulated_obj_params[name], self.gauss_params_fixed[name]], dim=0
                )
            return full_scene_params
        
        return articulated_obj_params


    def get_outputs(self, camera: Cameras) -> Dict[str, Union[torch.Tensor, List]]:
        """Takes in a camera and returns a dictionary of outputs with articulation."""
        if not isinstance(camera, Cameras):
            return {}

        # 1. Prepare articulated Gaussians (maintaining gradient flow)
        gaussians_to_render = self._get_gaussians_for_render(camera)

        # 2. Handle camera optimization
        optimized_camera_to_world = self.camera_optimizer.apply_to_camera(camera) if self.training else camera.camera_to_worlds

        # 3. Setup for rasterization
        colors_crop = torch.cat(
            (gaussians_to_render["features_dc"][:, None, :], gaussians_to_render["features_rest"]), dim=1
        )
        
        camera_scale_fac = self._get_downscale_factor()
        camera.rescale_output_resolution(1 / camera_scale_fac)
        viewmat = get_viewmat(optimized_camera_to_world)
        K = camera.get_intrinsics_matrices().cuda()
        W, H = int(camera.width.item()), int(camera.height.item())
        self.last_size = (H, W)
        camera.rescale_output_resolution(camera_scale_fac)

        # Determine render mode and SH degree
        render_mode = "RGB+ED" if self.config.output_depth_during_training or not self.training else "RGB"
        if self.config.sh_degree > 0:
            sh_degree_to_use = min(self.step // self.config.sh_degree_interval, self.config.sh_degree)
        else:
            colors_crop = torch.sigmoid(colors_crop).squeeze(1)
            sh_degree_to_use = None
        
        # 4. Rasterize (gradients flow through gaussians_to_render back to self.gauss_params)
        render, alpha, self.info = rasterization(
            means=gaussians_to_render["means"],      # Has gradients from self.gauss_params["means"]
            quats=gaussians_to_render["quats"],      # Has gradients from self.gauss_params["quats"]
            scales=torch.exp(gaussians_to_render["scales"]),
            opacities=torch.sigmoid(gaussians_to_render["opacities"]).squeeze(-1),
            colors=colors_crop,
            viewmats=viewmat,
            Ks=K,
            width=W,
            height=H,
            packed=False,
            near_plane=0.01,
            far_plane=1e10,
            render_mode=render_mode,
            sh_degree=sh_degree_to_use,
            sparse_grad=False,
            absgrad=self.strategy.absgrad if isinstance(self.strategy, DefaultStrategy) else False,
            rasterize_mode=self.config.rasterize_mode,
        )
        
        # 5. Strategy step (CRITICAL: This operates on the original trainable parameters)
        if self.training:
            self.strategy.step_pre_backward(
                self.gauss_params,  # Pass the original trainable parameters, not the articulated ones
                self.optimizers, 
                self.strategy_state, 
                self.step, 
                self.info
            )
        
        # 6. Post-processing
        background = self._get_background_color()
        rgb = render[..., :3] + (1 - alpha) * background
        rgb = torch.clamp(rgb, 0.0, 1.0)
        depth_im = render[..., 3:4] if render_mode == "RGB+ED" else None

        return {
            "rgb": rgb.squeeze(0),
            "depth": depth_im.squeeze(0) if depth_im is not None else None,
            "accumulation": alpha.squeeze(0),
            "background": background,
        }
    

def apply_joint_transform(means, quats, joint_pivot, joint_axis, joint_angle):
    """
    Apply revolute joint transformation while preserving gradients.
    This is the key function that must maintain the gradient connection.
    """
    from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_quaternion, quaternion_multiply
    
    if joint_angle == 0.0:
        return means, quats
    
    # All operations preserve gradients
    means_local = means - joint_pivot.unsqueeze(0)
    axis_angle = joint_axis * (-joint_angle)
    R = axis_angle_to_matrix(axis_angle.unsqueeze(0)).squeeze(0)  # [3, 3]
    
    # Transform positions (gradients preserved through matrix ops)
    means_rotated = torch.matmul(means_local, R.T) + joint_pivot.unsqueeze(0)
    
    # Transform orientations (gradients preserved through quaternion ops)
    joint_quat = matrix_to_quaternion(R.unsqueeze(0)).squeeze(0)  # [4]
    quats_rotated = quaternion_multiply(
        joint_quat.unsqueeze(0).expand_as(quats), 
        quats
    )
    # print(f"Joint axis: {joint_axis.cpu().numpy()}, Pivot: {joint_pivot.cpu().numpy()}")
    # print(f"Gaussian mean sample: {means[0].detach().cpu().numpy()}")

    
    return means_rotated, quats_rotated