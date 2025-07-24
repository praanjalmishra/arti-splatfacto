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


def slerp(q0: torch.Tensor, q1: torch.Tensor, t: float) -> torch.Tensor:
    """Spherical linear interpolation (SLERP) between two quaternions."""
    dot = torch.sum(q0 * q1, dim=-1, keepdim=True)

    # Use the shorter path by negating q1 if needed
    q1 = torch.where(dot < 0, -q1, q1)
    dot = torch.clamp(dot.abs(), 1e-6, 1.0)

    theta_0 = torch.acos(dot)  # Angle between input vectors
    sin_theta_0 = torch.sin(theta_0)

    if torch.any(sin_theta_0 < 1e-4):
        return F.normalize((1 - t) * q0 + t * q1, dim=-1)

    theta = theta_0 * t
    sin_theta = torch.sin(theta)

    s0 = torch.sin(theta_0 - theta) / sin_theta_0
    s1 = sin_theta / sin_theta_0

    return F.normalize(s0 * q0 + s1 * q1, dim=-1)

@dataclass
class ArtiSplatfactoModelConfig(SplatfactoModelConfig):


    _target: Type = field(default_factory=lambda: ArtiSplatfactoModel)
    obj_mask_file: Optional[Path] = None


class ArtiSplatfactoModel(SplatfactoModel):    

    config: ArtiSplatfactoModelConfig

    def populate_modules(self):
        """
        Populates the modules of the model.
        """
        super().populate_modules()

        def make_param(shape, requires_grad=True):
            return torch.nn.Parameter(torch.zeros(shape).float().cuda(), requires_grad=requires_grad)

        dim_sh = num_sh_bases(self.config.sh_degree)

        # Post transformation Gaussians (trainable)
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

        # Pre-transformation Gaussians (for object-specific loading)
        self.gauss_params_pre = torch.nn.ParameterDict({
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
        Initializes the model from a full-scene checkpoint, partitioning it
        into trainable (object) and fixed (background) sets.
        This is typically run only once at the start of fine-tuning.
        """
        print("🚀 Initializing from full scene: partitioning Gaussians...")

        self.obj_3d_seg = Object3DSeg.read_from_file(self.config.obj_mask_file, device=self.device)
        self.obj_3d_seg.refine_mask(dilate_k=2, erode_k=1)
        
        # Identify Gaussians inside the object mask
        all_means = state_dict["gauss_params.means"]
        obj_mask = self.obj_3d_seg.query(all_means.to(self.device), dilate=True).cpu()
        non_obj_mask = ~obj_mask

        GAUSSIAN_PARAM_NAMES: List[str] = ["means", "scales", "quats", "features_dc", "features_rest", "opacities"]

        
        # Partition data
        obj_data = {name: state_dict[f"gauss_params.{name}"][obj_mask] for name in GAUSSIAN_PARAM_NAMES}
        bg_data = {name: state_dict[f"gauss_params.{name}"][non_obj_mask] for name in GAUSSIAN_PARAM_NAMES}

        # 1. Save pre-transformation object Gaussians
        for name, data in obj_data.items():
            self.gauss_params_pre[name] = torch.nn.Parameter(data.to(self.device).detach(), requires_grad=False)

        # 2. Transform and load trainable object Gaussians
        pose = self.obj_3d_seg.pose_change.cpu()
        obj_data["means"], obj_data["quats"] = transform_gaussians(pose, obj_data["means"], obj_data["quats"])
        for name, data in obj_data.items():
            self.gauss_params[name].data = data.to(self.device)

        # 3. Load fixed background Gaussians
        for name, data in bg_data.items():
            self.gauss_params_fixed[name] = torch.nn.Parameter(data.to(self.device), requires_grad=False)
        print(f"✅ Partitioning complete. Trainable: {obj_data['means'].shape[0]}, Fixed: {bg_data['means'].shape[0]}")


    def load_state_dict(self, state_dict, **kwargs):
        """
        Load the state_dict into the model.
        """
        print(f"--- Loading state_dict (Training mode: {self.training}) ---")

        is_partitioned_checkpoint = "gauss_params_fixed.means" in state_dict
        GAUSSIAN_PARAM_NAMES: List[str] = ["means", "scales", "quats", "features_dc", "features_rest", "opacities"]

        if is_partitioned_checkpoint:
            print("Resuming from a partitioned checkpoint...")
            # Create a copy for the super() call to avoid modifying the original dict
            super_state_dict = state_dict.copy()
            
            # Manually load our custom parameter groups from the original state_dict
            for name in GAUSSIAN_PARAM_NAMES:
                for param_group_name in ["gauss_params_fixed", "gauss_params_pre"]:
                    key = f"{param_group_name}.{name}"
                    if key in state_dict:
                        # Ensure the parameter dictionary exists on the model
                        if not hasattr(self, param_group_name):
                            setattr(self, param_group_name, torch.nn.ParameterDict())
                        
                        # Load the data and create a new parameter
                        getattr(self, param_group_name)[name] = torch.nn.Parameter(
                            state_dict[key].to(self.device), requires_grad=False
                        )

                        # Remove the key from the dictionary we pass to super()
                        if key in super_state_dict:
                            del super_state_dict[key]
            
            # Load all remaining standard parameters (e.g., trainable gauss_params)
            super().load_state_dict(super_state_dict, **kwargs)
            print("✅ State restored successfully into separate groups.")

        elif self.config.obj_mask_file is not None:
            if self.training:
                self._initialize_and_partition(state_dict)
            else:
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

    def _get_gaussians_for_render(
        self, time: float = 1.0, interp_mode: str = "lerp"
    ) -> Dict[str, torch.Tensor]:
        """
        Selects, interpolates, and combines Gaussians for rendering.
        Allows LERP or SLERP interpolation between object states.
        """
        obj_params_post = self.gauss_params
        has_pre_state = hasattr(self, "gauss_params_pre") and self.gauss_params_pre["means"].shape[0] > 0

        if not self.training and has_pre_state and time < 1.0:
            print(f"Interpolating object state with time t={time:.2f} using {interp_mode.upper()}")
            t = time
            final_obj_params = {}

            for name in obj_params_post.keys():
                pre = self.gauss_params_pre[name]
                post = obj_params_post[name]

                if name == "quats" and interp_mode == "slerp":
                    final_obj_params[name] = slerp(pre, post, t)
                else:
                    final_obj_params[name] = (1 - t) * pre + t * post

            if interp_mode == "lerp":
                final_obj_params["quats"] = F.normalize(final_obj_params["quats"], dim=-1)

        else:
            final_obj_params = obj_params_post

        if hasattr(self, "gauss_params_fixed") and self.gauss_params_fixed["means"].shape[0] > 0:
            full_scene_params = {}
            for name in final_obj_params.keys():
                full_scene_params[name] = torch.cat(
                    [final_obj_params[name], self.gauss_params_fixed[name]], dim=0
                )
            return full_scene_params

        return final_obj_params


    def get_outputs(self, camera: Cameras) -> Dict[str, Union[torch.Tensor, List]]:
        """Takes in a camera and returns a dictionary of outputs."""
        if not isinstance(camera, Cameras):
            return {}

        # 1. Get time value for interpolation
        time_value = 1.0
        if hasattr(camera, "times") and camera.times is not None:
            time_value = float(camera.times.flatten()[0])

        # 2. Prepare Gaussians for rendering using the new helper method
        gaussians_to_render = self._get_gaussians_for_render(time=time_value, interp_mode="slerp")

        # 3. Handle camera optimization
        optimized_camera_to_world = self.camera_optimizer.apply_to_camera(camera) if self.training else camera.camera_to_worlds

        # As `load_state_dict` now handles merging, the crop logic and manual combining is simplified.
        # The crop logic is omitted here for clarity but can be added back if needed,
        # operating on the `gaussians_to_render` dictionary.
        
        # 4. Setup for rasterization
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
        
        # 5. Rasterize the scene
        render, alpha, self.info = rasterization(
            means=gaussians_to_render["means"],
            quats=gaussians_to_render["quats"],
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
        
        # 6. Post-processing
        if self.training:
            self.strategy.step_pre_backward(
                self.gauss_params, self.optimizers, self.strategy_state, self.step, self.info
            )
        
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

