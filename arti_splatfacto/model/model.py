from __future__ import annotations

from dataclasses import dataclass, field
from pickle import TRUE
import re
from typing import Dict, List, Type, Optional, Union, Tuple
from pathlib import Path

try:
    from gsplat.rendering import rasterization
except ImportError:
    print("Please install gsplat>=1.0.0")

import numpy
import torch
import torch.nn.functional as F
from torch.nn import Parameter
from gsplat.strategy import DefaultStrategy, MCMCStrategy
from arti_splatfacto.model.splatfacto import SplatfactoModelConfig, SplatfactoModel
from nerfstudio.engine.optimizers import Optimizers
from nerfstudio.utils.spherical_harmonics import RGB2SH, SH2RGB, num_sh_bases
from nerfstudio.model_components.lib_bilagrid import BilateralGrid, color_correct, slice, total_variation_loss
from arti_splatfacto.obj_3d_seg import Object3DSeg
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.utils.misc import torch_compile
from arti_splatfacto.gauss_utils import transform_gaussians, sample_gaussians, fit_gaussian_batch, rot2quat
from arti_splatfacto.utils.strategy import SpatialArtiStrategy
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_quaternion, quaternion_multiply
from arti_splatfacto.metrics import RGBMetrics, DepthMetrics
import torchvision.transforms.functional as TF

from arti_splatfacto.utils.debug_utils import decode_id_map, save_debug_id_maps, depth_debug
from arti_splatfacto.utils.img_utils import psnr_masked, crop_imgs_w_masks, compute_2D_bbox, batch_crop_resize,batch_crop_resize
from arti_splatfacto.utils.articulation_utils import apply_joint_transform, apply_joint_transform_prismatic, apply_articulation_to_optimizer_params

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

    joint_type: str = field(default="prismatic")  # "revolute" or "prismatic"

    continue_cull_post_densification: bool = True
    """If True, continue to cull problematic gaussians even after densification stops"""
    
    # Enhanced culling parameters
    cull_post_densification_every: int = 50
    """How often to cull post-densification (in steps)"""
    
    cull_boundary_gaussians: bool = True
    """If True, cull Gaussians that drift outside object boundaries"""

    use_depth: bool = True
    """If True, use depth information for culling and optimization"""

    depth_lambda: float = 0.2
    """Weighting factor for depth information in loss function"""

    output_depth_during_training: bool = True
    """If True, output depth information during training"""

    depth_debug_vis: bool = True


class ArtiSplatfactoModel(SplatfactoModel):    

    config: ArtiSplatfactoModelConfig

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Joint configuration
        self.joint_pivot = self.obj_3d_seg.joint_pivot.to(self.device)
        self.joint_axis = F.normalize(self.obj_3d_seg.joint_axis.to(self.device), dim=0)
        self.joint_type = self.obj_3d_seg.joint_type


        # Get joint angles from metadata if available
        self.metadata = kwargs.get("metadata", {})
        self.joint_angles = self.metadata.get("joint_angles", None)

    

    def populate_modules(self):
        """Populates the modules of the model."""
        super().populate_modules()
        print("Populating ArtiSplatfactoModel modules...")

        def make_param(shape, requires_grad=True):
            return torch.nn.Parameter(
                torch.empty(shape, device="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.float32),
                requires_grad=requires_grad,
            )

        dim_sh = num_sh_bases(self.config.sh_degree)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Use ParameterDict like base model
        self.gauss_params = torch.nn.ParameterDict({
            "means":         torch.nn.Parameter(torch.empty((0, 3), device=device, requires_grad=True)),
            "scales":        torch.nn.Parameter(torch.empty((0, 3), device=device, requires_grad=True)),
            "quats":         torch.nn.Parameter(torch.empty((0, 4), device=device, requires_grad=True)),
            "features_dc":   torch.nn.Parameter(torch.empty((0, 3), device=device, requires_grad=True)),
            "features_rest": torch.nn.Parameter(torch.empty((0, dim_sh - 1, 3), device=device, requires_grad=True)),
            "opacities":     torch.nn.Parameter(torch.empty((0, 1), device=device, requires_grad=True)),
        })

        # Canonical gaussians (trainable)
        self.gauss_params_canonical = torch.nn.ParameterDict({
            "means":         torch.nn.Parameter(torch.empty((0, 3), device=device, requires_grad=True)),
            "scales":        torch.nn.Parameter(torch.empty((0, 3), device=device, requires_grad=True)),
            "quats":         torch.nn.Parameter(torch.empty((0, 4), device=device, requires_grad=True)),
            "features_dc":   torch.nn.Parameter(torch.empty((0, 3), device=device, requires_grad=True)),
            "features_rest": torch.nn.Parameter(torch.empty((0, dim_sh - 1, 3), device=device, requires_grad=True)),
            "opacities":     torch.nn.Parameter(torch.empty((0, 1), device=device, requires_grad=True)),
        })

        # Fixed gaussians (non-trainable but still Parameters for saving)
        self.gauss_params_fixed = torch.nn.ParameterDict({
            "means":         torch.nn.Parameter(torch.empty((0, 3), device=device), requires_grad=False),
            "scales":        torch.nn.Parameter(torch.empty((0, 3), device=device), requires_grad=False),
            "quats":         torch.nn.Parameter(torch.empty((0, 4), device=device), requires_grad=False),
            "features_dc":   torch.nn.Parameter(torch.empty((0, 3), device=device), requires_grad=False),
            "features_rest": torch.nn.Parameter(torch.empty((0, dim_sh - 1, 3), device=device), requires_grad=False),
            "opacities":     torch.nn.Parameter(torch.empty((0, 1), device=device), requires_grad=False),
        })

        self.obj_3d_seg = Object3DSeg.load(self.config.obj_mask_file, device=device)
        

        self.rgb_metrics = RGBMetrics()
        self.depth_metrics = DepthMetrics()
        self.mse_loss = torch.nn.MSELoss()


    def state_dict(self, *args, **kwargs):
        state = super().state_dict(*args, **kwargs)

        # Save all parameter sets
        for name, param in self.gauss_params.items():
            state[f"gauss_params.{name}"] = param.data
        for name, param in self.gauss_params_canonical.items():
            state[f"gauss_params_canonical.{name}"] = param.data
        if hasattr(self, 'gauss_params_fixed') and self.gauss_params_fixed:
            for name, param in self.gauss_params_fixed.items():
                state[f"gauss_params_fixed.{name}"] = param.data

        return state

    def get_gaussian_param_groups(self) -> Dict[str, List[Parameter]]:
        """Return optimizer param groups for gaussians."""
        groups = {}
        
        obj_param_mapping = {
            "obj_means": "means",
            "obj_scales": "scales", 
            "obj_quats": "quats",
            "obj_features_dc": "features_dc",
            "obj_features_rest": "features_rest",
            "obj_opacities": "opacities",
        }
        
        for optimizer_name, internal_name in obj_param_mapping.items():
            if (hasattr(self, 'gauss_params') and 
                internal_name in self.gauss_params and 
                self.gauss_params[internal_name].numel() > 0):
                groups[optimizer_name] = [self.gauss_params[internal_name]]
                print(f"Added param group '{optimizer_name}': {self.gauss_params[internal_name].shape}")

        # Canonical parameters  "means" -> optimizer "canon_means"
        canon_param_mapping = {
            "canon_means": "means",
            "canon_scales": "scales",
            "canon_quats": "quats", 
            "canon_features_dc": "features_dc",
            "canon_features_rest": "features_rest",
            "canon_opacities": "opacities",
        }
        
        for optimizer_name, internal_name in canon_param_mapping.items():
            if (hasattr(self, 'gauss_params_canonical') and 
                internal_name in self.gauss_params_canonical and 
                self.gauss_params_canonical[internal_name].numel() > 0):
                groups[optimizer_name] = [self.gauss_params_canonical[internal_name]]
                print(f"Added param group '{optimizer_name}': {self.gauss_params_canonical[internal_name].shape}")
        
        print(f"[debug] Created {len(groups)} parameter groups")
        return groups
        
    
    def _initialize_and_partition(self, state_dict: Dict[str, torch.Tensor]):
        """
        From a full-scene checkpoint: split into trainable object + fixed background.
        (Canonical stays empty here; you can fill it later if you have an exposed mask.)
        """
        print("Initializing from full scene: partitioning Gaussians...")

        # self.obj_3d_seg = Object3DSeg.read_from_file(self.config.obj_mask_file, device=self.device)
        print("obj mask points:", self.obj_3d_seg)
        # self.obj_3d_seg.refine_mask(dilate_k=4, erode_k=1)

        GAUSS = ["means", "scales", "quats", "features_dc", "features_rest", "opacities"]

        all_means = state_dict["gauss_params.means"].to(self.device)
        obj_mask = self.obj_3d_seg.query_refine(all_means, grow=3, thresh=0.01, bbox_margin=0.01).to(torch.bool).cpu()
        bg_mask  = ~obj_mask

        for p in GAUSS:
            subset = state_dict[f"gauss_params.{p}"][obj_mask].to(self.device)
            param = torch.nn.Parameter(subset.clone().detach(), requires_grad=True)
            self.gauss_params[p] = param
            print(f"Object {p}: {param.shape}, requires_grad={param.requires_grad}")

        # Canonical gaussians (trainable)
        for p in GAUSS:
            subset = state_dict[f"gauss_params.{p}"][obj_mask].to(self.device)
            # Create new Parameter with requires_grad=True
            param = torch.nn.Parameter(subset.clone().detach(), requires_grad=True)
            self.gauss_params_canonical[p] = param
            print(f"Canonical {p}: {param.shape}, requires_grad={param.requires_grad}")

        # Fixed background gaussians (non-trainable)
        for p in GAUSS:
            subset = state_dict[f"gauss_params.{p}"][bg_mask].to(self.device)
            # Create new Parameter with requires_grad=False
            param = torch.nn.Parameter(subset.clone().detach(), requires_grad=False)
            self.gauss_params_fixed[p] = param
            print(f"Background {p}: {param.shape}, requires_grad={param.requires_grad}")


        n_obj = self.gauss_params["means"].shape[0]
        n_canon = self.gauss_params_canonical["means"].shape[0]
        n_bg = self.gauss_params_fixed["means"].shape[0]


        print(f"Partitioning complete. Trainable: {self.gauss_params['means'].shape[0]}, Canonical: {self.gauss_params_canonical['means'].shape[0]}, Fixed: {self.gauss_params_fixed['means'].shape[0]}")

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor], **kwargs):
        print(f"Loading state_dict (Training mode: {self.training})")
        assert self.config.obj_mask_file is not None and self.config.obj_mask_file.exists()

        # self.obj_3d_seg = Object3DSeg.read_from_file(self.config.obj_mask_file, device=self.device)
        if hasattr(self.obj_3d_seg, 'joint_axis') and self.obj_3d_seg.joint_axis is not None:
            self.joint_axis = self.obj_3d_seg.joint_axis.to(self.device)
            # self.joint_axis = torch.tensor([0.0, 1.0, 0.0], device=self.device)
            print(f"Updated joint axis from mask: {self.joint_axis}")
        if hasattr(self.obj_3d_seg, 'joint_pivot') and self.obj_3d_seg.joint_pivot is not None:
            self.joint_pivot = self.obj_3d_seg.joint_pivot.to(self.device)
            print(f"Updated joint pivot from mask: {self.joint_pivot}")
        if hasattr(self.obj_3d_seg, 'joint_angle') and self.obj_3d_seg.joint_angle is not None:
            self.max_joint_angle = self.obj_3d_seg.joint_angle.to(self.device)
            print(f"Updated joint angles from mask: {self.max_joint_angle}")


        GAUSS = ["means", "scales", "quats", "features_dc", "features_rest", "opacities"]

        if "means" in state_dict:
            for p in GAUSS:
                state_dict[f"gauss_params.{p}"] = state_dict[p]

        if not self.training:
            is_partitioned = any(k.startswith("gauss_params_fixed.") for k in state_dict)
        else:
            is_partitioned = "gauss_params_fixed.means" in state_dict

        if is_partitioned:
            self.gauss_params = torch.nn.ParameterDict()
            self.gauss_params_canonical = torch.nn.ParameterDict()
            self.gauss_params_fixed = torch.nn.ParameterDict()
            
            for p in GAUSS:
                if f"gauss_params.{p}" in state_dict:
                    tensor = state_dict[f"gauss_params.{p}"].to(self.device)
                    self.gauss_params[p] = torch.nn.Parameter(tensor.clone().detach(), requires_grad=True)
                    
                if f"gauss_params_canonical.{p}" in state_dict:
                    tensor = state_dict[f"gauss_params_canonical.{p}"].to(self.device)
                    self.gauss_params_canonical[p] = torch.nn.Parameter(tensor.clone().detach(), requires_grad=True)
                    
                if f"gauss_params_fixed.{p}" in state_dict:
                    tensor = state_dict[f"gauss_params_fixed.{p}"].to(self.device)
                    self.gauss_params_fixed[p] = torch.nn.Parameter(tensor.clone().detach(), requires_grad=False)

        else:
            print("Partitioning full scene into obj/canonical/bg...")
            self._initialize_and_partition(state_dict)

        # Load non-gaussian parameters
        non_gauss_state = {
            k: v for k, v in state_dict.items()
            if not (
                k.startswith("gauss_params.") or
                k.startswith("gauss_params_canonical.") or
                k.startswith("gauss_params_fixed.")
            )
        }

        super().load_state_dict(non_gauss_state, strict=False)
        self.step = 0
        
        print(f"Load complete — obj={self.gauss_params['means'].shape[0]}, "
            f"canon={self.gauss_params_canonical['means'].shape[0]}, "
            f"bg={self.gauss_params_fixed['means'].shape[0] if self.gauss_params_fixed else 0}")
        

    def step_cb(self, optimizers: Optimizers, step):
        self.step = step
        self.optimizers = optimizers.optimizers
        self.schedulers = optimizers.schedulers


    def step_post_backward(self, step):
        """Apply strategy to both object and canonical parameters separately"""
        assert step == self.step
        
        if not isinstance(self.strategy, DefaultStrategy):
            raise ValueError(f"Only DefaultStrategy supported, got {self.strategy}")

        def create_empty_info_with_absgrad():
            """Create empty info dict with proper absgrad tensor"""
            empty_ids = torch.empty(0, dtype=torch.long, device=self.device)
            
            # Create empty tensor for the gradient key with absgrad
            key = self.strategy.key_for_gradient  # Usually "means2d"
            if hasattr(self, 'combined_info') and self.combined_info and key in self.combined_info:
                ref_tensor = self.combined_info[key]
                empty_tensor = torch.empty((0,) + ref_tensor.shape[1:], 
                                        device=self.device, dtype=ref_tensor.dtype)
            else:
                # Fallback - means2d is typically (N, 2)
                empty_tensor = torch.empty((0, 2), device=self.device, dtype=torch.float32)
            
            empty_tensor.absgrad = torch.empty_like(empty_tensor)
            
            return {"gaussian_ids": empty_ids, key: empty_tensor}
        
        if hasattr(self, 'combined_info') and self.combined_info and self.combined_info.get("gaussian_ids") is not None:
            visible_ids = self.combined_info["gaussian_ids"]
            n_obj = self.n_obj_rendered
            
            # Split IDs: first n_obj belong to object, rest to canonical
            obj_mask = visible_ids < n_obj
            canon_mask = visible_ids >= n_obj
            
            # Create object info with proper absgrad handling
            if obj_mask.any():
                obj_visible_ids = visible_ids[obj_mask]
                obj_info = {"gaussian_ids": obj_visible_ids}
                
                for k, v in self.combined_info.items():
                    if k == "gaussian_ids":
                        continue
                        
                    if isinstance(v, torch.Tensor) and v.shape[0] == len(visible_ids):
                        # Extract object portion
                        obj_tensor = v[obj_mask].contiguous()
                        
                        # Handle absgrad if present
                        if hasattr(v, 'absgrad') and v.absgrad is not None:
                            obj_tensor.absgrad = v.absgrad[obj_mask].contiguous()
                        
                        obj_info[k] = obj_tensor
                    else:
                        obj_info[k] = v
            else:
                obj_info = create_empty_info_with_absgrad()
            
            # Create canonical info with proper absgrad handling
            if canon_mask.any():
                canon_visible_ids = visible_ids[canon_mask] - n_obj  # Adjust IDs to 0-based
                canon_info = {"gaussian_ids": canon_visible_ids}
                
                for k, v in self.combined_info.items():
                    if k == "gaussian_ids":
                        continue
                        
                    if isinstance(v, torch.Tensor) and v.shape[0] == len(visible_ids):
                        canon_tensor = v[canon_mask].contiguous()
                        

                        if hasattr(v, 'absgrad') and v.absgrad is not None:
                            canon_tensor.absgrad = v.absgrad[canon_mask].contiguous()
                        
                        canon_info[k] = canon_tensor
                    else:
                        canon_info[k] = v
            else:
                canon_info = create_empty_info_with_absgrad()
        else:
            obj_info = create_empty_info_with_absgrad()
            canon_info = create_empty_info_with_absgrad()
        
        # Verify absgrad is present before calling strategy
        gradient_key = self.strategy.key_for_gradient
        if gradient_key in obj_info and not hasattr(obj_info[gradient_key], 'absgrad'):
            print(f"WARNING: Object info missing absgrad for {gradient_key}")
            obj_info[gradient_key].absgrad = torch.empty_like(obj_info[gradient_key])
        
        if gradient_key in canon_info and not hasattr(canon_info[gradient_key], 'absgrad'):
            print(f"WARNING: Canonical info missing absgrad for {gradient_key}")
            canon_info[gradient_key].absgrad = torch.empty_like(canon_info[gradient_key])
        
        obj_optimizers = {name.replace('obj_', ''): opt for name, opt in self.optimizers.items() if name.startswith('obj_')}
        
        print(f"Object: {len(obj_info['gaussian_ids'])} visible Gaussians")
        n_obj_before = self.gauss_params['means'].shape[0]
        
        if obj_info["gaussian_ids"].numel() > 0:
            self.strategy.step_post_backward(
                params=self.gauss_params,
                optimizers=obj_optimizers,
                state=self.strategy_state,
                step=self.step,
                info=obj_info,
                packed=True,
            )
        else:
            print(f"[Debug] Step {step}: Skipping object strategy (0 visible object Gaussians)")

        
        n_obj_after = self.gauss_params['means'].shape[0]
        print(f"   Object strategy complete: {n_obj_before} → {n_obj_after} Gaussians")
        
        # 2. Apply strategy to CANONICAL parameters
        canon_optimizers = {name.replace('canon_', ''): opt for name, opt in self.optimizers.items() if name.startswith('canon_')}
        
        print(f"Canonical: {len(canon_info['gaussian_ids'])} visible Gaussians")
        n_canon_before = self.gauss_params_canonical['means'].shape[0]
        
        # Initialize canonical strategy state if needed
        if not hasattr(self, 'strategy_state_canonical'):
            self.strategy_state_canonical = self.strategy.initialize_state(scene_scale=0.1)

        if canon_info["gaussian_ids"].numel() > 0:
            self.strategy.step_post_backward(
                params=self.gauss_params_canonical,
                optimizers=canon_optimizers,
                state=self.strategy_state_canonical,
                step=self.step,
                info=canon_info,
                packed=True,
            )
        else:
            print(f"⚠️ Step {step}: Skipping canonical strategy (0 visible canonical Gaussians)")


    # get_loss_dict with background accumulation penalty
    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        gt_img = self.composite_with_background(self.get_gt_img(batch["image"]), outputs["background"])
        pred_img = outputs["rgb"]

        mask = batch.get("mask", None)

        if mask is not None:
            assert mask.shape[:2] == gt_img.shape[:2] == pred_img.shape[:2]
            gt_img = gt_img * mask
            pred_img = pred_img * mask
        

        # === Losses ===
        Ll1 = torch.abs(gt_img - pred_img).mean()
        simloss = 1 - self.ssim(gt_img.permute(2, 0, 1)[None, ...],
                                pred_img.permute(2, 0, 1)[None, ...])
        loss_dict = {
            "main_loss": (1 - self.config.ssim_lambda) * Ll1 + self.config.ssim_lambda * simloss,
        }

        # # Background accumulation penalty
        if mask is not None and "accumulation" in outputs:
            accumulation = outputs["accumulation"]
            background_mask = ~mask.bool()
            background_acc_loss = (background_mask * accumulation).mean()
            loss_dict["background_acc_penalty"] = 0.1 * background_acc_loss

        # Scale regularization
        if self.config.use_scale_regularization and self.step % 10 == 0:
            scales = torch.exp(self.gauss_params["scales"])
            scale_ratios = scales.max(dim=-1)[0] / (scales.min(dim=-1)[0] + 1e-8)
            ratio_penalty = torch.clamp(scale_ratios - self.config.max_gauss_ratio, min=0.0)
            size_penalty = torch.clamp(scales.max(dim=-1)[0] - 0.15, min=0.0)
            scale_reg = 0.1 * (ratio_penalty.mean() + size_penalty.mean())
        else:
            scale_reg = torch.tensor(0.0).to(self.device)
        loss_dict["scale_reg"] = scale_reg

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

        if self.config.use_depth and "depth_image" in batch:
            depth_out = outputs["depth"]
            depth_gt = self.get_gt_img(batch["depth_image"])

            if mask is not None:
                assert mask.shape[:2] == depth_out.shape[:2] == depth_gt.shape[:2]
                depth_out = depth_out * mask
                depth_gt = depth_gt * mask

            if self.config.depth_debug_vis and self.step % 1000 == 0:
                depth_debug(self.step, depth_out, depth_gt, mask)

            # Ensure valid values
            depth_out = torch.clamp(depth_out, min=1e-4)
            depth_gt = torch.clamp(depth_gt, min=1e-4)
            valid = torch.isfinite(depth_out) & torch.isfinite(depth_gt)
            if mask is not None:
                valid &= (mask > 0.5)

            # Log-scale L1 loss
            if valid.any():
                depth_loss = (torch.log(depth_out[valid]) - torch.log(depth_gt[valid])).abs().mean()
            else:
                depth_loss = torch.tensor(0.0, device=depth_out.device)

            loss_dict["depth_loss"] = self.config.depth_lambda * depth_loss

        if self.step % 1000 == 0 and not getattr(self, '_debug_saved_this_step', False):
            self._debug_saved_this_step = True
            save_debug_id_maps(self, batch)
            # import pdb; pdb.set_trace()
        elif self.step % 1000 != 0:
            self._debug_saved_this_step = False
        
        return loss_dict

    def get_metrics_dict(self, outputs, batch) -> Dict[str, torch.Tensor]:
        """
        Computes the metrics for the model.
        """
        d = self._get_downscale_factor()
        if d > 1:
            # use torchvision to resize
            newsize = (batch["image"].shape[0] // d, batch["image"].shape[1] // d)
            gt_img = TF.resize(
                batch["image"].permute(2, 0, 1), newsize, antialias=None
            ).permute(1, 2, 0)

            if "depth_image" in batch:
                depth_size = (
                    batch["depth_image"].shape[0] // d,
                    batch["depth_image"].shape[1] // d,
                )
                sensor_depth_gt = TF.resize(
                    batch["depth_image"].permute(2, 0, 1), depth_size, antialias=None
                ).permute(1, 2, 0)
        else:
            gt_img = batch["image"]
            if "depth_image" in batch:
                sensor_depth_gt = batch["depth_image"]

        metrics_dict = {}
        gt_rgb = gt_img.to(self.device)  # RGB or RGBA image
        predicted_rgb = (
            outputs["rgb"][0, ...] if outputs["rgb"].dim() == 4 else outputs["rgb"]
        )

        with torch.no_grad():
            (psnr, ssim, lpips) = self.rgb_metrics(
                gt_rgb.permute(2, 0, 1).unsqueeze(0),
                predicted_rgb.permute(2, 0, 1).unsqueeze(0).to(self.device),
            )
            rgb_mse = self.mse_loss(gt_rgb.permute(2, 0, 1), predicted_rgb.permute(2, 0, 1))
            rgb_metrics = {
                "rgb_mse": float(rgb_mse),
                "rgb_psnr": float(psnr.item()),
                "rgb_ssim": float(ssim),
                "rgb_lpips": float(lpips),
            }
            metrics_dict.update(rgb_metrics)

        metrics_dict["gaussian_count"] = self.num_points


        with torch.no_grad():
            if "depth_image" in batch:
                predicted_depth = outputs['depth']
                # ADD THIS CHECK:
                if predicted_depth is not None:
                    (abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3) = self.depth_metrics(
                        predicted_depth.permute(2, 0, 1), sensor_depth_gt.permute(2, 0, 1)
                    )

                    depth_metrics = {
                        "depth_abs_rel": float(abs_rel.item()),
                        "depth_sq_rel": float(sq_rel.item()),
                        "depth_rmse": float(rmse.item()),
                        "depth_rmse_log": float(rmse_log.item()),
                        "depth_a1": float(a1.item()),
                        "depth_a2": float(a2.item()),
                        "depth_a3": float(a3.item()),
                    }
                    metrics_dict.update(depth_metrics)


        # track scales
        metrics_dict.update(
            {"avg_min_scale": torch.nanmean(torch.exp(self.scales[..., -1]))}
        )

        return metrics_dict

    def get_joint_angle_for_camera(self, camera: Cameras) -> float:
        """Extract joint angle from camera.times (interpolated) or metadata (fixed)"""

        if hasattr(camera, 'times') and camera.times is not None and self.joint_angles is not None:
            time_val = float(camera.times.flatten()[0])
            num_frames = len(self.joint_angles)
            frame_idx = int(time_val * (num_frames - 1))
            frame_idx = max(0, min(frame_idx, num_frames - 1))
            joint_angle = self.joint_angles[frame_idx].item()
            # print(f"Time {time_val:.3f} → Joint angle {joint_angle:.3f} rad")
            return joint_angle

        if hasattr(camera, 'metadata') and camera.metadata is not None:
            joint_angle = camera.metadata.get("joint_angle", 0.0)
            return joint_angle

        return 0.0
    

    def get_gaussians_for_render(self, camera: Cameras) -> Dict[str, torch.Tensor]:
        """
        Prepare Gaussians for rendering with per-frame articulation.
        Training: object (articulated) + canonical (identity)
        Eval: object (articulated) + canonical (identity) + background
        """
        
        joint_angle = self.get_joint_angle_for_camera(camera)
        
        articulated_obj_params = apply_articulation_to_optimizer_params(self, joint_angle)

        if self.training:
            
            combined_params = {}
            for name in articulated_obj_params.keys():
                obj_tensor = articulated_obj_params[name]
                canon_tensor = self.gauss_params_canonical[name]  # Canonical at identity pose
                
                # Concatenate along the first dimension (number of Gaussians)
                combined_params[name] = torch.cat([obj_tensor, canon_tensor], dim=0)
            
            n_obj = articulated_obj_params['means'].shape[0]
            n_canon = self.gauss_params_canonical['means'].shape[0]
            total = combined_params['means'].shape[0]
            
            # print(f"[Training render]: obj({n_obj}) + canon({n_canon}) = {total} total")
            # print(f"   Object: articulated at {joint_angle:.3f} rad")
            # print(f"   Canonical: identity pose")

            return combined_params
        else:
            # Evaluation mode: object (articulated) + canonical (identity) + background
            print(f"[ Eval mode]: combining articulated object + canonical + background")
            
            # Start with articulated object + canonical
            combined_params = {}
            for name in articulated_obj_params.keys():
                obj_tensor = articulated_obj_params[name]
                canon_tensor = self.gauss_params_canonical[name].data  # Use .data for eval
                combined_params[name] = torch.cat([obj_tensor, canon_tensor], dim=0)
            
            # Add background if available
            if (hasattr(self, "gauss_params_fixed") and 
                self.gauss_params_fixed is not None and 
                len(self.gauss_params_fixed) > 0 and
                self.gauss_params_fixed["means"].shape[0] > 0):
                
                full_scene_params = {}
                for name in combined_params.keys():
                    combined_tensor = combined_params[name]
                    bg_tensor = self.gauss_params_fixed[name].data
                    
                    if combined_tensor.device != bg_tensor.device:
                        bg_tensor = bg_tensor.to(combined_tensor.device)
                    
                    full_scene_params[name] = torch.cat([combined_tensor, bg_tensor], dim=0)
                
                n_obj = articulated_obj_params['means'].shape[0]
                n_canon = self.gauss_params_canonical['means'].shape[0]
                n_bg = self.gauss_params_fixed['means'].shape[0]
                total = full_scene_params['means'].shape[0]
                
                print(f"[Eval render check]:")
                print(f"   Object: {n_obj}, Canonical: {n_canon}, Background: {n_bg}, Total: {total}")
                
                return full_scene_params
            else:
                print("[!!] No background gaussians found - rendering object + canonical only")
                return combined_params


    # def _apply_articulation_to_canonical_params(self, joint_angle: float) -> Dict[str, torch.Tensor]:
    #     """
    #     Apply per-frame articulation to the canonical object parameters.
    #     """
    #     if joint_angle == 0.0:
    #         return {name: param.data for name, param in self.gauss_params.items()}
        
    #     means_articulated, quats_articulated = apply_joint_transform(
    #         means=self.gauss_params["means"].data,  # Use .data
    #         quats=self.gauss_params["quats"].data,  # Use .data
    #         joint_pivot=self.joint_pivot.to(self.device),
    #         joint_axis=self.joint_axis.to(self.device),
    #         joint_angle=joint_angle
    #     )
        
    #     articulated_params = {}
    #     for name, param in self.gauss_params.items():
    #         if name == "means":
    #             articulated_params[name] = means_articulated
    #         elif name == "quats":
    #             articulated_params[name] = quats_articulated
    #         else:
    #             articulated_params[name] = param.data  # Use .data
        
    #     return articulated_params


    def get_outputs(self, camera: Cameras, render_id_map: bool = False) -> Dict[str, Union[torch.Tensor, List]]:
        """Takes in a camera and returns a dictionary of outputs with articulation."""
        if not isinstance(camera, Cameras):
            return {}

        gaussians_to_render = self.get_gaussians_for_render(camera)
        self._current_camera = camera 

        joint_angle = self.get_joint_angle_for_camera(camera)
        
        # DEBUG: Verify what we're actually rendering
        n_obj = self.gauss_params['means'].shape[0]
        n_canon = self.gauss_params_canonical['means'].shape[0]
        actual_count = gaussians_to_render['means'].shape[0]
        
        if self.training:
            expected_count = n_obj + n_canon  # Both object and canonical
            print(f"Training render check:")
            print(f"   Expected (obj + canon): {expected_count} ({n_obj} + {n_canon})")
            print(f"   Actually rendering: {actual_count}")
        else:
            bg_count = self.gauss_params_fixed['means'].shape[0] if hasattr(self, 'gauss_params_fixed') else 0
            print(f" Eval render check:")
            print(f"   Object: {n_obj}, Canonical: {n_canon}, Background: {bg_count}, Total: {actual_count}")
        
        optimized_camera_to_world = self.camera_optimizer.apply_to_camera(camera) if self.training else camera.camera_to_worlds
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

        # Determine render mode
        if render_id_map:
            # Object: ID 0-255 (red channel)
            # Canonical: ID 256-511 (green channel) 
            # Background: ID 512+ (blue channel)
            
            id_colors = torch.zeros((actual_count, 3), device=gaussians_to_render["means"].device)
            
            # Object Gaussians (0 to n_obj-1): encode in red channel
            if n_obj > 0:
                obj_ids = torch.arange(n_obj, device=id_colors.device, dtype=torch.float32)
                id_colors[:n_obj, 0] = (obj_ids % 256) / 255.0  # Red channel
                id_colors[:n_obj, 1] = 0.0  # Green = 0
                id_colors[:n_obj, 2] = 0.0  # Blue = 0
            
            # Canonical Gaussians (n_obj to n_obj+n_canon-1): encode in green channel
            if n_canon > 0:
                canon_start = n_obj
                canon_end = n_obj + n_canon
                canon_ids = torch.arange(n_canon, device=id_colors.device, dtype=torch.float32)
                id_colors[canon_start:canon_end, 0] = 0.0  # Red = 0
                id_colors[canon_start:canon_end, 1] = (canon_ids % 256) / 255.0  # Green channel
                id_colors[canon_start:canon_end, 2] = 0.0  # Blue = 0
            
            # Background Gaussians (remaining): encode in blue channel
            if actual_count > n_obj + n_canon:
                bg_start = n_obj + n_canon
                bg_count = actual_count - bg_start
                bg_ids = torch.arange(bg_count, device=id_colors.device, dtype=torch.float32)
                id_colors[bg_start:, 0] = 0.0  # Red = 0
                id_colors[bg_start:, 1] = 0.0  # Green = 0
                id_colors[bg_start:, 2] = (bg_ids % 256) / 255.0  # Blue channel
            
            colors_for_render = id_colors
            render_mode = "RGB"
            sh_degree_to_use = None  # No SH for ID maps
            
            if actual_count > n_obj + n_canon:
                print(f"   Background: {n_obj+n_canon}-{actual_count-1} (blue channel)")
        else:
            # Normal RGB rendering
            render_mode = "RGB+ED" if self.config.output_depth_during_training or not self.training else "RGB"
            if self.config.sh_degree > 0:
                sh_degree_to_use = min(self.step // self.config.sh_degree_interval, self.config.sh_degree)
                colors_for_render = colors_crop
            else:
                colors_for_render = torch.sigmoid(colors_crop).squeeze(1)
                sh_degree_to_use = None

        render, alpha, self.info = rasterization(
            means=gaussians_to_render["means"],
            quats=gaussians_to_render["quats"],
            scales=torch.exp(gaussians_to_render["scales"]),
            opacities=torch.sigmoid(gaussians_to_render["opacities"]).squeeze(-1),
            colors=colors_for_render,
            viewmats=viewmat,
            Ks=K,
            width=W,
            height=H,
            packed=True,
            near_plane=0.01,
            far_plane=1e10,
            render_mode=render_mode,
            sh_degree=sh_degree_to_use,
            sparse_grad=False,
            absgrad=self.strategy.absgrad if isinstance(self.strategy, DefaultStrategy) else False,
            rasterize_mode=self.config.rasterize_mode,
        )
        

        # inject joint information for the new mask aware strategy
        if self.info is not None:
            self.info["joint_angle"] = joint_angle
            self.info["joint_pivot"] = self.joint_pivot
            self.info["joint_axis"] = self.joint_axis
            self.info["joint_type"] = self.joint_type
            if self.step % 100 == 0:  # Log occasionally
                print(f"Injected joint_angle={joint_angle:.3f} into render info")
        
        if self.training:
            self.strategy.step_pre_backward(
                params=self.gauss_params, 
                optimizers=self.optimizers,
                state=self.strategy_state,
                step=self.step,
                info=self.info  
            )

        # Debug info
        n_rendered = gaussians_to_render["means"].shape[0]
        print(f"Rendering {n_rendered} Gaussians")
        if self.info.get("gaussian_ids") is not None:
            # print(f"Gaussian IDs shape: {self.info['gaussian_ids'].shape}")
            # print(f"ID range: {self.info['gaussian_ids'].min()} to {self.info['gaussian_ids'].max()}")
            
            # Store info for strategy processing (will be split in step_post_backward)
            self.combined_info = self.info
            self.n_obj_rendered = n_obj
        else:
            print("gaussian_ids is None - NO VISIBLE GAUSSIANS!")
            self.combined_info = None
            self.n_obj_rendered = n_obj

        if render_id_map:
            # For ID maps, return raw render 
            background = self._get_background_color() 
            outputs = {
                "rgb": torch.clamp(render[..., :3], 0.0, 1.0).squeeze(0),
                "depth": None,
                "accumulation": alpha.squeeze(0),
                "background": background,
                "id_map": render[..., :3].squeeze(0),  # Raw ID colors
            }
            
            # Also decode the ID information for debugging
            id_debug = decode_id_map(render[..., :3].squeeze(0), n_obj, n_canon)
            outputs.update(id_debug)
            
            return outputs
        else:
            # Normal RGB rendering with background
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


    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        """Writes the test image outputs.

        Args:
            image_idx: Index of the image.
            step: Current step.
            batch: Batch of data.
            outputs: Outputs of the model.

        Returns:
            A dictionary of metrics.
        """
        gt_rgb = self.composite_with_background(self.get_gt_img(batch["image"]), outputs["background"])
        predicted_rgb = outputs["rgb"]
        cc_rgb = None

        combined_rgb = torch.cat([gt_rgb, predicted_rgb], dim=1)

        if self.config.color_corrected_metrics:
            cc_rgb = color_correct(predicted_rgb, gt_rgb)
            cc_rgb = torch.moveaxis(cc_rgb, -1, 0)[None, ...]

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        gt_rgb = torch.moveaxis(gt_rgb, -1, 0)[None, ...]
        predicted_rgb = torch.moveaxis(predicted_rgb, -1, 0)[None, ...]

        psnr = self.psnr(gt_rgb, predicted_rgb)
        ssim = self.ssim(gt_rgb, predicted_rgb)
        lpips = self.lpips(gt_rgb, predicted_rgb)

        # all of these metrics will be logged as scalars
        metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim)}  # type: ignore
        metrics_dict["lpips"] = float(lpips)

        if self.config.color_corrected_metrics:
            assert cc_rgb is not None
            cc_psnr = self.psnr(gt_rgb, cc_rgb)
            cc_ssim = self.ssim(gt_rgb, cc_rgb)
            cc_lpips = self.lpips(gt_rgb, cc_rgb)
            metrics_dict["cc_psnr"] = float(cc_psnr.item())
            metrics_dict["cc_ssim"] = float(cc_ssim)
            metrics_dict["cc_lpips"] = float(cc_lpips)

        time_val = float(batch["time"])
        mask_pre = batch.get("mask_pre", None)
        mask_post = batch.get("mask_post", None)

        if mask_pre is not None:
            mask_pre = self._downscale_if_required(mask_pre.to(self.device))
        if mask_post is not None:
            mask_post = self._downscale_if_required(mask_post.to(self.device))

        # --- Mask selection logic (same as training) ---
        if time_val <= 0.25 and mask_post is not None:
            mask = mask_post
        elif time_val >= 0.25 and mask_pre is not None and mask_post is not None:
            mask = torch.clamp(mask_pre + mask_post, 0.0, 1.0)
        else:
            mask = None 

        # print(f"mask shape: {mask.shape if mask is not None else None}")
        # print(f"gt_rgb shape: {gt_rgb.shape}")
        # print(f"predicted_rgb shape: {predicted_rgb.shape}")
        # print(mask.shape, mask.dtype)

        import os
        import torchvision.utils as vutils

        if mask is not None:
            if mask.ndim == 3 and mask.shape[-1] == 1:
                mask = mask.permute(2, 0, 1).unsqueeze(0).bool()  # [H,W,1] -> [1,1,H,W]
            elif mask.ndim == 2:
                mask = mask.unsqueeze(0).unsqueeze(0).bool()      # [H,W] -> [1,1,H,W]

            # Expand for RGB masking
            mask_expanded = mask.expand(-1, 3, -1, -1)
            debug_gt_masked = gt_rgb * mask_expanded
            debug_pred_masked = predicted_rgb * mask_expanded

            debug = False  # Set to True to enable debug saving
            
            # === Save only every 1000 steps ===
            if self.step % 1000 == 0 and debug:
                debug_dir = os.path.join("/local/home/pmishra/cvg/arti-splatfacto", "debug")
                os.makedirs(debug_dir, exist_ok=True)
                img_idx = int(batch["image_idx"])
                vutils.save_image(debug_gt_masked, f"{debug_dir}/step{self.step:06d}_gt_masked_idx{img_idx}.png")
                vutils.save_image(debug_pred_masked, f"{debug_dir}/step{self.step:06d}_pred_masked_idx{img_idx}.png")
                vutils.save_image(mask.float(), f"{debug_dir}/step{self.step:06d}_mask_idx{img_idx}.png")
            

            # Metrics
            psnr_masked = self.psnr_masked(gt_rgb, predicted_rgb, mask)
            metrics_dict["psnr_masked"] = float(psnr_masked.item())

            gt_rgb_crop = crop_imgs_w_masks(gt_rgb, mask)
            pred_rgb_crop = crop_imgs_w_masks(predicted_rgb, mask)

            metrics_dict["ssim_masked"] = float(self.ssim(gt_rgb_crop, pred_rgb_crop))
            metrics_dict["lpips_masked"] = float(self.lpips(gt_rgb_crop, pred_rgb_crop))

        # Save combined RGB for visualization (optional)
        images_dict = {"img": combined_rgb}

        return metrics_dict, images_dict
