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

    joint_type: str = field(default="prismatic")

    continue_cull_post_densification: bool = True
    """If True, continue to cull problematic gaussians even after densification stops"""
    
    # Enhanced culling parameters
    cull_post_densification_every: int = 50
    """How often to cull post-densification (in steps)"""
    
    cull_boundary_gaussians: bool = True
    """If True, cull Gaussians that drift outside object boundaries"""

    use_depth: bool = False
    """If True, use depth information for culling and optimization"""

    depth_lambda: float = 0.5
    """Weighting factor for depth information in loss function"""

    output_depth_during_training: bool = True
    """If True, output depth information during training"""


class ArtiSplatfactoModel(SplatfactoModel):    

    config: ArtiSplatfactoModelConfig

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Joint configuration
        self.joint_pivot = torch.tensor(self.config.joint_pivot, dtype=torch.float32, device=self.device)
        self.joint_axis = torch.tensor(self.config.joint_axis, dtype=torch.float32, device=self.device)
        self.joint_axis = F.normalize(self.joint_axis, dim=0)
        self.joint_type = self.config.joint_type

        # Get joint angles from metadata if available
        self.metadata = kwargs.get("metadata", {})
        self.joint_angles = self.metadata.get("joint_angles", None)

        self._needs_optimizer_recreation = False

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

        self.obj_3d_seg = Object3DSeg.read_from_file(self.config.obj_mask_file, device=self.device)
        print("obj mask points:", self.obj_3d_seg)
        self.obj_3d_seg.refine_mask(dilate_k=4, erode_k=1)

        GAUSS = ["means", "scales", "quats", "features_dc", "features_rest", "opacities"]

        all_means = state_dict["gauss_params.means"].to(self.device)
        obj_mask = self.obj_3d_seg.query_refine(all_means, grow=1, thresh=0.01, bbox_margin=0.01).to(torch.bool).cpu()
        # obj_mask = self.obj_3d_seg.query(all_means).to(torch.bool).cpu()
        bg_mask  = ~obj_mask

        for p in GAUSS:
            subset = state_dict[f"gauss_params.{p}"][obj_mask].to(self.device)
            # Create new Parameter with requires_grad=True
            param = torch.nn.Parameter(subset.clone().detach(), requires_grad=True)
            self.gauss_params[p] = param
            print(f"✅ Object {p}: {param.shape}, requires_grad={param.requires_grad}")

        # Canonical gaussians (trainable)
        for p in GAUSS:
            subset = state_dict[f"gauss_params.{p}"][obj_mask].to(self.device)
            # Create new Parameter with requires_grad=True
            param = torch.nn.Parameter(subset.clone().detach(), requires_grad=True)
            self.gauss_params_canonical[p] = param
            print(f"✅ Canonical {p}: {param.shape}, requires_grad={param.requires_grad}")

        # Fixed background gaussians (non-trainable)
        for p in GAUSS:
            subset = state_dict[f"gauss_params.{p}"][bg_mask].to(self.device)
            # Create new Parameter with requires_grad=False
            param = torch.nn.Parameter(subset.clone().detach(), requires_grad=False)
            self.gauss_params_fixed[p] = param
            print(f"✅ Background {p}: {param.shape}, requires_grad={param.requires_grad}")

        # IDs: reuse if present, else allocate

        n_obj = self.gauss_params["means"].shape[0]
        n_canon = self.gauss_params_canonical["means"].shape[0]
        n_bg = self.gauss_params_fixed["means"].shape[0]


        print(f"Partitioning complete. Trainable: {self.gauss_params['means'].shape[0]}, Canonical: {self.gauss_params_canonical['means'].shape[0]}, Fixed: {self.gauss_params_fixed['means'].shape[0]}")

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor], **kwargs):
        print(f"Loading state_dict (Training mode: {self.training})")
        assert self.config.obj_mask_file is not None and self.config.obj_mask_file.exists()

        self.obj_3d_seg = Object3DSeg.read_from_file(self.config.obj_mask_file, device=self.device)
        if hasattr(self.obj_3d_seg, 'joint_axis') and self.obj_3d_seg.joint_axis is not None:
            self.joint_axis = self.obj_3d_seg.joint_axis.to(self.device)
            print(f"Updated joint axis from mask: {self.joint_axis}")
        if hasattr(self.obj_3d_seg, 'joint_pivot') and self.obj_3d_seg.joint_pivot is not None:
            self.joint_pivot = self.obj_3d_seg.joint_pivot.to(self.device)
            print(f"Updated joint pivot from mask: {self.joint_pivot}")
        if hasattr(self.obj_3d_seg, 'joint_angle') and self.obj_3d_seg.joint_angle is not None:
            self.max_joint_angle = self.obj_3d_seg.joint_angle.to(self.device)
            print(f"Updated joint angles from mask: {self.max_joint_angle}")


        # # Align mask (extended → rest)
        # if self.config.joint_type == "prismatic":
        #     self.max_joint_angle = torch.tensor(0.3, device=self.device)
        #     translation = -self.joint_axis * self.max_joint_angle

        #     # Shift the mask bbox
        #     self.obj_3d_seg.bbox_min = self.obj_3d_seg.bbox_min + translation
        #     self.obj_3d_seg.bbox_max = self.obj_3d_seg.bbox_max + translation

        #     # NOTE: voxel grid stays the same (local coords inside bbox).
        #     # We just shift the bbox so query() and query_refine() match the Gaussians.
        #     print(f"[Mask Alignment] Shifted mask bbox into rest space by {translation}")


        GAUSS = ["means", "scales", "quats", "features_dc", "features_rest", "opacities"]

        # Backward compatibility for old checkpoints
        if "means" in state_dict:
            for p in GAUSS:
                state_dict[f"gauss_params.{p}"] = state_dict[p]

        if not self.training:
            # At eval, trust checkpoint background if any fixed gaussians exist
            is_partitioned = any(k.startswith("gauss_params_fixed.") for k in state_dict)
        else:
            is_partitioned = "gauss_params_fixed.means" in state_dict

        if is_partitioned:
            # Directly load partitioned checkpoint
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
        
        # print(f" Applying strategy to both object and canonical parameters")
        # print(f"   Object: {self.gauss_params['means'].shape[0]} Gaussians")
        # print(f"   Canonical: {self.gauss_params_canonical['means'].shape[0]} Gaussians")

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
            
            # CRITICAL: Attach absgrad attribute
            empty_tensor.absgrad = torch.empty_like(empty_tensor)
            
            return {"gaussian_ids": empty_ids, key: empty_tensor}
        
        # Split the combined info for object and canonical
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
                # No visible object Gaussians
                obj_info = create_empty_info_with_absgrad()
            
            # Create canonical info with proper absgrad handling
            if canon_mask.any():
                canon_visible_ids = visible_ids[canon_mask] - n_obj  # Adjust IDs to 0-based
                canon_info = {"gaussian_ids": canon_visible_ids}
                
                for k, v in self.combined_info.items():
                    if k == "gaussian_ids":
                        continue
                        
                    if isinstance(v, torch.Tensor) and v.shape[0] == len(visible_ids):
                        # Extract canonical portion
                        canon_tensor = v[canon_mask].contiguous()
                        
                        # Handle absgrad if present
                        if hasattr(v, 'absgrad') and v.absgrad is not None:
                            canon_tensor.absgrad = v.absgrad[canon_mask].contiguous()
                        
                        canon_info[k] = canon_tensor
                    else:
                        canon_info[k] = v
            else:
                # No visible canonical Gaussians
                canon_info = create_empty_info_with_absgrad()
        else:
            # No visible Gaussians at all
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
        
        # 1. Apply strategy to OBJECT parameters
        obj_optimizers = {name.replace('obj_', ''): opt for name, opt in self.optimizers.items() if name.startswith('obj_')}
        
        print(f"Object: {len(obj_info['gaussian_ids'])} visible Gaussians")
        n_obj_before = self.gauss_params['means'].shape[0]
        
        self.strategy.step_post_backward(
            params=self.gauss_params,
            optimizers=obj_optimizers,
            state=self.strategy_state,
            step=self.step,
            info=obj_info,
            packed=True,
        )
        
        n_obj_after = self.gauss_params['means'].shape[0]
        print(f"   Object strategy complete: {n_obj_before} → {n_obj_after} Gaussians")
        
        # 2. Apply strategy to CANONICAL parameters
        canon_optimizers = {name.replace('canon_', ''): opt for name, opt in self.optimizers.items() if name.startswith('canon_')}
        
        print(f"Canonical: {len(canon_info['gaussian_ids'])} visible Gaussians")
        n_canon_before = self.gauss_params_canonical['means'].shape[0]
        
        # Initialize canonical strategy state if needed
        if not hasattr(self, 'strategy_state_canonical'):
            self.strategy_state_canonical = self.strategy.initialize_state(scene_scale=0.1)

        self.strategy.step_post_backward(
            params=self.gauss_params_canonical,
            optimizers=canon_optimizers,
            state=self.strategy_state_canonical,
            step=self.step,
            info=canon_info,
            packed=True,
        )
        
        n_canon_after = self.gauss_params_canonical['means'].shape[0]
        # print(f"   Canonical strategy complete: {n_canon_before} → {n_canon_after} Gaussians")
        
        # print(f" Both strategies complete:")
        # print(f"   Object: {n_obj_before} → {n_obj_after}")
        # print(f"   Canonical: {n_canon_before} → {n_canon_after}")



    # get_loss_dict with background accumulation penalty
    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        gt_img = self.composite_with_background(self.get_gt_img(batch["image"]), outputs["background"])
        pred_img = outputs["rgb"]

        time_val = float(batch["time"])  
        mask_pre = batch.get("mask_pre", None)
        mask_post = batch.get("mask_post", None)

        if mask_pre is not None:
            mask_pre = self._downscale_if_required(mask_pre.to(self.device))
        if mask_post is not None:
            mask_post = self._downscale_if_required(mask_post.to(self.device))

        # If door is still mostly closed, supervise only with mask_post
        if time_val <= 0.25 and mask_post is not None:
            mask = mask_post
        # If door is opening/opened, use mask_post (revealed area + door)
        elif time_val >= 0.25 and mask_pre is not None and mask_post is not None:
            mask = torch.clamp(mask_pre + mask_post, 0.0, 1.0)
        else:
            mask = None

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

        # Background accumulation penalty
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
            depth_out = outputs['depth']
            depth_batch = self.get_gt_img(batch["depth_image"])

            if mask is not None:
                assert mask.shape[:2] == depth_out.shape[:2] == depth_batch.shape[:2]
                # depth_out = depth_out * mask
                # depth_batch = depth_batch * mask

            debug = False  # Set to True to enable debug saving
         

            if self.step % 1000 == 0 and debug:
                import os
                import torchvision.utils as vutils
                debug_dir = os.path.join("/local/home/pmishra/cvg/arti-splatfacto", "debug_depth")
                os.makedirs(debug_dir, exist_ok=True)
                
                # Print depth value ranges for debugging
                print(f"Step {self.step} - Depth Analysis:")
                print(f"  Predicted depth range: {depth_out.min():.4f} to {depth_out.max():.4f}")
                print(f"  Ground truth depth range: {depth_batch.min():.4f} to {depth_batch.max():.4f}")
                print(f"  Depth ratio (pred/gt): {depth_out.mean()/depth_batch.mean():.4f}")
                
                
                # Normalize for visualization (0-1 range)
                if depth_out.max() > depth_out.min():
                    depth_out_norm = (depth_out - depth_out.min()) / (depth_out.max() - depth_out.min())
                else:
                    depth_out_norm = torch.zeros_like(depth_out)
                    
                if depth_batch.max() > depth_batch.min():
                    depth_gt_norm = (depth_batch - depth_batch.min()) / (depth_batch.max() - depth_batch.min())
                else:
                    depth_gt_norm = torch.zeros_like(depth_batch)
                
                # Save images (convert HWC to CHW for save_image)
                vutils.save_image(depth_out_norm.permute(2,0,1), f"{debug_dir}/step{self.step:06d}_pred_depth.png")
                vutils.save_image(depth_gt_norm.permute(2,0,1), f"{debug_dir}/step{self.step:06d}_gt_depth.png")
                
                # Save difference map
                diff = torch.abs(depth_out_norm - depth_gt_norm)
                vutils.save_image(diff.permute(2,0,1), f"{debug_dir}/step{self.step:06d}_depth_diff.png")
                
                if mask is not None:
                    mask_to_save = mask.float().permute(2,0,1)  # (1,H,W), float in [0,1]
                    vutils.save_image(mask_to_save, f"{debug_dir}/step{self.step:06d}_mask.png")

                # import pdb; pdb.set_trace()
            
            valid_mask = torch.isfinite(depth_out) & torch.isfinite(depth_batch)
            valid_depth_out = depth_out[valid_mask]
            valid_depth_batch = depth_batch[valid_mask]

            # if valid_depth_out.numel() > 0: 
            #     numerator = torch.sum(valid_depth_out * valid_depth_batch)
            #     denominator = torch.sum(valid_depth_out * valid_depth_out)
            #     scale = numerator / (denominator + 1e-8)
            #     scaled_pred_depth = scale * valid_depth_out
            #     depth_loss = torch.abs(scaled_pred_depth - valid_depth_batch).mean()
            # else:
            #     depth_loss = torch.tensor(0.0, device=depth_out.device) 
            depth_loss = (torch.log(valid_depth_out) - torch.log(valid_depth_batch)).abs().mean()

            loss_dict['depth_loss'] = self.config.depth_lambda * depth_loss

        if self.step % 1000 == 0 and not getattr(self, '_debug_saved_this_step', False):
            self._debug_saved_this_step = True
            self._save_debug_id_maps(batch)
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

    def _get_joint_angle_for_camera(self, camera: Cameras) -> float:
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
    

    def _get_gaussians_for_render(self, camera: Cameras) -> Dict[str, torch.Tensor]:
        """
        Prepare Gaussians for rendering with per-frame articulation.
        Training: object (articulated) + canonical (identity)
        Eval: object (articulated) + canonical (identity) + background
        """
        # DEBUG: Check gradients at start
        # for name, param in self.gauss_params.items():
        #     print(f"   {name}: requires_grad={param.requires_grad}")
        
        joint_angle = self._get_joint_angle_for_camera(camera)
        
        # Apply articulation to object parameters
        # print(f"[render gauss] Applying articulation with angle {joint_angle}")
        articulated_obj_params = self._apply_articulation_to_optimizer_params(joint_angle)

        if self.training:
            # Training mode: combine articulated object + canonical (at identity pose)
            # print(f" Training mode: combining object + canonical Gaussians")
            
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
                print("⚠️  No background gaussians found - rendering object + canonical only")
                return combined_params
    
    def _apply_articulation_to_optimizer_params(self, joint_angle: float) -> Dict[str, torch.Tensor]:
        articulated_params = {}

        # Only apply articulation to object Gaussians
        obj_means = self.gauss_params["means"]
        obj_quats = self.gauss_params["quats"]

        if joint_angle != 0.0:
            if self.joint_type == "revolute":
                obj_means, obj_quats = apply_joint_transform(
                    means=obj_means, quats=obj_quats,
                    joint_pivot=self.joint_pivot,
                    joint_axis=self.joint_axis,
                    joint_angle=joint_angle
                )
            elif self.joint_type == "prismatic":
                obj_means, obj_quats = apply_joint_transform_prismatic(
                    means=obj_means, quats=obj_quats,
                    joint_pivot=self.joint_pivot,
                    joint_axis=self.joint_axis,
                    joint_disp=joint_angle
                )

        # Rebuild params: only object Gaussians articulated
        for name, param in self.gauss_params.items():
            if name == "means":
                articulated_params[name] = obj_means
            elif name == "quats":
                articulated_params[name] = obj_quats
            else:
                articulated_params[name] = param

        return articulated_params


    def _apply_articulation_to_canonical_params(self, joint_angle: float) -> Dict[str, torch.Tensor]:
        """
        Apply per-frame articulation to the canonical object parameters.
        """
        if joint_angle == 0.0:
            return {name: param.data for name, param in self.gauss_params.items()}
        
        means_articulated, quats_articulated = apply_joint_transform(
            means=self.gauss_params["means"].data,  # Use .data
            quats=self.gauss_params["quats"].data,  # Use .data
            joint_pivot=self.joint_pivot.to(self.device),
            joint_axis=self.joint_axis.to(self.device),
            joint_angle=joint_angle
        )
        
        articulated_params = {}
        for name, param in self.gauss_params.items():
            if name == "means":
                articulated_params[name] = means_articulated
            elif name == "quats":
                articulated_params[name] = quats_articulated
            else:
                articulated_params[name] = param.data  # Use .data
        
        return articulated_params


    def get_outputs(self, camera: Cameras, render_id_map: bool = False) -> Dict[str, Union[torch.Tensor, List]]:
        """Takes in a camera and returns a dictionary of outputs with articulation."""
        if not isinstance(camera, Cameras):
            return {}

        gaussians_to_render = self._get_gaussians_for_render(camera)
        self._current_camera = camera 
        
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
            # For ID map rendering, we need to create pseudo-colors that encode the Gaussian type
            # Object: ID 0-255 (red channel)
            # Canonical: ID 256-511 (green channel) 
            # Background: ID 512+ (blue channel)
            
            # Create type-based colors instead of learned colors
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
            
            # print(f" Rendering ID map with {actual_count} Gaussians:")
            # print(f"   Object: 0-{n_obj-1} (red channel)")
            # print(f"   Canonical: {n_obj}-{n_obj+n_canon-1} (green channel)")
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
            # For ID maps, return raw render without background blending
            background = self._get_background_color() 
            outputs = {
                "rgb": torch.clamp(render[..., :3], 0.0, 1.0).squeeze(0),
                "depth": None,
                "accumulation": alpha.squeeze(0),
                "background": background,
                "id_map": render[..., :3].squeeze(0),  # Raw ID colors
            }
            
            # Also decode the ID information for debugging
            id_debug = self._decode_id_map(render[..., :3].squeeze(0), n_obj, n_canon)
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

    def _decode_id_map(self, id_map: torch.Tensor, n_obj: int, n_canon: int) -> Dict[str, torch.Tensor]:
        """
        Decode the ID map to extract masks for different Gaussian types.
        
        Args:
            id_map: (H, W, 3) tensor with encoded IDs
            n_obj: Number of object Gaussians
            n_canon: Number of canonical Gaussians
        
        Returns:
            Dictionary with decoded masks and statistics
        """
        H, W = id_map.shape[:2]
        
        # Extract channel values and convert back to IDs
        red_vals = (id_map[..., 0] * 255).round().long()
        green_vals = (id_map[..., 1] * 255).round().long()
        blue_vals = (id_map[..., 2] * 255).round().long()
        
        # Create masks for each type
        obj_mask = (red_vals > 0) & (green_vals == 0) & (blue_vals == 0)
        canon_mask = (red_vals == 0) & (green_vals > 0) & (blue_vals == 0)
        bg_mask = (red_vals == 0) & (green_vals == 0) & (blue_vals > 0)
        
        # Count pixels for each type
        obj_pixels = obj_mask.sum().item()
        canon_pixels = canon_mask.sum().item()
        bg_pixels = bg_mask.sum().item()
        total_pixels = H * W
        
        # print(f"🔍 ID Map Analysis:")
        # print(f"   Object pixels: {obj_pixels} ({obj_pixels/total_pixels*100:.1f}%)")
        # print(f"   Canonical pixels: {canon_pixels} ({canon_pixels/total_pixels*100:.1f}%)")
        # print(f"   Background pixels: {bg_pixels} ({bg_pixels/total_pixels*100:.1f}%)")
        # print(f"   Empty pixels: {total_pixels - obj_pixels - canon_pixels - bg_pixels}")
        
        # Check for floaters (simplified to avoid memory issues)
        if obj_pixels > 0:
            # Simple check: count object pixels in border regions
            H, W = obj_mask.shape
            border_width = 20
            
            # Create border mask (pixels near edges)
            border_mask = torch.zeros_like(obj_mask)
            border_mask[:border_width, :] = 1  # Top
            border_mask[-border_width:, :] = 1  # Bottom
            border_mask[:, :border_width] = 1  # Left
            border_mask[:, -border_width:] = 1  # Right
            
            # Count object pixels in border regions (potential floaters)
            border_obj_pixels = (obj_mask & border_mask).sum().item()
            border_ratio = border_obj_pixels / obj_pixels if obj_pixels > 0 else 0
            
            if border_ratio > 0.1:  # More than 10% in border regions
                # print(f"⚠️  Found {border_obj_pixels} object pixels in border regions ({border_ratio:.1%})")
        
        return {
            "obj_mask": obj_mask.float(),
            "canon_mask": canon_mask.float(), 
            "bg_mask": bg_mask.float(),
            "obj_pixel_count": obj_pixels,
            "canon_pixel_count": canon_pixels,
            "bg_pixel_count": bg_pixels,
        }
    

    def _save_debug_id_maps(self, batch):
        """Save ID maps for debugging floater Gaussians"""
        import os
        import torchvision.utils as vutils
        
        debug_dir = "/local/home/pmishra/cvg/arti-splatfacto/arti_debug"  
        os.makedirs(debug_dir, exist_ok=True)
        
        try:
            # Get camera from batch
            camera = getattr(self, '_current_camera', None)
            if camera is None:
                print("No camera found in batch for debug saving")
                return
                
            # Render ID map
            debug_outputs = self.get_outputs(camera, render_id_map=True)
            
            id_map = debug_outputs["id_map"]
            obj_mask = debug_outputs["obj_mask"] 
            canon_mask = debug_outputs["canon_mask"]
            
            vutils.save_image(id_map.permute(2,0,1), f"{debug_dir}/step{self.step:06d}_id_map.png")
            vutils.save_image(obj_mask.unsqueeze(0), f"{debug_dir}/step{self.step:06d}_obj_mask.png") 
            vutils.save_image(canon_mask.unsqueeze(0), f"{debug_dir}/step{self.step:06d}_canon_mask.png")
            
            # Save normal RGB render (with pink background issue)
            normal_outputs = self.get_outputs(camera, render_id_map=False)
            rgb_rendered = normal_outputs["rgb"]
            vutils.save_image(rgb_rendered.permute(2,0,1), f"{debug_dir}/step{self.step:06d}_rgb_rendered.png")
            
            # Save ground truth image from batch (actual camera image)
            gt_img = self.get_gt_img(batch["image"])
            if self._get_downscale_factor() > 1:
                # Match the downscaling applied to rendered image
                import torchvision.transforms.functional as TF
                d = self._get_downscale_factor()
                newsize = (gt_img.shape[0] // d, gt_img.shape[1] // d)
                gt_img = TF.resize(gt_img.permute(2, 0, 1), newsize, antialias=None).permute(1, 2, 0)
            
            vutils.save_image(gt_img.permute(2,0,1), f"{debug_dir}/step{self.step:06d}_gt_image.png")
            
            
        except Exception as e:
            print(f"Failed to save debug ID maps: {e}")


    def psnr_masked(self, image, rgb, mask):
        assert mask.dtype == torch.bool
        # mask: [1, 1, H, W], image/rgb: [1, 3, H, W]
        mask = mask.expand(-1, 3, -1, -1)   # expand channel dim only
        return self.psnr(image[mask], rgb[mask])


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

        

def crop_imgs_w_masks(images, masks, resize=(256, 256)):
    """
    Crop the images to the smallest bbox containing the masks
    """
    # Masks to bboxes
    bboxes = []
    for mask in masks:
        point_coords = torch.nonzero(mask.squeeze())[:, [1, 0]]
        bbox = compute_2D_bbox(point_coords.unsqueeze(0)).float()
        bboxes.append(bbox)
    bboxes = torch.cat(bboxes, dim=0)
    imgs_cropped = batch_crop_resize(images, bboxes, *resize)
    return imgs_cropped   


def compute_2D_bbox(points):
    """
    Compute bboxes for a batch of 2D points
    """
    assert len(points.shape) == 3
    mins, _ = torch.min(points, dim=1)
    maxs, _= torch.max(points, dim=1)
    bboxes = torch.cat((mins, maxs), dim=1)
    return bboxes       

def batch_crop_resize(
    img, rois, out_H, out_W, aligned=True, interpolation="bilinear"
):
    """
    Crop and resize images
    """
    assert len(img.shape) >= 3 and img.shape[-3] == 3, \
        "Error: Image size must be (*, 3, H, W)"
    assert rois.shape[-1] == 4, "Error: Bboxes should be Bx4"
    roi_idx = torch.arange(rois.size(0)).view(-1, 1).to(rois)
    rois = torch.cat((roi_idx, rois), dim=-1)
    # Crop and resize
    output_size = (out_H, out_W)
    from torchvision.ops import RoIAlign, RoIPool
    if interpolation == "bilinear":
        op = RoIAlign(output_size, 1.0, 0, aligned=aligned)
    elif interpolation == "nearest":
        op = RoIPool(output_size, 1.0)  #
    else:
        raise ValueError(f"Wrong interpolation type: {interpolation}")
    return op(img, rois)    

def apply_joint_transform(means, quats, joint_pivot, joint_axis, joint_angle):
    """
    Apply revolute joint transformation while preserving gradients.
    """
    
    # if joint_angle == 0.0:
    #     return means, quats
    
    # # DEBUG: Check input gradients
    # print(f" apply_joint_transform input:")
    # print(f"   means requires_grad: {means.requires_grad}")
    # print(f"   quats requires_grad: {quats.requires_grad}")
    
    # CRITICAL: Ensure joint_pivot and joint_axis don't break gradients
    if not joint_pivot.requires_grad:
        joint_pivot = joint_pivot.detach()  # Explicitly detach constants
    if not joint_axis.requires_grad:
        joint_axis = joint_axis.detach()    # Explicitly detach constants
    
    means_local = means - joint_pivot.unsqueeze(0)
    axis_angle = joint_axis * (-joint_angle)
    R = axis_angle_to_matrix(axis_angle.unsqueeze(0)).squeeze(0)  # [3, 3]
    
    means_rotated = torch.matmul(means_local, R.T) + joint_pivot.unsqueeze(0)
    
    joint_quat = matrix_to_quaternion(R.unsqueeze(0)).squeeze(0)  # [4]
    quats_rotated = quaternion_multiply(
        joint_quat.unsqueeze(0).expand_as(quats), 
        quats
    )
    
    # # DEBUG: Check output gradients
    # print(f"apply_joint_transform output:")
    # print(f"   means_rotated requires_grad: {means_rotated.requires_grad}")
    # print(f"   quats_rotated requires_grad: {quats_rotated.requires_grad}")

    # import pdb; pdb.set_trace()  # Debugging breakpoint
    
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
