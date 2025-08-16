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

    continue_cull_post_densification: bool = True
    """If True, continue to cull problematic gaussians even after densification stops"""
    
    # Enhanced culling parameters
    cull_post_densification_every: int = 50
    """How often to cull post-densification (in steps)"""
    
    cull_boundary_gaussians: bool = True
    """If True, cull Gaussians that drift outside object boundaries"""

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
        print("Populating ArtiSplatfactoModel modules...")

        def make_param(shape, requires_grad=True):
            return torch.nn.Parameter(
                torch.empty(shape, device="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.float32),
                requires_grad=requires_grad,
            )

        dim_sh = num_sh_bases(self.config.sh_degree)

        self.gauss_params = torch.nn.ParameterDict({
            "means":         make_param((0, 3)),
            "scales":        make_param((0, 3)),
            "quats":         make_param((0, 4)),
            "features_dc":   make_param((0, 3)),
            "features_rest": make_param((0, dim_sh - 1, 3)),
            "opacities":     make_param((0, 1)),
        })

        # # CANONICAL object state (for exposing revealed part)
        self.gauss_params_canonical = torch.nn.ParameterDict({
            "means":         make_param((0, 3)),
            "scales":        make_param((0, 3)),
            "quats":         make_param((0, 4)),
            "features_dc":   make_param((0, 3)),
            "features_rest": make_param((0, dim_sh - 1, 3)),
            "opacities":     make_param((0, 1)),
        })

        # Fixed (non-trainable) Gaussians 
        self.gauss_params_fixed = {}

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.register_buffer("gauss_ids_obj", torch.empty((0,), dtype=torch.long, device=device))
        self.register_buffer("gauss_ids_canon", torch.empty((0,), dtype=torch.long, device=device))
        self.register_buffer("gauss_ids_fixed", torch.empty((0,), dtype=torch.long, device=device))
        self.register_buffer("next_gauss_id", torch.tensor(0, dtype=torch.long, device=device))

        print(f"densification strategy: {self.strategy}")
        print(f"densification parameter: {self.config.warmup_length}, {self.config.stop_split_at}")
        

    def alloc_ids(self, n: int):
        """Allocate Gaussian IDs."""
        if n <= 0:
            return torch.empty((0,), dtype=torch.long, device=self.gauss_ids_obj.device)
        start_id = int(self.next_gauss_id.item())
        ids = torch.arange(start_id, start_id + n, dtype=torch.long, device=self.gauss_ids_obj.device)
        self.next_gauss_id += n
        return ids
    
    @property
    def ids_obj(self):
        """ IDs of trainable gaussians (object only) """
        return self.gauss_ids_obj

    
    @property
    def ids_all(self):
        """IDs of all gaussians for rendering (trainable obj + canonical + fixed)"""
        return torch.cat(
            [self.gauss_ids_obj, self.gauss_ids_canon, self.gauss_ids_fixed],
            dim=0
        )


    # def state_dict(self, *args, **kwargs):
    #     state = super().state_dict(*args, **kwargs)
    #     if hasattr(self, "gauss_params_fixed"):
    #         for name, param in self.gauss_params_fixed.items():
    #             state[f"gauss_params_fixed.{name}"] = param.data

    #     return state
    
    def _initialize_and_partition(self, state_dict: Dict[str, torch.Tensor]):
        """
        From a full-scene checkpoint: split into trainable object + fixed background.
        (Canonical stays empty here; you can fill it later if you have an exposed mask.)
        """
        print("Initializing from full scene: partitioning Gaussians...")

        self.obj_3d_seg = Object3DSeg.read_from_file(self.config.obj_mask_file, device=self.device)
        self.obj_3d_seg.refine_mask(dilate_k=4, erode_k=1)

        GAUSS = ["means", "scales", "quats", "features_dc", "features_rest", "opacities"]

        all_means = state_dict["gauss_params.means"].to(self.device)
        obj_mask = self.obj_3d_seg.query_refine(all_means, grow=1, thresh=0.01, bbox_margin=0.01).to(torch.bool).cpu()
        # obj_mask = self.obj_3d_seg.query(all_means).to(torch.bool).cpu()
        bg_mask  = ~obj_mask

        # trainable object subset
        for p in GAUSS:
            subset = state_dict[f"gauss_params.{p}"][obj_mask].to(self.device)
            self.gauss_params[p] = torch.nn.Parameter(subset)

        # trainable canonical subset
        for p in GAUSS:
            subset = state_dict[f"gauss_params.{p}"][obj_mask].to(self.device)
            self.gauss_params_canonical[p] = torch.nn.Parameter(subset)

        # Fixed background (plain tensors)
        self.gauss_params_fixed = {p: state_dict[f"gauss_params.{p}"][bg_mask].to(self.device) for p in GAUSS}

        # IDs: reuse if present, else allocate

        n_obj = self.gauss_params["means"].shape[0]
        n_canon = self.gauss_params_canonical["means"].shape[0]
        n_bg = self.gauss_params_fixed["means"].shape[0]

        if "gauss_ids" in state_dict:
            ids_all = state_dict["gauss_ids"].to(self.device)
            self.gauss_ids_obj   = ids_all[obj_mask.to(ids_all.device)]
            self.gauss_ids_fixed = ids_all[bg_mask .to(ids_all.device)]
            self.gauss_ids_canonical = ids_all[obj_mask.to(ids_all.device)]
            self.next_gauss_id   = torch.tensor(int(ids_all.max().item()) + 1, dtype=torch.long, device=self.device)
        else:
            self.gauss_ids_obj   = self.alloc_ids(n_obj)
            self.gauss_ids_canonical = self.alloc_ids(n_canon)
            self.gauss_ids_fixed = self.alloc_ids(n_bg)
            

        print(f"Partitioning complete. Trainable: {self.gauss_params['means'].shape[0]}, Canonical: {self.gauss_params_canonical['means'].shape[0]}, Fixed: {self.gauss_params_fixed['means'].shape[0]}")
        print(f"obj ID <{self.gauss_ids_obj}>, fixed ID <{self.gauss_ids_fixed}>, canon ID <{self.gauss_ids_canonical}>")

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

        GAUSS = ["means", "scales", "quats", "features_dc", "features_rest", "opacities"]

        # Backward compatibility for old checkpoints
        if "means" in state_dict:
            for p in GAUSS:
                state_dict[f"gauss_params.{p}"] = state_dict[p]

        is_partitioned = "gauss_params_fixed.means" in state_dict

        if is_partitioned:
            # Directly load partitioned checkpoint (already separated into 3 groups)
            for p in GAUSS:
                if f"gauss_params.{p}" in state_dict:
                    self.gauss_params[p] = torch.nn.Parameter(state_dict[f"gauss_params.{p}"].to(self.device))
                if f"gauss_params_canonical.{p}" in state_dict:
                    self.gauss_params_canonical[p] = torch.nn.Parameter(state_dict[f"gauss_params_canonical.{p}"].to(self.device))
                if f"gauss_params_fixed.{p}" in state_dict:
                    self.gauss_params_fixed[p] = state_dict[f"gauss_params_fixed.{p}"].to(self.device)

            # IDs if present
            self.gauss_ids_obj        = state_dict.get("gauss_ids_obj", torch.empty((0,), dtype=torch.long, device=self.device)).to(self.device)
            self.gauss_ids_canon  = state_dict.get("gauss_ids_canon", torch.empty((0,), dtype=torch.long, device=self.device)).to(self.device)
            self.gauss_ids_fixed  = state_dict.get("gauss_ids_fixed", torch.empty((0,), dtype=torch.long, device=self.device)).to(self.device)

        else:
            print("Partitioning full scene into obj/canonical/bg...")
            self._initialize_and_partition(state_dict)

        # Load non-gaussian parameters
        non_gauss_state = {
            k: v for k, v in state_dict.items()
            if not (
                k.startswith("gauss_params.") or
                k.startswith("gauss_params_canonical.") or
                k.startswith("gauss_params_fixed.") or
                k.startswith("gauss_ids")
            )
        }

        super().load_state_dict(non_gauss_state, strict=False)
        self.step = 0
        
        print(f"Load complete — obj={self.gauss_params['means'].shape[0]}, "
            f"canon={self.gauss_params_canonical['means'].shape[0]}, "
            f"bg={self.gauss_params_fixed['means'].shape[0] if self.gauss_params_fixed else 0}")


    def _update_optimizer_param_references(self):
        """
        CRITICAL: Update optimizer parameter references after loading checkpoint.
        This fixes the 'in_optimizer=False' issue by making optimizers point to the new parameters.
        """
        if not hasattr(self, 'optimizers') or not self.optimizers:
            print("⚠️  No optimizers found - skipping parameter reference update")
            return
            
        print("🔧 Updating optimizer parameter references...")
        
        # Map of parameter names to their new tensors
        param_mapping = {
            "means": self.gauss_params["means"],
            "scales": self.gauss_params["scales"], 
            "quats": self.gauss_params["quats"],
            "features_dc": self.gauss_params["features_dc"],
            "features_rest": self.gauss_params["features_rest"],
            "opacities": self.gauss_params["opacities"],
        }
        
        for param_name, new_param in param_mapping.items():
            if param_name in self.optimizers:
                optimizer = self.optimizers[param_name]
                
                # Update the parameter reference in the optimizer
                for group in optimizer.param_groups:
                    if len(group['params']) > 0:
                        # Replace the old parameter with the new one
                        old_param = group['params'][0]
                        group['params'][0] = new_param
                        
                        print(f"  {param_name}: {old_param.shape} → {new_param.shape}")
                        
                        # Clear optimizer state for the old parameter and initialize for new
                        if old_param in optimizer.state:
                            del optimizer.state[old_param]
                        optimizer.state[new_param] = {}
        
        print("Optimizer parameter references updated")


    # Add this method to be called after optimizers are set up
    def step_cb(self, optimizers: Optimizers, step):
        """Called by trainer when optimizers are ready"""
        self.step = step
        self.optimizers = optimizers.optimizers
        self.schedulers = optimizers.schedulers
        
        # CRITICAL: Update optimizer parameter references if we've loaded a checkpoint
        if hasattr(self, 'gauss_params') and self.gauss_params['means'].shape[0] > 0:
            self._update_optimizer_param_references()


    # Enhanced step_post_backward implementation
    def step_post_backward(self, step):
        """Strategy step after backward pass with continue_cull_post_densification"""
        print(f"🚨 step_post_backward CALLED at step {step}")
        assert step == self.step

        print(f"Strategy step_post_backward with {self.gauss_params['means'].shape[0]} object Gaussians")

        
        print(f"Found {len(self.info['gaussian_ids'])} visible Gaussians")

        n_gaussians_before = self.gauss_params['means'].shape[0]
        print(f"Before strategy: obj={n_gaussians_before}")

        # Determine if we should do densification
        should_densify = step < self.config.stop_split_at
        
        if should_densify:
            print("Applying densification strategy")
            # Apply regular strategy operations (densification + culling)
            if isinstance(self.strategy, DefaultStrategy):
                self.strategy.step_post_backward(
                    params=self.gauss_params,
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    step=self.step,
                    info=self.info,
                    packed=True,
                )
            elif isinstance(self.strategy, MCMCStrategy):
                self.strategy.step_post_backward(
                    params=self.gauss_params,  
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    step=step,
                    info=self.info,
                    lr=self.schedulers["means"].get_last_lr()[0],
                )
            else:
                raise ValueError(f"Unknown strategy {self.strategy}")
        
        elif self.config.continue_cull_post_densification:
            print("✂️  Post-densification culling phase")
            # Only do culling, no more densification
            if step % self.config.cull_post_densification_every == 0:
                self._cull_post_densification()
        
        n_gaussians_after = self.gauss_params['means'].shape[0]
        print(f"After strategy: obj={n_gaussians_after}")


    def _cull_post_densification(self):
        """
        Cull problematic Gaussians after densification has stopped
        Similar to Street Gaussians approach
        """
        if not hasattr(self, 'gauss_params') or self.gauss_params['means'].shape[0] == 0:
            return
        
        print("Analyzing Gaussians for post-densification culling...")
        
        # Get current parameters
        means = self.gauss_params["means"]
        scales = torch.exp(self.gauss_params["scales"])
        opacities = torch.sigmoid(self.gauss_params["opacities"])
        
        n_total = len(means)
        cull_masks = {}
        
        low_opacity_mask = opacities.squeeze() < self.config.cull_alpha_thresh
        cull_masks["low_opacity"] = low_opacity_mask
        
        large_scale_mask = scales.max(dim=-1)[0] > self.config.cull_scale_thresh
        cull_masks["large_scale"] = large_scale_mask
        
        aspect_ratios = scales.max(dim=-1)[0] / (scales.min(dim=-1)[0] + 1e-8)
        spiky_mask = aspect_ratios > self.config.max_gauss_ratio
        cull_masks["spiky"] = spiky_mask
        
        boundary_mask = torch.zeros_like(low_opacity_mask)
        if (self.config.cull_boundary_gaussians and 
            hasattr(self, 'obj_3d_seg') and 
            self.obj_3d_seg is not None):
            
            obj_confidence = self.obj_3d_seg.query(means, thresh=0.0)
            boundary_mask = obj_confidence < 0.3  # Outside object with low confidence
            cull_masks["boundary"] = boundary_mask
        
        huge_mask = scales.max(dim=-1)[0] > 0.5  # 50cm - emergency threshold
        cull_masks["huge"] = huge_mask
        
        # Combine all culling criteria
        final_cull_mask = torch.zeros_like(low_opacity_mask)
        for criterion, mask in cull_masks.items():
            final_cull_mask |= mask
        
        n_cull = final_cull_mask.sum().item()
        
        if n_cull > 0:
            print(f"Culling breakdown:")
            for criterion, mask in cull_masks.items():
                count = mask.sum().item()
                if count > 0:
                    print(f"   - {criterion}: {count}")
            
            print(f"🗑️  Culling {n_cull}/{n_total} Gaussians ({n_cull/n_total*100:.1f}%)")
            
            # Apply culling
            keep_mask = ~final_cull_mask
            for param_name, param in self.gauss_params.items():
                if param.numel() > 0:
                    new_param = param[keep_mask].contiguous()
                    self.gauss_params[param_name] = torch.nn.Parameter(new_param)
            
            # Update optimizer references
            self._update_optimizer_param_references()
            
            n_remaining = keep_mask.sum().item()
            print(f" Remaining Gaussians: {n_remaining}")
            
            # Update strategy state if it exists
            if hasattr(self, 'strategy_state') and self.strategy_state is not None:
                self._update_strategy_state_after_culling(keep_mask)
        else:
            print(" No Gaussians need culling")


    def _update_strategy_state_after_culling(self, keep_mask):
        """Update strategy state arrays after culling"""
        if not hasattr(self, 'strategy_state') or self.strategy_state is None:
            return
        
        state = self.strategy_state
        
        # Update state arrays that track per-Gaussian statistics
        for key in ["grad2d", "count", "radii"]:
            if key in state and state[key] is not None:
                if len(state[key]) == len(keep_mask):
                    state[key] = state[key][keep_mask]
                    print(f"🔧 Updated strategy state '{key}': {len(keep_mask)} → {len(state[key])}")


    # Enhanced get_loss_dict with background accumulation penalty
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

        return loss_dict

    

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
        """
        joint_angle = self._get_joint_angle_for_camera(camera)
        
        if self.training:
            print(f"Training mode: rendering {self.gauss_params['means'].shape[0]} trainable Gaussians")
            
            if joint_angle != 0.0:
                articulated_params = self._apply_articulation_to_optimizer_params(joint_angle)
                return articulated_params
            else:
                return {name: param for name, param in self.gauss_params.items()}

        # During evaluation: can use full scene (trainable + fixed)
        joint_angle = self._get_joint_angle_for_camera(camera)
        articulated_obj_params = self._apply_articulation_to_canonical_params(joint_angle)
        
        if hasattr(self, "gauss_params_fixed") and self.gauss_params_fixed["means"].shape[0] > 0:
            full_scene_params = {}
            for name in articulated_obj_params.keys():
                full_scene_params[name] = torch.cat(
                    [articulated_obj_params[name], self.gauss_params_fixed[name]], dim=0
                )
            print(f"Eval mode: rendering {full_scene_params['means'].shape[0]} total Gaussians")
            return full_scene_params

        return articulated_obj_params

    def _apply_articulation_to_optimizer_params(self, joint_angle: float) -> Dict[str, torch.Tensor]:
        """
        Apply articulation directly to optimizer parameters during training.
        This ensures the strategy operations work on the same tensors.
        """
        if joint_angle == 0.0:
            return {name: param for name, param in self.gauss_params.items()}
        
        means_articulated, quats_articulated = apply_joint_transform(
            means=self.gauss_params["means"],  
            quats=self.gauss_params["quats"],  
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
                articulated_params[name] = param
        
        return articulated_params

    def _apply_articulation_to_canonical_params(self, joint_angle: float) -> Dict[str, torch.Tensor]:
        """
        Apply per-frame articulation to the canonical object parameters.

        """
        if joint_angle == 0.0:
            return {name: param for name, param in self.gauss_params.items()}
        
        means_articulated, quats_articulated = apply_joint_transform(
            means=self.gauss_params["means"],  
            quats=self.gauss_params["quats"],  
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
                articulated_params[name] = param
        
        return articulated_params

    def get_outputs(self, camera: Cameras) -> Dict[str, Union[torch.Tensor, List]]:
        """Takes in a camera and returns a dictionary of outputs with articulation."""
        if not isinstance(camera, Cameras):
            return {}

        gaussians_to_render = self._get_gaussians_for_render(camera)
        
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

        render_mode = "RGB+ED" if self.config.output_depth_during_training or not self.training else "RGB"
        if self.config.sh_degree > 0:
            sh_degree_to_use = min(self.step // self.config.sh_degree_interval, self.config.sh_degree)
        else:
            colors_crop = torch.sigmoid(colors_crop).squeeze(1)
            sh_degree_to_use = None
        
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
            print(f"Gaussian IDs shape: {self.info['gaussian_ids'].shape}")
            print(f"ID range: {self.info['gaussian_ids'].min()} to {self.info['gaussian_ids'].max()}")
        else:
            print("gaussian_ids is None - NO VISIBLE GAUSSIANS!")
            if self.training and self.step % 100 == 0:
                print("🚨 This means all Gaussians are being culled!")
                print("   Check: camera position, Gaussian positions, opacities, scales")

        if self.training:
            print(f"Strategy gets {self.gauss_params['means'].shape[0]} Gaussians (object only)")
            
            # Ensure perfect match between render and strategy
            assert self.gauss_params['means'].shape[0] == n_rendered, \
                f"Mismatch: rendered {n_rendered}, strategy gets {self.gauss_params['means'].shape[0]}"
            
            self.strategy.step_pre_backward(
                self.gauss_params,
                self.optimizers,
                self.strategy_state,
                self.step,
                self.info
            )
            print(f"Strategy step pre-backward complete at step {self.step}")

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
