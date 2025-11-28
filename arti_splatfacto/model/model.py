from __future__ import annotations

from dataclasses import dataclass, field
from pickle import TRUE
import re
from typing import Dict, List, Type, Optional, Union, Tuple
from pathlib import Path

from flask import g

try:
    from gsplat.rendering import rasterization
except ImportError:
    print("Please install gsplat>=1.0.0")

from attrs import has
import numpy as np
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

from arti_splatfacto.utils.debug_utils import decode_id_map, save_debug_id_maps, save_depth_debug, save_normal_debug
from arti_splatfacto.utils.img_utils import psnr_masked, crop_imgs_w_masks, compute_2D_bbox, batch_crop_resize,batch_crop_resize
from arti_splatfacto.utils.articulation_utils import apply_joint_transform, apply_joint_transform_prismatic, apply_articulation_to_optimizer_params
from arti_splatfacto.utils.normal_utils import normal_from_depth_image
from arti_splatfacto.utils.loss_utils import opacity_loss
from arti_splatfacto.utils.depth_loss import DepthLoss, compute_scale_and_shift
from nerfstudio.utils.rich_utils import CONSOLE



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

    ### normal regularization parameters
    use_normal_reg: bool = False
    smooth_normals: bool = False  
    normal_debug_vis: bool = False  
    normal_lambda: float = 0.05 


    ### opacity regularization
    use_opacity_regularization: bool = False
    opacity_lambda_obj: float = 0.01
    opacity_lambda_fixed: float = 0.01

    joint_correction_lambda: float = 0.01
    active_joint_id: str = "joint_0"
    training_mode: str = field(default="articulation")
    """Training mode: 'articulation' or 'recovery'"""

    lambda_L1_fixed_opacity: float = 0.001




class ArtiSplatfactoModel(SplatfactoModel):    

    config: ArtiSplatfactoModelConfig

    def __init__(self, *args, **kwargs):
        self.metadata = kwargs.pop("metadata", {}) or {}
        super().__init__(*args, **kwargs)
        

        self.joint_type = self.obj_3d_seg.joint_type 


        if self.config.use_depth:
            self.depth_loss_fn = DepthLoss(alpha=0.5, scales=4)


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

        self.all_gauss_params_obj = torch.nn.ModuleDict()
        self.all_gauss_params_canon = torch.nn.ModuleDict()

        # Fixed gaussians (non-trainable but still Parameters for saving)
        self.gauss_params_fixed = torch.nn.ParameterDict({
            "means":         torch.nn.Parameter(torch.empty((0, 3), device=device), requires_grad=False),
            "scales":        torch.nn.Parameter(torch.empty((0, 3), device=device), requires_grad=False),
            "quats":         torch.nn.Parameter(torch.empty((0, 4), device=device), requires_grad=False),
            "features_dc":   torch.nn.Parameter(torch.empty((0, 3), device=device), requires_grad=False),
            "features_rest": torch.nn.Parameter(torch.empty((0, dim_sh - 1, 3), device=device), requires_grad=False),
            "opacities":     torch.nn.Parameter(torch.empty((0, 1), device=device), requires_grad=False),
        })

        self.all_joint_params = torch.nn.ModuleDict()

        self.obj_3d_seg = Object3DSeg.load(self.config.obj_mask_file, device=device)
        active_id = self.config.active_joint_id

        self.all_joint_params[active_id] = torch.nn.ParameterDict()

        initial_pivot = self.obj_3d_seg.joint_pivot.to(device)
        initial_axis = F.normalize(self.obj_3d_seg.joint_axis.to(device), dim=0)

        self.register_buffer(f'initial_joint_pivot_{active_id}', initial_pivot.clone())
        
        self.all_joint_params[active_id]["pivot"] = torch.nn.Parameter(initial_pivot.clone(), requires_grad=True)
        self.all_joint_params[active_id]["axis_raw"] = torch.nn.Parameter(initial_axis.clone(), requires_grad=True)
        
        joint_angles_meta = self.metadata.get("joint_angles", [])
        num_frames = len(joint_angles_meta)

        if num_frames > 0:
            initial_angles = torch.as_tensor(joint_angles_meta, device=device, dtype=torch.float32)
            
            self.all_joint_params[active_id]["angles"] = torch.nn.Parameter(
                initial_angles.clone(),
                requires_grad=True
            )
            
            # Store initial for reference/regularization
            self.register_buffer(f'initial_angles_{active_id}', initial_angles.clone())

        # --- Backward Compatibility Pointers ---
        self.joint_pivot = self.all_joint_params[active_id]["pivot"]
        self.joint_axis_raw = self.all_joint_params[active_id]["axis_raw"]

        if num_frames > 0:
            self.joint_angles_learned = self.all_joint_params[active_id]["angles"]
            self.initial_angles = getattr(self, f'initial_angles_{active_id}')

        self.rgb_metrics = RGBMetrics()
        self.depth_metrics = DepthMetrics()
        self.mse_loss = torch.nn.MSELoss()


    @property
    def joint_axis(self):
        """Normalized joint axis"""
        if hasattr(self, 'joint_axis_raw'):
            return F.normalize(self.joint_axis_raw, dim=0)
        return None

    @property
    def joint_angles(self):
        """Return the learned angles directly."""
        if hasattr(self, 'joint_angles_learned'):
            return self.joint_angles_learned
        return torch.tensor([0.0], device=self.device)

    @property
    def joint_limits(self):
        """Limits are just the min/max of learned angles."""
        if hasattr(self, 'joint_angles_learned'):
            angles = self.joint_angles_learned
            return torch.stack([angles.min(), angles.max()])
        # Fallback to metadata
        return torch.tensor(self.obj_3d_seg.joint_limits, device=self.device)

    @property  
    def joint_angles_normalized(self):
        """Normalized angles for stable optimization [0, 1]"""
        angles = self.joint_angles
        joint_min, joint_max = self.joint_limits
        # Avoid division by zero
        range_val = joint_max - joint_min
        if range_val < 1e-6:
            return torch.zeros_like(angles)
        return (angles - joint_min) / range_val

    @property
    def angle_range(self):
        """Observed range from learned angles."""
        limits = self.joint_limits
        return limits[1] - limits[0]



    def state_dict(self, *args, **kwargs):
        state = super().state_dict(*args, **kwargs)
        
        # Save all joint-specific Gaussian sets
        for joint_id, params in self.all_gauss_params_obj.items():
            for name, param in params.items():
                state[f"all_gauss_params_obj.{joint_id}.{name}"] = param.data

        for joint_id, params in self.all_gauss_params_canon.items():
            for name, param in params.items():
                state[f"all_gauss_params_canon.{joint_id}.{name}"] = param.data

        # Save all joint-specific Articulation Parameters
        for joint_id, params in self.all_joint_params.items():
            for name, param in params.items():
                state[f"all_joint_params.{joint_id}.{name}"] = param.data
        
        # Save Background Gaussians (Remains the same)
        if hasattr(self, 'gauss_params_fixed') and self.gauss_params_fixed:
            for name, param in self.gauss_params_fixed.items():
                state[f"gauss_params_fixed.{name}"] = param.data

        joint_meta = {}
        for joint_id in self.all_joint_params.keys():
            joint_meta[joint_id] = {}
            
            # Save derived limits
            if joint_id == self.config.active_joint_id:
                limits = self.joint_limits
                joint_meta[joint_id]["limits"] = limits.detach().cpu()
            
            # Save type
            type_attr = f"joint_type_{joint_id}"
            if hasattr(self, type_attr):
                joint_meta[joint_id]["type"] = getattr(self, type_attr)
        
        state["joint_metadata_absolute"] = joint_meta
        
        return state


    def configure_training_stage(self):
        """
        Configure model parameters based on training_mode and active_joint_id.
        """
        if self.config.training_mode == "recovery":
            self.setup_recovery_stage()
        elif self.config.training_mode == "articulation":
            self.setup_articulation_stage()
        else:
            raise ValueError(f"Unknown training_mode: {self.config.training_mode}")


    def setup_articulation_stage(self):
        """
        Configure model for articulation stage (Joint N):
        - Unfreeze geometry/radiance/joint params for the ACTIVE joint.
        - Freeze geometry/radiance/joint params for ALL previous joints.
        - Freeze ALL background params.
        """
        active_id = self.config.active_joint_id
        GAUSS = ["means", "scales", "quats", "features_dc", "features_rest", "opacities"]
        
        CONSOLE.print("\n" + "="*70)
        CONSOLE.print(f"[bold cyan]CONFIGURING MODEL FOR ARTICULATION STAGE: {active_id}[/bold cyan]")
        CONSOLE.print("="*70)
        
        # --- 1. Configure Background Gaussians (Always Frozen in Articulation) ---
        for param_name in GAUSS:
            if param_name in self.gauss_params_fixed:
                self.gauss_params_fixed[param_name].requires_grad = False
        CONSOLE.print(f"  [red]✓ Frozen ALL Background Gaussians ({self.gauss_params_fixed['means'].shape[0]} pts)[/red]")

        # --- 2. Configure ALL Joint-Specific Gaussians ---
        for joint_id in self.all_gauss_params_obj.keys():
            is_active = (joint_id == active_id)
            
            # Articulated Object Gaussians
            for param_name in GAUSS:
                # Active joint: Trainable for geometry and radiance
                # Previous joints: Frozen
                requires_grad = is_active
                self.all_gauss_params_obj[joint_id][param_name].requires_grad = requires_grad
                self.all_gauss_params_canon[joint_id][param_name].requires_grad = requires_grad
            
            status = "Trainable" if is_active else "Frozen"
            color = "green" if is_active else "yellow"
            CONSOLE.print(f"  [{color}]✓ {status} Object/Canonical Gaussians for {joint_id}[/{color}]")

        # --- 3. Configure ALL Joint-Specific Parameters ---
        JOINT_PARAMS = ["pivot", "axis_raw", "max_angle", "angle_deltas"]
        for joint_id in self.all_joint_params.keys():
            is_active = (joint_id == active_id)

            for param_name in JOINT_PARAMS:
                if param_name in self.all_joint_params[joint_id]:
                    # Active joint: Trainable for geometry (pivot/axis) and corrections
                    # Previous joints: Frozen
                    requires_grad = is_active
                    self.all_joint_params[joint_id][param_name].requires_grad = requires_grad
                    
            status = "Trainable" if is_active else "Frozen"
            color = "green" if is_active else "yellow"
            CONSOLE.print(f"  [{color}]✓ {status} Articulation Parameters for {joint_id}[/{color}]")

        CONSOLE.print("\n[bold green]ARTICULATION STAGE CONFIGURATION COMPLETE[/bold green]")
        CONSOLE.print(f"Active Joint: {active_id} | Others: FROZEN | Background: FROZEN")
        CONSOLE.print("="*70 + "\n")

    def setup_recovery_stage(self):
        """
        Configure model for recovery stage (Joint N):
        - Freeze ALL geometry (means, scales, quats) for all joints and background.
        - Enable ONLY ACTIVE JOINT radiance (features_dc, features_rest, opacities) and background.
        - Freeze ALL joint parameters (pivot, axis, deltas, etc.) for all joints.
        """
        GEOMETRY_PARAMS = ["means", "scales", "quats"]
        RADIANCE_PARAMS = ["features_dc", "features_rest", "opacities"]
        JOINT_PARAMS = ["pivot", "axis_raw", "max_angle", "angle_deltas"]
        active_id = getattr(self.config, "active_joint_id", None)
        
        CONSOLE.print("\n" + "="*70)
        CONSOLE.print("[bold yellow]CONFIGURING MODEL FOR RECOVERY STAGE (RADIANCE OPTIMIZATION)[/bold yellow]")
        CONSOLE.print("="*70)

        # --- 1. Configure ALL Joint-Specific Gaussians ---
        for joint_id in self.all_gauss_params_obj.keys():
            is_active = (joint_id == active_id)
            color = "green" if is_active else "yellow"
            CONSOLE.print(f"\n[{color}]Configuring Gaussians for {joint_id}[/{color}]")

            # Freeze geometry for all joints
            for param_name in GEOMETRY_PARAMS:
                self.all_gauss_params_obj[joint_id][param_name].requires_grad = False
                self.all_gauss_params_canon[joint_id][param_name].requires_grad = False

            if is_active:
                self.all_gauss_params_obj[joint_id]["means"].requires_grad = True
                self.all_gauss_params_obj[joint_id]["quats"].requires_grad = True
                self.all_gauss_params_canon[joint_id]["means"].requires_grad = True
                self.all_gauss_params_canon[joint_id]["quats"].requires_grad = True
                CONSOLE.print(f"  [green]✓ Enabled MEAN updates for {joint_id}[/green]")
            else:
                self.all_gauss_params_obj[joint_id]["means"].requires_grad = False
                self.all_gauss_params_canon[joint_id]["means"].requires_grad = False
            


            CONSOLE.print(f"  ✓ Froze Geometry ({GEOMETRY_PARAMS}) for Object/Canonical")

            # Enable radiance only for active joint
            for param_name in RADIANCE_PARAMS:
                requires_grad = is_active
                self.all_gauss_params_obj[joint_id][param_name].requires_grad = True
                self.all_gauss_params_canon[joint_id][param_name].requires_grad = True
            if is_active:
                CONSOLE.print(f"  [green]✓ Enabled Radiance ({RADIANCE_PARAMS}) for ACTIVE joint {joint_id}[/green]")
            else:
                CONSOLE.print(f"  [yellow]✗ Frozen Radiance ({RADIANCE_PARAMS}) for previous joint {joint_id}[/yellow]")

        # --- 2. Configure Background Gaussians ---
        CONSOLE.print("\n[yellow]Configuring Background Gaussians[/yellow]")
        for param_name in GEOMETRY_PARAMS:
            if param_name in self.gauss_params_fixed:
                self.gauss_params_fixed[param_name].requires_grad = False
        CONSOLE.print(f"  ✓ Froze Geometry ({GEOMETRY_PARAMS}) for Background")
        
        for param_name in RADIANCE_PARAMS:
            if param_name in self.gauss_params_fixed:
                self.gauss_params_fixed[param_name].requires_grad = True
        CONSOLE.print(f"  [bold green]✓ Enabled Radiance ({RADIANCE_PARAMS}) for Background[/bold green]")

        # --- 3. Configure Joint Parameters (Always Frozen) ---
        CONSOLE.print("\n[cyan]Freezing ALL articulation parameters...[/cyan]")
        for joint_id in self.all_joint_params.keys():
            for param_name in JOINT_PARAMS:
                if param_name in self.all_joint_params[joint_id]:
                    self.all_joint_params[joint_id][param_name].requires_grad = False
                    CONSOLE.print(f"  ✓ Froze {joint_id} {param_name}")

        # --- 4. Final Summary ---
        CONSOLE.print("\n" + "="*70)
        CONSOLE.print(f"[bold green]RECOVERY STAGE CONFIGURATION COMPLETE[/bold green]")
        CONSOLE.print(f"Active Joint: [bold]{active_id}[/bold] Radiance = Trainable")
        CONSOLE.print("All Geometry: FROZEN | Other Joints: FROZEN | Background: Radiance Trainable")
        CONSOLE.print("="*70 + "\n")

    def get_gaussian_param_groups(self) -> Dict[str, List[Parameter]]:
        """Return optimizer param groups based on training mode."""
        
        if self.config.training_mode == "recovery":
            return self.get_recovery_param_groups()
        else:
            return self.get_articulation_param_groups()
        
    def get_articulation_param_groups(self) -> Dict[str, List[Parameter]]:
        """Return parameter groups for the ACTIVE joint's articulation training."""
        groups = {}
        active_id = self.config.active_joint_id
        GAUSS = ["means", "scales", "quats", "features_dc", "features_rest", "opacities"]
        
        CONSOLE.print(f"\n[cyan]Building articulation parameter groups for ACTIVE joint: {active_id}...[/cyan]")

        # --- 1. Active Joint Gaussian Parameters (Object & Canonical) ---
        if active_id in self.all_gauss_params_obj:
            obj_params = self.all_gauss_params_obj[active_id]
            canon_params = self.all_gauss_params_canon[active_id]
            
            for param_name in GAUSS:
                # Object Gaussians
                opt_name = f"obj_{param_name}"
                if param_name in obj_params and obj_params[param_name].requires_grad:
                    groups[opt_name] = [obj_params[param_name]]
                    CONSOLE.print(f"  ✓ {opt_name}: {obj_params[param_name].shape}")

                # Canonical Gaussians
                opt_name = f"canon_{param_name}"
                if param_name in canon_params and canon_params[param_name].requires_grad:
                    groups[opt_name] = [canon_params[param_name]]
                    CONSOLE.print(f"  ✓ {opt_name}: {canon_params[param_name].shape}")

        # --- 2. Active Joint Articulation Parameters ---
        if active_id in self.all_joint_params:
            joint_params = self.all_joint_params[active_id]
            
            # Pivot and Axis
            joint_geom_params = {
                "joint_pivot": "pivot",
                "joint_axis": "axis_raw",
            }
            
            for opt_name, internal_name in joint_geom_params.items():
                if internal_name in joint_params and joint_params[internal_name].requires_grad:
                    groups[opt_name] = [joint_params[internal_name]]
                    CONSOLE.print(f"  ✓ {opt_name}: {joint_params[internal_name].shape}")

            # SIMPLIFIED: Just per-frame angles
            if "angles" in joint_params and joint_params["angles"].requires_grad:
                groups["joint_angles"] = [joint_params["angles"]]
                CONSOLE.print(f"  ✓ joint_angles: {joint_params['angles'].shape}")

        CONSOLE.print(f"\n[green]Total: {len(groups)} articulation parameter groups[/green]\n")
        return groups


    def get_recovery_param_groups(self) -> Dict[str, List[Parameter]]:
            """
            Parameter groups for recovery training (ALL radiance).
            Groups are named using generic config keys (e.g., 'bg_features_dc') 
            to match the optimizer configuration.
            """
            groups = {}
            RADIANCE_PARAMS = ["features_dc", "features_rest", "opacities"]
            
            # Define the static optimizer keys we will use for grouping the radiance parameters
            PARAM_GROUP_MAPPING = {
                "features_dc": "bg_features_dc",
                "features_rest": "bg_features_rest",
                "opacities": "bg_opacities",
            }
            
            CONSOLE.print("\n[cyan]Building recovery parameter groups (Radiance for ALL)...[/cyan]")

            # --- 1. All Joint Gaussian Radiance Parameters ---
            for joint_id in self.all_gauss_params_obj.keys():
                obj_params = self.all_gauss_params_obj[joint_id]
                canon_params = self.all_gauss_params_canon[joint_id]
                
                for param_name in RADIANCE_PARAMS:
                    opt_key = PARAM_GROUP_MAPPING[param_name] # e.g., 'bg_features_dc'
                    
                    # Object Radiance
                    if param_name in obj_params and obj_params[param_name].requires_grad:
                        if opt_key not in groups: groups[opt_key] = []
                        groups[opt_key].append(obj_params[param_name])
                        CONSOLE.print(f"  ✓ Added {joint_id} Object {param_name} to {opt_key} group")

                    # Canonical Radiance
                    if param_name in canon_params and canon_params[param_name].requires_grad:
                        if opt_key not in groups: groups[opt_key] = []
                        groups[opt_key].append(canon_params[param_name])
                        CONSOLE.print(f"  ✓ Added {joint_id} Canonical {param_name} to {opt_key} group")
            
            # --- 2. Background Radiance Parameters ---
            for param_name in RADIANCE_PARAMS:
                opt_key = PARAM_GROUP_MAPPING[param_name]
                if param_name in self.gauss_params_fixed and self.gauss_params_fixed[param_name].requires_grad:
                    if opt_key not in groups: groups[opt_key] = []
                    groups[opt_key].append(self.gauss_params_fixed[param_name])
                    CONSOLE.print(f"  [bold green]✓ Added Background {param_name} to {opt_key} group[/bold green]")
            
            # --- 3. Per-frame articulation (if enabled/trainable) ---
            # If 'joint_t_values' or similar per-frame parameters are still trainable, they should 
            # map to an existing config key, likely 'joint_t_values' or 'camera_opt'.
            # Assuming joint_t_values is the correct key if it was used in the config.
            # This section remains largely dependent on your specific config keys.

            if hasattr(self, 'joint_t_raw') and self.joint_t_raw.requires_grad:
                # Assumes 'joint_t_values' is a key in the config
                groups["joint_t_values"] = [self.joint_t_raw] 
                CONSOLE.print(f"  ✓ joint_t_values (Legacy): {self.joint_t_raw.shape}")
            
            CONSOLE.print(f"\n[green]Total: {len(groups)} recovery parameter groups[/green]\n")
            return groups
    
    def _initialize_and_partition(self, state_dict: Dict[str, torch.Tensor]):
        """
        Partition Gaussians for the active joint.
        
        - For joint_0: Splits full scene into object + background
        - For joint_1+: Splits ONLY from background (preserves previous joints)
        """
        print("Initializing from full scene: partitioning Gaussians...")
        active_id = self.config.active_joint_id
        print(f"obj mask points for {active_id}:", self.obj_3d_seg)
        
        GAUSS = ["means", "scales", "quats", "features_dc", "features_rest", "opacities"]
        device = self.device
        
        # Check if this is the first joint or a subsequent one
        is_first_joint = (active_id == "joint_0" or len(self.all_gauss_params_obj) == 0)
        
        if is_first_joint:
            # --- Case 1: First Joint - Partition from Full Scene ---
            print(f"  First joint: partitioning from full scene")
            
            all_means = state_dict["gauss_params.means"].to(device)
            obj_mask = self.obj_3d_seg.query_refine_ellipsoid(
                all_means, grow=2, thresh=0.01, bbox_margin=0.00
            ).to(torch.bool).cpu()
            bg_mask = ~obj_mask
            
            print(f"  Mask: {obj_mask.sum()} object points, {bg_mask.sum()} background points")
            
            # Create new ParameterDicts for the active joint
            self.all_gauss_params_obj[active_id] = torch.nn.ParameterDict()
            self.all_gauss_params_canon[active_id] = torch.nn.ParameterDict()
            
            for p in GAUSS:
                subset_obj = state_dict[f"gauss_params.{p}"][obj_mask].to(device)
                subset_bg = state_dict[f"gauss_params.{p}"][bg_mask].to(device)
                
                # Object Gaussians (trainable)
                self.all_gauss_params_obj[active_id][p] = torch.nn.Parameter(
                    subset_obj.clone().detach(), requires_grad=True
                )
                
                # Canonical Gaussians (trainable)
                self.all_gauss_params_canon[active_id][p] = torch.nn.Parameter(
                    subset_obj.clone().detach(), requires_grad=True
                )
                
                # Fixed background
                self.gauss_params_fixed[p] = torch.nn.Parameter(
                    subset_bg.clone().detach(), requires_grad=False
                )
                
                if p == "means":
                    print(f"  [{active_id} Object] {p}: {subset_obj.shape}")
                    print(f"  [{active_id} Canon] {p}: {subset_obj.shape}")
                    print(f"  [Background] {p}: {subset_bg.shape}")
        
        else:
            # --- Case 2: Subsequent Joint - Partition from Background Only ---
            print(f"  Sequential joint: partitioning from background only")
            print(f"  Existing joints: {list(self.all_gauss_params_obj.keys())}")
            
            all_means = state_dict["gauss_params.means"].to(device)
            total_input = all_means.shape[0]
            
            # Calculate how many Gaussians belong to previous joints
            prev_joints_count = sum(
                self.all_gauss_params_obj[jid]["means"].shape[0] 
                for jid in self.all_gauss_params_obj.keys() 
                if jid != active_id
            )
            
            # The background Gaussians are at the beginning of the input
            bg_start_idx = 0
            bg_end_idx = total_input - prev_joints_count
            
            if bg_end_idx <= 0:
                raise ValueError(
                    f"Invalid partition: total_input={total_input}, "
                    f"prev_joints_count={prev_joints_count}. "
                    f"Background should be positive!"
                )
            
            print(f"  Total input: {total_input:,} Gaussians")
            print(f"  Previous joints: {prev_joints_count:,} Gaussians")
            print(f"  Background range: [0:{bg_end_idx:,}]")
            
            # Extract only background means for mask query
            bg_means_only = all_means[bg_start_idx:bg_end_idx]
            
            # Query mask on background only
            obj_mask_in_bg = self.obj_3d_seg.query_refine_ellipsoid(
                bg_means_only, grow=3, thresh=0.01, bbox_margin=0.0
            ).to(torch.bool).cpu()
            
            remaining_bg_mask = ~obj_mask_in_bg
            
            print(f"  Mask query on background: {obj_mask_in_bg.sum()} new object points, "
                f"{remaining_bg_mask.sum()} remaining background points")
            
            # Create new ParameterDicts for the active joint
            self.all_gauss_params_obj[active_id] = torch.nn.ParameterDict()
            self.all_gauss_params_canon[active_id] = torch.nn.ParameterDict()
            
            for p in GAUSS:
                # Extract background portion only
                bg_data = state_dict[f"gauss_params.{p}"][bg_start_idx:bg_end_idx].to(device)
                
                # Split background into new joint + remaining background
                new_joint_data = bg_data[obj_mask_in_bg]
                remaining_bg_data = bg_data[remaining_bg_mask]
                
                # Object Gaussians (trainable)
                self.all_gauss_params_obj[active_id][p] = torch.nn.Parameter(
                    new_joint_data.clone().detach(), requires_grad=True
                )
                
                # Canonical Gaussians (trainable)
                self.all_gauss_params_canon[active_id][p] = torch.nn.Parameter(
                    new_joint_data.clone().detach(), requires_grad=True
                )
                
                # Update background
                self.gauss_params_fixed[p] = torch.nn.Parameter(
                    remaining_bg_data.clone().detach(), requires_grad=False
                )
                
                if p == "means":
                    print(f"  [{active_id} Object] {p}: {new_joint_data.shape}")
                    print(f"  [{active_id} Canon] {p}: {new_joint_data.shape}")
                    print(f"  [Updated Background] {p}: {remaining_bg_data.shape}")
        
        # Set pointers for the active joint
        self.gauss_params = self.all_gauss_params_obj[active_id]
        self.gauss_params_canonical = self.all_gauss_params_canon[active_id]
        
        # Summary
        n_obj = self.all_gauss_params_obj[active_id]["means"].shape[0]
        n_canon = self.all_gauss_params_canon[active_id]["means"].shape[0]
        n_bg = self.gauss_params_fixed["means"].shape[0]
        
        print(f"\n{'='*70}")
        print(f"Partitioning complete for {active_id}")
        print(f"  Object: {n_obj:,} Gaussians")
        print(f"  Canonical: {n_canon:,} Gaussians")
        print(f"  Background: {n_bg:,} Gaussians")
        print(f"  All joints: {list(self.all_gauss_params_obj.keys())}")
        print(f"{'='*70}\n")

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor], **kwargs):

        if hasattr(self, '_skip_load_state_dict') and self._skip_load_state_dict:
            return  # Silent skip - no log needed
        
        active_id = self.config.active_joint_id
        GAUSS = ["means", "scales", "quats", "features_dc", "features_rest", "opacities"]
        device = self.device
        
        # --- Helper Function: Identify Legacy Parameters ---
        def is_legacy_single_joint_param(key: str) -> bool:
            """Dynamically detect old single-joint parameter format."""
            legacy_base_names = [
                "joint_pivot", "joint_axis_raw", "max_joint_angle", "min_joint_angle", 
                "joint_angle_deltas", "angle_range_log"
            ]
            
            if key in legacy_base_names:
                return True
            
            if re.match(r"joint_angles_prior_joint_\d+$", key):
                return True
            if re.match(r"joint_limits_joint_\d+$", key):
                return True
            if re.match(r"initial_angle_range_joint_\d+$", key):
                return True
            
            return False
        
        # --- 1. Detect Partitioning Status ---
        is_multi_joint_partitioned = any(k.startswith("all_gauss_params_obj.") for k in state_dict)
        is_single_joint_partitioned = "gauss_params_fixed.means" in state_dict
        
        # Check if we need to partition the active joint
        needs_active_joint_partition = (
            self.config.training_mode == "articulation" and 
            "gauss_params.means" in state_dict
        )

        # Backwards compatibility for single-joint format
        if "means" in state_dict and not is_multi_joint_partitioned:
            for p in GAUSS:
                state_dict[f"gauss_params.{p}"] = state_dict[p]

        # --- 2. Load Previously Trained Joints and Background ---
        if is_multi_joint_partitioned or is_single_joint_partitioned:
            print(f"Loading previously trained joints from partitioned checkpoint...")
            
            # --- Load ALL Previously Trained Gaussian Sets ---
            for key, tensor in state_dict.items():
                # Skip non-tensor entries (e.g., joint_metadata_absolute dict)
                if not torch.is_tensor(tensor):
                    continue
                tensor = tensor.to(device)

                
                # Load joint-specific Gaussians
                match = re.match(r"all_gauss_params_(obj|canon)\.(joint_\d+)\.(" + "|".join(GAUSS) + r")$", key)
                if match:
                    set_type, joint_id, param_name = match.groups()
                    
                    # Skip the active joint - it will be re-partitioned
                    if joint_id == active_id and needs_active_joint_partition:
                        continue
                    
                    target_dict = self.all_gauss_params_obj if set_type == 'obj' else self.all_gauss_params_canon
                    
                    if joint_id not in target_dict:
                        target_dict[joint_id] = torch.nn.ParameterDict()
                    
                    # Previous joints are always frozen
                    target_dict[joint_id][param_name] = torch.nn.Parameter(
                        tensor.clone().detach(), 
                        requires_grad=False
                    )

                # Load Background Gaussians
                elif key.startswith("gauss_params_fixed."):
                    param_name = key.split('.')[-1]
                    requires_grad = (self.training and self.config.training_mode == "recovery")
                    self.gauss_params_fixed[param_name] = torch.nn.Parameter(
                        tensor.clone().detach(), 
                        requires_grad=requires_grad
                    )

        # --- 3. Partition Active Joint ---
        if needs_active_joint_partition:
            print(f"{'='*70}")
            if active_id == "joint_0":
                print(f"INITIAL PARTITIONING for {active_id} (first joint)")
            else:
                print(f"SEQUENTIAL PARTITIONING for {active_id}")
            print(f"{'='*70}")
            
            # CRITICAL FIX: Reconstruct the full scene for partitioning
            if active_id != "joint_0" and is_multi_joint_partitioned:
                # For joint_1+, reconstruct: background + all_previous_joints
                print(f"Reconstructing full scene for partitioning...")
                
                combined_data = {p: [] for p in GAUSS}
                
                # Add background
                if hasattr(self, 'gauss_params_fixed') and 'means' in self.gauss_params_fixed:
                    for p in GAUSS:
                        combined_data[p].append(self.gauss_params_fixed[p].data)
                    bg_count = self.gauss_params_fixed['means'].shape[0]
                    print(f"  Added {bg_count:,} background Gaussians")
                
                # Add all previous joints
                prev_joint_count = 0
                for joint_id in sorted(self.all_gauss_params_obj.keys()):
                    if joint_id != active_id:  # Skip active joint (doesn't exist yet)
                        for p in GAUSS:
                            combined_data[p].append(self.all_gauss_params_obj[joint_id][p].data)
                        count = self.all_gauss_params_obj[joint_id]['means'].shape[0]
                        prev_joint_count += count
                        print(f"  Added {count:,} Gaussians from {joint_id}")
                
                # Concatenate everything
                partition_state = {}
                for p in GAUSS:
                    partition_state[f"gauss_params.{p}"] = torch.cat(combined_data[p], dim=0)
                
                total_points = partition_state["gauss_params.means"].shape[0]
                print(f"  Total reconstructed scene: {total_points:,} Gaussians")
                print(f"  (Background: {bg_count:,} + Previous joints: {prev_joint_count:,})")
                
            else:
                # For joint_0, use the original full scene directly
                partition_state = {}
                for p in GAUSS:
                    key = f"gauss_params.{p}"
                    if key in state_dict:
                        partition_state[key] = state_dict[key]
                
                if "gauss_params.means" in partition_state:
                    print(f"  Using full scene: {partition_state['gauss_params.means'].shape[0]:,} Gaussians")
            
            # Now partition the active joint from the reconstructed full scene
            print(f"Querying 3D segmentation for {active_id}...")
            self._initialize_and_partition(partition_state)
            
            print(f"✓ Partitioned {len(self.all_gauss_params_obj[active_id]['means']):,} Gaussians for {active_id}")
            print(f"{'='*70}\n")



        FRAME_DEPENDENT_PARAMS = {"angle_deltas", "angles"}

        for key, tensor in state_dict.items():
            if key.startswith("all_joint_params."):
                parts = key.split('.')
                if len(parts) < 3:
                    continue
                    
                joint_id = parts[1]
                param_name = parts[2]
                
                # Ensure joint entry exists
                if joint_id not in self.all_joint_params:
                    self.all_joint_params[joint_id] = torch.nn.ParameterDict()
                
                # === CRITICAL FIX: Skip frame-dependent params with mismatched shapes ===
                if param_name in FRAME_DEPENDENT_PARAMS:
                    # Check if parameter exists in current model
                    if param_name in self.all_joint_params[joint_id]:
                        current_shape = self.all_joint_params[joint_id][param_name].shape
                        checkpoint_shape = tensor.shape
                        
                        if current_shape != checkpoint_shape:
                            print(f"⚠️  Skipping {key}: shape mismatch "
                                f"(checkpoint: {checkpoint_shape}, current: {current_shape}). "
                                f"Using freshly initialized values.")
                            continue
                    else:
                        joint_angles_meta = self.metadata.get("joint_angles", [])
                        expected_frames = len(joint_angles_meta)
                        checkpoint_frames = tensor.shape[0]
                        
                        if expected_frames != checkpoint_frames:
                            print(f"⚠️  Skipping {key}: frame count mismatch "
                                f"(checkpoint: {checkpoint_frames} frames, dataset: {expected_frames} frames). "
                                f"Using freshly initialized values.")
                            continue
                
                # Determine trainability: only active joint in articulation mode
                requires_grad = (
                    joint_id == active_id and 
                    self.training and 
                    self.config.training_mode == "articulation"
                )
                
                if param_name not in self.all_joint_params[joint_id]:
                    self.all_joint_params[joint_id][param_name] = torch.nn.Parameter(
                        tensor.to(device), 
                        requires_grad=requires_grad
                    )
                else:
                    self.all_joint_params[joint_id][param_name].data.copy_(tensor.to(device))
                    self.all_joint_params[joint_id][param_name].requires_grad = requires_grad
        
        # --- 5. Set Active Joint Pointers EARLY (Before configure_training_stage) ---
        
        # Set Gaussian pointers
        if active_id in self.all_gauss_params_obj:
            self.gauss_params = self.all_gauss_params_obj[active_id]
            self.gauss_params_canonical = self.all_gauss_params_canon[active_id]
        else:
            print(f"⚠️  Warning: Active joint {active_id} not found. Using empty placeholders.")
            self.gauss_params = torch.nn.ParameterDict({
                p: torch.nn.Parameter(
                    torch.empty((0, 3 if p not in ["quats", "opacities"] else (4 if p == "quats" else 1)), device=device), 
                    requires_grad=False
                ) for p in GAUSS
            })
            self.gauss_params_canonical = self.gauss_params

        # Set joint parameter pointers
        if active_id in self.all_joint_params:
            self.joint_pivot = self.all_joint_params[active_id]["pivot"]
            self.joint_axis_raw = self.all_joint_params[active_id]["axis_raw"]
            
            # Handle different angle parameter formats
            if "angles" in self.all_joint_params[active_id]:
                # NEW FORMAT: Direct per-frame angles
                self.joint_angles_learned = self.all_joint_params[active_id]["angles"]
                print(f"✓ Loaded {len(self.joint_angles_learned)} learned angles for {active_id}")
                
                # Set initial_angles if buffer exists
                initial_angles_attr = f'initial_angles_{active_id}'
                if hasattr(self, initial_angles_attr):
                    self.initial_angles = getattr(self, initial_angles_attr)
            
            elif "angle_deltas" in self.all_joint_params[active_id]:
                # OLD FORMAT: Prior + deltas (backward compatibility)
                print(f"⚠️  Converting old delta format to direct angles for {active_id}")
                prior_attr = f'joint_angles_prior_{active_id}'
                
                if hasattr(self, prior_attr):
                    prior = getattr(self, prior_attr)
                    deltas = self.all_joint_params[active_id]["angle_deltas"]
                    
                    # Combine prior + deltas into direct angles
                    combined_angles = prior + deltas.data
                    
                    self.all_joint_params[active_id]["angles"] = torch.nn.Parameter(
                        combined_angles,
                        requires_grad=True
                    )
                    self.joint_angles_learned = self.all_joint_params[active_id]["angles"]
                    print(f"  Converted {len(combined_angles)} angles")
                    
                    # Store initial for reference
                    self.register_buffer(f'initial_angles_{active_id}', prior.clone())
                    self.initial_angles = getattr(self, f'initial_angles_{active_id}')
                else:
                    print(f"  ⚠️  Could not convert: no prior found for {active_id}")
            
            else:
                # NO ANGLES: Fresh joint or angles were skipped due to frame mismatch
                print(f"⚠️  No angles found for {active_id}")
                
                # Check if we have freshly initialized angles from populate_modules
                if "angles" in self.all_joint_params[active_id]:
                    self.joint_angles_learned = self.all_joint_params[active_id]["angles"]
                    print(f"  Using freshly initialized {len(self.joint_angles_learned)} angles from metadata")
                    
                    # Set initial_angles if buffer exists
                    initial_angles_attr = f'initial_angles_{active_id}'
                    if hasattr(self, initial_angles_attr):
                        self.initial_angles = getattr(self, initial_angles_attr)
                else:
                    print(f"  ⚠️  Warning: No angle parameters exist for {active_id}")


        print(f"\n[Buffer Check] Filtering frame-dependent buffers...")

        # Find all frame-dependent buffer keys in checkpoint
        frame_dependent_buffer_patterns = [
            r"initial_angles_joint_\d+$",
            r"joint_angles_prior_joint_\d+$",
            r"initial_angle_range_joint_\d+$",
        ]

        buffers_to_check = []
        for key in list(state_dict.keys()):
            for pattern in frame_dependent_buffer_patterns:
                if re.match(pattern, key):
                    buffers_to_check.append(key)
                    break

        # Check each buffer and remove if shape mismatch
        for buffer_key in buffers_to_check:
            checkpoint_buffer = state_dict[buffer_key]
            
            if not torch.is_tensor(checkpoint_buffer):
                continue
            
            # Check if buffer exists in current model
            if hasattr(self, buffer_key):
                current_buffer = getattr(self, buffer_key)
                
                if current_buffer.shape != checkpoint_buffer.shape:
                    print(f"  ⚠️  Skipping {buffer_key}: shape mismatch "
                        f"(checkpoint: {checkpoint_buffer.shape}, current: {current_buffer.shape})")
                    del state_dict[buffer_key]
                else:
                    print(f"  ✓ {buffer_key}: shape matches")
            else:
                # Buffer doesn't exist in current model yet
                # Check if it will have the right shape when created
                joint_angles_meta = self.metadata.get("joint_angles", [])
                expected_frames = len(joint_angles_meta)
                checkpoint_frames = checkpoint_buffer.shape[0]
                
                if expected_frames != checkpoint_frames:
                    print(f"  ⚠️  Skipping {buffer_key}: frame count mismatch "
                        f"(checkpoint: {checkpoint_frames}, dataset: {expected_frames})")
                    del state_dict[buffer_key]

        # --- 6. Load General Non-Gaussian State (Parent Class Attributes) ---

        # Define all patterns/names to exclude from parent load
        def should_skip_param(key: str) -> bool:
            """Check if parameter should be skipped during parent load."""
            
            # Skip Gaussian parameters (handled separately)
            if key.startswith(("gauss_params.", "gauss_params_canonical.", "gauss_params_fixed.", "all_gauss_params_")):
                return True
            
            # Skip joint parameters (handled separately)
            if key.startswith("all_joint_params."):
                return True
            
            # Skip pointer attributes (set manually after loading)
            pointer_attrs = {
                "joint_angles_learned",
                "initial_angles", 
                "joint_pivot",
                "joint_axis_raw",
                "angle_range_log",
                "gauss_params",
                "gauss_params_canonical"
            }
            if key in pointer_attrs:
                return True
            
            # Skip legacy single-joint format
            if is_legacy_single_joint_param(key):
                return True
            
            # Skip frame-dependent buffers (backup - already removed above)
            frame_buffer_patterns = [
                r"initial_angles_joint_\d+$",
                r"joint_angles_prior_joint_\d+$",
                r"initial_angle_range_joint_\d+$",
            ]
            for pattern in frame_buffer_patterns:
                if re.match(pattern, key):
                    return True
            
            return False

        non_gauss_state = {
            k: v for k, v in state_dict.items() 
            if not should_skip_param(k)
        }

        super().load_state_dict(non_gauss_state, strict=False)
        
        # --- 7. Configure Training Stage (Needs Pointers to Be Set) ---
        self.step = 0
        self.configure_training_stage()
        
        # --- 8. Load Joint Metadata ---
        if "joint_metadata_absolute" in state_dict:
            joint_meta = state_dict["joint_metadata_absolute"]  # Don't pop - might be dict
            if isinstance(joint_meta, dict):
                for joint_id, meta in joint_meta.items():
                    if "limits" in meta and torch.is_tensor(meta["limits"]):
                        setattr(self, f"joint_limits_{joint_id}", meta["limits"].to(self.device))
                    if "type" in meta:
                        setattr(self, f"joint_type_{joint_id}", meta["type"])
        
        # --- 9. Status Report (Safe Access) ---
        print(f"\n{'='*60}")
        print(f"✓ Checkpoint Load Complete")
        print(f"{'='*60}")
        print(f"Active Joint: {active_id}")
        print(f"Training Mode: {self.config.training_mode}")
        
        # Safe access to active Gaussians
        if hasattr(self, 'gauss_params') and 'means' in self.gauss_params:
            print(f"Active Gaussians: {self.gauss_params['means'].shape[0]:,}")
        elif active_id in self.all_gauss_params_obj and 'means' in self.all_gauss_params_obj[active_id]:
            print(f"Active Gaussians: {self.all_gauss_params_obj[active_id]['means'].shape[0]:,}")
        
        all_joint_ids = sorted(set(self.all_gauss_params_obj.keys()) | set(self.all_joint_params.keys()))
        print(f"\nLoaded Joints ({len(all_joint_ids)}):")
        print(f"{'-'*60}")
        
        for joint_id in all_joint_ids:
            if joint_id in self.all_gauss_params_obj:
                n_gaussians = self.all_gauss_params_obj[joint_id]["means"].shape[0]
                is_frozen = not self.all_gauss_params_obj[joint_id]["means"].requires_grad
                gauss_status = f"{n_gaussians:,} Gaussians ({'FROZEN' if is_frozen else 'TRAINABLE'})"
            else:
                gauss_status = "No Gaussians"
            
            if joint_id in self.all_joint_params:
                param_names = list(self.all_joint_params[joint_id].keys())
                if len(param_names) > 0:
                    params_frozen = not self.all_joint_params[joint_id][param_names[0]].requires_grad
                    param_status = f"{len(param_names)} params ({'FROZEN' if params_frozen else 'TRAINABLE'})"
                else:
                    param_status = "No params"
            else:
                param_status = "No params"
            
            marker = "→" if joint_id == active_id else " "
            print(f"{marker} {joint_id:12s} | {gauss_status:30s} | {param_status}")
        
        if hasattr(self, 'gauss_params_fixed') and 'means' in self.gauss_params_fixed:
            n_bg = self.gauss_params_fixed['means'].shape[0]
            bg_frozen = not self.gauss_params_fixed['means'].requires_grad
            print(f"\n  Background     | {n_bg:,} Gaussians ({'FROZEN' if bg_frozen else 'TRAINABLE'})")
        
        print(f"{'='*60}\n")

        
    def step_cb(self, optimizers: Optimizers, step):
        self.step = step
        self.optimizers = optimizers.optimizers
        self.schedulers = optimizers.schedulers

    def step_post_backward(self, step):
        """
        Apply densification strategy only to the ACTIVE joint's object and canonical parameters,
        relying on the prioritized render order.
        """
        assert step == self.step

        # === RECOVERY MODE: Skip all geometric refinement ===
        if self.config.training_mode == "recovery":
            if step % 500 == 0:
                CONSOLE.print(f"[yellow]Step {step}: Recovery mode - skipping densification[/yellow]")
                CONSOLE.print("  • Geometry frozen (no splits/clones/prunes)")
                CONSOLE.print("  • Radiance optimization only")

                # Debug: check if background is getting gradients
                if hasattr(self, "gauss_params_fixed"):
                    bg_features_grad = self.gauss_params_fixed["features_dc"].grad
                    if bg_features_grad is not None:
                        CONSOLE.print(f"  ✓ Background getting gradients: {bg_features_grad.norm():.6e}")
                    else:
                        CONSOLE.print("  [red]✗ Background NOT getting gradients![/red]")
            return

        # === ARTICULATION MODE: Densification for ACTIVE Joint Only ===
        if not isinstance(self.strategy, DefaultStrategy):
            raise ValueError(f"Only DefaultStrategy supported, got {self.strategy}")

        # --- Local vars ---
        n_active_obj = self.n_active_obj
        n_obj_total = self.n_obj_total
        n_active_canon = self.gauss_params_canonical["means"].shape[0]

        # --- Helper: empty info dict ---
        def create_empty_info_with_absgrad():
            empty_ids = torch.empty(0, dtype=torch.long, device=self.device)
            key = self.strategy.key_for_gradient
            if hasattr(self, "combined_info") and self.combined_info and key in self.combined_info:
                ref_tensor = self.combined_info[key]
                empty_tensor = torch.empty((0,) + ref_tensor.shape[1:], device=self.device, dtype=ref_tensor.dtype)
            else:
                empty_tensor = torch.empty((0, 2), device=self.device, dtype=torch.float32)
            empty_tensor.absgrad = torch.empty_like(empty_tensor)
            return {"gaussian_ids": empty_ids, key: empty_tensor}

        obj_info = create_empty_info_with_absgrad()
        canon_info = create_empty_info_with_absgrad()

        # --- Process combined info from last render ---
        if hasattr(self, "combined_info") and self.combined_info and self.combined_info.get("gaussian_ids") is not None:
            visible_ids = self.combined_info["gaussian_ids"]

            # ===============================================================
            # 1. ACTIVE OBJECT GAUSSIANS
            # ===============================================================
            obj_mask = visible_ids < n_active_obj
            if obj_mask.any():
                obj_visible_ids = visible_ids[obj_mask]
                obj_info = {"gaussian_ids": obj_visible_ids}

                # Copy matching tensors (and absgrad where available)
                for k, v in self.combined_info.items():
                    if k == "gaussian_ids":
                        continue
                    if isinstance(v, torch.Tensor) and v.shape[0] == len(visible_ids):
                        obj_tensor = v[obj_mask].contiguous()
                        if hasattr(v, "absgrad") and v.absgrad is not None:
                            obj_tensor.absgrad = v.absgrad[obj_mask].contiguous()
                        obj_info[k] = obj_tensor
                    else:
                        obj_info[k] = v

                # --- Inject required fields for gsplat ---
                size_tensor_ref = obj_info["radii"] if "radii" in obj_info else obj_info[self.strategy.key_for_gradient]

                if "width" not in obj_info:
                    obj_info["width"] = torch.zeros_like(size_tensor_ref)
                    obj_info["height"] = torch.zeros_like(size_tensor_ref)

                if "n_cameras" not in obj_info:
                    obj_info["n_cameras"] = torch.tensor(1, device=self.device)

            # ===============================================================
            # 2. ACTIVE CANONICAL GAUSSIANS
            # ===============================================================
            canon_start_id = n_obj_total
            canon_mask = (visible_ids >= canon_start_id) & (visible_ids < canon_start_id + n_active_canon)
            if canon_mask.any():
                canon_visible_ids = visible_ids[canon_mask] - canon_start_id
                canon_info = {"gaussian_ids": canon_visible_ids}

                for k, v in self.combined_info.items():
                    if k == "gaussian_ids":
                        continue
                    if isinstance(v, torch.Tensor) and v.shape[0] == len(visible_ids):
                        canon_tensor = v[canon_mask].contiguous()
                        if hasattr(v, "absgrad") and v.absgrad is not None:
                            canon_tensor.absgrad = v.absgrad[canon_mask].contiguous()
                        canon_info[k] = canon_tensor
                    else:
                        canon_info[k] = v

                size_tensor_ref = canon_info["radii"] if "radii" in canon_info else canon_info[self.strategy.key_for_gradient]

                if "width" not in canon_info:
                    canon_info["width"] = torch.zeros_like(size_tensor_ref)
                    canon_info["height"] = torch.zeros_like(size_tensor_ref)

                if "n_cameras" not in canon_info:
                    canon_info["n_cameras"] = torch.tensor(1, device=self.device)

        # ===============================================================
        # 3. APPLY STRATEGY TO ACTIVE JOINT'S PARAMETERS
        # ===============================================================
        obj_optimizers = {name.replace("obj_", ""): opt for name, opt in self.optimizers.items() if name.startswith("obj_")}
        canon_optimizers = {name.replace("canon_", ""): opt for name, opt in self.optimizers.items() if name.startswith("canon_")}

        active_id = self.config.active_joint_id

        # --- Object densification ---
        # print("obj_info keys:", list(obj_info.keys()))
        n_obj_before = self.gauss_params["means"].shape[0]
        if obj_info["gaussian_ids"].numel() > 0:
            self.strategy.step_post_backward(
                params=self.gauss_params,
                optimizers=obj_optimizers,
                state=self.strategy_state,
                step=self.step,
                info=obj_info,
                packed=True,
            )
        n_obj_after = self.gauss_params["means"].shape[0]

        # --- Canonical densification ---
        n_canon_before = self.gauss_params_canonical["means"].shape[0]
        if not hasattr(self, "strategy_state_canonical"):
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
        n_canon_after = self.gauss_params_canonical["means"].shape[0]


    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        """Route to appropriate loss function based on training mode"""
        if self.config.training_mode == "recovery":
            return self.get_loss_dict_recovery(outputs, batch, metrics_dict)
        else:
            return self.get_loss_dict_articulation(outputs, batch, metrics_dict)
    


    def get_loss_dict_recovery(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        """Computes and returns the losses dict.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
            metrics_dict: dictionary of metrics, some of which we can use for loss
        """
        gt_img = self.composite_with_background(self.get_gt_img(batch["image"]), outputs["background"])
        pred_img = outputs["rgb"]


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

        # Losses for mcmc
        if self.config.strategy == "mcmc":
            if self.config.mcmc_opacity_reg > 0.0:
                mcmc_opacity_reg = (
                    self.config.mcmc_opacity_reg * torch.abs(torch.sigmoid(self.gauss_params["opacities"])).mean()
                )
                loss_dict["mcmc_opacity_reg"] = mcmc_opacity_reg
            if self.config.mcmc_scale_reg > 0.0:
                mcmc_scale_reg = self.config.mcmc_scale_reg * torch.abs(torch.exp(self.gauss_params["scales"])).mean()
                loss_dict["mcmc_scale_reg"] = mcmc_scale_reg

        if self.training:
            # Add loss from camera optimizer
            self.camera_optimizer.get_loss_dict(loss_dict)
            if self.config.use_bilateral_grid:
                loss_dict["tv_loss"] = 10 * total_variation_loss(self.bil_grids.grids)

        if self.config.use_depth and "depth_image" in batch and outputs.get("depth") is not None:

            gt_depth = self._downscale_if_required(batch["depth_image"])
            gt_depth = gt_depth.to(self.device)
            depth_mask = gt_depth > 0
            if depth_mask.any():
                depth_loss = torch.nn.functional.l1_loss(
                    outputs["depth"][depth_mask], 
                    gt_depth[depth_mask]
                )
                loss_dict["depth_loss"] = depth_loss * self.config.depth_lambda

        if self.config.use_depth and "depth_image" in batch:
            depth_out = outputs["depth"]
            depth_gt = self.get_gt_img(batch["depth_image"])
            
            depth_out_loss = depth_out.squeeze(-1).unsqueeze(0)
            depth_gt_loss = depth_gt.squeeze(-1).unsqueeze(0)
            
            mask_loss = torch.ones_like(depth_out_loss)
            
            depth_loss = self.depth_loss_fn(depth_out_loss, depth_gt_loss, mask_loss)
            
            loss_dict["depth_loss"] = depth_loss * self.config.depth_lambda

        if self.config.use_opacity_regularization and self.training:
            
            all_obj_opacities = []
            all_fixed_opacities = []
            active_id = self.config.active_joint_id
            
            for joint_id in self.all_gauss_params_obj.keys():
                
                params = self.all_gauss_params_obj[joint_id]
                
                # Use .requires_grad on the opacities parameter itself to determine inclusion.
                # This correctly includes previous joints if their radiance was unfrozen (Fix B).
                if params["opacities"].requires_grad:
                    all_obj_opacities.append(torch.sigmoid(params["opacities"]))
                

            # --- 2. Collect Opacities from Fixed (Background) Gaussians ---
            if "opacities" in self.gauss_params_fixed and self.gauss_params_fixed["opacities"].requires_grad:
                all_fixed_opacities.append(torch.sigmoid(self.gauss_params_fixed["opacities"]))
            
            
            # --- 3. Compute Loss for Each Group if data exists ---
            
            total_obj_loss = torch.tensor(0.0, device=self.device)
            total_fixed_loss = torch.tensor(0.0, device=self.device)
            
            if all_obj_opacities:
                obj_opacities = torch.cat(all_obj_opacities, dim=0)
                total_obj_loss = opacity_loss(obj_opacities)
                


            if all_fixed_opacities:
                fixed_opacities = torch.cat(all_fixed_opacities, dim=0)
                
                total_fixed_loss_base = opacity_loss(fixed_opacities)
                
                L1_fixed_penalty = fixed_opacities.mean()
                
                total_fixed_loss = total_fixed_loss_base + (
                    self.config.lambda_L1_fixed_opacity * L1_fixed_penalty
                )

            loss_dict["opacity_reg"] = (
                self.config.opacity_lambda_obj * total_obj_loss +
                self.config.opacity_lambda_fixed * total_fixed_loss
            )

        return loss_dict


    def get_loss_dict_articulation(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        gt_img = self.composite_with_background(self.get_gt_img(batch["image"]), outputs["background"])
        pred_img = outputs["rgb"]
        mask = batch.get("mask", None)

        # print("batch keys:", batch.keys())
        # import pdb; pdb.set_trace()

        
        if "mask" in batch:
            # batch["mask"] : [H, W, 1]
            mask = self._downscale_if_required(batch["mask"])
            mask = mask.to(self.device)
            assert mask.shape[:2] == gt_img.shape[:2] == pred_img.shape[:2]
            gt_img = gt_img * mask
            pred_img = pred_img * mask
        
        # === Main Losses ===
        Ll1 = torch.abs(gt_img - pred_img).mean()
        simloss = 1 - self.ssim(gt_img.permute(2, 0, 1)[None, ...],
                                pred_img.permute(2, 0, 1)[None, ...])
        loss_dict = {
            "main_loss": (1 - self.config.ssim_lambda) * Ll1 + self.config.ssim_lambda * simloss,
        }

        active_scales = torch.cat([self.gauss_params["scales"], self.gauss_params_canonical["scales"]], dim=0)
        active_opacities = torch.cat([self.gauss_params["opacities"], self.gauss_params_canonical["opacities"]], dim=0)
        
        # === Background accumulation penalty ===
        if mask is not None and "accumulation" in outputs:
            accumulation = outputs["accumulation"]
            background_mask = ~mask.bool()
            background_acc_loss = (background_mask * accumulation).mean()
            loss_dict["background_acc_penalty"] = 0.5 * background_acc_loss
        
        # === Scale regularization ===
        if self.config.use_scale_regularization and self.step % 10 == 0:
            # Use combined active scales
            scales = torch.exp(active_scales)
            scale_ratios = scales.max(dim=-1)[0] / (scales.min(dim=-1)[0] + 1e-8)
            ratio_penalty = torch.clamp(scale_ratios - self.config.max_gauss_ratio, min=0.0)
            size_penalty = torch.clamp(scales.max(dim=-1)[0] - 0.15, min=0.0)
            scale_reg = 0.1 * (ratio_penalty.mean() + size_penalty.mean())
        else:
            scale_reg = torch.tensor(0.0).to(self.device)
        loss_dict["scale_reg"] = scale_reg
        
        # === MCMC extras ===
        if self.config.strategy == "mcmc":
            if self.config.mcmc_opacity_reg > 0.0:
                loss_dict["mcmc_opacity_reg"] = (
                    self.config.mcmc_opacity_reg * torch.abs(torch.sigmoid(self.gauss_params["opacities"])).mean()
                )
            if self.config.mcmc_scale_reg > 0.0:
                loss_dict["mcmc_scale_reg"] = (
                    self.config.mcmc_scale_reg * torch.abs(torch.exp(self.gauss_params["scales"])).mean()
                )

        # === Camera + bilateral grid ===
        if self.training:
            self.camera_optimizer.get_loss_dict(loss_dict)
        if self.config.use_bilateral_grid:
            loss_dict["tv_loss"] = 10 * total_variation_loss(self.bil_grids.grids)
        
        if self.config.use_depth and "depth_image" in batch:
            depth_out = outputs["depth"]
            depth_gt = self.get_gt_img(batch["depth_image"])
            
            depth_out_loss = depth_out.squeeze(-1).unsqueeze(0)
            depth_gt_loss = depth_gt.squeeze(-1).unsqueeze(0)
            
            if mask is not None:
                mask_loss = mask.squeeze(-1).unsqueeze(0)
            else:
                mask_loss = torch.ones_like(depth_out_loss)
            
            depth_loss = self.depth_loss_fn(depth_out_loss, depth_gt_loss, mask_loss)
            
            # Add warmup schedule
            if self.step < 1000:
                depth_weight = 0.0
            elif self.step < 3000:
                # Ramp from 0 to final weight
                progress = (self.step - 1000) / 2000
                depth_weight = progress * self.config.depth_lambda
            else:
                depth_weight = self.config.depth_lambda
            
            loss_dict["depth_loss"] = depth_weight * depth_loss
            
            # Original Debug logging
            if self.step % 500 == 0:
                print(f"[Depth - Step {self.step}] Loss: {depth_loss:.6f}")

        if hasattr(self, "joint_angles_learned") and self.joint_angles_learned is not None:
            num_frames = len(self.joint_angles_learned)
            
            # --- 1. Temporal Smoothness ---
            # Encourage smooth motion between consecutive frames
            temporal_reg = torch.tensor(0.0, device=self.device)
            if num_frames > 1:
                forward_diff = self.joint_angles_learned[1:] - self.joint_angles_learned[:-1]
                temporal_reg = torch.mean(forward_diff**2)
            loss_dict["joint_temporal_smooth"] = 0.05 * temporal_reg
            
            # --- 2. Acceleration Penalty ---
            # Prevent jerky motion (second-order smoothness)
            accel_reg = torch.tensor(0.0, device=self.device)
            if num_frames > 2:
                accel = self.joint_angles_learned[:-2] - 2*self.joint_angles_learned[1:-1] + self.joint_angles_learned[2:]
                accel_reg = torch.mean(accel**2)
            loss_dict["joint_acceleration"] = 0.02 * accel_reg
            
            # --- 3. Range Bounds ---
            # Ensure physically plausible range
            joint_type = self.obj_3d_seg.joint_type
            current_range = self.angle_range
            
            if joint_type == "revolute":
                min_range = 0.2   # ~11 degrees minimum
                max_range = 3.14  # ~180 degrees maximum
            else:  # prismatic
                min_range = 0.01  # 1cm minimum
                max_range = 0.5   # 50cm maximum
            
            range_penalty = (
                torch.relu(min_range - current_range) ** 2 +
                torch.relu(current_range - max_range) ** 2
            )
            loss_dict["joint_range_bounds"] = 0.05 * range_penalty
            

            # if hasattr(self, 'initial_angles'):
            #     drift = self.joint_angles_learned - self.initial_angles
            #     drift_loss = torch.mean(drift**2)
            #     loss_dict["joint_drift_from_init"] = 0.01 * drift_loss
            
            # --- Total Joint Regularization ---
            total_joint_reg = (
                loss_dict["joint_temporal_smooth"] +
                loss_dict["joint_acceleration"] +
                loss_dict["joint_range_bounds"]
            )
            
            # if "joint_drift_from_init" in loss_dict:
            #     total_joint_reg += loss_dict["joint_drift_from_init"]
            
            loss_dict["joint_regularization"] = total_joint_reg
    


        if self.config.use_opacity_regularization and self.training:
            
            all_obj_opacities = []
            all_fixed_opacities = []
            active_id = self.config.active_joint_id
            
            # --- 1. Collect Opacities from ALL Joints (Object & Canonical) ---
            for joint_id in self.all_gauss_params_obj.keys():
                
                # Check trainability for each joint parameter
                # We assume that in Articulation mode, only the active joint's radiance is trainable, 
                # OR, based on the recommended fix, all radiance is trainable.
                # The crucial check is that the .data must flow if .grad exists.
                
                params = self.all_gauss_params_obj[joint_id]
                
                # Use .requires_grad on the opacities parameter itself to determine inclusion.
                # This correctly includes previous joints if their radiance was unfrozen (Fix B).
                if params["opacities"].requires_grad:
                    all_obj_opacities.append(torch.sigmoid(params["opacities"]))
                

            # --- 2. Collect Opacities from Fixed (Background) Gaussians ---
            if "opacities" in self.gauss_params_fixed and self.gauss_params_fixed["opacities"].requires_grad:
                all_fixed_opacities.append(torch.sigmoid(self.gauss_params_fixed["opacities"]))
            
            
            # --- 3. Compute Loss for Each Group if data exists ---
            
            total_obj_loss = torch.tensor(0.0, device=self.device)
            total_fixed_loss = torch.tensor(0.0, device=self.device)
            
            if all_obj_opacities:
                obj_opacities = torch.cat(all_obj_opacities, dim=0)
                total_obj_loss = opacity_loss(obj_opacities)
                
            # if all_canon_opacities:
            #     canon_opacities = torch.cat(all_canon_opacities, dim=0)
            #     total_canon_loss = opacity_loss(canon_opacities)
                
            if all_fixed_opacities:
                fixed_opacities = torch.cat(all_fixed_opacities, dim=0)
                # Use obj lambda for consistency, or define a new fixed_lambda
                total_fixed_loss = opacity_loss(fixed_opacities)


            loss_dict["opacity_reg"] = (
                self.config.opacity_lambda_obj * total_obj_loss +
                self.config.opacity_lambda_fixed * total_fixed_loss
            )


        # === Normal Regularization ===
        if self.config.use_normal_reg and "depth_image" in batch and self.training:
            # Extract intrinsics
            fx = batch["fx"].item() if isinstance(batch["fx"], torch.Tensor) else batch["fx"]
            fy = batch["fy"].item() if isinstance(batch["fy"], torch.Tensor) else batch["fy"]
            cx = batch["cx"].item() if isinstance(batch["cx"], torch.Tensor) else batch["cx"]
            cy = batch["cy"].item() if isinstance(batch["cy"], torch.Tensor) else batch["cy"]
            c2w = batch["c2w"]
            
            # Get depths
            depth_gt = self.get_gt_img(batch["depth_image"])
            depth_out = outputs["depth"]
            img_size = (depth_gt.shape[1], depth_gt.shape[0])  # (W, H)
            
            # Get scale and shift from depth loss
            depth_out_loss = depth_out.squeeze(-1).unsqueeze(0)
            depth_gt_loss = depth_gt.squeeze(-1).unsqueeze(0)
            mask = batch.get("mask", None)
            
            if mask is not None:
                mask_loss = mask.squeeze(-1).unsqueeze(0)
            else:
                mask_loss = torch.ones_like(depth_out_loss)
            
            scale, shift = compute_scale_and_shift(depth_out_loss, depth_gt_loss, mask_loss)
            
            # Apply scale and shift to predicted depth
            depth_out_aligned = scale.item() * depth_out + shift.item()
            
            normals_gt = normal_from_depth_image(
                depths=depth_gt.squeeze(-1),
                fx=fx, fy=fy, cx=cx, cy=cy,
                img_size=img_size,
                c2w=c2w,
                device=self.device,
                smooth=self.config.smooth_normals
            )
            
            # Compute normals from ALIGNED predicted depth
            normals_pred = normal_from_depth_image(
                depths=depth_out_aligned.squeeze(-1),  # Use aligned depth!
                fx=fx, fy=fy, cx=cx, cy=cy,
                img_size=img_size,
                c2w=c2w,
                device=self.device,
                smooth=False
            )
            
            # Create valid mask (DON'T zero out normals, just track valid regions)
            if mask is not None:
                valid_mask = (mask > 0.5).squeeze(-1)
            else:
                valid_mask = torch.ones(normals_gt.shape[:2], dtype=torch.bool, device=self.device)
            
            # Filter out invalid normals (near-zero from padding/edges)
            valid_normals_gt = torch.norm(normals_gt, dim=-1) > 0.1
            valid_normals_pred = torch.norm(normals_pred, dim=-1) > 0.1
            valid_mask = valid_mask & valid_normals_gt & valid_normals_pred
            
            if valid_mask.any():
                # Extract valid normals
                normals_gt_valid = normals_gt[valid_mask]
                normals_pred_valid = normals_pred[valid_mask]
                
                # Cosine similarity loss (1 - |dot product|)
                dot_product = (normals_pred_valid * normals_gt_valid).sum(dim=-1)
                normal_loss = 1 - dot_product.abs()
                normal_loss = normal_loss.mean()
                
                loss_dict["normal_loss"] = self.config.normal_lambda * normal_loss
                
                # Debug info
                if self.step % 1000 == 0:
                    mean_dot = dot_product.mean().item()
                    mean_angle = torch.acos(dot_product.abs().clamp(-1, 1)).mean().item() * 180 / 3.14159
                    print(f"[Normal - Step {self.step}] "
                        f"Loss: {normal_loss.item():.4f}, "
                        f"Mean dot: {mean_dot:.4f}, "
                        f"Mean angle: {mean_angle:.1f}°, "
                        f"Valid pixels: {valid_mask.sum()}")
                    
                    # Optional: save normal debug
                    if self.config.normal_debug_vis:
                        save_normal_debug(self.step, normals_gt, normals_pred, mask)
            else:
                loss_dict["normal_loss"] = torch.tensor(0.0, device=self.device)
                if self.step % 1000 == 0:
                    print(f"[Normal - Step {self.step}] Warning: No valid normals for loss computation")
            
        
        # if self.step % 500 == 0 and not getattr(self, '_debug_saved_this_step', False):
        #     self._debug_saved_this_step = True
        #     save_debug_id_maps(self, batch)

        # elif self.step % 100 != 0:
        #     self._debug_saved_this_step = False


        
        return loss_dict



    def get_joint_angle_for_camera(self, camera: Cameras, joint_id: str = None):
        """
        Return a differentiable joint angle tensor for a specific joint.
        
        Args:
            camera: Camera object with metadata
            joint_id: Specific joint ID (e.g., 'joint_0', 'joint_1'). 
                    If None, uses active_joint_id.
        """
        # Use active joint if not specified
        if joint_id is None:
            joint_id = self.config.active_joint_id
        
        # --- Case 1: Per-joint metadata (NEW multi-joint format) ---
        if hasattr(camera, "metadata") and camera.metadata is not None:
            # Try per-joint key first (e.g., "joint_angles_joint_0")
            angle_key = f"joint_angles_{joint_id}"
            angle_val = camera.metadata.get(angle_key, None)
            
            if angle_val is not None:
                if torch.is_tensor(angle_val):
                    return angle_val.to(self.device).float()
                else:
                    return torch.tensor([float(angle_val)], device=self.device)
        
        # --- Case 2: time-based interpolation (training with learned angles) ---
        # Check if this joint has learned angles
        if joint_id in self.all_joint_params and "angles" in self.all_joint_params[joint_id]:
            if hasattr(camera, "times") and camera.times is not None:
                # Use the learned angles for this specific joint
                joint_angles = self.all_joint_params[joint_id]["angles"]
                time_val = camera.times.flatten()[0]
                num_frames = len(joint_angles)
                
                # Interpolate
                idx_f = time_val * (num_frames - 1)
                idx0 = torch.floor(idx_f).long().clamp(0, num_frames - 2)
                idx1 = idx0 + 1
                w = idx_f - idx0.float()
                angle0 = joint_angles[idx0]
                angle1 = joint_angles[idx1]
                angle = (1.0 - w) * angle0 + w * angle1
                return angle
        

        if hasattr(self, 'all_joint_params') and joint_id in self.all_joint_params:
            if hasattr(self, f'joint_limits_{joint_id}'):
                limits = getattr(self, f'joint_limits_{joint_id}')
                
                # ORIGINAL (Midpoint fallback):
                # mid_val = (limits[0] + limits[1]) / 2.0 
                
                min_val = limits[0] 
                
                return torch.tensor([min_val], device=self.device, dtype=torch.float32) # <-- USE MIN
        
    
        return torch.tensor([0.0], device=self.device, dtype=torch.float32)


    def get_gaussians_for_render(self, camera: Cameras) -> Dict[str, torch.Tensor]:
        """
        Prepares Gaussians for rendering with proper multi-joint support.
        
        Training (Articulation):
        - Only render ACTIVE joint (obj + canonical)
        - Skip all other joints
        - No background
        
        Training (Recovery):
        - Render ALL joints (frozen geometry, trainable radiance)
        - Render background (trainable radiance)
        
        Eval/Inference:
        - Render ALL joints with their respective articulations
        - Render background
        """
        GAUSS = ["means", "scales", "quats", "features_dc", "features_rest", "opacities"]
        GEOMETRY = ["means", "scales", "quats"]
        RADIANCE = ["features_dc", "features_rest", "opacities"]
        
        active_id = self.config.active_joint_id
        is_articulation_training = (self.training and self.config.training_mode == "articulation")
        is_recovery_training = (self.training and self.config.training_mode == "recovery")
        is_eval = not self.training
        
        obj_sets_to_combine = []
        canon_sets_to_combine = []
        
        # --- Process ALL Joints ---
        for joint_id in sorted(self.all_gauss_params_obj.keys()):
            is_active = (joint_id == active_id)
            
            # === ARTICULATION TRAINING: Only render active joint ===
            if is_articulation_training and not is_active:
                continue  # Skip non-active joints
            
            # Get joint angle (each joint may have different frame counts)
            joint_angle = self.get_joint_angle_for_camera(camera, joint_id=joint_id)
            
            obj_params_raw = self.all_gauss_params_obj[joint_id]
            canon_params = self.all_gauss_params_canon[joint_id]
            
            # === Active Joint (Always Included) ===
            if is_active:
                # Use apply_articulation_to_optimizer_params for active joint
                # (works with self.gauss_params which points to active joint)
                articulated_obj = apply_articulation_to_optimizer_params(self, joint_angle)
                obj_sets_to_combine.append(articulated_obj)
                
                # Canonical (from self.gauss_params_canonical)
                canon_active = {k: self.gauss_params_canonical[k] for k in self.gauss_params_canonical.keys()}
                canon_sets_to_combine.append(canon_active)
            
            # === Non-Active Joints (Recovery or Eval) ===
            else:
                # Prepare articulated object Gaussians with proper gradient control
                if is_recovery_training:
                    # Recovery: Freeze geometry, keep radiance trainable
                    obj_params_mixed = {}
                    for k in GAUSS:
                        if k in GEOMETRY:
                            obj_params_mixed[k] = obj_params_raw[k].detach()
                        else:  # Radiance
                            obj_params_mixed[k] = obj_params_raw[k]
                elif is_eval:
                    # Eval: Detach everything (no gradients needed)
                    obj_params_mixed = {k: obj_params_raw[k].detach() for k in GAUSS}
                else:
                    # Shouldn't reach here, but safe fallback
                    obj_params_mixed = {k: obj_params_raw[k].detach() for k in GAUSS}
                
                # Apply articulation for this joint
                articulated_obj = self._apply_articulation_to_joint(
                    obj_params_mixed,
                    joint_id, 
                    joint_angle.detach()  # Always detach angle for non-active joints
                )
                obj_sets_to_combine.append(articulated_obj)
                
                # Canonical Gaussians
                if is_recovery_training:
                    # Recovery: Keep radiance trainable
                    canon_other = {k: canon_params[k] for k in canon_params.keys()}
                else:
                    # Eval: Detach everything
                    canon_other = {k: canon_params[k].detach() for k in canon_params.keys()}
                canon_sets_to_combine.append(canon_other)
        
        # --- Combine All Joints ---
        if not obj_sets_to_combine:
            # Shouldn't happen, but safe fallback
            print("⚠️  Warning: No joints to render!")
            combined_params = {
                p: torch.empty((0, 3 if p not in ["quats", "opacities"] else (4 if p == "quats" else 1)), 
                            device=self.device) 
                for p in GAUSS
            }
        else:
            # Combine all articulated object groups
            combined_obj_params = {p: torch.cat([d[p] for d in obj_sets_to_combine], dim=0) for p in GAUSS}
            
            # Combine all canonical groups
            combined_canon_params = {p: torch.cat([d[p] for d in canon_sets_to_combine], dim=0) for p in GAUSS}

            # Final combination: All Articulated Objects + All Canonicals
            combined_params = {p: torch.cat([combined_obj_params[p], combined_canon_params[p]], dim=0) for p in GAUSS}
        
        # --- Store Counts for step_post_backward & Debugging ---
        if hasattr(self, 'gauss_params') and 'means' in self.gauss_params:
            self.n_active_obj = self.gauss_params['means'].shape[0]
        else:
            self.n_active_obj = 0
        
        self.n_obj_total = combined_obj_params['means'].shape[0] if obj_sets_to_combine else 0
        self.n_canon_total = combined_canon_params['means'].shape[0] if canon_sets_to_combine else 0

        # --- Add Background ---
        include_background = is_recovery_training or is_eval
        
        if include_background:
            if (hasattr(self, "gauss_params_fixed") and self.gauss_params_fixed and 
                self.gauss_params_fixed["means"].shape[0] > 0):
                full_scene_params = {}
                for name in combined_params.keys():
                    combined_tensor = combined_params[name]
                    bg_tensor = self.gauss_params_fixed[name]
                    
                    # Detach background in eval mode
                    if is_eval:
                        bg_tensor = bg_tensor.data
                    
                    # Ensure same device
                    if combined_tensor.device != bg_tensor.device:
                        bg_tensor = bg_tensor.to(combined_tensor.device)
                    
                    full_scene_params[name] = torch.cat([combined_tensor, bg_tensor], dim=0)
                
                return full_scene_params
        
        # Return combined joints without background
        return combined_params


    def _apply_articulation_to_joint(self, obj_params: Dict, joint_id: str, joint_angle: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Apply articulation transform to a specific joint's parameters.
        This is a helper for rendering non-active joints.
        """
        # Get joint-specific parameters
        joint_params = self.all_joint_params[joint_id]
        pivot = joint_params["pivot"]
        axis_raw = joint_params["axis_raw"]
        
        # Normalize axis
        axis = axis_raw / (torch.norm(axis_raw) + 1e-8)
        
        # Get joint type
        joint_type_attr = f'joint_type_{joint_id}'
        joint_type = getattr(self, joint_type_attr, 'revolute')
        
        # Extract means
        means = obj_params["means"]
        
        # Apply transform
        if joint_type == "revolute":
            # Rotation matrix from axis-angle
            angle_scalar = joint_angle.squeeze()
            K = torch.tensor([
                [0, -axis[2], axis[1]],
                [axis[2], 0, -axis[0]],
                [-axis[1], axis[0], 0]
            ], device=means.device, dtype=means.dtype)
            
            R = torch.eye(3, device=means.device, dtype=means.dtype) + \
                torch.sin(angle_scalar) * K + \
                (1 - torch.cos(angle_scalar)) * (K @ K)
            
            # Transform means
            centered = means - pivot.unsqueeze(0)
            rotated = centered @ R.T
            transformed_means = rotated + pivot.unsqueeze(0)
            
        elif joint_type == "prismatic":
            # Translation along axis
            translation = joint_angle.squeeze() * axis
            transformed_means = means + translation.unsqueeze(0)
        else:
            transformed_means = means
        
        # Return articulated parameters
        return {
            "means": transformed_means,
            "scales": obj_params["scales"],
            "quats": obj_params["quats"],
            "features_dc": obj_params["features_dc"],
            "features_rest": obj_params["features_rest"],
            "opacities": obj_params["opacities"],
        }

    # def forward(self, camera: Cameras) -> Dict[str, torch.Tensor]:
    #     """Override to accept Cameras instead of RayBundles."""
    #     # import pdb; pdb.set_trace()
    #     return self.get_outputs(camera)

    def get_outputs(self, camera: Cameras, render_id_map: bool = False) -> Dict[str, Union[torch.Tensor, List]]:
        """Takes in a camera and returns a dictionary of outputs with articulation."""
        if not isinstance(camera, Cameras):
            return {}
        gaussians_to_render = self.get_gaussians_for_render(camera)
        self._current_camera = camera 


        n_obj_total = self.n_obj_total if hasattr(self, 'n_obj_total') else self.gauss_params['means'].shape[0]
        n_canon_total = self.n_canon_total if hasattr(self, 'n_canon_total') else self.gauss_params_canonical['means'].shape[0]
        actual_count = gaussians_to_render['means'].shape[0]
        
        if self.training:
            expected_count = n_obj_total + n_canon_total  # All object + all canonical
            print(f"Training render check (Multi-Joint):")
            print(f"   Expected (All Obj + All Canon): {expected_count} ({n_obj_total} + {n_canon_total})")
            print(f"   Actually rendering: {actual_count}")
        else:
            bg_count = self.gauss_params_fixed['means'].shape[0] if hasattr(self, 'gauss_params_fixed') else 0
            print(f" Eval render check (Multi-Joint):")
            print(f"   All Obj: {n_obj_total}, All Canon: {n_canon_total}, Background: {bg_count}, Total: {actual_count}")
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

        if render_id_map:
            # Object: ID 0-255 (red channel) - applies to ALL articulated objects
            # Canonical: ID 256-511 (green channel) - applies to ALL canonical objects
            # Background: ID 512+ (blue channel)
            
            id_colors = torch.zeros((actual_count, 3), device=gaussians_to_render["means"].device)
            
            # Use total counts: n_obj_total and n_canon_total
            
            # All Object Gaussians (0 to n_obj_total-1): encode in red channel
            if n_obj_total > 0:
                obj_ids = torch.arange(n_obj_total, device=id_colors.device, dtype=torch.float32)
                id_colors[:n_obj_total, 0] = (obj_ids % 256) / 255.0  # Red channel
                id_colors[:n_obj_total, 1] = 0.0  # Green = 0
                id_colors[:n_obj_total, 2] = 0.0  # Blue = 0
            
            # All Canonical Gaussians (n_obj_total to n_obj_total + n_canon_total - 1): encode in green channel
            if n_canon_total > 0:
                canon_start = n_obj_total
                canon_end = n_obj_total + n_canon_total
                canon_ids = torch.arange(n_canon_total, device=id_colors.device, dtype=torch.float32)
                id_colors[canon_start:canon_end, 0] = 0.0  # Red = 0
                id_colors[canon_start:canon_end, 1] = (canon_ids % 256) / 255.0  # Green channel
                id_colors[canon_start:canon_end, 2] = 0.0  # Blue = 0
            
            # Background Gaussians (remaining): encode in blue channel
            if actual_count > n_obj_total + n_canon_total:
                bg_start = n_obj_total + n_canon_total
                bg_count = actual_count - bg_start
                bg_ids = torch.arange(bg_count, device=id_colors.device, dtype=torch.float32)
                id_colors[bg_start:, 0] = 0.0  # Red = 0
                id_colors[bg_start:, 1] = 0.0  # Green = 0
                id_colors[bg_start:, 2] = (bg_ids % 256) / 255.0  # Blue channel
            
            colors_for_render = id_colors
            render_mode = "RGB"
            sh_degree_to_use = None  # No SH for ID maps
            if actual_count > n_obj_total + n_canon_total:
                print(f"   Background: {n_obj_total+n_canon_total}-{actual_count-1} (blue channel)")
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
        
        
        if self.training and self.config.training_mode == "articulation":
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
            self.n_obj_rendered = n_obj_total
        else:
            print("gaussian_ids is None - NO VISIBLE GAUSSIANS!")
            self.combined_info = None
            self.n_obj_rendered = n_obj_total

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
            id_debug = decode_id_map(render[..., :3].squeeze(0), n_obj_total, n_canon_total)
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


    def check_joint_gradients(self):
        """Debug function to verify joint parameters have gradients"""
        print("\n=== Joint Parameter Gradient Check ===")
        
        if hasattr(self, 'joint_pivot'):
            print(f"joint_pivot: requires_grad={self.joint_pivot.requires_grad}, "
                f"grad={'exists' if self.joint_pivot.grad is not None else 'None'}")
            if self.joint_pivot.grad is not None:
                print(f"  grad norm: {self.joint_pivot.grad.norm().item():.6e}")
        
        if hasattr(self, 'joint_axis_raw'):
            print(f"joint_axis_raw: requires_grad={self.joint_axis_raw.requires_grad}, "
                f"grad={'exists' if self.joint_axis_raw.grad is not None else 'None'}")
            if self.joint_axis_raw.grad is not None:
                print(f"  grad norm: {self.joint_axis_raw.grad.norm().item():.6e}")
        
        if hasattr(self, 'max_joint_angle'):
            print(f"max_joint_angle: requires_grad={self.max_joint_angle.requires_grad}, "
                f"grad={'exists' if self.max_joint_angle.grad is not None else 'None'}")
            if self.max_joint_angle.grad is not None:
                print(f"  grad value: {self.max_joint_angle.grad.item():.6e}")

        if hasattr(self, 'min_joint_angle'):
            print(f"min_joint_angle: requires_grad={self.min_joint_angle.requires_grad}, "
                f"grad={'exists' if self.min_joint_angle.grad is not None else 'None'}")
            if self.min_joint_angle.grad is not None:
                print(f"  grad value: {self.min_joint_angle.grad.item():.6e}")
        
        if hasattr(self, 'joint_t_raw'):
            print(f"joint_t_raw: requires_grad={self.joint_t_raw.requires_grad}, "
                f"grad={'exists' if self.joint_t_raw.grad is not None else 'None'}")
            if self.joint_t_raw.grad is not None:
                print(f"  grad norm: {self.joint_t_raw.grad.norm().item():.6e}")
                print(f"  grad nonzero: {(self.joint_t_raw.grad.abs() > 1e-8).sum()}/{len(self.joint_t_raw)}")
   
    def get_metrics_dict(self, outputs, batch) -> Dict[str, torch.Tensor]:
        """
        Computes comprehensive metrics for the model with detailed joint parameter tracking.
        """
        d = self._get_downscale_factor()
        if d > 1:
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
        gt_rgb = gt_img.to(self.device)
        predicted_rgb = (
            outputs["rgb"][0, ...] if outputs["rgb"].dim() == 4 else outputs["rgb"]
        )

        # === RGB Metrics ===
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

        # === Gaussian Count Metrics ===
        metrics_dict["gaussian_count"] = self.num_points
        
        # Breakdown by type
        if hasattr(self, 'gauss_params') and self.gauss_params:
            metrics_dict["gaussian_count_object"] = self.gauss_params['means'].shape[0]
        if hasattr(self, 'gauss_params_canonical') and self.gauss_params_canonical:
            metrics_dict["gaussian_count_canonical"] = self.gauss_params_canonical['means'].shape[0]
        if hasattr(self, 'gauss_params_fixed') and self.gauss_params_fixed:
            metrics_dict["gaussian_count_background"] = self.gauss_params_fixed['means'].shape[0]

        # === Depth Metrics ===
        with torch.no_grad():
            if "depth_image" in batch:
                predicted_depth = outputs.get('depth')
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

        # === Gaussian Scale Metrics ===
        with torch.no_grad():
            if hasattr(self, 'scales'):
                metrics_dict["avg_min_scale"] = float(torch.nanmean(torch.exp(self.scales[..., -1])))
                metrics_dict["avg_max_scale"] = float(torch.nanmean(torch.exp(self.scales[..., 0])))
                metrics_dict["avg_scale_ratio"] = float(
                    torch.nanmean(torch.exp(self.scales[..., 0]) / (torch.exp(self.scales[..., -1]) + 1e-8))
                )

        if hasattr(self, "joint_angles") and self.joint_angles is not None:
            with torch.no_grad():
                # Joint angle statistics
                mean_angle = self.joint_angles.mean().item()
                std_angle = self.joint_angles.std().item()
                min_angle = self.joint_angles.min().item()
                max_angle = self.joint_angles.max().item()
                angle_range = max_angle - min_angle
                
                metrics_dict.update({
                    "joint/angle_mean": mean_angle,
                    "joint/angle_std": std_angle,
                    "joint/angle_min": min_angle,
                    "joint/angle_max": max_angle,
                    "joint/angle_range": angle_range,
                    "joint/angle_range_degrees": angle_range * 180 / 3.14159,
                })
                
                # NEW: Range parameter tracking (replaces min/max params)
                if hasattr(self, 'angle_range_log'):
                    learned_range = torch.exp(self.angle_range_log).item()
                    metrics_dict["joint/learned_range"] = float(learned_range)
                    metrics_dict["joint/learned_range_degrees"] = float(learned_range * 180 / 3.14159)
                    
                    # Compare learned range to observed range
                    range_utilization = angle_range / (learned_range + 1e-8)
                    metrics_dict["joint/range_utilization"] = float(range_utilization)
                
                # NEW: First frame tracking (anchor point)
                first_frame_angle = self.joint_angles[0].item()
                metrics_dict["joint/first_frame_angle"] = float(first_frame_angle)
                metrics_dict["joint/first_frame_angle_degrees"] = float(first_frame_angle * 180 / 3.14159)
                
                # Derived limits from properties
                joint_limits = self.joint_limits
                metrics_dict["joint/derived_min"] = float(joint_limits[0].item())
                metrics_dict["joint/derived_max"] = float(joint_limits[1].item())
                metrics_dict["joint/derived_min_degrees"] = float(joint_limits[0].item() * 180 / 3.14159)
                metrics_dict["joint/derived_max_degrees"] = float(joint_limits[1].item() * 180 / 3.14159)
                
                # Pivot tracking with enhanced metrics
                if hasattr(self, 'joint_pivot'):
                    pivot = self.joint_pivot
                    metrics_dict.update({
                        "joint/pivot_x": float(pivot[0].item()),
                        "joint/pivot_y": float(pivot[1].item()),
                        "joint/pivot_z": float(pivot[2].item()),
                        "joint/pivot_norm": float(pivot.norm().item()),
                    })
                    
                    # Pivot drift from initialization
                    active_id = self.config.active_joint_id
                    initial_pivot_attr = f'initial_joint_pivot_{active_id}'
                    if hasattr(self, initial_pivot_attr):
                        initial_pivot = getattr(self, initial_pivot_attr)
                        pivot_diff = pivot - initial_pivot
                        pivot_drift = pivot_diff.norm().item()
                        
                        metrics_dict.update({
                            "joint/pivot_drift": float(pivot_drift),
                            "joint/pivot_drift_x": float(pivot_diff[0].item()),
                            "joint/pivot_drift_y": float(pivot_diff[1].item()),
                            "joint/pivot_drift_z": float(pivot_diff[2].item()),
                        })
                        
                        # Log if drift is significant
                        if pivot_drift > 0.01 and self.step % 1000 == 0:
                            print(f"[Pivot Drift - Step {self.step}] "
                                f"Total: {pivot_drift:.4f}m, "
                                f"ΔX: {pivot_diff[0].item():.4f}, "
                                f"ΔY: {pivot_diff[1].item():.4f}, "
                                f"ΔZ: {pivot_diff[2].item():.4f}")
                
                # Axis tracking with enhanced metrics
                if hasattr(self, 'joint_axis'):
                    axis = self.joint_axis
                    axis_norm = axis.norm().item()
                    
                    metrics_dict.update({
                        "joint/axis_x": float(axis[0].item()),
                        "joint/axis_y": float(axis[1].item()),
                        "joint/axis_z": float(axis[2].item()),
                        "joint/axis_norm": float(axis_norm),
                    })
                    
                    # Axis drift from initial (assuming you have initial axis stored)
                    active_id = self.config.active_joint_id
                    # Compute initial axis from metadata
                    initial_axis_raw = F.normalize(self.obj_3d_seg.joint_axis.to(self.device), dim=0)
                    
                    # Angle between current and initial axis (cosine similarity)
                    axis_similarity = torch.dot(axis, initial_axis_raw).item()
                    axis_angle_diff = torch.acos(torch.clamp(torch.tensor(axis_similarity), -1.0, 1.0)).item()
                    
                    metrics_dict.update({
                        "joint/axis_similarity": float(axis_similarity),
                        "joint/axis_angle_diff_rad": float(axis_angle_diff),
                        "joint/axis_angle_diff_deg": float(axis_angle_diff * 180 / 3.14159),
                    })
                    
                    # Warn if axis has drifted significantly
                    if axis_angle_diff > 0.1 and self.step % 1000 == 0:  # > ~5.7 degrees
                        print(f"[Axis Drift - Step {self.step}] "
                            f"Angle diff: {axis_angle_diff * 180 / 3.14159:.1f}°, "
                            f"Current: [{axis[0].item():.3f}, {axis[1].item():.3f}, {axis[2].item():.3f}], "
                            f"Initial: [{initial_axis_raw[0].item():.3f}, {initial_axis_raw[1].item():.3f}, {initial_axis_raw[2].item():.3f}]")
                    
                    # Check if axis is properly normalized
                    norm_error = abs(axis_norm - 1.0)
                    metrics_dict["joint/axis_norm_error"] = float(norm_error)
                    if norm_error > 0.01 and self.step % 1000 == 0:
                        print(f"[Axis Normalization - Step {self.step}] Warning: norm={axis_norm:.6f} (should be 1.0)")
                
                # Per-frame correction statistics
                if hasattr(self, 'joint_angle_deltas'):
                    deltas = self.joint_angle_deltas
                    metrics_dict.update({
                        "joint/delta_mean": float(deltas.mean().item()),
                        "joint/delta_std": float(deltas.std().item()),
                        "joint/delta_max": float(deltas.max().item()),
                        "joint/delta_min": float(deltas.min().item()),
                        "joint/delta_abs_max": float(deltas.abs().max().item()),
                        "joint/delta_abs_mean": float(deltas.abs().mean().item()),
                    })
                    
                    # First frame delta (should be small since it anchors min_angle)
                    first_delta = deltas[0].item()
                    metrics_dict["joint/first_frame_delta"] = float(first_delta)
                    
                    if abs(first_delta) > 0.1 and self.step % 500 == 0:
                        print(f"[First Frame Delta - Step {self.step}] "
                            f"Warning: δ[0]={first_delta:.4f} (should be near 0 for stable anchoring)")
                
                # Current frame info
                if 'time' in batch:
                    time_val = float(batch['time'])
                    num_frames = len(self.joint_angles)
                    frame_idx = int(time_val * (num_frames - 1))
                    frame_idx = max(0, min(frame_idx, num_frames - 1))
                    
                    current_angle = self.joint_angles[frame_idx].item()
                    metrics_dict.update({
                        "joint/current_frame_idx": float(frame_idx),
                        "joint/current_frame_time": float(time_val),
                        "joint/current_frame_angle": float(current_angle),
                        "joint/current_frame_angle_degrees": float(current_angle * 180 / 3.14159),
                    })
                    
                    # Current frame's correction
                    if hasattr(self, 'joint_angle_deltas'):
                        current_delta = self.joint_angle_deltas[frame_idx].item()
                        current_prior = self.joint_angles_prior[frame_idx].item()
                        metrics_dict.update({
                            "joint/current_frame_delta": float(current_delta),
                            "joint/current_frame_prior": float(current_prior),
                        })

        # === Gradient Tracking for Pivot/Axis (every 100 steps) ===
        if self.step % 100 == 0:
            grad_metrics = {}
            
            # Joint parameter gradients
            if hasattr(self, 'joint_pivot') and self.joint_pivot.grad is not None:
                pivot_grad = self.joint_pivot.grad
                grad_metrics["gradients/joint_pivot_norm"] = float(pivot_grad.norm().item())
                grad_metrics["gradients/joint_pivot_mean"] = float(pivot_grad.abs().mean().item())
                grad_metrics["gradients/joint_pivot_max"] = float(pivot_grad.abs().max().item())
                grad_metrics["gradients/joint_pivot_x"] = float(pivot_grad[0].item())
                grad_metrics["gradients/joint_pivot_y"] = float(pivot_grad[1].item())
                grad_metrics["gradients/joint_pivot_z"] = float(pivot_grad[2].item())
                
                # Check if gradients are flowing
                if pivot_grad.norm().item() < 1e-8:
                    print(f"[Gradient Check - Step {self.step}] ⚠️  Pivot gradients near zero!")
            else:
                if hasattr(self, 'joint_pivot') and self.step % 500 == 0:
                    print(f"[Gradient Check - Step {self.step}] ⚠️  No pivot gradients!")
            
            if hasattr(self, 'joint_axis_raw') and self.joint_axis_raw.grad is not None:
                axis_grad = self.joint_axis_raw.grad
                grad_metrics["gradients/joint_axis_norm"] = float(axis_grad.norm().item())
                grad_metrics["gradients/joint_axis_mean"] = float(axis_grad.abs().mean().item())
                grad_metrics["gradients/joint_axis_x"] = float(axis_grad[0].item())
                grad_metrics["gradients/joint_axis_y"] = float(axis_grad[1].item())
                grad_metrics["gradients/joint_axis_z"] = float(axis_grad[2].item())
                
                if axis_grad.norm().item() < 1e-8:
                    print(f"[Gradient Check - Step {self.step}] ⚠️  Axis gradients near zero!")
            else:
                if hasattr(self, 'joint_axis_raw') and self.step % 500 == 0:
                    print(f"[Gradient Check - Step {self.step}] ⚠️  No axis gradients!")
            
            if hasattr(self, 'angle_range_log') and self.angle_range_log.grad is not None:
                range_grad = self.angle_range_log.grad
                grad_metrics["gradients/range_log_value"] = float(range_grad.item())
                grad_metrics["gradients/range_log_abs"] = float(abs(range_grad.item()))
                
                if abs(range_grad.item()) < 1e-8:
                    print(f"[Gradient Check - Step {self.step}] ⚠️  Range gradients near zero!")
            
            if hasattr(self, 'joint_angle_deltas') and self.joint_angle_deltas.grad is not None:
                delta_grad = self.joint_angle_deltas.grad
                grad_metrics["gradients/delta_norm"] = float(delta_grad.norm().item())
                grad_metrics["gradients/delta_mean"] = float(delta_grad.abs().mean().item())
                grad_metrics["gradients/delta_max"] = float(delta_grad.abs().max().item())
                
                # Count frames with significant gradients
                significant_grads = (delta_grad.abs() > 1e-8).sum().item()
                grad_metrics["gradients/delta_frames_with_grad"] = float(significant_grads)
                grad_metrics["gradients/delta_frames_grad_pct"] = float(significant_grads / len(delta_grad) * 100)
            
            # Gaussian parameter gradients (for comparison)
            if hasattr(self, 'gauss_params'):
                if self.gauss_params['means'].grad is not None:
                    grad_metrics["gradients/gauss_means_norm"] = float(
                        self.gauss_params['means'].grad.norm().item()
                    )
                if self.gauss_params['opacities'].grad is not None:
                    grad_metrics["gradients/gauss_opacities_norm"] = float(
                        self.gauss_params['opacities'].grad.norm().item()
                    )
            
            metrics_dict.update(grad_metrics)
        # === Learning Rate Tracking (every 100 steps) ===
        if self.step % 100 == 0 and hasattr(self, 'optimizers'):
            lr_metrics = {}
            for name, opt in self.optimizers.items():
                if 'joint' in name:
                    for i, param_group in enumerate(opt.param_groups):
                        lr_key = f"learning_rates/{name}_group{i}" if len(opt.param_groups) > 1 else f"learning_rates/{name}"
                        lr_metrics[lr_key] = float(param_group['lr'])
            metrics_dict.update(lr_metrics)

        # === Training Coverage Tracking ===
        if hasattr(self, 'joint_t_values') and 'time' in batch:
            # Track which frames have been trained
            if not hasattr(self, '_trained_frames'):
                self._trained_frames = set()
                self._frame_visit_count = torch.zeros(len(self.joint_t_values), device=self.device)
            
            time_val = float(batch['time'])
            frame_idx = int(time_val * (len(self.joint_t_values) - 1))
            self._trained_frames.add(frame_idx)
            self._frame_visit_count[frame_idx] += 1
            
            if self.step % 100 == 0:
                coverage_metrics = {
                    "training/unique_frames_seen": float(len(self._trained_frames)),
                    "training/coverage_pct": float(len(self._trained_frames) / len(self.joint_t_values) * 100),
                    "training/avg_frame_visits": float(self._frame_visit_count.mean().item()),
                    "training/min_frame_visits": float(self._frame_visit_count.min().item()),
                    "training/max_frame_visits": float(self._frame_visit_count.max().item()),
                }
                metrics_dict.update(coverage_metrics)

        return metrics_dict