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

    depth_lambda: float = 0.4
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
    opacity_lambda_canon: float = 5e-4

    joint_correction_lambda: float = 0.01
    active_joint_id: str = "joint_0"
    training_mode: str = field(default="articulation")
    """Training mode: 'articulation' or 'recovery'"""




class ArtiSplatfactoModel(SplatfactoModel):    

    config: ArtiSplatfactoModelConfig

    def __init__(self, *args, **kwargs):
        self.metadata = kwargs.pop("metadata", {}) or {}
        super().__init__(*args, **kwargs)
        

        self.joint_type = self.obj_3d_seg.joint_type 


        if self.config.use_depth:
            self.depth_loss_fn = DepthLoss()


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

        # # Use ParameterDict like base model
        # self.gauss_params = torch.nn.ParameterDict({
        #     "means":         torch.nn.Parameter(torch.empty((0, 3), device=device, requires_grad=True)),
        #     "scales":        torch.nn.Parameter(torch.empty((0, 3), device=device, requires_grad=True)),
        #     "quats":         torch.nn.Parameter(torch.empty((0, 4), device=device, requires_grad=True)),
        #     "features_dc":   torch.nn.Parameter(torch.empty((0, 3), device=device, requires_grad=True)),
        #     "features_rest": torch.nn.Parameter(torch.empty((0, dim_sh - 1, 3), device=device, requires_grad=True)),
        #     "opacities":     torch.nn.Parameter(torch.empty((0, 1), device=device, requires_grad=True)),
        # })

        # # Canonical gaussians (trainable)
        # self.gauss_params_canonical = torch.nn.ParameterDict({
        #     "means":         torch.nn.Parameter(torch.empty((0, 3), device=device, requires_grad=True)),
        #     "scales":        torch.nn.Parameter(torch.empty((0, 3), device=device, requires_grad=True)),
        #     "quats":         torch.nn.Parameter(torch.empty((0, 4), device=device, requires_grad=True)),
        #     "features_dc":   torch.nn.Parameter(torch.empty((0, 3), device=device, requires_grad=True)),
        #     "features_rest": torch.nn.Parameter(torch.empty((0, dim_sh - 1, 3), device=device, requires_grad=True)),
        #     "opacities":     torch.nn.Parameter(torch.empty((0, 1), device=device, requires_grad=True)),
        # })

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

        # Create storage for the CURRENT active joint's parameters
        self.all_joint_params[active_id] = torch.nn.ParameterDict()

        initial_pivot = self.obj_3d_seg.joint_pivot.to(device)
        initial_axis = F.normalize(self.obj_3d_seg.joint_axis.to(device), dim=0)
        initial_max_angle = self.obj_3d_seg.joint_limits[1]

        # Store initial pivot as a buffer associated with the active joint (optional, but cleaner)
        self.register_buffer(f'initial_joint_pivot_{active_id}', initial_pivot.clone())
        
        # Joint geometry parameters (will be added to the active ParameterDict)
        self.all_joint_params[active_id]["pivot"] = torch.nn.Parameter(initial_pivot.clone(), requires_grad=True)
        self.all_joint_params[active_id]["axis_raw"] = torch.nn.Parameter(initial_axis.clone(), requires_grad=True)
        self.all_joint_params[active_id]["max_angle"] = torch.nn.Parameter(
            torch.tensor(initial_max_angle, device=device, dtype=torch.float32),
            requires_grad=True
        )
        
        # Handle per-frame joint angle corrections (must be registered on the model)
        joint_angles_meta = self.metadata.get("joint_angles", [])
        num_frames = len(joint_angles_meta)

        if num_frames > 0:
            prior_tensor = torch.as_tensor(joint_angles_meta, device=device, dtype=torch.float32)
            self.register_buffer(f'joint_angles_prior_{active_id}', prior_tensor)
            
            self.all_joint_params[active_id]["angle_deltas"] = torch.nn.Parameter(torch.zeros(num_frames, device=device))
            
            joint_min, joint_max = self.obj_3d_seg.joint_limits[0], self.obj_3d_seg.joint_limits[1]
            self.register_buffer(f'joint_limits_{active_id}', torch.tensor([joint_min, joint_max], device=device))

        # --- Backward Compatibility Pointers ---
        # Create pointers to the active joint's parameters for existing methods 
        # like get_gaussian_param_groups, get_outputs, etc.
        self.joint_pivot = self.all_joint_params[active_id]["pivot"]
        self.joint_axis_raw = self.all_joint_params[active_id]["axis_raw"]
        self.max_joint_angle = self.all_joint_params[active_id]["max_angle"]
        if num_frames > 0:
            self.joint_angle_deltas = self.all_joint_params[active_id]["angle_deltas"]
            self.joint_angles_prior = getattr(self, f'joint_angles_prior_{active_id}')
            self.joint_limits = getattr(self, f'joint_limits_{active_id}')

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
        """Final angles = prior + corrections, clamped to limits"""
        if hasattr(self, 'joint_angle_deltas'):
            raw_angles = self.joint_angles_prior + self.joint_angle_deltas
            return torch.clamp(raw_angles, self.joint_limits[0], self.joint_limits[1])
        # Fallback/Eval logic
        return torch.tensor([0.0], device=self.device)

    @property  
    def joint_angles_normalized(self):
        """Normalized angles for stable optimization [0, 1]"""
        angles = self.joint_angles
        joint_min, joint_max = self.joint_limits[0], self.joint_limits[1]
        return (angles - joint_min) / (joint_max - joint_min)

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
        - Enable ALL radiance (features_dc, features_rest, opacities) for all joints and background.
        - Freeze ALL joint parameters (pivot, axis, deltas, etc.) for all joints.
        """
        GEOMETRY_PARAMS = ["means", "scales", "quats"]
        RADIANCE_PARAMS = ["features_dc", "features_rest", "opacities"]
        JOINT_PARAMS = ["pivot", "axis_raw", "max_angle", "angle_deltas"]
        
        CONSOLE.print("\n" + "="*70)
        CONSOLE.print("[bold yellow]CONFIGURING MODEL FOR RECOVERY STAGE (RADIANCE OPTIMIZATION)[/bold yellow]")
        CONSOLE.print("="*70)
        
        # --- 1. Configure ALL Joint-Specific Gaussians (Freeze Geo, Unfreeze Radiance) ---
        for joint_id in self.all_gauss_params_obj.keys():
            CONSOLE.print(f"\n[yellow]Configuring Gaussians for {joint_id}[/yellow]")
            
            for param_name in GEOMETRY_PARAMS:
                # **Fix**: Freeze Geometry
                self.all_gauss_params_obj[joint_id][param_name].requires_grad = False
                self.all_gauss_params_canon[joint_id][param_name].requires_grad = False
            CONSOLE.print(f"  ✓ Froze Geometry ({GEOMETRY_PARAMS}) for Object/Canonical")

            for param_name in RADIANCE_PARAMS:
                # Unfreeze Radiance
                self.all_gauss_params_obj[joint_id][param_name].requires_grad = True
                self.all_gauss_params_canon[joint_id][param_name].requires_grad = True
            CONSOLE.print(f"  [green]✓ Enabled Radiance ({RADIANCE_PARAMS}) for Object/Canonical[/green]")
        
        # --- 2. Configure Background Gaussians (Freeze Geo, Unfreeze Radiance) ---
        CONSOLE.print("\n[yellow]Configuring Background Gaussians[/yellow]")
        for param_name in GEOMETRY_PARAMS:
            if param_name in self.gauss_params_fixed:
                self.gauss_params_fixed[param_name].requires_grad = False
        CONSOLE.print(f"  ✓ Froze Geometry ({GEOMETRY_PARAMS}) for Background")
        
        for param_name in RADIANCE_PARAMS:
            if param_name in self.gauss_params_fixed:
                self.gauss_params_fixed[param_name].requires_grad = True
        CONSOLE.print(f"  [bold green]✓ Enabled Radiance ({RADIANCE_PARAMS}) for Background[/bold green]")

        # --- 3. Configure ALL Joint-Specific Parameters (Always Frozen) ---
        CONSOLE.print("\n[cyan]Freezing ALL joint geometry parameters...[/cyan]")
        for joint_id in self.all_joint_params.keys():
            for param_name in JOINT_PARAMS:
                if param_name in self.all_joint_params[joint_id]:
                    self.all_joint_params[joint_id][param_name].requires_grad = False
                    CONSOLE.print(f"  ✓ Froze {joint_id} {param_name}")
        
        # --- 4. Final Summary ---
        CONSOLE.print("\n" + "="*70)
        CONSOLE.print("[bold green]RECOVERY STAGE CONFIGURATION COMPLETE[/bold green]")
        CONSOLE.print("ALL Geometry: FROZEN | ALL Radiance: TRAINABLE | ALL Articulation: FROZEN")
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

        # --- 2. Active Joint Articulation Parameters (Pivot, Axis, Deltas) ---
        if active_id in self.all_joint_params:
            joint_params = self.all_joint_params[active_id]
            
            # Pivot, Axis, Max Angle
            joint_geom_params = {
                "joint_pivot": "pivot",
                "joint_axis": "axis_raw",
                "max_joint_angle": "max_angle",
            }
            
            for opt_name, internal_name in joint_geom_params.items():
                if internal_name in joint_params and joint_params[internal_name].requires_grad:
                    groups[opt_name] = [joint_params[internal_name]]
                    CONSOLE.print(f"  ✓ {opt_name}: {joint_params[internal_name].shape}")

            # Joint Corrections
            if "angle_deltas" in joint_params and joint_params["angle_deltas"].requires_grad:
                groups["joint_corrections"] = [joint_params["angle_deltas"]]
                CONSOLE.print(f"  ✓ joint_corrections: {joint_params['angle_deltas'].shape}")

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
        From a full-scene checkpoint: split into trainable object + fixed background.
        This runs only for the first joint (e.g., 'joint_0').
        """
        print("Initializing from full scene: partitioning Gaussians...")

        active_id = self.config.active_joint_id
        
        print(f"obj mask points for {active_id}:", self.obj_3d_seg)

        GAUSS = ["means", "scales", "quats", "features_dc", "features_rest", "opacities"]
        device = self.device

        all_means = state_dict["gauss_params.means"].to(device)
        obj_mask = self.obj_3d_seg.query_refine(all_means, grow=3, thresh=0.01, bbox_margin=0.01).to(torch.bool).cpu()
        bg_mask  = ~obj_mask

        print(f"check the mask {obj_mask.sum()} object points, {bg_mask.sum()} background points")
        # import pdb; pdb.set_trace()
        
        # --- Create new ParameterDicts for the active joint ---
        self.all_gauss_params_obj[active_id] = torch.nn.ParameterDict()
        self.all_gauss_params_canon[active_id] = torch.nn.ParameterDict()

        for p in GAUSS:
            subset_obj = state_dict[f"gauss_params.{p}"][obj_mask].to(device)
            subset_bg = state_dict[f"gauss_params.{p}"][bg_mask].to(device)
            
            # Object Gaussians (trainable for active joint)
            param_obj = torch.nn.Parameter(subset_obj.clone().detach(), requires_grad=True)
            self.all_gauss_params_obj[active_id][p] = param_obj
            
            # Canonical Gaussians (trainable for active joint)
            param_canon = torch.nn.Parameter(subset_obj.clone().detach(), requires_grad=True)
            self.all_gauss_params_canon[active_id][p] = param_canon
            
            # Fixed background gaussians (non-trainable)
            param_fixed = torch.nn.Parameter(subset_bg.clone().detach(), requires_grad=False)
            self.gauss_params_fixed[p] = param_fixed
            
            # Set pointers for backward compatibility (only for the active joint)
            if p == "means":
                self.gauss_params = self.all_gauss_params_obj[active_id]
                self.gauss_params_canonical = self.all_gauss_params_canon[active_id]
            
            print(f"[{active_id} Object] {p}: {param_obj.shape}, requires_grad={param_obj.requires_grad}")
            print(f"[{active_id} Canon] {p}: {param_canon.shape}, requires_grad={param_canon.requires_grad}")


        n_obj = self.all_gauss_params_obj[active_id]["means"].shape[0]
        n_canon = self.all_gauss_params_canon[active_id]["means"].shape[0]
        n_bg = self.gauss_params_fixed["means"].shape[0]

        print(f"Partitioning complete. {active_id} Obj: {n_obj}, {active_id} Canon: {n_canon}, Fixed BG: {n_bg}")


    def load_state_dict(self, state_dict: Dict[str, torch.Tensor], **kwargs):
        print(f"Loading state_dict (Training mode: {self.training})")
        assert self.config.obj_mask_file is not None and self.config.obj_mask_file.exists()

        active_id = self.config.active_joint_id
        GAUSS = ["means", "scales", "quats", "features_dc", "features_rest", "opacities"]
        device = self.device
        
        # --- Helper Function: Identify Legacy Parameters ---
        def is_legacy_single_joint_param(key: str) -> bool:
            """Dynamically detect old single-joint parameter format."""
            legacy_base_names = [
                "joint_pivot", "joint_axis_raw", "max_joint_angle", "joint_angle_deltas"
            ]
            
            if key in legacy_base_names:
                return True
            
            if re.match(r"joint_angles_prior_joint_\d+$", key):
                return True
            if re.match(r"joint_limits_joint_\d+$", key):
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
                # NOTE: We do NOT include gauss_params.means because it only contains
                # the previous active joint's Gaussians, which we already loaded above
                print(f"Reconstructing full scene for partitioning...")
                
                combined_data = {p: [] for p in GAUSS}
                
                # Add background
                if hasattr(self, 'gauss_params_fixed') and 'means' in self.gauss_params_fixed:
                    for p in GAUSS:
                        combined_data[p].append(self.gauss_params_fixed[p].data)
                    bg_count = self.gauss_params_fixed['means'].shape[0]
                    print(f"  Added {bg_count:,} background Gaussians")
                
                # Add all previous joints (these are the complete movable body so far)
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
            
            print(f"✓ Partitioned {self.gauss_params['means'].shape[0]:,} Gaussians for {active_id}")
            print(f"{'='*70}\n")

        # --- 4. Load Joint Articulation Parameters ---
        
        FRAME_DEPENDENT_PARAMS = {"angle_deltas"}
        
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
                
                # Skip frame-dependent parameters if shape mismatch
                if param_name in FRAME_DEPENDENT_PARAMS:
                    if param_name in self.all_joint_params[joint_id]:
                        current_shape = self.all_joint_params[joint_id][param_name].shape
                        checkpoint_shape = tensor.shape
                        
                        if current_shape != checkpoint_shape:
                            print(f"⚠️  Skipping {key}: shape mismatch "
                                f"(checkpoint: {checkpoint_shape}, current: {current_shape}). "
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
        
        # --- 5. Set Active Joint Pointers ---
        
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
        
        if active_id in self.all_joint_params:
            self.joint_pivot = self.all_joint_params[active_id]["pivot"]
            self.joint_axis_raw = self.all_joint_params[active_id]["axis_raw"]
            self.max_joint_angle = self.all_joint_params[active_id]["max_angle"]
            
            if "angle_deltas" in self.all_joint_params[active_id]:
                self.joint_angle_deltas = self.all_joint_params[active_id]["angle_deltas"]
                self.joint_angles_prior = getattr(self, f'joint_angles_prior_{active_id}', None)
                self.joint_limits = getattr(self, f'joint_limits_{active_id}', None)
        
        # --- 6. Load General Non-Gaussian State ---
        
        non_gauss_state = {
            k: v for k, v in state_dict.items() 
            if not (
                k.startswith("gauss_params.") or 
                k.startswith("gauss_params_canonical.") or 
                k.startswith("gauss_params_fixed.") or 
                k.startswith("all_gauss_params_") or
                k.startswith("all_joint_params.") or
                is_legacy_single_joint_param(k)
            )
        }
        
        super().load_state_dict(non_gauss_state, strict=False)
        
        self.step = 0
        self.configure_training_stage()
        
        # --- 7. Status Report ---
        print(f"\n{'='*60}")
        print(f"✓ Checkpoint Load Complete")
        print(f"{'='*60}")
        print(f"Active Joint: {active_id}")
        print(f"Training Mode: {self.config.training_mode}")
        print(f"Active Gaussians: {self.gauss_params['means'].shape[0]:,}")
        
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
                params_frozen = not self.all_joint_params[joint_id][param_names[0]].requires_grad
                param_status = f"{len(param_names)} params ({'FROZEN' if params_frozen else 'TRAINABLE'})"
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
                loss_dict["depth_loss"] = depth_loss * 0.1

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
        
        # === Depth Loss ===
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
            
            with torch.no_grad():
                from arti_splatfacto.utils.depth_loss import compute_scale_and_shift
                scale, shift = compute_scale_and_shift(depth_out_loss, depth_gt_loss, mask_loss)
            
            if self.config.depth_debug_vis and self.step % 1000 == 0:
                save_depth_debug(self.step, depth_out, depth_gt, mask, 
                                scale=scale.item(), shift=shift.item())
            
            loss_dict["depth_loss"] = self.config.depth_lambda * depth_loss
            
            # Debug logging
            if self.step % 2000 == 0:
                print(f"[Depth - Step {self.step}] Loss: {depth_loss:.6f}, "
                    f"Scale: {scale.item():.4f}, Shift: {shift.item():.4f}")

        if hasattr(self, "joint_angle_deltas") and self.joint_angle_deltas is not None:
            # Get current frame info
            time_val = batch['time']
            num_frames = len(self.joint_angle_deltas)
            frame_idx = int(time_val * (num_frames - 1))
            frame_idx = max(0, min(frame_idx, num_frames - 1))
            
            # Soft regularization on corrections (spring back to prior)
            delta_regularization = torch.mean(self.joint_angle_deltas**2)
            loss_dict["joint_correction_reg"] = self.config.joint_correction_lambda * delta_regularization
            
            # Optional: Temporal smoothness between adjacent frame corrections
            smooth_penalty = torch.tensor(0.0, device=self.device)
            if frame_idx > 0:
                prev_delta = self.joint_angle_deltas[frame_idx - 1]
                current_delta = self.joint_angle_deltas[frame_idx]
                smooth_penalty = (current_delta - prev_delta) ** 2
            
            # Very light temporal smoothness (corrections should change gradually)
            temporal_reg = 0.01 * smooth_penalty
            loss_dict["joint_temporal_reg"] = temporal_reg
            
            # Debug logging
            if self.step % 100 == 0:
                current_correction = self.joint_angle_deltas[frame_idx].item()
                max_correction = torch.abs(self.joint_angle_deltas).max().item()
                mean_correction = torch.abs(self.joint_angle_deltas).mean().item()
                
                # Current physical angle
                current_angle = self.joint_angles[frame_idx].item()
                prior_angle = self.joint_angles_prior[frame_idx].item()
                
                print(f"[Joint Corrections - Step {self.step}]")
                print(f"  Frame {frame_idx}: Prior={prior_angle:.3f}, Final={current_angle:.3f}, Δ={current_correction:.4f}")
                print(f"  Correction stats: Max={max_correction:.4f}, Mean={mean_correction:.4f}")
            
        if self.config.use_opacity_regularization and self.training:

            obj_opacity = torch.sigmoid(self.gauss_params["opacities"])
            canon_opacity = torch.sigmoid(self.gauss_params_canonical["opacities"])

            loss_obj_opacity = opacity_loss(obj_opacity)
            loss_canon_opacity = opacity_loss(canon_opacity)

            loss_dict["opacity_reg"] = (
                self.config.opacity_lambda_obj * loss_obj_opacity +
                self.config.opacity_lambda_canon * loss_canon_opacity
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




    
    def get_joint_angle_for_camera(self, camera: Cameras):
        """Return a differentiable joint angle tensor."""
        if hasattr(camera, "metadata") and camera.metadata is not None:
            angle_val = camera.metadata.get("joint_angles", None)
            if angle_val is not None:
                if torch.is_tensor(angle_val):
                    return angle_val.to(self.device).float()
                else:
                    return torch.tensor([float(angle_val)], device=self.device)

        # --- Case 2: time-based interpolation (training / sequences) ---
        if hasattr(camera, "times") and camera.times is not None:
            time_val = camera.times.flatten()[0]
            num_frames = len(self.joint_angles)
            idx_f = time_val * (num_frames - 1)
            idx0 = torch.floor(idx_f).long().clamp(0, num_frames - 2)
            idx1 = idx0 + 1
            w = idx_f - idx0.float()
            angle0 = self.joint_angles[idx0]
            angle1 = self.joint_angles[idx1]
            angle = (1.0 - w) * angle0 + w * angle1
            return angle

        # --- Default ---
        return torch.tensor([0.0], device=self.device, dtype=torch.float32)
    
    def get_gaussians_for_render(self, camera: Cameras) -> Dict[str, torch.Tensor]:
        """
        Prepares Gaussians for rendering by applying articulation transforms and 
        filtering parameters based on the current training mode.
        
        Render Order (Articulation Mode): [Active Obj, Active Canon]
        Render Order (Recovery/Eval Mode): [Active Obj, Other Objs, Active Canon, Other Canons, BG]
        """
        GAUSS = ["means", "scales", "quats", "features_dc", "features_rest", "opacities"]
        active_id = self.config.active_joint_id
        
        # --- 1. Get Transform for Active Joint ---
        # The angle is calculated using the active joint's parameter structure
        joint_angle_active = self.get_joint_angle_for_camera(camera)
        
        # Articulate the active joint's Object Gaussians (using the pointers)
        articulated_obj_active = apply_articulation_to_optimizer_params(self, joint_angle_active)
        
        # Get the active joint's Canonical Gaussians
        canon_active = {k: self.gauss_params_canonical[k].data if not self.training else self.gauss_params_canonical[k] for k in self.gauss_params_canonical.keys()}

        # --- 2. Determine Sets to Render ---
        is_articulation_training = (self.training and self.config.training_mode == "articulation")
        
        # Lists to hold the final groups to be combined
        obj_sets_to_combine = [articulated_obj_active]
        canon_sets_to_combine = [canon_active]
        
        # Collect ALL Other Articulated Object and Canonical Gaussians (FROZEN/PASSIVE)
        if not is_articulation_training:
            # Only include frozen sets during Recovery/Eval
            for joint_id in sorted(self.all_gauss_params_obj.keys()):
                if joint_id == active_id:
                    continue

                obj_params = self.all_gauss_params_obj[joint_id]
                canon_params = self.all_gauss_params_canon[joint_id]
                
                # Render the frozen articulated state for previous joints
                obj_sets_to_combine.append({k: obj_params[k].data if not self.training else obj_params[k] for k in obj_params.keys()})
                
                # Render the frozen canonical state for previous joints
                canon_sets_to_combine.append({k: canon_params[k].data if not self.training else canon_params[k] for k in canon_params.keys()})
            
        # --- 3. Combine in Render Order: [All Objs] then [All Canons] ---
        
        # Combine all articulated object groups (Active Obj is always first)
        combined_obj_params = {p: torch.cat([d[p] for d in obj_sets_to_combine], dim=0) for p in GAUSS}
        
        # Combine all canonical groups (Active Canon is always first)
        combined_canon_params = {p: torch.cat([d[p] for d in canon_sets_to_combine], dim=0) for p in GAUSS}

        # Final combination: All Articulated Objects + All Canonicals
        combined_params = {p: torch.cat([combined_obj_params[p], combined_canon_params[p]], dim=0) for p in GAUSS}
        
        # --- 4. Store Counts for step_post_backward & Debugging (CRITICAL) ---
        self.n_active_obj = articulated_obj_active['means'].shape[0]
        self.n_obj_total = combined_obj_params['means'].shape[0]
        self.n_canon_total = combined_canon_params['means'].shape[0]

        # --- 5. Add Background (Only in Recovery/Eval) ---
        include_background = (not self.training or self.config.training_mode == "recovery")
        
        if include_background:
            if (hasattr(self, "gauss_params_fixed") and self.gauss_params_fixed and self.gauss_params_fixed["means"].shape[0] > 0):
                full_scene_params = {}
                for name in combined_params.keys():
                    combined_tensor = combined_params[name]
                    bg_tensor = self.gauss_params_fixed[name]
                    
                    if not self.training:
                        bg_tensor = bg_tensor.data
                    if combined_tensor.device != bg_tensor.device:
                        bg_tensor = bg_tensor.to(combined_tensor.device)
                    
                    full_scene_params[name] = torch.cat([combined_tensor, bg_tensor], dim=0)
                
                return full_scene_params
            
        # Return only the combined articulated/canonical sets (either filtered or full)
        return combined_params

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

        # === Joint Parameter Tracking ===
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
                
                # Max angle parameter
                if hasattr(self, 'max_joint_angle'):
                    metrics_dict["joint/max_angle_param"] = float(self.max_joint_angle.item())
                    metrics_dict["joint/max_angle_param_degrees"] = float(self.max_joint_angle.item() * 180 / 3.14159)
                
                # T-values statistics (progress from closed to open)
                if hasattr(self, 'joint_t_values'):
                    t_vals = self.joint_t_values
                    metrics_dict.update({
                        "joint/t_mean": float(t_vals.mean().item()),
                        "joint/t_std": float(t_vals.std().item()),
                        "joint/t_min": float(t_vals.min().item()),
                        "joint/t_max": float(t_vals.max().item()),
                    })
                
                # Pivot tracking
                if hasattr(self, 'joint_pivot'):
                    metrics_dict.update({
                        "joint/pivot_x": float(self.joint_pivot[0].item()),
                        "joint/pivot_y": float(self.joint_pivot[1].item()),
                        "joint/pivot_z": float(self.joint_pivot[2].item()),
                    })
                    
                    # Pivot drift from initialization
                    if hasattr(self, 'initial_joint_pivot'):
                        pivot_drift = (self.joint_pivot - self.initial_joint_pivot).norm().item()
                        metrics_dict["joint/pivot_drift"] = float(pivot_drift)
                
                # Axis tracking
                if hasattr(self, 'joint_axis'):
                    axis = self.joint_axis
                    metrics_dict.update({
                        "joint/axis_x": float(axis[0].item()),
                        "joint/axis_y": float(axis[1].item()),
                        "joint/axis_z": float(axis[2].item()),
                        "joint/axis_norm": float(axis.norm().item()),  # Should be ~1.0
                    })
                
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
                    
                    if hasattr(self, 'joint_t_values'):
                        current_t = self.joint_t_values[frame_idx].item()
                        metrics_dict["joint/current_frame_t"] = float(current_t)

        # === Gradient Tracking (every 100 steps) ===
        if self.step % 100 == 0:
            grad_metrics = {}
            
            # Joint parameter gradients
            if hasattr(self, 'joint_pivot') and self.joint_pivot.grad is not None:
                grad_metrics["gradients/joint_pivot_norm"] = float(self.joint_pivot.grad.norm().item())
                grad_metrics["gradients/joint_pivot_mean"] = float(self.joint_pivot.grad.abs().mean().item())
                grad_metrics["gradients/joint_pivot_max"] = float(self.joint_pivot.grad.abs().max().item())
            
            if hasattr(self, 'joint_axis_raw') and self.joint_axis_raw.grad is not None:
                grad_metrics["gradients/joint_axis_norm"] = float(self.joint_axis_raw.grad.norm().item())
                grad_metrics["gradients/joint_axis_mean"] = float(self.joint_axis_raw.grad.abs().mean().item())
            
            if hasattr(self, 'max_joint_angle') and self.max_joint_angle.grad is not None:
                grad_metrics["gradients/max_angle_value"] = float(self.max_joint_angle.grad.item())
                grad_metrics["gradients/max_angle_abs"] = float(abs(self.max_joint_angle.grad.item()))
            
            if hasattr(self, 'joint_t_raw') and self.joint_t_raw.grad is not None:
                t_grad = self.joint_t_raw.grad
                grad_metrics["gradients/t_raw_norm"] = float(t_grad.norm().item())
                grad_metrics["gradients/t_raw_mean"] = float(t_grad.abs().mean().item())
                grad_metrics["gradients/t_raw_max"] = float(t_grad.abs().max().item())
                
                # Count frames with significant gradients
                significant_grads = (t_grad.abs() > 1e-8).sum().item()
                grad_metrics["gradients/t_frames_with_grad"] = float(significant_grads)
                grad_metrics["gradients/t_frames_grad_pct"] = float(significant_grads / len(t_grad) * 100)
            
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