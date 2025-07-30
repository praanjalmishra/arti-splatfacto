from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Type, Optional, Union
from pathlib import Path

try:
    from gsplat.rendering import rasterization
except ImportError:
    print("Please install gsplat>=1.0.0")

from importlib_metadata import metadata
import torch
import torch.nn.functional as F
from gsplat.strategy import DefaultStrategy, MCMCStrategy
from nerfstudio.models.splatfacto import SplatfactoModelConfig, SplatfactoModel
from nerfstudio.engine.optimizers import Optimizers
from nerfstudio.utils.spherical_harmonics import RGB2SH, SH2RGB, num_sh_bases
from nerfstudio.model_components.lib_bilagrid import BilateralGrid, color_correct, slice, total_variation_loss
from arti_splatfacto.obj_3d_seg import Object3DSeg
from arti_splatfacto.scene_3d import Scene3D
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.utils.misc import torch_compile
from arti_splatfacto.gauss_utils import transform_gaussians, sample_gaussians, fit_gaussian_batch, rot2quat
from pytorch3d.transforms import quaternion_multiply

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

    q1 = torch.where(dot < 0, -q1, q1)
    dot = torch.clamp(dot.abs(), 1e-6, 1.0)

    theta_0 = torch.acos(dot)  
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

    # background_model: SplatfactoModelConfig = field(default_factory=lambda: SplatfactoModelConfig)
    # object_model: SplatfactoModelConfig = field(default_factory=lambda: SplatfactoModelConfig)

    # # obj_mask_file: Optional[Path] = None
    # fourier_features_dim: int = 5
    # fourier_features_scale: int = 1


class ArtiSplatfactoModel(SplatfactoModel):    

    config: ArtiSplatfactoModelConfig

    def populate_modules(self):

        super().populate_modules()

        # Clean up: deregister inherited Gaussian parameter attributes (e.g., self.means, self.quats, etc.)
        # for gs_param in list(self.gauss_params.keys()):
        #     self._parameters.pop(gs_param, None)   # safe if it was a registered param
        #     self._buffers.pop(gs_param, None)      # safe if it was a registered buffer
        #     if hasattr(self, gs_param):
        #         delattr(self, gs_param)            # now safe to delete
        #     setattr(self, gs_param, None)  

        def make_param(shape, requires_grad=True):
            return torch.nn.Parameter(torch.zeros(shape).float().cuda(), requires_grad=requires_grad)

        dim_sh = num_sh_bases(self.config.sh_degree)

        # Canonical (trainable object Gaussians in canonical frame)
        self.gauss_params_canonical = torch.nn.ParameterDict({
            "means":         make_param((0, 3), requires_grad=True),
            "scales":        make_param((0, 3), requires_grad=True),
            "quats":         make_param((0, 4), requires_grad=True),
            "features_dc":   make_param((0, 3), requires_grad=True),
            "features_rest": make_param((0, dim_sh - 1, 3), requires_grad=True),
            "opacities":     make_param((0, 1), requires_grad=True),
            })
        
        self.gauss_params_fixed = torch.nn.ParameterDict({
            "means":         make_param((0, 3), requires_grad=False),
            "scales":        make_param((0, 3), requires_grad=False),
            "quats":         make_param((0, 4), requires_grad=False),
            "features_dc":   make_param((0, 3), requires_grad=False),
            "features_rest": make_param((0, dim_sh - 1, 3), requires_grad=False),
            "opacities":     make_param((0, 1), requires_grad=False),
        })

        metadata = self.kwargs.get("metadata", {})

        if "scene_path" not in metadata:
            raise ValueError("Scene path not found in metadata. Please provide a valid scene path.")
        else:
            print(f"Loading scene from path: {metadata['scene_path']}")
            device = self.gauss_params_canonical["means"].device
            self.scene = Scene3D.from_directory(metadata["scene_path"], device=device)

        
        # assert "scene" in self.kwargs["metadata"], "Scene is not available in metadata!!!"
        # self.scene: Scene3D = self.kwargs["metadata"]["scene"]
        self.register_buffer("gaussian_obj_ids", torch.empty(0, dtype=torch.long))

        # self.all_models = torch.nn.ModuleDict()
        
        # setup bg model
        # self.config.background_model.sh_degree = self.config.sh_degree
        # self.all_models["background"] = self.config.background_model.setup(
        #     scene_box=self.scene_box,
        #     num_train_data=self.num_train_data,
        #     **self.kwargs
        # )

        # # setup object models
        # for obj_id, obj in self.scene.objects.items():
        #     self.config.object_model.sh_degree = self.config.sh_degree
        #     obj_model = self.config.object_model.setup(
        #         scene_box=self.scene_box,
        #         num_train_data=self.num_train_data,
        #         object_id=obj_id,
        #         **self.kwargs
        #     )
        #     self.all_models[f"object_{obj_id}"] = obj_model

        # setup object model
        # print(f"✅ Scene3D with {len(self.scene.objects)} objects loaded.")

    # @property
    # def background_model(self) -> SplatfactoModel:
    #     return self.all_models["background"]

    # def get_object_model(self, object_id: int) -> SplatfactoModel:
    #     return self.all_models[f"object_{object_id}"]

    def state_dict(self, *args, **kwargs):
        """
        Returns the state dictionary for the model.
        
        The super() call automatically includes all registered ParameterDicts 
        (gauss_params_canonical, gauss_params_fixed) and buffers (gaussian_obj_ids).
        """
        return super().state_dict(*args, **kwargs)
    
    def _initialize_and_partition(self, state_dict: Dict[str, torch.Tensor]):
        """
        Initializes the model from a full-scene checkpoint, partitioning Gaussians
        into trainable objects (in canonical space) and a fixed background.
        """
        print("🚀 Initializing from full scene: partitioning Gaussians...")

        scene: Scene3D = self.scene
        all_means = state_dict["gauss_params.means"].to(self.device)
        N = all_means.shape[0]

        # Use a list to collect Gaussian parameters for each object
        obj_gaussians_list = []
        obj_ids_list = []
        full_obj_mask = torch.zeros(N, dtype=torch.bool, device=self.device)

        GAUSSIAN_PARAM_NAMES = ["means", "scales", "quats", "features_dc", "features_rest", "opacities"]
        # Note: state_dict keys from splatfacto are prefixed with `_`
        STATE_DICT_MAP = {k: f"gauss_params.{k}" for k in GAUSSIAN_PARAM_NAMES}
        STATE_DICT_MAP["features_dc"] = "gauss_params.features_dc" # Has different shape, handle carefully
        STATE_DICT_MAP["features_rest"] = "gauss_params.features_rest"


        for obj_id, obj in scene.objects.items():
            # print(f"  🔹 Object ID: {obj_id} | Type: {type(obj)} | Content: {obj}")
            obj_mask = obj.query(all_means, dilate=False) # Use Scene3D's query method per object
            if obj_mask.sum() == 0:
                print(f"⚠️ [Warning] Object {obj_id} has no Gaussians assigned.")
                continue

            print(f"Found {obj_mask.sum()} Gaussians for object {obj_id}.")
            full_obj_mask |= obj_mask

            # Extract Gaussians belonging to this object from the world frame
            obj_data_world = {
                name: state_dict[STATE_DICT_MAP[name]].to(self.device)[obj_mask] for name in GAUSSIAN_PARAM_NAMES
            }

            # Reshape features_dc if necessary (splatfacto stores it as (N, 3))
            if obj_data_world["features_dc"].dim() == 2:
                obj_data_world["features_dc"] = obj_data_world["features_dc"].unsqueeze(1)


            # Get the transform to move Gaussians from world to this object's canonical frame.
            # This is the inverse of the transform that places the canonical object in the world.
            world_to_canonical_T = torch.inverse(obj.get_pivot_to_origin_transform().to(self.device))

            # Transform means and quaternions to the canonical frame
            means_canonical, quats_canonical = transform_gaussians(
                world_to_canonical_T, obj_data_world["means"], obj_data_world["quats"]
            )

            # Store the canonical Gaussians for this object
            obj_data_canonical = obj_data_world
            obj_data_canonical["means"] = means_canonical
            obj_data_canonical["quats"] = quats_canonical
            obj_gaussians_list.append(obj_data_canonical)
            obj_ids_list.append(torch.full((obj_mask.sum(),), obj_id, device=self.device, dtype=torch.long))

        # --- Populate Trainable Object Parameters ---
        if obj_gaussians_list:
            # Concatenate all collected object Gaussians
            final_obj_gaussians = {
                name: torch.cat([d[name] for d in obj_gaussians_list], dim=0) for name in GAUSSIAN_PARAM_NAMES
            }
            # Assign to the model's canonical parameter dictionary
            for name, data in final_obj_gaussians.items():
                self.gauss_params_canonical[name] = torch.nn.Parameter(data, requires_grad=True)

            self.gaussian_obj_ids = torch.cat(obj_ids_list, dim=0)

        # --- Populate Fixed Background Parameters ---
        bg_mask = ~full_obj_mask
        bg_data = {
            name: state_dict[STATE_DICT_MAP[name]].to(self.device)[bg_mask] for name in GAUSSIAN_PARAM_NAMES
        }
        # Reshape features_dc for background
        if bg_data["features_dc"].dim() == 2:
            bg_data["features_dc"] = bg_data["features_dc"].unsqueeze(1)

        for name, data in bg_data.items():
            self.gauss_params_fixed[name] = torch.nn.Parameter(data, requires_grad=False)

        num_obj_gaussians = self.gauss_params_canonical["means"].shape[0]
        num_bg_gaussians = self.gauss_params_fixed["means"].shape[0]
        print(f"✅ Partitioning complete. Trainable Objects: {num_obj_gaussians}, Fixed Background: {num_bg_gaussians}")


    def load_state_dict(self, state_dict, **kwargs):
        """
        Loads the state_dict. If it's a raw splatfacto checkpoint, it triggers
        the partitioning process. If it's an already partitioned ArtiSplatfacto
        checkpoint, it loads the parameters into their respective groups.
        """
        is_partitioned_checkpoint = "gauss_params_fixed.means" in state_dict

        if self.training and not is_partitioned_checkpoint:
            # This is the first run with a full scene checkpoint, so partition it.
            self._initialize_and_partition(state_dict)
        else:
            # Loading a pre-partitioned checkpoint or running in inference mode.
            if is_partitioned_checkpoint:
                print("✅ Resuming from a partitioned ArtiSplatfacto checkpoint...")
                # Load canonical object params
                for name, param in self.gauss_params_canonical.items():
                    param.data = state_dict[f"gauss_params_canonical.{name}"]
                # Load fixed background params
                for name, param in self.gauss_params_fixed.items():
                    param.data = state_dict[f"gauss_params_fixed.{name}"]
                # Load object IDs
                self.gaussian_obj_ids = state_dict["gaussian_obj_ids"]
            else:
                # Loading a full scene for inference without partitioning
                print("⚡️ Loading full scene for inference. No partitioning will be performed.")
                super().load_state_dict(state_dict, **kwargs)

    
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

    def _get_gaussians_for_render(self, joint_angles: Dict[str, float]) -> Dict[str, torch.Tensor]:
        """
        Applies forward kinematics to canonical object Gaussians to pose them for the
        current frame, then merges them with the static background Gaussians.
        """
        # Return only background if no objects exist
        if self.gauss_params_canonical["means"].shape[0] == 0:
            return self.gauss_params_fixed

        # 1. Apply forward kinematics (FK) to transform canonical object Gaussians
        # The Scene3D object handles all the underlying matrix math for FK.
        posed = self.scene.apply_articulations(
            self.gauss_params_canonical["means"],
            self.gauss_params_canonical["quats"],
            joint_angles
        )

        posed_obj_params = {
            "means": posed["means"],
            "quats": posed["quats"],
            "scales": self.gauss_params_canonical["scales"],
            "features_dc": self.gauss_params_canonical["features_dc"],
            "features_rest": self.gauss_params_canonical["features_rest"],
            "opacities": self.gauss_params_canonical["opacities"],
        }

        # 2. Merge posed object Gaussians with static background Gaussians
        if self.gauss_params_fixed["means"].shape[0] > 0:
            full_scene_params = {
                name: torch.cat([posed_obj_params[name], self.gauss_params_fixed[name]], dim=0)
                for name in posed_obj_params.keys()
            }
            return full_scene_params
        else:
            return posed_obj_params


    def get_outputs(self, camera: Cameras) -> Dict[str, Union[torch.Tensor, List]]:
        """Takes in a camera and returns a dictionary of outputs for rendering."""
        if not isinstance(camera, Cameras):
            return {}
        
        if not hasattr(self, "scene"):
            assert hasattr(self, "metadata"), "Model metadata not set"
            assert "scene" in self.metadata, "Scene not found in metadata"
            self.scene = self.metadata["scene"]
            print(" Scene initialized from metadata")

        # 1. Get joint angles for the current frame from the scene definition
        # We use camera.times as a proxy for the frame's timestamp or index.
        time_value = 0.0  # Default to canonical pose (t=0)
        if hasattr(camera, "times") and camera.times is not None:
            time_value = float(camera.times.flatten()[0])

        print("camera metadata:", camera.metadata)
        joint_angles = camera.metadata.get("joint_angle", None)
        timestamp = camera.metadata.get("time", None)

        import pdb; pdb.set_trace()

        


        # metadata = self.kwargs["metadata"]
        # import pdb; pdb.set_trace()
        # if "joint_angle" in metadata and "time" in metadata:
        #     # If joint angles are provided in metadata, use them directly
        #     joint_angles = metadata["joint_angle"]
        #     timestamp = metadata["time"]
        
        # assert joint_angles is not None, "Joint angles missing for current frame!"
        # assert timestamp is not None, "Timestamp missing for current frame!"

        # print(f"joint_angles: {joint_angles}")
        # print(f"timestamp: {timestamp}")

        # import pdb; pdb.set_trace()
        # Scene3D should provide the joint angles for this specific time

        # 2. Prepare Gaussians for rendering by applying FK
        gaussians_to_render = self._get_gaussians_for_render(joint_angles)
        
        # If no gaussians (e.g., empty scene), return black image
        if not gaussians_to_render or gaussians_to_render["means"].shape[0] == 0:
            H, W = int(camera.height.item()), int(camera.width.item())
            return {"rgb": torch.zeros((H, W, 3), device=self.device)}


        # 3. Handle camera optimization
        optimized_camera_to_world = self.camera_optimizer.apply_to_camera(camera) if self.training else camera.camera_to_worlds

        # 4. Setup for rasterization
        colors_crop = torch.cat(
            (gaussians_to_render["features_dc"], gaussians_to_render["features_rest"]), dim=1
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

        # 6. Post-processing and returning outputs
        if self.training:
            # Note: Densification and other strategies should only apply to trainable params
            self.strategy.step_pre_backward(
                self.gauss_params_canonical, self.optimizers, self.strategy_state, self.step, self.info
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
