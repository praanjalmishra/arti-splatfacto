from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Type, Optional, Union
from pathlib import Path

try:
    from gsplat.rendering import rasterization
except ImportError:
    print("Please install gsplat>=1.0.0")

import torch
from nerfstudio.models.splatfacto import SplatfactoModelConfig, SplatfactoModel
from nerfstudio.utils.spherical_harmonics import RGB2SH, SH2RGB, num_sh_bases
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


    _target: Type = field(default_factory=lambda: SplatfactoModel)
    obj_mask_file: Optional[Path] = None
    # background_color: Literal["random", "black", "white"] = "random"


class ArtiSplatfactoModel(SplatfactoModel):    

    config: ArtiSplatfactoModelConfig

    def populate_modules(self):
        """
        Populates the modules of the model.

        returns:
            None
        """
        super().populate_modules()

        means = torch.zeros((1, 3)).float().cuda()  
        scales = torch.zeros((1, 3)).float().cuda()
        quats = torch.zeros((1, 4)).float().cuda()
        dim_sh = num_sh_bases(self.config.sh_degree)
        features_dc = torch.zeros((1, 3)).float().cuda()
        features_rest = torch.zeros((1, dim_sh-1, 3)).float().cuda()
        opacities = torch.zeros((1, 1)).float().cuda()
        
        # Convert to parameters
        self.gauss_params = torch.nn.ParameterDict(
            {
                "means": torch.nn.Parameter(means),
                "scales": torch.nn.Parameter(scales),
                "quats": torch.nn.Parameter(quats),
                "features_dc": torch.nn.Parameter(features_dc),
                "features_rest": torch.nn.Parameter(features_rest),
                "opacities": torch.nn.Parameter(opacities),
            }
        )
    

    def load_state_dict(self, dict, **kwargs):
        print(f"!!! Loading state_dict, training={self.training}")
        
        # check if we need to do object-specific loading
        needs_object_filtering = (
            self.config.obj_mask_file is not None and 
            "gauss_params.means" in dict and 
            dict["gauss_params.means"].shape[0] > 1  # More than our placeholder size
        )
        
        if needs_object_filtering:
            print("!!! Detected object-filtered checkpoint, handling special loading...")
            
            # Load object mask (needed for both training and inference)
            if isinstance(self.config.obj_mask_file, Path):
                self.obj_3d_seg = Object3DSeg.read_from_file(self.config.obj_mask_file, device=self.device)
                corners = self.obj_3d_seg.get_all_corners()
                print(f"Voxel coordinates count: {corners.shape[0]}")
                print(f"Loaded object mask from {self.config.obj_mask_file}")

                # Before refinement
                voxel_before = self.obj_3d_seg.voxel.clone()
                num_voxels_before = voxel_before.sum().item()
                print(f"[Before refinement] Non-zero voxels: {num_voxels_before}")

                # Apply refinement
                self.obj_3d_seg.refine_mask(dilate_k=2, erode_k=1)
                

                # After refinement
                voxel_after = self.obj_3d_seg.voxel
                num_voxels_after = voxel_after.sum().item()
                new_voxels = (voxel_after & ~voxel_before).sum().item()
                removed_voxels = (voxel_before & ~voxel_after).sum().item()

                print(f"[After refinement] Non-zero voxels: {num_voxels_after}")
                print(f"    + Voxels added:   {new_voxels}")
                print(f"    - Voxels removed: {removed_voxels}")
                print(f"    Δ Change:         {num_voxels_after - num_voxels_before}")


            else:
                raise ValueError(f"Unknown type of obj_mask_file {self.config.obj_mask_file}")
            
            # Handle backwards compatibility
            if "means" in dict:
                for p in ["means", "scales", "quats", "features_dc", "features_rest", "opacities"]:
                    dict[f"gauss_params.{p}"] = dict[p]
            
            if self.training:
                # TRAINING MODE: Filter and transform gaussians
                print("!!! Training mode: filtering and transforming gaussians")
                
                self.obj_mask = self.obj_3d_seg.query(dict["gauss_params.means"].cuda(), dilate=True).cpu()
                print(f"Gaussians inside object mask: {self.obj_mask.sum().item()}")
                
                if self.obj_mask.sum() == 0:
                    print("[red]No gaussians inside the object mask![/red]")
                    return

                # Transform gaussians
                pose = self.obj_3d_seg.pose_change
                dict["gauss_params.means"][self.obj_mask], dict["gauss_params.quats"][self.obj_mask] = \
                    transform_gaussians(pose.cpu(), dict["gauss_params.means"][self.obj_mask], dict["gauss_params.quats"][self.obj_mask])

                # Filter to only object gaussians
                filtered_data = {}
                for name in ["means", "scales", "quats", "features_dc", "features_rest", "opacities"]:
                    filtered_data[name] = dict[f"gauss_params.{name}"][self.obj_mask]
                
                # Resize existing parameters IN-PLACE
                for name, new_data in filtered_data.items():
                    existing_param = self.gauss_params[name]
                    existing_param.data = new_data.to(existing_param.device).detach()
                    print(f"Resized {name}: {existing_param.shape}")
                
                # Store fixed gaussians
                self.gauss_params_fixed = {}
                non_obj_mask = ~self.obj_mask
                for name in ["means", "scales", "quats", "features_dc", "features_rest", "opacities"]:
                    self.gauss_params_fixed[name] = dict[f"gauss_params.{name}"][non_obj_mask].to(self.device)
                
                print(f"[INFO] Loaded {filtered_data['means'].shape[0]} Gaussians for fine-tuning.")



                # num_seed = 3000  # Or more, based on resolution needs
                # bbox_min = filtered_data["means"].min(dim=0).values.to(self.device)
                # bbox_max = filtered_data["means"].max(dim=0).values.to(self.device)

                # # Uniform random seed in bounding box
                # new_means = torch.rand((num_seed, 3), device=self.device) * (bbox_max - bbox_min) + bbox_min
                # new_scales = torch.log(torch.ones((num_seed, 3), device=self.device) * 0.01)
                # new_quats = torch.nn.functional.normalize(torch.randn((num_seed, 4), device=self.device), dim=-1)
                # new_features_dc = torch.rand((num_seed, 3), device=self.device) * 0.5 + 0.25  # mid-gray
                # new_features_rest = torch.zeros((num_seed, self.features_rest.shape[1], 3), device=self.device)
                # new_opacities = torch.full((num_seed, 1), -2.0, device=self.device)  # low initial opacity

                # # Concatenate to existing object Gaussians
                # for name, new_tensor in {
                #     "means": new_means,
                #     "scales": new_scales,
                #     "quats": new_quats,
                #     "features_dc": new_features_dc,
                #     "features_rest": new_features_rest,
                #     "opacities": new_opacities,
                # }.items():
                #     existing = getattr(self, name)
                #     new_full = torch.cat([existing, new_tensor], dim=0)
                #     self.gauss_params[name].data = new_full

                
            else:
                # INFERENCE MODE: Load all gaussians directly
                print("!!! Inference mode: loading all gaussians directly")
                
                # Resize model to match checkpoint size
                checkpoint_size = dict["gauss_params.means"].shape[0]
                print(f"Resizing model from {self.means.shape[0]} to {checkpoint_size} gaussians")
                
                for name in ["means", "scales", "quats", "features_dc", "features_rest", "opacities"]:
                    checkpoint_data = dict[f"gauss_params.{name}"]
                    existing_param = self.gauss_params[name]
                    existing_param.data = checkpoint_data.to(existing_param.device).detach()
                    print(f"Loaded {name}: {existing_param.shape}")
            
            self.step = 0
            
        else:
            # Normal loading (original checkpoint without object filtering)
            print("!!! Normal checkpoint loading")
            super().load_state_dict(dict, **kwargs)
            self.step = 0

        # Debug info
        print(f"=== FINAL STATE ===")
        print(f"Gaussians loaded: {self.means.shape[0]}")
        print(f"Fixed gaussians: {getattr(self, 'gauss_params_fixed', {}).get('means', torch.tensor([])).shape[0] if hasattr(self, 'gauss_params_fixed') else 0}")
        print(f"Training mode: {self.training}")
        print(f"=== END ===")


    def get_outputs(self, camera: Cameras) -> Dict[str, Union[torch.Tensor, List]]:
        """Takes in a camera and returns a dictionary of outputs.

        Args:
            camera: The camera(s) for which output images are rendered. It should have
            all the needed information to compute the outputs.

        Returns:
            Outputs of model. (ie. rendered colors)
        """
        # print(f"!!! get_outputs called at step {getattr(self, 'step', 'unknown')}")

        if not isinstance(camera, Cameras):
            print("Called get_outputs with not a camera")
            return {}

        if self.training:
            assert camera.shape[0] == 1, "Only one camera at a time"
            optimized_camera_to_world = self.camera_optimizer.apply_to_camera(camera)
        else:
            optimized_camera_to_world = camera.camera_to_worlds

        # cropping
        if self.crop_box is not None and not self.training:
            crop_ids = self.crop_box.within(self.means).squeeze()
            if crop_ids.sum() == 0:
                return self.get_empty_outputs(
                    int(camera.width.item()), int(camera.height.item()), self.background_color
                )
        else:
            crop_ids = None

        if crop_ids is not None:
            opacities_crop = self.opacities[crop_ids]
            means_crop = self.means[crop_ids]
            features_dc_crop = self.features_dc[crop_ids]
            features_rest_crop = self.features_rest[crop_ids]
            scales_crop = self.scales[crop_ids]
            quats_crop = self.quats[crop_ids]
        else:
            opacities_crop = self.opacities
            means_crop = self.means
            features_dc_crop = self.features_dc
            features_rest_crop = self.features_rest
            scales_crop = self.scales
            quats_crop = self.quats

            if hasattr(self, "gauss_params_fixed") and self.training:
                # print(f"!!! Combining {self.means.shape[0]} trainable + {self.gauss_params_fixed['means'].shape[0]} fixed gaussians")
                
                opacities_crop = torch.cat([opacities_crop, self.gauss_params_fixed["opacities"]], dim=0)
                means_crop = torch.cat([means_crop, self.gauss_params_fixed["means"]], dim=0)
                features_dc_crop = torch.cat([features_dc_crop, self.gauss_params_fixed["features_dc"]], dim=0)
                features_rest_crop = torch.cat([features_rest_crop, self.gauss_params_fixed["features_rest"]], dim=0)
                scales_crop = torch.cat([scales_crop, self.gauss_params_fixed["scales"]], dim=0)
                quats_crop = torch.cat([quats_crop, self.gauss_params_fixed["quats"]], dim=0)
            elif hasattr(self, "gauss_params_fixed"):
                # During evaluation, also include fixed gaussians
                # print(f"!!! (Eval) Combining {self.means.shape[0]} trainable + {self.gauss_params_fixed['means'].shape[0]} fixed gaussians")
                
                opacities_crop = torch.cat([opacities_crop, self.gauss_params_fixed["opacities"]], dim=0)
                means_crop = torch.cat([means_crop, self.gauss_params_fixed["means"]], dim=0)
                features_dc_crop = torch.cat([features_dc_crop, self.gauss_params_fixed["features_dc"]], dim=0)
                features_rest_crop = torch.cat([features_rest_crop, self.gauss_params_fixed["features_rest"]], dim=0)
                scales_crop = torch.cat([scales_crop, self.gauss_params_fixed["scales"]], dim=0)
                quats_crop = torch.cat([quats_crop, self.gauss_params_fixed["quats"]], dim=0)


        colors_crop = torch.cat((features_dc_crop[:, None, :], features_rest_crop), dim=1)

        camera_scale_fac = self._get_downscale_factor()
        camera.rescale_output_resolution(1 / camera_scale_fac)
        viewmat = get_viewmat(optimized_camera_to_world)
        K = camera.get_intrinsics_matrices().cuda()
        W, H = int(camera.width.item()), int(camera.height.item())
        self.last_size = (H, W)
        camera.rescale_output_resolution(camera_scale_fac)  # type: ignore

        # apply the compensation of screen space blurring to gaussians
        if self.config.rasterize_mode not in ["antialiased", "classic"]:
            raise ValueError("Unknown rasterize_mode: %s", self.config.rasterize_mode)

        if self.config.output_depth_during_training or not self.training:
            render_mode = "RGB+ED"
        else:
            render_mode = "RGB"

        if self.config.sh_degree > 0:
            sh_degree_to_use = min(self.step // self.config.sh_degree_interval, self.config.sh_degree)
        else:
            colors_crop = torch.sigmoid(colors_crop).squeeze(1)  # [N, 1, 3] -> [N, 3]
            sh_degree_to_use = None

        render, alpha, self.info = rasterization(
            means=means_crop,
            quats=quats_crop,  # rasterization does normalization internally
            scales=torch.exp(scales_crop),
            opacities=torch.sigmoid(opacities_crop).squeeze(-1),
            colors=colors_crop,
            viewmats=viewmat,  # [1, 4, 4]
            Ks=K,  # [1, 3, 3]
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
            # set some threshold to disregrad small gaussians for faster rendering.
            # radius_clip=3.0,
        )
        if self.training:
            self.strategy.step_pre_backward(
                self.gauss_params, self.optimizers, self.strategy_state, self.step, self.info
            )
        alpha = alpha[:, ...]

        background = self._get_background_color()
        rgb = render[:, ..., :3] + (1 - alpha) * background
        rgb = torch.clamp(rgb, 0.0, 1.0)

        # apply bilateral grid
        if self.config.use_bilateral_grid and self.training:
            if camera.metadata is not None and "cam_idx" in camera.metadata:
                rgb = self._apply_bilateral_grid(rgb, camera.metadata["cam_idx"], H, W)

        if render_mode == "RGB+ED":
            depth_im = render[:, ..., 3:4]
            depth_im = torch.where(alpha > 0, depth_im, depth_im.detach().max()).squeeze(0)
        else:
            depth_im = None

        if background.shape[0] == 3 and not self.training:
            background = background.expand(H, W, 3)

        return {
            "rgb": rgb.squeeze(0),  # type: ignore
            "depth": depth_im,  # type: ignore
            "accumulation": alpha.squeeze(0),  # type: ignore
            "background": background,  # type: ignore
        }  # type: ignore

