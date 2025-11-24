from __future__ import annotations
from dataclasses import dataclass, field
from importlib import metadata
from typing import Literal, Type, Optional, Dict, Any
from cycler import V
import test
import torch
from torch.cuda.amp.grad_scaler import GradScaler

from nerfstudio.utils import profiler
from nerfstudio.pipelines.base_pipeline import VanillaPipeline, VanillaPipelineConfig
from nerfstudio.data.datamanagers.base_datamanager import (
    DataManager,
    DataManagerConfig,
    VanillaDataManager,
)
from arti_splatfacto.data.datamanager import FullImageDatamanagerConfig, FullImageDatamanager
from nerfstudio.models.base_model import Model, ModelConfig
from arti_splatfacto.model.model import ArtiSplatfactoModelConfig

@dataclass
class ArtiSplatfactoPipelineConfig(VanillaPipelineConfig):
    """
    Configuration class for ArtiSplatfactoPipeline.
    """
    
    _target: Type = field(default_factory=lambda: ArtiSplatfactoPipeline)
    datamanager: DataManagerConfig = field(default_factory=FullImageDatamanagerConfig)
    model: ModelConfig = field(default_factory=ArtiSplatfactoModelConfig)


class ArtiSplatfactoPipeline(VanillaPipeline):
    """Custom pipeline for ArtiSplatfacto with correct checkpoint handling."""

    def __init__(
        self,
        config: ArtiSplatfactoPipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        grad_scaler: Optional[GradScaler] = None,
    ):
        super().__init__(
            config=config,
            device=device,
            test_mode=test_mode,
            world_size=world_size,
            local_rank=local_rank,
            grad_scaler=grad_scaler,
        )

    def state_dict(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Return complete pipeline state including model's full joint data.
        """
        # Get base pipeline state (includes flattened model keys)
        state = super().state_dict(*args, **kwargs)
        
        # CRITICAL: Save complete model state separately
        # This calls your model's state_dict() which handles all the joint saving
        state["model_full"] = self.model.state_dict()
        
        # Optional: Add metadata for debugging
        state["pipeline_metadata"] = {
            "active_joint_id": self.model.config.active_joint_id,
            "training_mode": self.model.config.training_mode,
            "num_joints": len(self.model.all_joint_params),
            "joint_ids": list(self.model.all_joint_params.keys()),
        }
        
        print(f"[Pipeline] Saved state with {len(self.model.all_joint_params)} joints: {list(self.model.all_joint_params.keys())}")
        
        return state
    
    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True) -> None:
        """
        Override to prevent double-loading of model state.
        The model is already loaded in load_pipeline(), so we skip it here.
        """
        self.model._skip_load_state_dict = True
        # Filter out ALL model keys to prevent double-loading
        filtered_state = {}
        skipped_keys = []
        
        for key, value in state_dict.items():
            if (key.startswith("_model") or           # Vanilla format: _model.gauss_params.means
                key == "model_full" or                 # Articulated format: model_full
                key == "pipeline_metadata" or          # Metadata
                key.startswith("gauss_params.") or     # Direct gauss params
                key.startswith("gauss_params_canonical.") or
                key.startswith("gauss_params_fixed.") or
                key.startswith("all_gauss_params_") or
                key.startswith("all_joint_params.")):
                skipped_keys.append(key)
                continue
            
            filtered_state[key] = value
        
        if skipped_keys:
            print(f"[Pipeline] Skipping {len(skipped_keys)} model keys (already loaded in load_pipeline)")
        
        super().load_state_dict(filtered_state, strict=False)
        if hasattr(self.model, '_skip_load_state_dict'):
            delattr(self.model, '_skip_load_state_dict')

    def load_pipeline(self, loaded_state: Dict[str, Any], step: int) -> None:
        """Load checkpoint while properly restoring joint-aware model."""
        
        # Check if this is a vanilla checkpoint or articulated checkpoint
        has_model_full = "model_full" in loaded_state
        has_vanilla_model_params = any(k.startswith("_model.gauss_params.") for k in loaded_state.keys())
        
        # STEP 1: Load Model State
        if has_model_full:
            # Articulated checkpoint with separate model_full
            print("[Pipeline] Loading full model state (per-joint params)...")
            self.model.load_state_dict(loaded_state["model_full"], strict=False)
            
            # Log what was loaded
            if "pipeline_metadata" in loaded_state:
                meta = loaded_state["pipeline_metadata"]
                print(f"  ✓ Loaded {meta['num_joints']} joints: {meta['joint_ids']}")
                print(f"  ✓ Active joint: {meta['active_joint_id']}")
                print(f"  ✓ Training mode: {meta['training_mode']}")
        
        elif has_vanilla_model_params:
            print("[Pipeline] Detected VANILLA 3DGS checkpoint - extracting model parameters...")
            
            # Extract model parameters from the pipeline state
            # Keys are like "_model.gauss_params.means" -> need to become "gauss_params.means"
            model_state = {}
            for key, value in loaded_state.items():
                # Clean DDP prefix first
                clean_key = key[len("module."):] if key.startswith("module.") else key
                
                # Extract model parameters (remove "_model." prefix)
                if clean_key.startswith("_model."):
                    # Remove '_model.' prefix: "_model.gauss_params.means" -> "gauss_params.means"
                    param_key = clean_key[len("_model."):]
                    model_state[param_key] = value
            
            print(f"[Pipeline] Extracted {len(model_state)} model parameters")
            print(f"[Pipeline] Sample keys: {list(model_state.keys())[:5]}")
            
            # Check if we have the expected gaussian parameters
            if "gauss_params.means" in model_state:
                print(f"[Pipeline] Found {model_state['gauss_params.means'].shape[0]:,} Gaussians to partition")
            
            # Now load the vanilla model state (this will trigger partitioning)
            if model_state:
                self.model.load_state_dict(model_state, strict=False)
            else:
                print("[Pipeline] ⚠️  No model parameters found in checkpoint!")
        
        else:
            print("[Pipeline] ⚠️  Unknown checkpoint format - no model_full or vanilla params found")
        
        self.model.update_to_step(step)
        
        # STEP 3: Load Remaining Pipeline State (non-model components)
        clean_state = {
            (key[len("module."):] if key.startswith("module.") else key): value
            for key, value in loaded_state.items()
        }
        
        # This will automatically filter out model keys (via our load_state_dict override)
        self.load_state_dict(clean_state, strict=False)
        
        print(f"[Pipeline] ✓ Pipeline loaded at step {step}")



    @profiler.time_function
    def get_eval_image_metrics_and_images(self, step: int):
        self.eval()
        camera, batch = self.datamanager.next_eval_image(step)
        outputs = self.model.get_outputs_for_camera(camera)
        metrics_dict, images_dict = self.model.get_image_metrics_and_images(outputs, batch)
        metrics_dict["num_rays"] = (camera.height * camera.width * camera.size).item()
        self.train()
        return metrics_dict, images_dict

    @profiler.time_function
    def get_average_image_metrics(
        self,
        data_loader,
        image_prefix: str,
        step: Optional[int] = None,
        output_path=None,
        get_std=False,
    ):
        self.eval()
        metrics_dict_list = []
        for camera, batch in data_loader:
            outputs = self.model.get_outputs_for_camera(camera=camera)
            metrics_dict, _ = self.model.get_image_metrics_and_images(outputs, batch)
            metrics_dict_list.append(metrics_dict)
        self.train()
        return super().get_average_image_metrics(
            data_loader=data_loader,
            image_prefix=image_prefix,
            step=step,
            output_path=output_path,
            get_std=get_std,
        )