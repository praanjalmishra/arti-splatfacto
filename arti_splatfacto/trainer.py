from __future__ import annotations

from dataclasses import dataclass, field
from typing import Type, Dict, Any, Optional, Literal
from pathlib import Path
import torch
import os

from nerfstudio.engine.trainer import Trainer, TrainerConfig
from nerfstudio.engine.optimizers import Optimizers
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.utils import writer, profiler
from nerfstudio.engine.callbacks import TrainingCallbackAttributes
from nerfstudio.viewer_legacy.server.viewer_state import ViewerLegacyState
from nerfstudio.viewer.viewer import Viewer as ViewerState
import dataclasses

@dataclass
class ArtiSplatfactoTrainerConfig(TrainerConfig):
    """Custom trainer config for ArtiSplatfacto with optimizer remapping """
    
    _target: Type = field(default_factory=lambda: ArtiSplatfactoTrainer)


class ArtiSplatfactoTrainer(Trainer):
    """Custom trainer that handles optimizer remapping for fine-tuning from vanilla Splatfacto"""
    
    def setup(self, test_mode: Literal["test", "val", "inference"] = "val") -> None:
        """Override setup to ensure proper order: pipeline -> checkpoint loading -> optimizers"""
        self.pipeline = self.config.pipeline.setup(
            device=self.device,
            test_mode=test_mode,
            world_size=self.world_size,
            local_rank=self.local_rank,
            grad_scaler=self.grad_scaler,
        )
        
        self._load_checkpoint()
        
        self.optimizers = self.setup_optimizers()
        
        # set up viewer
        viewer_log_path = self.base_dir / self.config.viewer.relative_log_filename
        self.viewer_state, banner_messages = None, None

        if self.config.is_viewer_legacy_enabled() and self.local_rank == 0:
            datapath = self.config.data
            if datapath is None:
                datapath = self.base_dir
            self.viewer_state = ViewerLegacyState(
                self.config.viewer,
                log_filename=viewer_log_path,
                datapath=datapath,
                pipeline=self.pipeline,
                trainer=self,
                train_lock=self.train_lock,
            )
            banner_messages = [f"Legacy viewer at: {self.viewer_state.viewer_url}"]
        if self.config.is_viewer_enabled() and self.local_rank == 0:
            
            datapath = self.config.data
            if datapath is None:
                datapath = self.base_dir
            self.viewer_state = ViewerState(
                self.config.viewer,
                log_filename=viewer_log_path,
                datapath=datapath,
                pipeline=self.pipeline,
                trainer=self,
                train_lock=self.train_lock,
                share=self.config.viewer.make_share_url,
            )
            banner_messages = self.viewer_state.viewer_info
        self._check_viewer_warnings()

        self.callbacks = self.pipeline.get_training_callbacks(
            TrainingCallbackAttributes(
                optimizers=self.optimizers, grad_scaler=self.grad_scaler, pipeline=self.pipeline, trainer=self
            )
        )


        writer_log_path = self.base_dir / self.config.logging.relative_log_dir
        writer.setup_event_writer(
            self.config.is_wandb_enabled(),
            self.config.is_tensorboard_enabled(),
            self.config.is_comet_enabled(),
            log_dir=writer_log_path,
            experiment_name=self.config.experiment_name,
            project_name=self.config.project_name,
        )
        writer.setup_local_writer(
            self.config.logging, max_iter=self.config.max_num_iterations, banner_messages=banner_messages
        )
        
        writer.put_config(name="config", config_dict=dataclasses.asdict(self.config), step=0)
        profiler.setup_profiler(self.config.logging, writer_log_path)
    
    def _load_checkpoint(self) -> None:
        """Override checkpoint loading, only model weights, skip optimizers for now"""
        load_dir = self.config.load_dir
        load_checkpoint = self.config.load_checkpoint
        
        if load_dir is not None:
            load_step = self.config.load_step
            if load_step is None:
                print("Loading latest Nerfstudio checkpoint from load_dir...")
                load_step = sorted(int(x[x.find("-") + 1 : x.find(".")]) for x in os.listdir(load_dir))[-1]
            load_path: Path = load_dir / f"step-{load_step:09d}.ckpt"
            assert load_path.exists(), f"Checkpoint {load_path} does not exist"
            loaded_state = torch.load(load_path, map_location="cpu")
            self._start_step = loaded_state["step"] + 1
            
            # Load pipeline only - this populates the model parameters
            self.pipeline.load_pipeline(loaded_state["pipeline"], loaded_state["step"])
            
            # Store optimizer states for later loading
            self._loaded_optimizer_states = loaded_state.get("optimizers", {})
            self._loaded_scheduler_states = loaded_state.get("schedulers", {})
            
            # Load gradient scaler
            self.grad_scaler.load_state_dict(loaded_state["scalers"])
            
            CONSOLE.print(f"Done loading Nerfstudio checkpoint from {load_path}")
            
        elif load_checkpoint is not None:
            assert load_checkpoint.exists(), f"Checkpoint {load_checkpoint} does not exist"
            loaded_state = torch.load(load_checkpoint, map_location="cpu")
            self._start_step = loaded_state["step"] + 1
            
            # Load pipeline only
            self.pipeline.load_pipeline(loaded_state["pipeline"], loaded_state["step"])
            
            # Store optimizer states for later loading
            self._loaded_optimizer_states = loaded_state.get("optimizers", {})
            self._loaded_scheduler_states = loaded_state.get("schedulers", {})
            
            # Load gradient scaler
            self.grad_scaler.load_state_dict(loaded_state["scalers"])
            
            CONSOLE.print(f"Done loading Nerfstudio checkpoint from {load_checkpoint}")
        else:
            CONSOLE.print("No Nerfstudio checkpoint to load, so training from scratch.")
            self._loaded_optimizer_states = {}
            self._loaded_scheduler_states = {}

    def setup_optimizers(self) -> Optimizers:
        """Set up optimizers and load states with remapping if available"""
        # First, set up fresh optimizers with current parameter groups
        optimizer_config = self.config.optimizers.copy()
        param_groups = self.pipeline.get_param_groups()
        optimizers = Optimizers(optimizer_config, param_groups)
        
        # If we have loaded optimizer states, apply remapping
        if hasattr(self, '_loaded_optimizer_states') and self._loaded_optimizer_states:
            # self._load_optimizers_with_remapping(self._loaded_optimizer_states, optimizers)
            print(" skipping optimizer remapping")

        if hasattr(self, '_loaded_scheduler_states') and self._loaded_scheduler_states:
            # self._load_schedulers_with_remapping(self._loaded_scheduler_states, optimizers)
            print(" skipping scheduler remapping")

        return optimizers

    # def _load_optimizers_with_remapping(self, loaded_optimizers: Dict[str, Any], optimizers: Optimizers) -> None:
    #     """Override checkpoint loading to handle optimizer remapping"""
    #     load_dir = self.config.load_dir
    #     load_checkpoint = self.config.load_checkpoint
        
    #     if load_dir is not None:
    #         load_step = self.config.load_step
    #         if load_step is None:
    #             print("Loading latest Nerfstudio checkpoint from load_dir...")
    #             # NOTE: this is specific to the checkpoint name format
    #             load_step = sorted(int(x[x.find("-") + 1 : x.find(".")]) for x in os.listdir(load_dir))[-1]
    #         load_path: Path = load_dir / f"step-{load_step:09d}.ckpt"
    #         assert load_path.exists(), f"Checkpoint {load_path} does not exist"
    #         loaded_state = torch.load(load_path, map_location="cpu")
    #         self._start_step = loaded_state["step"] + 1
            
    #         # Load pipeline first
    #         self.pipeline.load_pipeline(loaded_state["pipeline"], loaded_state["step"])
            
    #         # Handle optimizer loading with remapping
    #         self._load_optimizers_with_remapping(loaded_state.get("optimizers", {}))
            
    #         # Load schedulers if available
    #         if "schedulers" in loaded_state and self.config.load_scheduler:
    #             self._load_schedulers_with_remapping(loaded_state["schedulers"])
            
    #         # Load gradient scaler
    #         self.grad_scaler.load_state_dict(loaded_state["scalers"])
            
    #         CONSOLE.print(f"Done loading Nerfstudio checkpoint from {load_path}")
            
    #     elif load_checkpoint is not None:
    #         assert load_checkpoint.exists(), f"Checkpoint {load_checkpoint} does not exist"
    #         loaded_state = torch.load(load_checkpoint, map_location="cpu")
    #         self._start_step = loaded_state["step"] + 1
            
    #         # Load pipeline first
    #         self.pipeline.load_pipeline(loaded_state["pipeline"], loaded_state["step"])
            
    #         # Handle optimizer loading with remapping
    #         self._load_optimizers_with_remapping(loaded_state.get("optimizers", {}))
            
    #         # Load schedulers if available
    #         if "schedulers" in loaded_state and self.config.load_scheduler:
    #             self._load_schedulers_with_remapping(loaded_state["schedulers"])
            
    #         # Load gradient scaler
    #         self.grad_scaler.load_state_dict(loaded_state["scalers"])
            
    #         CONSOLE.print(f"Done loading Nerfstudio checkpoint from {load_checkpoint}")
    #     else:
    #         CONSOLE.print("No Nerfstudio checkpoint to load, so training from scratch.")

    # def _load_optimizers_with_remapping(self, loaded_optimizers: Dict[str, Any], optimizers: Optimizers) -> None:
    #     """Load optimizers with key remapping for fine-tuning compatibility"""
    #     print(f"🔄 Loading optimizers with remapping...")
    #     print(f"   Loaded optimizer keys: {list(loaded_optimizers.keys())}")
    #     print(f"   Expected optimizer keys: {list(optimizers.optimizers.keys())}")
        
    #     # Define the mapping from old keys to new keys
    #     key_mapping = {
    #         # Map vanilla Splatfacto keys to object parameters only
    #         'means': 'obj_means',
    #         'scales': 'obj_scales', 
    #         'quats': 'obj_quats',
    #         'features_dc': 'obj_features_dc',
    #         'features_rest': 'obj_features_rest',
    #         'opacities': 'obj_opacities',
    #     }
        
    #     # Apply remapping and load optimizer states
    #     for old_key, optimizer_state in loaded_optimizers.items():
    #         target_key = None
            
    #         if old_key in key_mapping:
    #             # Map Gaussian parameters to object parameters
    #             target_key = key_mapping[old_key]
    #         elif old_key in optimizers.optimizers:
    #             # Keep non-Gaussian optimizers as-is (camera_opt, bilateral_grid, etc.)
    #             target_key = old_key
            
    #         if target_key and target_key in optimizers.optimizers:
    #             try:
    #                 optimizers.optimizers[target_key].load_state_dict(optimizer_state)
    #                 print(f"   ✅ Loaded optimizer state: '{old_key}' -> '{target_key}'")
    #             except Exception as e:
    #                 print(f"   ⚠️  Failed to load optimizer '{target_key}': {e}")
    #                 print(f"       Starting with fresh optimizer state for '{target_key}'")
    #         else:
    #             print(f"   ⚠️  Skipped unknown optimizer '{old_key}'")
        
    #     # Note: Canonical optimizers start fresh (no old state to load)
    #     canon_optimizers = [k for k in optimizers.optimizers.keys() if k.startswith('canon_')]
    #     if canon_optimizers:
    #         print(f"   💡 Canonical optimizers start fresh: {canon_optimizers}")

    # def _load_schedulers_with_remapping(self, loaded_schedulers: Dict[str, Any], optimizers: Optimizers) -> None:
    #     """Load schedulers with key remapping"""
    #     print(f"🔄 Loading schedulers with remapping...")
        
    #     # Same mapping as optimizers
    #     key_mapping = {
    #         'means': 'obj_means',
    #         'scales': 'obj_scales', 
    #         'quats': 'obj_quats',
    #         'features_dc': 'obj_features_dc',
    #         'features_rest': 'obj_features_rest',
    #         'opacities': 'obj_opacities',
    #     }
        
    #     for old_key, scheduler_state in loaded_schedulers.items():
    #         target_key = None
            
    #         if old_key in key_mapping:
    #             target_key = key_mapping[old_key]
    #         elif old_key in optimizers.schedulers:
    #             target_key = old_key
            
    #         if target_key and target_key in optimizers.schedulers:
    #             try:
    #                 optimizers.schedulers[target_key].load_state_dict(scheduler_state)
    #                 print(f"   ✅ Loaded scheduler state: '{old_key}' -> '{target_key}'")
    #             except Exception as e:
    #                 print(f"   ⚠️  Failed to load scheduler '{target_key}': {e}")
    #                 print(f"       Starting with fresh scheduler state for '{target_key}'")
    #         else:
    #             print(f"   ⚠️  Skipped unknown scheduler '{old_key}'")