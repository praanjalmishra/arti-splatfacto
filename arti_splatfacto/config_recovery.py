from __future__ import annotations

from arti_splatfacto.data.dataparser import ArtiSplatfactoDataParserConfig
from arti_splatfacto.model.model import ArtiSplatfactoModelConfig
from arti_splatfacto.trainer import ArtiSplatfactoTrainerConfig, ArtiSplatfactoTrainer
from nerfstudio.data.datasets.depth_dataset import DepthDataset
from arti_splatfacto.data.dataset import ArtiDataset, DepthArtiDataset
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.plugins.types import MethodSpecification
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from arti_splatfacto.data.datamanager import ArtiSplatfactoManagerConfig, ArtiSplatfactoDataManager
from nerfstudio.data.datamanagers.full_images_datamanager import FullImageDatamanagerConfig, FullImageDatamanager

arti_splatfacto_recovery_config = MethodSpecification(
    config=ArtiSplatfactoTrainerConfig(
        method_name="arti_splatfacto_recovery",
        steps_per_eval_batch=100,
        steps_per_save=1000,
        max_num_iterations=10000,  # Shorter recovery phase
        mixed_precision=False,
        pipeline=VanillaPipelineConfig(
            datamanager=FullImageDatamanagerConfig(
                _target=FullImageDatamanager[DepthArtiDataset],
                dataparser=ArtiSplatfactoDataParserConfig(load_dynamic_objects=True),
                cache_images_type="uint8",
            ),
            model=ArtiSplatfactoModelConfig(
                training_mode="recovery",  # KEY DIFFERENCE
                
                refine_every=100,  
                stop_split_at=0,
                
                use_scale_regularization=False,  
                output_depth_during_training=True,
                use_depth=True,
                depth_lambda=0.1,  
            )
        ),
        optimizers={
            # === Object Radiance Only ===
            "obj_features_dc": {
                "optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15),  # Lower LR
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=0.0001,
                    max_steps=10000,
                ),
            },
            "obj_features_rest": {
                "optimizer": AdamOptimizerConfig(lr=0.0005, eps=1e-15),  # Lower LR
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=0.00005,
                    max_steps=10000,
                ),
            },
            "obj_opacities": {
                "optimizer": AdamOptimizerConfig(lr=0.01, eps=1e-15),  # Lower LR
                "scheduler": None,
            },
            
            # === Canonical Radiance Only ===
            "canon_features_dc": {
                "optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=0.0001,
                    max_steps=10000,
                ),
            },
            "canon_features_rest": {
                "optimizer": AdamOptimizerConfig(lr=0.0005, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=0.00005,
                    max_steps=10000,
                ),
            },
            "canon_opacities": {
                "optimizer": AdamOptimizerConfig(lr=0.01, eps=1e-15),
                "scheduler": None,
            },
            
            # === Background Radiance (THE KEY ADDITION) ===
            "bg_features_dc": {
                "optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=0.0001,
                    max_steps=10000,
                ),
            },
            "bg_features_rest": {
                "optimizer": AdamOptimizerConfig(lr=0.0005, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=0.00005,
                    max_steps=10000,
                ),
            },
            "bg_opacities": {
                "optimizer": AdamOptimizerConfig(lr=0.01, eps=1e-15),
                "scheduler": None,
            },
            
            # === Per-frame Articulation Refinement ===
            "joint_t_values": {
                "optimizer": AdamOptimizerConfig(lr=0.005, eps=1e-15),  # Lower LR
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=0.001,
                    max_steps=10000,
                ),
            },
            
            # Keep camera optimizer for minor adjustments
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-5, eps=1e-15),  # Very low LR
                "scheduler": None,
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="ArtiSplatfacto - Stage 3: Radiance Recovery",
)