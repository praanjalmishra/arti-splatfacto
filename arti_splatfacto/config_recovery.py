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
from arti_splatfacto.pipeline import ArtiSplatfactoPipeline, ArtiSplatfactoPipelineConfig

arti_splatfacto_recovery_config = MethodSpecification(
    config=ArtiSplatfactoTrainerConfig(
        method_name="arti_splatfacto_recovery",
        steps_per_eval_batch=100,
        steps_per_save=1000,
        max_num_iterations=10000, 
        mixed_precision=False,
        pipeline=ArtiSplatfactoPipelineConfig(
            datamanager=FullImageDatamanagerConfig(
                _target=FullImageDatamanager[DepthArtiDataset],
                dataparser=ArtiSplatfactoDataParserConfig(load_dynamic_objects=True),
                cache_images_type="uint8",
            ),
            model=ArtiSplatfactoModelConfig(
                training_mode="recovery", 
                refine_every=50,  
                stop_split_at=0,
                warmup_length=100,
                use_scale_regularization=False,  
                output_depth_during_training=True,
                use_depth=True,
                depth_lambda=1.0,   
            )
        ),
        optimizers={
            "obj_means": {
                "optimizer": AdamOptimizerConfig(lr=5e-4, eps=1e-15),  # Low LR!
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-6, max_steps=10000),
            },
            "obj_quats": {
                "optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-15),
                "scheduler": None,
            },  
            "obj_scales": {
                "optimizer": AdamOptimizerConfig(lr=0.005, eps=1e-15),
                "scheduler": None,
            },
            "obj_features_dc": {
                "optimizer": AdamOptimizerConfig(lr=0.005, eps=1e-15),  # Lower LR
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
            
            "canon_means": {
                "optimizer": AdamOptimizerConfig(lr=5e-4, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-6, max_steps=10000),
            },
            "canon_quats": {
                "optimizer": AdamOptimizerConfig(lr=0.0005, eps=1e-15),
                "scheduler": None,
            }, 
            "canon_scales": {
                "optimizer": AdamOptimizerConfig(lr=0.0005, eps=1e-15),
                "scheduler": None,
            },          
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

            "bg_means": {
                "optimizer": AdamOptimizerConfig(lr=5e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-6, max_steps=10000),
            },
            "bg_quats": {
                "optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-15),
                "scheduler": None,
            },            
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