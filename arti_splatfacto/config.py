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

arti_splatfacto_config = MethodSpecification(
    config=ArtiSplatfactoTrainerConfig(
        method_name="arti_splatfacto", 
        steps_per_eval_batch=100,
        steps_per_save=2000,
        max_num_iterations=30000,
        mixed_precision=False,
        pipeline=VanillaPipelineConfig(
            datamanager=FullImageDatamanagerConfig(
                _target=FullImageDatamanager[DepthArtiDataset],
                dataparser=ArtiSplatfactoDataParserConfig(load_dynamic_objects=True),
                cache_images_type="uint8",
            ),
            model=ArtiSplatfactoModelConfig(
                refine_every=10,              # More frequent (was 20)
                cull_alpha_thresh=0.001,      # Less aggressive culling (was 0.005)
                densify_grad_thresh=0.0005,   # More sensitive (was 0.0008)
                densify_size_thresh=0.008,    # Split smaller Gaussians (was 0.01)
                split_screen_size=0.08,       # Larger screen size threshold (was 0.05)
                warmup_length=500,            # Keep same
                stop_split_at=35000,          # Extend densification period (was 25000)
                reset_alpha_every=15,         # More frequent alpha reset (was 20)
                use_scale_regularization=True,
                use_depth = True,

            )
        ),
        optimizers={
            # === Object groups ===
            "obj_means": {
                "optimizer": AdamOptimizerConfig(lr=1.6e-4, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1.6e-6,
                    max_steps=30000,
                ),
            },
            "obj_features_dc": {
                "optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-15),
                "scheduler": None,
            },
            "obj_features_rest": {
                "optimizer": AdamOptimizerConfig(lr=0.0025 / 20, eps=1e-15),
                "scheduler": None,
            },
            "obj_opacities": {
                "optimizer": AdamOptimizerConfig(lr=0.05, eps=1e-15),
                "scheduler": None,
            },
            "obj_scales": {
                "optimizer": AdamOptimizerConfig(lr=0.005, eps=1e-15),
                "scheduler": None,
            },
            "obj_quats": {
                "optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15),
                "scheduler": None,
            },

            # === Canonical groups ===
            "canon_means": {
                "optimizer": AdamOptimizerConfig(lr=2e-4, eps=1e-15), 
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1.6e-5,
                    max_steps=30000,
                ),
            },
            "canon_features_dc": {
                "optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-15),
                "scheduler": None,
            },
            "canon_features_rest": {
                "optimizer": AdamOptimizerConfig(lr=0.0025 / 20, eps=1e-15),
                "scheduler": None,
            },
            "canon_opacities": {
                "optimizer": AdamOptimizerConfig(lr=0.05, eps=1e-15),
                "scheduler": None,
            },
            "canon_scales": {
                "optimizer": AdamOptimizerConfig(lr=0.005, eps=1e-15),
                "scheduler": None,
            },
            "canon_quats": {
                "optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15),
                "scheduler": None,
            },
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-4, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=5e-7, max_steps=30000, warmup_steps=1000, lr_pre_warmup=0
                ),
            },
            "bilateral_grid": {
                "optimizer": AdamOptimizerConfig(lr=2e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1e-4, max_steps=30000, warmup_steps=1000, lr_pre_warmup=0
                ),
            },
            #### joint param optimizer
            "joint_pivot": {
                "optimizer": AdamOptimizerConfig(lr=0.0001, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1e-6, max_steps=30000, warmup_steps=1000, lr_pre_warmup=0
                ),
            },
            "joint_axis": {
                "optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1e-6, max_steps=30000, warmup_steps=1000, lr_pre_warmup=0
                ),
            },
            "joint_angles": {
                "optimizer": AdamOptimizerConfig(lr=0.01, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=5e-3,      
                    max_steps=30000,
                    warmup_steps=0,     
                    lr_pre_warmup=0.01 
                ),
            },
            "joint_corrections": {
                "optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=5e-3,      
                    max_steps=30000,
                    warmup_steps=0,     
                    lr_pre_warmup=0.01 
                ),
            },
            "max_joint_angle": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1e-5,
                    max_steps=30000,
                ),
            },      
        },
            viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
            vis="viewer",
        ),
        description="A fine-tuning variant of Splatfacto",
)
