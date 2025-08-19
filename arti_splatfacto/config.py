from __future__ import annotations

from arti_splatfacto.data.dataparser import ArtiSplatfactoDataParserConfig
from arti_splatfacto.model.model import ArtiSplatfactoModelConfig
from arti_splatfacto.trainer import ArtiSplatfactoTrainerConfig, ArtiSplatfactoTrainer


from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.plugins.types import MethodSpecification
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from arti_splatfacto.data.datamanager import ArtiSplatfactoManagerConfig

arti_splatfacto_config = MethodSpecification(
    config=ArtiSplatfactoTrainerConfig(
        method_name="arti_splatfacto", 
        steps_per_eval_batch=100,
        steps_per_save=2000,
        max_num_iterations=30000,
        mixed_precision=False,
        pipeline=VanillaPipelineConfig(
            datamanager=ArtiSplatfactoManagerConfig(
                dataparser=ArtiSplatfactoDataParserConfig(load_dynamic_objects=True),
                cache_images_type="uint8",
            ),
            model=ArtiSplatfactoModelConfig(
                refine_every=20,
                cull_alpha_thresh=0.05,     
                densify_grad_thresh=0.002,
                warmup_length=22000,
                stop_split_at=27000,
            ),
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
                "optimizer": AdamOptimizerConfig(lr=8e-5, eps=1e-15),  # maybe slower
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=8e-7,
                    max_steps=30000,
                ),
            },
            "canon_features_dc": {
                "optimizer": AdamOptimizerConfig(lr=0.00125, eps=1e-15),
                "scheduler": None,
            },
            "canon_features_rest": {
                "optimizer": AdamOptimizerConfig(lr=0.00125 / 20, eps=1e-15),
                "scheduler": None,
            },
            "canon_opacities": {
                "optimizer": AdamOptimizerConfig(lr=0.025, eps=1e-15),
                "scheduler": None,
            },
            "canon_scales": {
                "optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-15),
                "scheduler": None,
            },
            "canon_quats": {
                "optimizer": AdamOptimizerConfig(lr=5e-4, eps=1e-15),
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
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="A fine-tuning variant of Splatfacto",
)


# from __future__ import annotations

# from arti_splatfacto.data.dataparser import ArtiSplatfactoDataParserConfig
# from arti_splatfacto.model.model import ArtiSplatfactoModelConfig


# from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
# from nerfstudio.configs.base_config import ViewerConfig
# from nerfstudio.engine.optimizers import AdamOptimizerConfig
# from nerfstudio.engine.schedulers import (
#     ExponentialDecaySchedulerConfig,
# )
# from nerfstudio.engine.trainer import TrainerConfig
# from nerfstudio.plugins.types import MethodSpecification
# from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
# # from arti_splatfacto.data.datamanager import FullImageDatamanagerConfig, FullImageDatamanager
# from arti_splatfacto.data.datamanager import ArtiSplatfactoManagerConfig

# # arti_splatfacto_config = MethodSpecification(
# #     config=TrainerConfig(
# #         method_name="arti_splatfacto", 
# #         steps_per_eval_batch=100,
# #         steps_per_save=2000,
# #         max_num_iterations=30000,
# #         mixed_precision=False,
# #         pipeline=VanillaPipelineConfig(
# #             datamanager=ArtiSplatfactoManagerConfig(
# #                 dataparser=ArtiSplatfactoDataParserConfig(load_dynamic_objects=True),
# #                 cache_images_type="uint8",
# #             ),
# #             model=ArtiSplatfactoModelConfig(
# #                 refine_every=10,
# #                 cull_alpha_thresh=0.05,     
# #                 densify_grad_thresh=0.002,
# #                 warmup_length=20500,
# #                 stop_split_at=30000,

# #             ),
# #         ),
# #         optimizers={
# #             "means": {
# #                 "optimizer": AdamOptimizerConfig(lr=1.6e-4, eps=1e-15),
# #                 "scheduler": ExponentialDecaySchedulerConfig(
# #                     lr_final=1.6e-6,
# #                     max_steps=30000,
# #                 ),
# #             },
# #             "features_dc": {
# #                 "optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-15),
# #                 "scheduler": None,
# #             },
# #             "features_rest": {
# #                 "optimizer": AdamOptimizerConfig(lr=0.0025 / 20, eps=1e-15),
# #                 "scheduler": None,
# #             },
# #             "opacities": {
# #                 "optimizer": AdamOptimizerConfig(lr=0.05, eps=1e-15),
# #                 "scheduler": None,
# #             },
# #             "scales": {
# #                 "optimizer": AdamOptimizerConfig(lr=0.005, eps=1e-15),
# #                 "scheduler": None,
# #             },
# #             "quats": {"optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15), "scheduler": None},
# #             "camera_opt": {
# #                 "optimizer": AdamOptimizerConfig(lr=1e-4, eps=1e-15),
# #                 "scheduler": ExponentialDecaySchedulerConfig(
# #                     lr_final=5e-7, max_steps=30000, warmup_steps=1000, lr_pre_warmup=0
# #                 ),
# #             },
# #             "bilateral_grid": {
# #                 "optimizer": AdamOptimizerConfig(lr=2e-3, eps=1e-15),
# #                 "scheduler": ExponentialDecaySchedulerConfig(
# #                     lr_final=1e-4, max_steps=30000, warmup_steps=1000, lr_pre_warmup=0
# #                 ),
# #             },
# #         },
# #         viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
# #         vis="viewer",
# #     ),
# #     description="A fine-tuning variant of Splatfacto",
# # )


