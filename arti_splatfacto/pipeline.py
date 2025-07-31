from __future__ import annotations
from dataclasses import dataclass, field
from importlib import metadata
from typing import Literal, Type, Optional
from cycler import V
import test
import torch
from torch.cuda.amp.grad_scaler import GradScaler

import gra
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
    """
    A custom pipeline class that extends the VanillaPipeline class.
    """

    def __init__(
        self,
        config: ArtiSplatfactoPipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        grad_scaler: Optional[GradScaler] = None,
    ):
        super(VanillaPipeline, self).__init__()
        self.config = config
        self.test_mode = test_mode
        self.datamanager: DataManager = config.datamanager.setup(
            device=device,
            test_mode=test_mode,
            world_size=world_size,
            local_rank=local_rank,
        )
        self.datamanager.to(device)
        metadata =self.datamanager.train_dataset.metadata
