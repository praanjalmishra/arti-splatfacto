"""
Simplified DataManager for ArtiSplatfacto
"""

from importlib import metadata
import random
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, Literal, Tuple, Type, Union, Generic

import torch
from arti_splatfacto.data.dataset import ArtiDataset
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.datamanagers.full_images_datamanager import (
    FullImageDatamanager,
    FullImageDatamanagerConfig,
)
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.data.datamanagers.base_datamanager import DataManager, DataManagerConfig, TDataset


@dataclass
class ArtiSplatfactoManagerConfig(FullImageDatamanagerConfig):
    """DataManager Config"""
    _target: Type = field(default_factory=lambda: ArtiSplatfactoDataManager)
    camera_res_scale_factor: float = 1.0


class ArtiSplatfactoDataManager(FullImageDatamanager[TDataset], Generic[TDataset]):
    """Simplified DataManager for ArtiSplatfacto"""

    config: ArtiSplatfactoManagerConfig
    train_dataset: ArtiDataset
    eval_dataset: ArtiDataset

    def __init__(
        self,
        config: ArtiSplatfactoManagerConfig,
        device: Union[torch.device, str] = "cpu",
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        **kwargs,
    ):
        self.config = config
        super().__init__(
            config=config,
            device=device,
            test_mode=test_mode,
            world_size=world_size,
            local_rank=local_rank,
            **kwargs,
        )
        self.image_idx = 0
        metadata = self.train_dataparser_outputs.metadata

    def create_train_dataset(self) -> InputDataset:
        return ArtiDataset(
            dataparser_outputs=self.train_dataparser_outputs,
            scale_factor=self.config.camera_res_scale_factor,
        )

    def create_eval_dataset(self) -> InputDataset:
        return ArtiDataset(
            dataparser_outputs=self.dataparser.get_dataparser_outputs(split=self.test_split),
            scale_factor=self.config.camera_res_scale_factor,
        )

    def next_train(self, step: int) -> Tuple[Cameras, Dict]:
        """Returns the next training image and camera."""
        self.image_idx = self.train_unseen_cameras.pop(0)
        if not self.train_unseen_cameras:
            self.train_unseen_cameras = list(range(len(self.train_dataset)))

        data = deepcopy(self.cached_train[self.image_idx])
        data["image"] = data["image"].to(self.device)

        camera = self.train_dataset.cameras[self.image_idx : self.image_idx + 1].to(self.device)
        if camera.metadata is None:
            camera.metadata = {}
        camera.metadata["cam_idx"] = self.image_idx

        return camera, data

    def next_eval(self, step: int) -> Tuple[Cameras, Dict]:
        """Returns the next evaluation image and camera."""
        image_idx = self.eval_unseen_cameras.pop(0)
        if not self.eval_unseen_cameras:
            self.eval_unseen_cameras = list(range(len(self.eval_dataset)))

        data = deepcopy(self.cached_eval[image_idx])
        data["image"] = data["image"].to(self.device)

        camera = self.eval_dataset.cameras[image_idx : image_idx + 1].to(self.device)
        if camera.metadata is None:
            camera.metadata = {}
        camera.metadata["cam_idx"] = image_idx

        return camera, data

    def next_eval_image(self, step: int) -> Tuple[Cameras, Dict]:
        """Returns a random evaluation image (no image reuse tracking)."""
        image_idx = random.randint(0, len(self.eval_dataset) - 1)
        data = deepcopy(self.cached_eval[image_idx])
        data["image"] = data["image"].to(self.device)

        camera = self.eval_dataset.cameras[image_idx : image_idx + 1].to(self.device)
        return camera, data
