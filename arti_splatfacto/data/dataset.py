"""
InputDataset that loads joint angles and times
"""

from pathlib import Path
from typing import Dict, Literal, Optional, Union

import numpy as np
import torch
from PIL import Image

from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.data.utils.data_utils import get_image_mask_tensor_from_path

class ArtiDataset(InputDataset):
    """Dataset with joint_angle, time, and pre/post mask support."""

    def __init__(self, dataparser_outputs: DataparserOutputs, scale_factor: float = 1.0):
        super().__init__(dataparser_outputs, scale_factor)
        self.dp_metadata = dataparser_outputs.metadata or {}
        self.has_joint_angle = "joint_angles" in self.dp_metadata
        self.has_time = "times" in self.dp_metadata
        self.has_mask_pre = "mask_pre_filenames" in self.dp_metadata
        self.has_mask_post = "mask_post_filenames" in self.dp_metadata

    def get_metadata(self, data: Dict) -> Dict:
        """Returns per-frame metadata."""
        image_idx = data["image_idx"]
        metadata = {}

        if self.has_joint_angle:
            metadata["joint_angle"] = (
                self.dp_metadata["joint_angles"][image_idx].item()
                if self.dp_metadata["joint_angles"] is not None else None
            )
        if self.has_time:
            metadata["time"] = (
                self.dp_metadata["times"][image_idx].item()
                if self.dp_metadata["times"] is not None else None
            )
        if self.has_mask_pre:
            mask_pre_path = self.dp_metadata["mask_pre_filenames"][image_idx]
            if mask_pre_path is not None:
                metadata["mask_pre"] = get_image_mask_tensor_from_path(
                    filepath=mask_pre_path, scale_factor=self.scale_factor
                )
        if self.has_mask_post:
            mask_post_path = self.dp_metadata["mask_post_filenames"][image_idx]
            if mask_post_path is not None:
                metadata["mask_post"] = get_image_mask_tensor_from_path(
                    filepath=mask_post_path, scale_factor=self.scale_factor
                )


        return metadata