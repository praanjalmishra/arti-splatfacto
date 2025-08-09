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

class ArtiDataset(InputDataset):
    """Dataset with joint_angle and time support."""

    def __init__(self, dataparser_outputs: DataparserOutputs, scale_factor: float = 1.0):
        super().__init__(dataparser_outputs, scale_factor)

        self.dp_metadata = dataparser_outputs.metadata or {}
        self.has_joint_angle = "joint_angles" in self.dp_metadata
        self.has_time = "times" in self.dp_metadata

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
        return metadata


