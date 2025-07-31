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

        self.has_joint_angle = any("joint_angle" in (getattr(cam, "metadata", {}) or {}) for cam in self.cameras)
        self.has_time = any("time" in (getattr(cam, "metadata", {}) or {}) for cam in self.cameras)

    def get_metadata(self, data: Dict) -> Dict:
        """Returns per-frame metadata."""
        image_idx = data["image_idx"]
        cam = self.cameras[image_idx]

        metadata = {}
        if self.has_joint_angle:
            metadata["joint_angle"] = cam.metadata.get("joint_angle", None)
        if self.has_time:
            metadata["time"] = cam.metadata.get("time", None)
        return metadata

