"""
InputDataset that loads joint angles and times
"""

from pathlib import Path
from typing import Dict, Literal, Optional, Union

import numpy as np
import torch
import cv2
from PIL import Image

from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.data.datasets.depth_dataset import DepthDataset
from nerfstudio.data.utils.data_utils import get_image_mask_tensor_from_path

class ArtiDataset(InputDataset):
    """Dataset with joint_angle, time, and mask support."""

    def __init__(self, dataparser_outputs: DataparserOutputs, scale_factor: float = 1.0):
        super().__init__(dataparser_outputs, scale_factor)
        self.dp_metadata = dataparser_outputs.metadata or {}
        self.has_joint_angle = self.dp_metadata.get("joint_angles") is not None
        self.has_time = self.dp_metadata.get("times") is not None
        self.has_mask = self.dp_metadata.get("mask") is not None
        self.has_mask_pre = self.dp_metadata.get("mask_pre_filenames") is not None
        self.has_mask_post = self.dp_metadata.get("mask_post_filenames") is not None

    def get_metadata(self, data: Dict) -> Dict:
        """Returns per-frame metadata, including intrinsics + pose."""
        image_idx = data["image_idx"]
        metadata = {}

        cam = self.cameras[image_idx]
        metadata.update({
            "fx": cam.fx.item() if isinstance(cam.fx, torch.Tensor) else float(cam.fx),
            "fy": cam.fy.item() if isinstance(cam.fy, torch.Tensor) else float(cam.fy),
            "cx": cam.cx.item() if isinstance(cam.cx, torch.Tensor) else float(cam.cx),
            "cy": cam.cy.item() if isinstance(cam.cy, torch.Tensor) else float(cam.cy),
            "c2w": cam.camera_to_worlds.detach().clone(),  # (4, 4)
        })

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

        if self.has_mask:
            mask_path = self.dp_metadata["mask"][image_idx]
            if mask_path is not None:
                metadata["mask"] = get_image_mask_tensor_from_path(
                    filepath=mask_path, scale_factor=self.scale_factor
                )

        return metadata

    
class DepthArtiDataset(DepthDataset, ArtiDataset):
    """Dataset with depth, joint_angle, time, and pre/post mask support."""

    def __init__(
        self, 
        dataparser_outputs: DataparserOutputs, 
        scale_factor: float = 1.0,
        cache_compressed_images: bool = False):
        DepthDataset.__init__(self, dataparser_outputs, scale_factor)
        ArtiDataset.__init__(self, dataparser_outputs, scale_factor)

    def get_metadata(self, data: Dict) -> Dict:
        # Start with depth metadata
        metadata = DepthDataset.get_metadata(self, data)
        # Merge in arti-specific metadata
        arti_metadata = ArtiDataset.get_metadata(self, data)
        metadata.update(arti_metadata)
        return metadata