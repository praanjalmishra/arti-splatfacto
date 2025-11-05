"""
 This file implements PosedRGBDItem and Record3D dataset in 
    USA-Net (https://github.com/codekansas/usa) project
 Codes are basically adapted from:
    1. https://github.com/codekansas/usa/blob/master/usa/tasks/datasets/posed_rgbd.py
    2. https://github.com/codekansas/usa/blob/master/usa/tasks/datasets/r3d.py

 
 License:
 MIT License

 Copyright (c) 2023 Ben Bolte

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in all
 copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 SOFTWARE.
"""


from __future__ import annotations
import pickle as pkl
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torchvision.transforms.functional as V

import json
import re
from zipfile import ZipFile

import cv2
import liblzfse
from PIL import Image
from quaternion import as_rotation_matrix, quaternion
from torch.utils.data.dataset import Dataset
from tqdm import tqdm
from torch import Tensor
from typing import NamedTuple

class PosedRGBDItem(NamedTuple):
    """Defines a posed RGB image.

    We presume the images and depths to be the distorted images, meaning that
    the depth plane should be flat rather than radial.
    """

    image: Tensor
    depth: Tensor
    mask: Tensor
    intrinsics: Tensor
    pose: Tensor

    def check(self) -> None:
        # Image should have shape (C, H, W)
        assert self.image.dim() == 3
        assert self.image.dtype == torch.float32
        # Depth should have shape (1, H, W)
        assert self.depth.dim() == 3
        assert self.depth.shape[0] == 1
        assert self.depth.dtype == torch.float32
        # Depth shape should match image shape.
        assert self.depth.shape[1:] == self.image.shape[1:]
        assert self.mask.shape[1:] == self.image.shape[1:]
        # Intrinsics should have shape (3, 3)
        assert self.intrinsics.shape == (3, 3)
        assert self.intrinsics.dtype == torch.float64
        # Pose should have shape (4, 4)
        assert self.pose.shape == (4, 4)
        assert self.pose.dtype == torch.float64


@dataclass(frozen=True)
class Metadata:
    rgb_shape: tuple[int, int]
    depth_shape: tuple[int, int]
    fps: int
    timestamps: np.ndarray  # (T) the camera frame timestamps
    intrinsics: np.ndarray  # (3, 3) intrinsics matrix
    poses: np.ndarray  # (T, 4, 4) camera pose matrices
    start_pose: np.ndarray  # (4, 4) initial camera pose matrix


def as_pose_matrix(pose: list[float]) -> np.ndarray:
    """Converts a list of pose parameters to a pose matrix.

    Args:
        pose: The list of pose parameters, (qx, qy, qz, qw, px, py, pz)

    Returns:
        A (4, 4) pose matrix
    """
    mat = np.eye(4, dtype=np.float64)
    qx, qy, qz, qw, px, py, pz = pose
    mat[:3, :3] = as_rotation_matrix(quaternion(qw, qx, qy, qz))
    mat[:3, 3] = [px, py, pz]
    return mat


def read_metadata(r3d_file: ZipFile, use_depth_shape: bool) -> Metadata:
    """Read metadata from R3D file."""
    with r3d_file.open("metadata", "r") as f:
        metadata_dict = json.load(f)

    if "dh" in metadata_dict and "dw" in metadata_dict:
        depth_shape = (metadata_dict["dh"], metadata_dict["dw"])
    else:
        depth_shape = (256, 192)
        print(f"WARNING: depth parameters not found! using default {depth_shape=}")

    if "frameTimestamps" in metadata_dict:
        timestamps = np.array(metadata_dict["frameTimestamps"], dtype=np.float64)
    else:
        print("WARNING: timestamps not found!")
        timestamps = np.arange(len(metadata_dict["poses"])) / metadata_dict["fps"]

    metadata = Metadata(
        rgb_shape=(metadata_dict["h"], metadata_dict["w"]),
        depth_shape=depth_shape,
        fps=metadata_dict["fps"],
        timestamps=timestamps,
        intrinsics=np.array(metadata_dict["K"], dtype=np.float64).reshape(3, 3).T,
        poses=np.stack([as_pose_matrix(pose) for pose in metadata_dict["poses"]], axis=0),
        start_pose=as_pose_matrix(metadata_dict["initPose"]),
    )

    if use_depth_shape:
        metadata.intrinsics[0, :] *= metadata.depth_shape[1] / metadata.rgb_shape[1]
        metadata.intrinsics[1, :] *= metadata.depth_shape[0] / metadata.rgb_shape[0]

    assert metadata.timestamps.shape[0] == metadata.poses.shape[0]
    assert metadata.poses.shape[1:] == (4, 4)
    assert metadata.start_pose.shape == (4, 4)

    return metadata


class R3DDataset(Dataset[PosedRGBDItem]):
    """
    Streaming version of R3D Dataset that loads frames on-demand.
    
    This version is memory-efficient and suitable for:
    - Large captures with many frames
    - Full-resolution RGB (use_depth_shape=False)
    
    """
    
    def __init__(
        self,
        path,
        *,
        use_depth_shape: bool = True,
        keep_zipfile_open: bool = False,
        downsample_factor: int = 1,
    ) -> None:
        """
        Initialize  R3D dataset.

        Args:
            path: The path to the .r3d file
            use_depth_shape: If True, images are resized to depth shape (192x256);
                           if False, depth is resized to RGB shape (more memory per frame)
            keep_zipfile_open: If True, keeps ZipFile handle open for faster access.
            downsample_factor: Factor by which to downsample images. 

        """
        path = Path(path)
        assert path.suffix == ".r3d", f"Invalid file suffix: {path.suffix} Expected `.r3d`"
        assert downsample_factor >= 1, f"downsample_factor must be >= 1, got {downsample_factor}"

        self.path = path
        self.use_depth_shape = use_depth_shape
        self.keep_zipfile_open = keep_zipfile_open
        self.downsample_factor = downsample_factor
        
        with ZipFile(path) as r3d_file:
            self.metadata = read_metadata(r3d_file, self.use_depth_shape)
        
        self.rgb_h, self.rgb_w = self.metadata.rgb_shape
        self.depth_h, self.depth_w = self.metadata.depth_shape
        base_h, base_w = (
            (self.depth_h, self.depth_w) if use_depth_shape else (self.rgb_h, self.rgb_w)
        )
        
        self.output_h = base_h // downsample_factor
        self.output_w = base_w // downsample_factor
        
        self.num_frames = self.metadata.timestamps.shape[0]
        
        self.intrinsics = self.metadata.intrinsics.copy()
        if downsample_factor > 1:
            # Scale focal lengths and principal point
            scale = 1.0 / downsample_factor
            self.intrinsics[0, :] *= scale  # fx and cx
            self.intrinsics[1, :] *= scale  # fy and cy
        
        affine_matrix_1 = np.array([
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1],
        ])
        self.poses = self.metadata.poses @ affine_matrix_1

        affine_matrix_2 = np.array([
            [1, 0, 0, 0],
            [0, 0, -1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
        ])
        self.poses = affine_matrix_2 @ self.poses
        
        self._zipfile_handle: Optional[ZipFile] = None
        if keep_zipfile_open:
            self._zipfile_handle = ZipFile(path)
            
        print(f"R3D Dataset initialized (streaming mode):")
        print(f"  Frames: {self.num_frames}")
        print(f"  RGB shape: {self.rgb_h}x{self.rgb_w}")
        print(f"  Depth shape: {self.depth_h}x{self.depth_w}")
        print(f"  Use depth shape: {use_depth_shape}")
        print(f"  Downsample factor: {downsample_factor}")
        print(f"  Final output shape: {self.output_h}x{self.output_w}")
        print(f"  Keep ZipFile open: {keep_zipfile_open}")

    def __len__(self) -> int:
        return self.num_frames

    def __del__(self):
        """Close ZipFile handle when object is destroyed."""
        if self._zipfile_handle is not None:
            self._zipfile_handle.close()

    def _get_zipfile(self) -> ZipFile:
        """Get ZipFile handle (either persistent or temporary)."""
        if self._zipfile_handle is not None:
            return self._zipfile_handle
        else:
            return ZipFile(self.path)

    def _load_frame_data(self, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load RGB, depth, and mask for a single frame.
        
        Args:
            index: Frame index
            
        Returns:
            Tuple of (rgb_array, depth_array, mask_array) with final output shape
        """
        need_to_close = self._zipfile_handle is None
        r3d_file = self._get_zipfile()
        
        try:
            img_fname = f"rgbd/{index}.jpg"
            with r3d_file.open(img_fname, "r") as img_f:
                img_arr = np.asarray(Image.open(img_f))
            
            assert img_arr.shape == (self.rgb_h, self.rgb_w, 3), \
                f"Expected shape ({self.rgb_h}, {self.rgb_w}, 3), got {img_arr.shape}"
            
            if self.use_depth_shape:
                img_arr = cv2.resize(img_arr, (self.depth_w, self.depth_h), interpolation=cv2.INTER_LINEAR)
            
            depth_fname = f"rgbd/{index}.depth"
            with r3d_file.open(depth_fname, "r") as depth_f:
                raw_bytes = depth_f.read()
                decompressed_bytes = liblzfse.decompress(raw_bytes)
                depth_arr = np.frombuffer(decompressed_bytes, dtype=np.float32).reshape(
                    self.depth_h, self.depth_w
                ).copy()
            
            depth_is_nan = np.isnan(depth_arr)
            depth_arr[depth_is_nan] = -1.0
            
            if not self.use_depth_shape:
                depth_arr = cv2.resize(depth_arr, (self.rgb_w, self.rgb_h), interpolation=cv2.INTER_NEAREST)
            
            conf_fname = f"rgbd/{index}.conf"
            try:
                with r3d_file.open(conf_fname, "r") as conf_f:
                    raw_bytes = conf_f.read()
                    decompressed_bytes = liblzfse.decompress(raw_bytes)
                    conf_arr = np.frombuffer(decompressed_bytes, dtype=np.uint8).reshape(
                        self.depth_h, self.depth_w
                    ).copy()
                
                conf_arr[depth_is_nan] = 0
                    
            except KeyError:
                conf_arr = np.zeros((self.depth_h, self.depth_w), dtype=np.uint8)
                conf_arr[depth_arr < 3] = 2
            
            if not self.use_depth_shape:
                conf_arr = cv2.resize(conf_arr, (self.rgb_w, self.rgb_h), interpolation=cv2.INTER_NEAREST)
            
            if self.downsample_factor > 1:
                img_arr = cv2.resize(
                    img_arr, 
                    (self.output_w, self.output_h), 
                    interpolation=cv2.INTER_AREA  
                )
                depth_arr = cv2.resize(
                    depth_arr, 
                    (self.output_w, self.output_h), 
                    interpolation=cv2.INTER_NEAREST  
                )
                conf_arr = cv2.resize(
                    conf_arr, 
                    (self.output_w, self.output_h), 
                    interpolation=cv2.INTER_NEAREST  
                )
            
            mask_arr = conf_arr != 2
            
            return img_arr, depth_arr, mask_arr
            
        finally:
            if need_to_close:
                r3d_file.close()

    def __getitem__(self, index: int) -> PosedRGBDItem:
        """
        Load a single frame on-demand.
        
        Args:
            index: Frame index
            
        Returns:
            PosedRGBDItem containing image, depth, mask, intrinsics, and pose
        """
        if index < 0 or index >= self.num_frames:
            raise IndexError(f"Index {index} out of range [0, {self.num_frames})")
        
        # Load frame data
        img_arr, depth_arr, mask_arr = self._load_frame_data(index)
        
        img = torch.from_numpy(img_arr).permute(2, 0, 1)  # (H, W, 3) -> (3, H, W)
        img = V.convert_image_dtype(img, torch.float32)
        
        depth = torch.from_numpy(depth_arr).unsqueeze(0)  # (H, W) -> (1, H, W)
        mask = torch.from_numpy(mask_arr).unsqueeze(0)    # (H, W) -> (1, H, W)
        
        intr = torch.from_numpy(self.intrinsics)
        pose = torch.from_numpy(self.poses[index])
        
        item = PosedRGBDItem(
            image=img,
            depth=depth,
            mask=mask,
            intrinsics=intr,
            pose=pose
        )
        
        return item
