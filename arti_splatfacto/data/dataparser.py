# from __future__ import annotations

# from dataclasses import dataclass, field
# from os import times
# from typing import Literal, Type, Optional
# from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig, Nerfstudio
# from nerfstudio.data.dataparsers.base_dataparser import DataParser, DataParserConfig, DataparserOutputs
# from nerfstudio.utils.io import load_from_json
# from arti_splatfacto.obj_3d_seg import Object3DSeg
# from arti_splatfacto.scene_3d import Scene3D

# import torch
# import json

# @dataclass
# class ArtiSplatfactoDataParserConfig(NerfstudioDataParserConfig):
#     _target: Type = field(default_factory=lambda: ArtiSplatfactoDataParser)
#     metadata_file: str = "transforms.json"
#     obj_mask_dir: str = "obj_masks/"
#     load_dynamic_objects: bool = True

# @dataclass
# class ArtiSplatfactoDataParser(Nerfstudio):
#     config: ArtiSplatfactoDataParserConfig
#     includes_time: bool = True


#     def _generate_dataparser_outputs(self, split="train"):
#         dataparser_outputs: DataparserOutputs = super()._generate_dataparser_outputs(split)

#         assert self.config.data.exists(), f"Data folder {self.config.data} does not exist."


#         # Load your transform.json
#         transform_path = self.config.data / self.config.metadata_file
#         transform_json = load_from_json(transform_path)

#         frames = transform_json.get("frames", [])
#         joint_angles = []
#         times = []

#         for frame in frames:
#             joint_angles.append(float(frame.get("joint_angle", 0.0)))
#             times.append(float(frame.get("time", 0.0)))

#         # Convert to tensors
#         joint_angles = torch.tensor(joint_angles, dtype=torch.float32)
#         times = torch.tensor(times, dtype=torch.float32)


#         # Store these in metadata
#         dataparser_outputs.metadata["joint_angle"] = joint_angles
#         dataparser_outputs.metadata["time"] = times

#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         if self.config.load_dynamic_objects:
#             # Load object segments
#             obj_seg_dir = self.config.data / self.config.obj_mask_dir
#             if obj_seg_dir.exists():
#                 scene = Scene3D.from_directory(obj_seg_dir, device=device)
#                 dataparser_outputs.metadata["scene"] = scene
#             else:
#                 raise FileNotFoundError(f"Object segment directory {obj_seg_dir} does not exist.")
            
#         print(f"metadata keys: {list(dataparser_outputs.metadata.keys())}")
#         print(f"joint_angle shape: {dataparser_outputs.metadata['joint_angle'].shape}")

#         return dataparser_outputs


"""Data parser for nerfstudio datasets."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Literal, Optional, Tuple, Type

import numpy as np
import torch
from PIL import Image

from nerfstudio.cameras import camera_utils
from nerfstudio.cameras.cameras import CAMERA_MODEL_TO_TYPE, Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import DataParser, DataParserConfig, DataparserOutputs
from nerfstudio.data.scene_box import SceneBox

from nerfstudio.utils.io import load_from_json
from nerfstudio.utils.rich_utils import CONSOLE


MAX_AUTO_RESOLUTION = 1600


@dataclass
class ArtiSplatfactoDataParserConfig(DataParserConfig):
    _target: Type = field(default_factory=lambda: ArtiSplatfactoDataParser)

    data: Path = Path()
    """Directory or explicit json file path specifying location of data."""
    scale_factor: float = 1.0
    """How much to scale the camera origins by."""
    downscale_factor: Optional[int] = None
    """How much to downscale images. If not set, images are chosen such that the max dimension is <1600px."""
    scene_scale: float = 1.0
    """How much to scale the region of interest by."""
    orientation_method: Literal["pca", "up", "vertical", "none"] = "up"
    """The method to use for orientation."""
    center_method: Literal["poses", "focus", "none"] = "poses"
    """The method to use to center the poses."""
    auto_scale_poses: bool = True
    """Whether to automatically scale the poses to fit in +/- 1 bounding box."""
    eval_mode: Literal["fraction", "filename", "interval", "all"] = "fraction"
    """
    The method to use for splitting the dataset into train and eval.
    Fraction splits based on a percentage for train and the remaining for eval.
    Filename splits based on filenames containing train/eval.
    Interval uses every nth frame for eval.
    All uses all the images for any split.
    """
    train_split_fraction: float = 0.9
    """The percentage of the dataset to use for training. Only used when eval_mode is train-split-fraction."""
    eval_interval: int = 8
    """The interval between frames to use for eval. Only used when eval_mode is eval-interval."""
    depth_unit_scale_factor: float = 1e-3
    """Scales the depth values to meters. Default value is 0.001 for a millimeter to meter conversion."""
    mask_color: Optional[Tuple[float, float, float]] = None
    """Replace the unknown pixels with this color. Relevant if you have a mask but still sample everywhere."""
    load_3D_points: bool = False
    """Whether to load the 3D points from the colmap reconstruction."""

    obj_mask_dir: str = "obj_masks/"
    load_dynamic_objects: bool = True


@dataclass
class ArtiSplatfactoDataParser(DataParser):
    """DataParser for ArtiSplatfacto scenes."""

    config: ArtiSplatfactoDataParserConfig
    downscale_factor: Optional[int] = None
    includes_time: bool = True

    def _generate_dataparser_outputs(self, split="train"):
        assert self.config.data.exists(), f"Data directory {self.config.data} does not exist."

        # Load transforms file
        meta = load_from_json(self.config.data / "transforms_post.json")
        data_dir = self.config.data
        frames = meta["frames"]

        # Initialize lists
        poses, image_filenames = [], []
        times, joint_angles = [], []

        for frame in frames:
            image_filenames.append(data_dir / frame["file_path"])
            poses.append(np.array(frame["transform_matrix"], dtype=np.float32))

            # Only load time and joint if enabled
            if self.config.load_dynamic_objects:
                times.append(frame.get("time", 0.0))
                joint_angles.append(frame.get("joint_angle", 0.0))

        poses = torch.from_numpy(np.array(poses, dtype=np.float32))

        if self.config.load_dynamic_objects:
            times = torch.tensor(times, dtype=torch.float32)
            joint_angles = torch.tensor(joint_angles, dtype=torch.float32)

        # Scale the translation component
        scale_factor = self.config.scale_factor
        poses[:, :3, 3] *= scale_factor

        N = len(image_filenames)

        if "camera_model" in meta:
            camera_type = CAMERA_MODEL_TO_TYPE[meta["camera_model"]]
        else:
            camera_type = CameraType.PERSPECTIVE

        # Intrinsics
        distortion_params = torch.tensor([
            meta.get("k1", 0.0), meta.get("k2", 0.0),
            meta.get("p1", 0.0), meta.get("p2", 0.0)
        ], dtype=torch.float32).expand(N, -1)

        cameras = Cameras(
            fx=meta["fl_x"],
            fy=meta["fl_y"],
            cx=meta["cx"],
            cy=meta["cy"],
            height=meta["h"],
            width=meta["w"],
            camera_to_worlds=poses[:, :3, :4],
            camera_type=camera_type,
            distortion_params=distortion_params,
            times=times if self.config.load_dynamic_objects and self.includes_time else None,
        )

        # per-camera metadata
        if self.config.load_dynamic_objects:
            for i, cam in enumerate(cameras.flatten()):
                if cam.metadata is None:
                    cam.metadata = {}
                cam.metadata["joint_angle"] = joint_angles[i].item()
                cam.metadata["time"] = times[i].item()


        # Scene bounding box
        aabb_scale = self.config.scene_scale
        scene_box = SceneBox(
            aabb=torch.tensor([[-aabb_scale] * 3, [aabb_scale] * 3], dtype=torch.float32)
        )

        # Collect metadata
        metadata = {
            "scene_path": str(data_dir / self.config.obj_mask_dir),
        }
        if self.config.load_dynamic_objects:
            metadata["joint_angles"] = joint_angles
            metadata["times"] = times

        print(f"metadata keys: {list(metadata.keys())}")
        print(f"joint angles shape: {metadata['joint_angles'].shape if 'joint_angles' in metadata else None}")
        print(f"times shape: {metadata['times'].shape if 'times' in metadata else None}")

        return DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            scene_box=scene_box,
            metadata=metadata,
            dataparser_scale=scale_factor,
            dataparser_transform=torch.eye(4),
        )