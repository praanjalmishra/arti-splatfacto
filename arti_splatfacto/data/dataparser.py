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
from nerfstudio.data.utils.dataparsers_utils import (
    get_train_eval_split_all,
    get_train_eval_split_filename,
    get_train_eval_split_fraction,
    get_train_eval_split_interval,
)

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
    """DataParser for ArtiSplatfacto scenes with per-frame depth and joint info."""
    config: ArtiSplatfactoDataParserConfig
    downscale_factor: Optional[int] = None
    includes_time: bool = True  # Cameras.times is supported

    def _generate_dataparser_outputs(self, split: str = "train") -> DataparserOutputs:
        assert self.config.data.exists(), f"Data directory {self.config.data} does not exist."

        # load metadata
        if self.config.data.suffix == ".json":
            meta = load_from_json(self.config.data)  
            data_dir = self.config.data.parent
        else:
            meta = load_from_json(self.config.data / "transforms_post.json")
            data_dir = self.config.data

        fx_fixed = "fl_x" in meta
        fy_fixed = "fl_y" in meta
        cx_fixed = "cx" in meta
        cy_fixed = "cy" in meta
        h_fixed  = "h" in meta
        w_fixed  = "w" in meta

        distort_fixed = False
        for k in ["distortion_params", "k1", "k2", "k3", "k4", "p1", "p2"]:
            if k in meta:
                distort_fixed = True
                break

        frames = meta["frames"]
        fnames_resolved = []
        for fr in frames:
            fp = Path(fr["file_path"])
            fnames_resolved.append(self._get_fname(fp, data_dir))
        order = np.argsort([str(p) for p in fnames_resolved])
        frames = [frames[i] for i in order]

        poses = []
        image_filenames, mask_filenames, depth_filenames = [], [], []
        mask_pre_filenames, mask_post_filenames = [], []

        fx_list, fy_list, cx_list, cy_list = [], [], [], []
        h_list, w_list, distort_list = [], [], []

        times_list, joint_list = [], []

        for fr in frames:
            img_path = self._get_fname(Path(fr["file_path"]), data_dir, downsample_folder_prefix="images_")
            image_filenames.append(img_path)

            poses.append(np.array(fr["transform_matrix"], dtype=np.float32))

            if not fx_fixed: fx_list.append(float(fr["fl_x"]))
            if not fy_fixed: fy_list.append(float(fr["fl_y"]))
            if not cx_fixed: cx_list.append(float(fr["cx"]))
            if not cy_fixed: cy_list.append(float(fr["cy"]))
            if not h_fixed:  h_list.append(int(fr["h"]))
            if not w_fixed:  w_list.append(int(fr["w"]))

            if not distort_fixed:
                if "distortion_params" in fr:
                    distort_list.append(torch.tensor(fr["distortion_params"], dtype=torch.float32))
                else:
                    distort_list.append(
                        camera_utils.get_distortion_params(
                            k1=float(fr.get("k1", 0.0)),
                            k2=float(fr.get("k2", 0.0)),
                            k3=float(fr.get("k3", 0.0)),
                            k4=float(fr.get("k4", 0.0)),
                            p1=float(fr.get("p1", 0.0)),
                            p2=float(fr.get("p2", 0.0)),
                        )
                    )

            if "depth_file_path" in fr and fr["depth_file_path"] is not None:
                depth_filenames.append(self._get_fname(Path(fr["depth_file_path"]), data_dir, "depths_"))
            else:
                depth_filenames.append(None)


            ## add mask path
            mask_rel = None
            if "mask_path" in fr:
                mask_rel = Path(fr["mask_path"])
            elif "mask_file_path" in fr:
                mask_rel = Path(fr["mask_file_path"])

            if mask_rel is not None and str(mask_rel) != "":
                mask_filenames.append(self._get_fname(mask_rel, data_dir, "mask"))
            else:
                mask_filenames.append(None)


            # NEW: separate pre- and post-masks
            mask_pre_rel = Path(fr.get("mask_pre_path", "")) if "mask_pre_path" in fr else None
            mask_post_rel = Path(fr.get("mask_post_path", "")) if "mask_post_path" in fr else None

            if mask_pre_rel is not None and str(mask_pre_rel) != "":
                mask_pre_filenames.append(self._get_fname(mask_pre_rel, data_dir, "masks_"))
            else:
                mask_pre_filenames.append(None)

            if mask_post_rel is not None and str(mask_post_rel) != "":
                mask_post_filenames.append(self._get_fname(mask_post_rel, data_dir, "masks_"))
            else:
                mask_post_filenames.append(None)


            if self.config.load_dynamic_objects:
                times_list.append(float(fr.get("time", 0.0)))
                joint_list.append(float(fr.get("joint_angle", 0.0)))

        poses = torch.from_numpy(np.asarray(poses, dtype=np.float32))

        times = torch.tensor(times_list, dtype=torch.float32) if self.config.load_dynamic_objects else None
        joint_angles = torch.tensor(joint_list, dtype=torch.float32) if self.config.load_dynamic_objects else None

        # eval split
        img_names_for_split = [str(p) for p in image_filenames]
        has_split_files_spec = any(f"{s}_filenames" in meta for s in ("train", "val", "test"))
        if f"{split}_filenames" in meta:
            split_set = set(str(self._get_fname(Path(x), data_dir)) for x in meta[f"{split}_filenames"])
            indices = np.array([i for i, p in enumerate(img_names_for_split) if p in split_set], dtype=np.int32)
            CONSOLE.log(f"[yellow] Dataset is overriding {split}_indices to {indices.tolist()}")
        elif has_split_files_spec:
            raise RuntimeError(f"The dataset's list of filenames for split {split} is missing.")
        else:
            if self.config.eval_mode == "fraction":
                i_train, i_eval = get_train_eval_split_fraction(image_filenames, self.config.train_split_fraction)
            elif self.config.eval_mode == "filename":
                i_train, i_eval = get_train_eval_split_filename(image_filenames)
            elif self.config.eval_mode == "interval":
                i_train, i_eval = get_train_eval_split_interval(image_filenames, self.config.eval_interval)
            elif self.config.eval_mode == "all":
                CONSOLE.log("[yellow] Using '--eval-mode=all'. Be careful with camera optimization.")
                i_train, i_eval = get_train_eval_split_all(image_filenames)
            else:
                raise ValueError(f"Unknown eval mode {self.config.eval_mode}")

            indices = i_train if split == "train" else i_eval

        idx = torch.as_tensor(indices, dtype=torch.long)

        if "orientation_override" in meta:
            orientation_method = meta["orientation_override"]
            CONSOLE.log(f"[yellow] Dataset is overriding orientation method to {orientation_method}")
        else:
            orientation_method = self.config.orientation_method

        poses_all, transform_matrix = camera_utils.auto_orient_and_center_poses(
            poses, method=orientation_method, center_method=self.config.center_method
        )

        scale_factor = 1.0
        if self.config.auto_scale_poses:
            scale_factor /= float(torch.max(torch.abs(poses_all[:, :3, 3])))
        scale_factor *= self.config.scale_factor
        poses_all[:, :3, 3] *= scale_factor

        image_filenames = [image_filenames[i] for i in indices]
        mask_filenames   = [mask_filenames[i] if mask_filenames[i] is not None else None for i in indices]
        mask_pre_filenames = [mask_pre_filenames[i] if mask_pre_filenames[i] is not None else None for i in indices]
        mask_post_filenames = [mask_post_filenames[i] if mask_post_filenames[i] is not None else None for i in indices]
        depth_filenames = [depth_filenames[i] for i in indices] if len(depth_filenames) > 0 else []
        poses = poses_all[idx]

        if self.config.load_dynamic_objects:
            times = times[idx] if times is not None else None
            joint_angles = joint_angles[idx] if joint_angles is not None else None

        camera_type = CAMERA_MODEL_TO_TYPE.get(meta.get("camera_model", ""), CameraType.PERSPECTIVE)

        # intrinsics tensors (fixed vs per-frame)
        fx = float(meta["fl_x"]) if fx_fixed else torch.tensor(fx_list, dtype=torch.float32)[idx]
        fy = float(meta["fl_y"]) if fy_fixed else torch.tensor(fy_list, dtype=torch.float32)[idx]
        cx = float(meta["cx"])  if cx_fixed else torch.tensor(cx_list, dtype=torch.float32)[idx]
        cy = float(meta["cy"])  if cy_fixed else torch.tensor(cy_list, dtype=torch.float32)[idx]
        H  = int(meta["h"])     if h_fixed  else torch.tensor(h_list, dtype=torch.int32)[idx]
        W  = int(meta["w"])     if w_fixed  else torch.tensor(w_list, dtype=torch.int32)[idx]

        if distort_fixed:
            distortion_params = (
                torch.tensor(meta["distortion_params"], dtype=torch.float32)
                if "distortion_params" in meta
                else camera_utils.get_distortion_params(
                    k1=float(meta.get("k1", 0.0)),
                    k2=float(meta.get("k2", 0.0)),
                    k3=float(meta.get("k3", 0.0)),
                    k4=float(meta.get("k4", 0.0)),
                    p1=float(meta.get("p1", 0.0)),
                    p2=float(meta.get("p2", 0.0)),
                )
            )
        else:
            distortion_params = torch.stack(distort_list, dim=0)[idx]

        cameras = Cameras(
            fx=fx, fy=fy, cx=cx, cy=cy,
            distortion_params=distortion_params,
            height=H, width=W,
            camera_to_worlds=poses[:, :3, :4],
            camera_type=camera_type,
            times=times if (self.config.load_dynamic_objects and self.includes_time) else None,
        )

        assert self.downscale_factor is not None
        cameras.rescale_output_resolution(scaling_factor=1.0 / self.downscale_factor)

        dataparser_transform = transform_matrix
        if "applied_scale" in meta:
            applied_scale = float(meta["applied_scale"])
            scale_factor *= applied_scale

        aabb_scale = self.config.scene_scale
        scene_box = SceneBox(
            aabb=torch.tensor([[-aabb_scale, -aabb_scale, -aabb_scale],
                               [ aabb_scale,  aabb_scale,  aabb_scale]], dtype=torch.float32)
        )

        # pack metadata
        metadata = {
            "depth_filenames": depth_filenames if any(x is not None for x in depth_filenames) else None,
            "depth_unit_scale_factor": self.config.depth_unit_scale_factor,
            "mask": mask_filenames if any(x is not None for x in mask_filenames) else None,
            # "mask_pre_filenames": mask_pre_filenames if any(x is not None for x in mask_pre_filenames) else None,
            # "mask_post_filenames": mask_post_filenames if any(x is not None for x in mask_post_filenames) else None,
            "scene_path": str(data_dir / self.config.obj_mask_dir),
        }
        if self.config.load_dynamic_objects:
            metadata["times"] = times  # (N,)
            metadata["joint_angles"] = joint_angles  # (N,)

        # print("=== ArtiSplatfactoDataParser DEBUG ===")
        # print(f"Split={split}, num_images={len(image_filenames)}")
        # for i in range(min(5, len(image_filenames))):
        #     print(f"[{i}] img={image_filenames[i]}")
        #     print(f"    depth={depth_filenames[i] if depth_filenames is not None else None}")
        #     print(f"    mask_pre={mask_pre_filenames[i]}")
        #     print(f"    mask_post={mask_post_filenames[i]}")
        #     if self.config.load_dynamic_objects:
        #         print(f"    time={times[i].item():.3f}, joint={joint_angles[i].item():.3f}")
        # print("======================================")

        return DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            scene_box=scene_box,
            mask_filenames=None,
            dataparser_scale=scale_factor,
            dataparser_transform=dataparser_transform,
            metadata=metadata,
        )

    def _get_fname(self, filepath: Path, data_dir: Path, downsample_folder_prefix="images_") -> Path:
        """Get the filename of the image file.
        downsample_folder_prefix can be used to point to auxiliary image data, e.g. masks

        filepath: the base file name of the transformations.
        data_dir: the directory of the data that contains the transform file
        downsample_folder_prefix: prefix of the newly generated downsampled images
        """

        if self.downscale_factor is None:
            if self.config.downscale_factor is None:
                test_img = Image.open(data_dir / filepath)
                h, w = test_img.size
                max_res = max(h, w)
                df = 0
                while True:
                    if (max_res / 2 ** (df)) <= MAX_AUTO_RESOLUTION:
                        break
                    if not (data_dir / f"{downsample_folder_prefix}{2 ** (df + 1)}" / filepath.name).exists():
                        break
                    df += 1

                self.downscale_factor = 2**df
                CONSOLE.log(f"Auto image downscale factor of {self.downscale_factor}")
            else:
                self.downscale_factor = self.config.downscale_factor
            assert self.downscale_factor is not None

        if self.downscale_factor > 1:
            return data_dir / f"{downsample_folder_prefix}{self.downscale_factor}" / filepath.name
        return data_dir / filepath