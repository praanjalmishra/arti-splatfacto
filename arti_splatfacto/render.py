#!/usr/bin/env python3
"""
Joint-Conditioned ArtiSplatfacto Inference Script with Camera Path Support
Uses Nerfstudio's rendering utilities for robust video generation.
"""
import json
import sys
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Optional, Union

import mediapy as media
import numpy as np
import torch
import tyro
import cv2
import math
from rich import box, style
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from typing_extensions import Annotated

from nerfstudio.cameras.camera_paths import get_path_from_json
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.model_components import renderers
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.utils import colormaps
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.utils.rich_utils import CONSOLE, ItersPerSecColumn



def format_joint_value(value: float, joint_type: str) -> str:
    """Format joint value for visualization based on joint type."""
    if joint_type == "revolute":
        return f"{value * 180.0 / math.pi:.1f} deg"
    elif joint_type == "prismatic":
        return f"{value:.3f} m"
    else:
        return f"{value:.3f}"


def get_default_joint_limits(joint_type: str):
    if joint_type == "revolute":
        return -math.pi, math.pi
    elif joint_type == "prismatic":
        return 0.0, 1.0   # safe generic travel range
    else:
        return -1.0, 1.0


def draw_joint_label(
    image: np.ndarray,
    text: str,
    *,
    margin: int = 30,
    color=(0, 255, 0),
):
    h, w = image.shape[:2]

    # Scale text relative to resolution
    font_scale = max(0.8, min(w / 800.0, 2.0))
    thickness = max(2, int(font_scale * 2))

    # Compute text size
    (text_w, text_h), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
    )

    # Top-left anchored position
    x = margin
    y = margin + text_h

    # Background box for readability
    cv2.rectangle(
        image,
        (x - 10, y - text_h - 10),
        (x + text_w + 10, y + baseline + 10),
        (0, 0, 0),
        thickness=-1,
    )

    # Draw text
    cv2.putText(
        image,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def render_joint_conditioned_trajectory(
    pipeline: Pipeline,
    cameras: Cameras,
    joint_angles: torch.Tensor,
    output_filename: Path,
    rendered_output_names: List[str],
    rendered_resolution_scaling_factor: float = 1.0,
    output_format: Literal["images", "video"] = "video",
    image_format: Literal["jpeg", "png"] = "png",
    jpeg_quality: int = 100,
    depth_near_plane: Optional[float] = None,
    depth_far_plane: Optional[float] = None,
    colormap_options: colormaps.ColormapOptions = colormaps.ColormapOptions(),
    fps: int = 30,
    joint_min: Optional[float] = None,
    joint_max: Optional[float] = None,
) -> None:
    """
    Render trajectory across multiple joint angles for articulated objects.
    
    Args:
        pipeline: Pipeline to evaluate with.
        cameras: Cameras to render (one or more camera poses).
        joint_angles: Tensor of joint angles to test at each camera pose.
        output_filename: Name of the output file/directory.
        rendered_output_names: List of outputs to visualize (e.g., ["rgb", "depth"]).
        rendered_resolution_scaling_factor: Scaling factor for camera resolution.
        output_format: "images" or "video".
        image_format: "jpeg" or "png".
        jpeg_quality: JPEG quality (0-100).
        depth_near_plane: Nearest depth for colormap. If None, use min value.
        depth_far_plane: Farthest depth for colormap. If None, use max value.
        colormap_options: Options for colormap.
        fps: Frames per second for video output.
    """
    CONSOLE.print("[bold green]Creating joint-conditioned trajectory " + output_format)
    
    # Verify model is joint-conditioned and get joint parameters
    model = pipeline.model

    # Detect Architecture
    if hasattr(model, 'all_joint_params') and model.all_joint_params:
        # Multi-joint architecture
        active_joint_id = getattr(model.config, 'active_joint_id', None)
        if active_joint_id is None:
            active_joint_id = sorted(model.all_joint_params.keys())[0]
            CONSOLE.print(f"[yellow]No active joint specified, using: {active_joint_id}[/yellow]")

        if active_joint_id not in model.all_joint_params:
            raise ValueError(
                f"Joint '{active_joint_id}' not found. "
                f"Available joints: {list(model.all_joint_params.keys())}"
            )

        joint_params = model.all_joint_params[active_joint_id]

        # Determine joint type
        joint_type_attr = f'joint_type_{active_joint_id}'
        if hasattr(model, joint_type_attr):
            joint_type = getattr(model, joint_type_attr)
        elif hasattr(model, 'joint_type'):
            joint_type = model.joint_type
        else:
            joint_type = "revolute"
            CONSOLE.print(f"[yellow]Joint type not found, assuming {joint_type}[/yellow]")

        # ---------------------------------
        # Only compute model limits if user
        # did NOT pass joint_min/max
        # ---------------------------------
        if joint_min is None or joint_max is None:
            if "angles" in joint_params:
                joint_min = joint_params["angles"].min().item()
                joint_max = joint_params["angles"].max().item()
            else:
                joint_min, joint_max = get_default_joint_limits(joint_type)

        else:
            # USER OVERRIDE
            CONSOLE.print(
                f"[green]Using USER-SPECIFIED joint limits "
                f"for '{active_joint_id}': [{joint_min:.3f}, {joint_max:.3f}][/green]"
            )

    # Legacy single-joint case
    elif hasattr(model, 'joint_angle_deltas'):
        active_joint_id = "joint_0"
        if joint_min is None or joint_max is None:
            joint_min = model.joint_limits[0].item()
            joint_max = model.joint_limits[1].item()
            CONSOLE.print(
                f"[cyan]Using MODEL joint limits (legacy): "
                f"[{joint_min:.3f}, {joint_max:.3f}]"
            )
        else:
            CONSOLE.print(
                f"[green]Using USER-SPECIFIED joint limits "
                f"[{joint_min:.3f}, {joint_max:.3f}]"
            )

        joint_type = model.joint_type
        CONSOLE.print("Using legacy single-joint architecture")

    else:
        raise ValueError("Model is not joint-conditioned!")

    
    CONSOLE.print(f"Joint type: {joint_type}")
    CONSOLE.print(f"Joint limits: [{joint_min:.3f}, {joint_max:.3f}]")
    CONSOLE.print(f"Testing {len(joint_angles)} joint angles across {len(cameras)} camera poses")
    
    cameras.rescale_output_resolution(rendered_resolution_scaling_factor)
    cameras = cameras.to(pipeline.device)
    
    # Calculate total frames
    total_frames = len(cameras) * len(joint_angles)
    
    progress = Progress(
        TextColumn(":movie_camera: Rendering :movie_camera:"),
        BarColumn(),
        TaskProgressColumn(
            text_format="[progress.percentage]{task.completed}/{task.total:>.0f}({task.percentage:>3.1f}%)",
            show_speed=True,
        ),
        ItersPerSecColumn(suffix="fps"),
        TimeRemainingColumn(elapsed_when_finished=False, compact=False),
        TimeElapsedColumn(),
    )
    
    output_image_dir = output_filename.parent / output_filename.stem
    if output_format == "images":
        output_image_dir.mkdir(parents=True, exist_ok=True)
    if output_format == "video":
        output_filename.parent.mkdir(parents=True, exist_ok=True)
    
    with ExitStack() as stack:
        writer = None
        
    with progress:
        num_cams = cameras.size
        total_frames = num_cams
        task = progress.add_task("Rendering", total=total_frames)
        frame_idx = 0

        for camera_idx in range(num_cams):
            progress.update(task, advance=1)

            camera = cameras[camera_idx : camera_idx + 1]

            # Choose joint angle for this camera index
            if len(joint_angles) == 1:
                # Constant angle over the whole path
                joint_angle = joint_angles[0]
            else:
                # Smooth interpolation from joint_min → joint_max along the path
                alpha = camera_idx / max(num_cams - 1, 1)  # in [0,1]
                joint_angle = joint_min + alpha * (joint_max - joint_min)

            unclamped_angle = float(joint_angle)
            clamped_angle = float(torch.clamp(torch.tensor(unclamped_angle), joint_min, joint_max))

            # Per-frame metadata for the active joint
            camera.metadata = {}
            angle_key = f"joint_angles_{active_joint_id}"
            camera.metadata[angle_key] = torch.tensor(
                [[clamped_angle]],
                device=model.device,
                dtype=torch.float32,
            )

            # Render
            with torch.no_grad():
                outputs = pipeline.model.get_outputs_for_camera(camera)

            render_image = []
            for rendered_output_name in rendered_output_names:
                if rendered_output_name not in outputs:
                    CONSOLE.rule("Error", style="red")
                    CONSOLE.print(
                        f"Could not find {rendered_output_name} in the model outputs",
                        justify="center",
                    )
                    CONSOLE.print(
                        f"Please set --rendered-output-names to one of: {outputs.keys()}",
                        justify="center",
                    )
                    sys.exit(1)

                output_image = outputs[rendered_output_name]
                is_depth = rendered_output_name.find("depth") != -1

                if is_depth:
                    output_image = (
                        colormaps.apply_depth_colormap(
                            output_image,
                            accumulation=outputs.get("accumulation"),
                            near_plane=depth_near_plane,
                            far_plane=depth_far_plane,
                            colormap_options=colormap_options,
                        )
                        .cpu()
                        .numpy()
                    )
                else:
                    output_image = (
                        colormaps.apply_colormap(
                            image=output_image,
                            colormap_options=colormap_options,
                        )
                        .cpu()
                        .numpy()
                    )

                render_image.append(output_image)

            # Concatenate outputs horizontally
            # render_image = np.concatenate(render_image, axis=1)
            # Stack outputs along x axis
            render_image = np.concatenate(render_image, axis=0)

            label_value = format_joint_value(clamped_angle, joint_type)
            label = f"{active_joint_id}: {label_value}"

            # Convert float32 render_image (0–1 or 0–255) to uint8 BGR
            if render_image.dtype != np.uint8:
                img_vis = (render_image * 255).astype(np.uint8)
            else:
                img_vis = render_image.copy()

            img_vis = cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR)

            draw_joint_label(img_vis, label)

            # Convert back to RGB for saving
            img_out = cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB)

            if output_format == "images":
                if image_format == "png":
                    media.write_image(
                        output_image_dir / f"{frame_idx:05d}.png",
                        img_out,
                        fmt="png",
                    )
                elif image_format == "jpeg":
                    media.write_image(
                        output_image_dir / f"{frame_idx:05d}.jpg",
                        img_out,
                        fmt="jpeg",
                        quality=jpeg_quality,
                    )

            if output_format == "video":
                if writer is None:
                    render_width = int(img_out.shape[1])
                    render_height = int(img_out.shape[0])
                    writer = stack.enter_context(
                        media.VideoWriter(
                            path=output_filename,
                            shape=(render_height, render_width),
                            fps=fps,
                        )
                    )
                writer.add_image(img_out)

            frame_idx += 1

    
    table = Table(
        title=None,
        show_header=False,
        box=box.MINIMAL,
        title_style=style.Style(bold=True),
    )
    if output_format == "video":
        table.add_row("Video", str(output_filename))
    else:
        table.add_row("Images", str(output_image_dir))
    
    CONSOLE.print(Panel(
        table,
        title="[bold][green]:tada: Render Complete :tada:[/bold]",
        expand=False
    ))


@dataclass
class RenderJointConditionedPath:
    """Render a camera path with varying joint angles for articulated objects."""
    
    load_config: Path
    """Path to config YAML file."""
    camera_path: Path = Path("camera_path.json")
    """Filename of the camera path to render."""
    output_path: Path = Path("renders/joint_render.mp4")
    """Path to output video file or directory."""
    
    # Joint angle parameters
    num_angles: int = 10
    """Number of joint angles to test at each camera pose."""
    angle_range: Optional[List[float]] = None
    """Custom angle range [min, max]. If None, use model's joint limits."""
    random_angles: bool = False
    """Use random instead of linearly spaced joint angles."""
    seed: Optional[int] = None
    """Random seed for reproducibility when using random angles."""
    
    # Rendering parameters
    rendered_output_names: List[str] = field(default_factory=lambda: ["rgb"])
    """Name of the renderer outputs to use. rgb, depth, etc. Concatenates them along x axis."""
    output_format: Literal["images", "video"] = "video"
    """How to save output data."""
    image_format: Literal["jpeg", "png"] = "png"
    """Image format for output."""
    jpeg_quality: int = 100
    """JPEG quality (0-100)."""
    downscale_factor: float = 1.0
    """Scaling factor to apply to the camera image resolution."""
    fps: int = 30
    """Frames per second for video output."""
    
    # Depth visualization parameters
    depth_near_plane: Optional[float] = None
    """Closest depth to consider when using the colormap for depth. If None, use min value."""
    depth_far_plane: Optional[float] = None
    """Furthest depth to consider when using the colormap for depth. If None, use max value."""
    colormap_options: colormaps.ColormapOptions = field(default_factory=colormaps.ColormapOptions)
    """Colormap options."""
    
    eval_num_rays_per_chunk: Optional[int] = None
    """Specifies number of rays per chunk during eval. If None, use the value in the config file."""
    
    def main(self) -> None:
        """Main function."""
        # Load pipeline
        _, pipeline, _, _ = eval_setup(
            self.load_config,
            eval_num_rays_per_chunk=self.eval_num_rays_per_chunk,
            test_mode="inference",
        )
        
        model = pipeline.model
        
        # Verify joint-conditioned model (supports both architectures)
        has_multi_joint = hasattr(model, 'all_joint_params') and model.all_joint_params
        has_single_joint = hasattr(model, 'joint_angle_deltas')
        
        if not (has_multi_joint or has_single_joint):
            CONSOLE.rule("Error", style="red")
            CONSOLE.print(
                "Model does not appear to be joint-conditioned!",
                justify="center"
            )
            CONSOLE.print(
                "Expected either 'all_joint_params' (multi-joint) or 'joint_angle_deltas' (single-joint)",
                justify="center"
            )
            sys.exit(1)
        
        # Get joint parameters
        if has_multi_joint:
            active_joint_id = getattr(model.config, 'active_joint_id', None)
            if active_joint_id is None:
                active_joint_id = sorted(model.all_joint_params.keys())[0]
            CONSOLE.print(f"[bold]Multi-joint model detected, using joint: {active_joint_id}")
            
            # ModuleDict uses dict-style indexing, not .get()
            if active_joint_id in model.all_joint_params:
                joint_params = model.all_joint_params[active_joint_id]
                if "angles" in joint_params:
                    default_min = joint_params["angles"].min().item()
                    default_max = joint_params["angles"].max().item()
                else:
                    default_min, default_max = -3.14, 3.14
            else:
                default_min, default_max = -3.14, 3.14
        else:
            CONSOLE.print("[bold]Single-joint model detected")
            default_min = model.joint_limits[0].item()
            default_max = model.joint_limits[1].item()
        
        # Load camera path
        CONSOLE.print(f"[bold]Loading camera path from {self.camera_path}")
        with open(self.camera_path, "r", encoding="utf-8") as f:
            camera_path_data = json.load(f)
        
        camera_path = get_path_from_json(camera_path_data)
        CONSOLE.print(f"Loaded {len(camera_path)} camera poses")
        
        # Determine joint angle range
        if self.angle_range is not None:
            joint_min, joint_max = self.angle_range
            CONSOLE.print(f"Using custom angle range: [{joint_min:.3f}, {joint_max:.3f}]")
        else:
            joint_min, joint_max = default_min, default_max
            CONSOLE.print(f"Using model's joint limits: [{joint_min:.3f}, {joint_max:.3f}]")
        
        # Create joint angles
        if self.seed is not None:
            torch.manual_seed(self.seed)
            CONSOLE.print(f"Using random seed: {self.seed}")
        
        if self.random_angles:
            joint_angles = torch.empty(self.num_angles).uniform_(joint_min, joint_max)
            joint_angles = torch.sort(joint_angles).values
            CONSOLE.print(f"🔀 Using {self.num_angles} random joint angles")
        else:
            joint_angles = torch.linspace(joint_min, joint_max, self.num_angles)
            CONSOLE.print(f"📈 Using {self.num_angles} linearly spaced joint angles")
        
        # Add .mp4 suffix to video output if none is specified
        if self.output_format == "video" and str(self.output_path.suffix) == "":
            self.output_path = self.output_path.with_suffix(".mp4")
        
        # Render
        render_joint_conditioned_trajectory(
            pipeline=pipeline,
            cameras=camera_path,
            joint_angles=joint_angles,
            output_filename=self.output_path,
            rendered_output_names=self.rendered_output_names,
            rendered_resolution_scaling_factor=1.0 / self.downscale_factor,
            output_format=self.output_format,
            image_format=self.image_format,
            jpeg_quality=self.jpeg_quality,
            depth_near_plane=self.depth_near_plane,
            depth_far_plane=self.depth_far_plane,
            colormap_options=self.colormap_options,
            fps=self.fps,
            joint_min=joint_min,
            joint_max=joint_max,
        )


@dataclass
class RenderJointConditionedDataset:
    """Render train/eval dataset images at multiple joint angles."""
    
    load_config: Path
    """Path to config YAML file."""
    output_path: Path = Path("renders/dataset")
    """Path to output directory."""
    
    # Dataset parameters
    split: Literal["train", "eval"] = "eval"
    """Which dataset split to render."""
    camera_indices: Optional[List[int]] = None
    """Specific camera indices to render. If None, render all."""
    num_cameras: int = 10
    """Number of cameras to sample if camera_indices is None."""
    
    # Joint angle parameters
    num_angles: int = 50
    """Number of joint angles to test at each camera pose."""
    angle_range: Optional[List[float]] = None
    """Custom angle range [min, max]. If None, use model's joint limits."""
    random_angles: bool = False
    """Use random instead of linearly spaced joint angles."""
    seed: Optional[int] = None
    """Random seed for reproducibility."""
    
    # Rendering parameters
    rendered_output_names: List[str] = field(default_factory=lambda: ["rgb"])
    """Name of the renderer outputs to use."""
    output_format: Literal["images", "video"] = "video"
    """How to save output data."""
    image_format: Literal["jpeg", "png"] = "png"
    """Image format."""
    jpeg_quality: int = 100
    """JPEG quality."""
    downscale_factor: float = 1.0
    """Scaling factor for resolution."""
    fps: int = 30
    """FPS for video."""
    
    depth_near_plane: Optional[float] = None
    """Near plane for depth colormap."""
    depth_far_plane: Optional[float] = None
    """Far plane for depth colormap."""
    colormap_options: colormaps.ColormapOptions = field(default_factory=colormaps.ColormapOptions)
    """Colormap options."""
    
    eval_num_rays_per_chunk: Optional[int] = None
    """Rays per chunk during eval."""
    
    def main(self) -> None:
        """Main function."""
        # Load pipeline
        _, pipeline, _, _ = eval_setup(
            self.load_config,
            eval_num_rays_per_chunk=self.eval_num_rays_per_chunk,
            test_mode="inference",
        )
        
        model = pipeline.model
        
        # Verify joint-conditioned model (supports both architectures)
        has_multi_joint = hasattr(model, 'all_joint_params') and model.all_joint_params
        has_single_joint = hasattr(model, 'joint_angle_deltas')
        
        if not (has_multi_joint or has_single_joint):
            CONSOLE.rule("Error", style="red")
            CONSOLE.print(
                "Model does not appear to be joint-conditioned!",
                justify="center"
            )
            CONSOLE.print(
                "Expected either 'all_joint_params' (multi-joint) or 'joint_angle_deltas' (single-joint)",
                justify="center"
            )
            sys.exit(1)
        
        # Get joint parameters
        if has_multi_joint:
            active_joint_id = getattr(model.config, 'active_joint_id', None)
            if active_joint_id is None:
                active_joint_id = sorted(model.all_joint_params.keys())[0]
            CONSOLE.print(f"[bold]Multi-joint model detected, using joint: {active_joint_id}")
            
            # ModuleDict uses dict-style indexing, not .get()
            if active_joint_id in model.all_joint_params:
                joint_params = model.all_joint_params[active_joint_id]
                if "angles" in joint_params:
                    default_min = joint_params["angles"].min().item()
                    default_max = joint_params["angles"].max().item()
                else:
                    default_min, default_max = -3.14, 3.14
            else:
                default_min, default_max = -3.14, 3.14
        else:
            CONSOLE.print("[bold]Single-joint model detected")
            default_min = model.joint_limits[0].item()
            default_max = model.joint_limits[1].item()
        
        # Get dataset
        if self.split == "train":
            if not hasattr(pipeline.datamanager, "train_dataset"):
                CONSOLE.rule("Error", style="red")
                CONSOLE.print("No train dataset found", justify="center")
                sys.exit(1)
            dataset = pipeline.datamanager.train_dataset
        else:
            if not hasattr(pipeline.datamanager, "eval_dataset"):
                CONSOLE.rule("Error", style="red")
                CONSOLE.print("No eval dataset found", justify="center")
                sys.exit(1)
            dataset = pipeline.datamanager.eval_dataset
        
        # Select cameras
        if self.camera_indices is not None:
            indices = self.camera_indices
        else:
            num_available = len(dataset)
            indices = np.linspace(0, num_available - 1, self.num_cameras, dtype=int).tolist()
        
        CONSOLE.print(f"Rendering {len(indices)} cameras from {self.split} dataset")
        
        # Create cameras object
        cameras_list = []
        for idx in indices:
            camera = dataset.cameras[idx:idx+1].to(model.device)
            
            # Preserve metadata
            if hasattr(dataset, 'metadata'):
                camera.metadata = {}
                metadata = dataset.metadata
                for key, value in metadata.items():
                    if torch.is_tensor(value) and value.ndim > 0:
                        camera.metadata[key] = value[idx:idx+1]
                    elif isinstance(value, (list, tuple)):
                        camera.metadata[key] = value[idx]
                    else:
                        camera.metadata[key] = value
            
            cameras_list.append(camera)
        
        # Stack into single Cameras object
        cameras = Cameras(
            camera_to_worlds=torch.cat([c.camera_to_worlds for c in cameras_list]),
            fx=torch.cat([c.fx for c in cameras_list]),
            fy=torch.cat([c.fy for c in cameras_list]),
            cx=torch.cat([c.cx for c in cameras_list]),
            cy=torch.cat([c.cy for c in cameras_list]),
            height=torch.cat([c.height for c in cameras_list]),
            width=torch.cat([c.width for c in cameras_list]),
            camera_type=torch.cat([c.camera_type for c in cameras_list]),
        )
        
        # Determine joint angles
        if self.angle_range is not None:
            joint_min, joint_max = self.angle_range
        else:
            joint_min, joint_max = default_min, default_max
        
        if self.seed is not None:
            torch.manual_seed(self.seed)
        
        if self.random_angles:
            joint_angles = torch.empty(self.num_angles).uniform_(joint_min, joint_max)
            joint_angles = torch.sort(joint_angles).values
        else:
            joint_angles = torch.linspace(joint_min, joint_max, self.num_angles)
        
        # Create output filename
        output_filename = self.output_path / f"{self.split}_joint_conditioned"
        if self.output_format == "video":
            output_filename = output_filename.with_suffix(".mp4")
        
        # Render
        render_joint_conditioned_trajectory(
            pipeline=pipeline,
            cameras=cameras,
            joint_angles=joint_angles,
            output_filename=output_filename,
            rendered_output_names=self.rendered_output_names,
            rendered_resolution_scaling_factor=1.0 / self.downscale_factor,
            output_format=self.output_format,
            image_format=self.image_format,
            jpeg_quality=self.jpeg_quality,
            depth_near_plane=self.depth_near_plane,
            depth_far_plane=self.depth_far_plane,
            colormap_options=self.colormap_options,
            fps=self.fps,
        )



Commands = tyro.conf.FlagConversionOff[
    Union[
        Annotated[RenderJointConditionedPath, tyro.conf.subcommand(name="camera-path")],
        Annotated[RenderJointConditionedDataset, tyro.conf.subcommand(name="dataset")],
    ]
]


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(Commands).main()


if __name__ == "__main__":
    entrypoint()