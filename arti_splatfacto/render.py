#!/usr/bin/env python3
"""
Joint-Conditioned ArtiSplatfacto Inference Script
Tests arbitrary joint angles for controllable articulation rendering.
"""
import cv2
import torch
import numpy as np
from pathlib import Path
import argparse
from typing import List
import matplotlib.pyplot as plt
import imageio

from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.cameras.cameras import Cameras


def load_model_and_cameras(config_path: Path, num_test_cameras: int = 3):
    """Load trained model and get test cameras."""
    print(f"\n[1/4] Loading checkpoint from {config_path}...")
    config, pipeline, checkpoint_path, step = eval_setup(config_path)
    model = pipeline.model
    model.eval()
    
    print(f"Loaded model at step {step}")
    print(f"Model type: {type(model).__name__}")
    
    # Verify joint-conditioned model
    if not hasattr(model, 'joint_angle_deltas'):
        raise ValueError("Model does not have joint_angle_deltas - not a joint-conditioned model!")
    
    print(f"Joint limits: {model.joint_limits.cpu().numpy()}")
    print(f"Prior angles range: [{model.joint_angles_prior.min():.3f}, {model.joint_angles_prior.max():.3f}]")
    print(f"Learned corrections range: [{model.joint_angle_deltas.min():.4f}, {model.joint_angle_deltas.max():.4f}]")
    
    # Get test cameras
    print(f"\n[2/4] Setting up {num_test_cameras} test cameras...")
    datamanager = pipeline.datamanager
    test_cameras = []
    
    if hasattr(datamanager, "train_dataset"):
        dataset = datamanager.train_dataset
        metadata = dataset.metadata  
        print(f"Dataset metadata keys: {list(metadata.keys())}")

        print(f"Using train dataset with {len(dataset)} images for test cameras.")
        print(f"Type: {type(dataset.cameras).__name__}")

        num_available = len(dataset)
        camera_indices = np.linspace(120, num_available - 1, num_test_cameras, dtype=int)
        
        for i, cam_idx in enumerate(camera_indices):

            # Slice 1 camera
            camera = dataset.cameras[cam_idx:cam_idx+1].to(model.device)

            # Attach metadata dictionary to this camera
            camera.metadata = {}

            for key, value in metadata.items():

                # Case 1: Per-frame tensor (e.g., times, joint_angles)
                if torch.is_tensor(value) and value.ndim > 0:
                    camera.metadata[key] = value[cam_idx:cam_idx+1]

                # Case 2: Per-frame filename lists (depth, mask)
                elif isinstance(value, (list, tuple)):
                    camera.metadata[key] = value[cam_idx]

                # Case 3: Shared scalar / global info (depth scale, scene_path)
                else:
                    camera.metadata[key] = value

            # Debug print
            t = float(camera.metadata.get("times", torch.tensor([0.0])).item())
            j = float(camera.metadata.get("joint_angles", torch.tensor([0.0])).item())
            print(f"  Camera {i+1}: index {cam_idx}, time={t:.4f}, joint={j:.4f}")

            test_cameras.append(camera)

    else:
        raise AttributeError("No train dataset found in datamanager.")
    
    return model, test_cameras



def test_arbitrary_joint_angles(model, test_cameras: List[Cameras], joint_angles: torch.Tensor, output_dir: Path):
    """Test rendering at arbitrary joint angles with angle overlay."""
    print(f"\n[3/4] Testing {len(joint_angles)} joint angles...")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    all_renders = []
    
    joint_min, joint_max = model.joint_limits[0].item(), model.joint_limits[1].item()
    joint_type = model.joint_type
    
    with torch.no_grad():
        for i, camera in enumerate(test_cameras):
            view_renders = []
            print(f"\n  Camera {i+1}/{len(test_cameras)}:")
            
            for j, joint_angle in enumerate(joint_angles):
                unclamped_angle = float(joint_angle)
                clamped_angle = float(torch.clamp(joint_angle, joint_min, joint_max))

                if not hasattr(camera, 'metadata'):
                    camera.metadata = {}

                camera.metadata["joint_angles"] = torch.tensor(
                    [clamped_angle], device=model.device, dtype=torch.float32
                )

                # Render
                outputs = model.get_outputs(camera)
                rgb = outputs["rgb"].detach().cpu().numpy()

                # Convert to uint8
                img = (rgb * 255).astype(np.uint8)

                # Overlay text (joint angle info)
                text = f"{joint_type}: {unclamped_angle:.3f}"
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.putText(img_bgr, text, (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

                # Save
                img_path = output_dir / f"camera_{i+1}_angle_{j+1}_{clamped_angle:.3f}.png"
                imageio.imwrite(img_path, img_rgb)

                print(f"    Joint angle {unclamped_angle:.3f} {joint_type} (clamped {clamped_angle:.3f}) -> {img_path.name}")

                view_renders.append(rgb)
            
            all_renders.append(view_renders)
    
    return all_renders


def test_arbitrary_joint_angles_depth(model, test_cameras: List[Cameras], joint_angles: torch.Tensor, output_dir: Path):
    """Test rendering depth at arbitrary joint angles with angle overlay."""
    print(f"\n[3/4-Depth] Testing {len(joint_angles)} joint angles (Depth Rendering)...")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    all_depths = []
    
    joint_min, joint_max = model.joint_limits[0].item(), model.joint_limits[1].item()
    joint_type = model.joint_type
    
    with torch.no_grad():
        for i, camera in enumerate(test_cameras):
            view_depths = []
            print(f"\n  Camera {i+1}/{len(test_cameras)}:")
            
            for j, joint_angle in enumerate(joint_angles):
                unclamped_angle = float(joint_angle)
                clamped_angle = float(torch.clamp(joint_angle, joint_min, joint_max))

                if not hasattr(camera, 'metadata'):
                    camera.metadata = {}

                camera.metadata["joint_angles"] = torch.tensor(
                    [clamped_angle], device=model.device, dtype=torch.float32
                )

                # Render
                outputs = model.get_outputs(camera)
                if "depth" not in outputs:
                    raise KeyError("Model outputs do not contain 'depth'. Make sure your model returns depth maps.")
                
                depth = outputs["depth"].detach().cpu().numpy().squeeze()
                view_depths.append(depth)

                # Normalize for visualization (0–1 range)
                depth_vis = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
                depth_img = (depth_vis * 255).astype(np.uint8)

                # Apply colormap for better visibility
                depth_color = cv2.applyColorMap(depth_img, cv2.COLORMAP_INFERNO)

                # Overlay joint angle info (in degrees)
                # text = f"{joint_type}: {math.degrees(unclamped_angle):.1f}°"
                # cv2.putText(depth_color, text, (20, 40),
                #             cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)

                # Save images
                img_path = output_dir / f"depth_camera_{i+1}_angle_{j+1}_{clamped_angle:.3f}.png"
                # npy_path = output_dir / f"depth_camera_{i+1}_angle_{j+1}_{clamped_angle:.3f}.npy"

                imageio.imwrite(img_path, cv2.cvtColor(depth_color, cv2.COLOR_BGR2RGB))
                # np.save(npy_path, depth)

                # print(f"    Joint angle {math.degrees(unclamped_angle):.1f}° ({joint_type}) "
                #       f"(clamped {math.degrees(clamped_angle):.1f}°) -> {img_path.name}")

            all_depths.append(view_depths)
    
    return all_depths

def create_visualization_grid(all_renders: List[List[np.ndarray]], joint_angles: torch.Tensor, 
                             model, output_dir: Path):
    """Create a visualization grid showing all camera views and joint angles."""
    print(f"\n[4/4] Creating visualization grid...")
    
    num_cameras = len(all_renders)
    num_angles = len(joint_angles)
    joint_type = model.joint_type
    joint_min, joint_max = model.joint_limits[0].item(), model.joint_limits[1].item()
    
    # Create figure
    fig, axes = plt.subplots(num_cameras, num_angles, figsize=(3*num_angles, 3*num_cameras))
    if num_cameras == 1:
        axes = axes.reshape(1, -1)
    if num_angles == 1:
        axes = axes.reshape(-1, 1)
    
    for cam_idx in range(num_cameras):
        for angle_idx in range(num_angles):
            ax = axes[cam_idx, angle_idx]
            
            # Display image
            rgb = all_renders[cam_idx][angle_idx]
            ax.imshow(rgb)
            ax.axis('off')
            
            # Title with joint angle
            clamped_angle = torch.clamp(joint_angles[angle_idx], joint_min, joint_max).item()
            if cam_idx == 0:  # Only show angle on top row
                ax.set_title(f'{clamped_angle:.3f}', fontsize=12, pad=10)
            
            # Camera label on left
            if angle_idx == 0:
                ax.set_ylabel(f'Camera {cam_idx+1}', fontsize=12, rotation=90, labelpad=20)
    
    # Overall title
    fig.suptitle(f'Joint-Conditioned Articulation Test\n'
                 f'Joint Type: {joint_type}, Range: [{joint_min:.3f}, {joint_max:.3f}]', 
                 fontsize=14, y=0.95)
    
    plt.tight_layout()
    grid_path = output_dir / "joint_angle_test_grid.png"
    plt.savefig(grid_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization grid saved: {grid_path}")



def main():
    parser = argparse.ArgumentParser(description="Test joint-conditioned ArtiSplatfacto with arbitrary joint angles")
    parser.add_argument("config_path", type=Path, help="Path to config.yml")
    parser.add_argument("--output_dir", type=Path, default="./joint_angle_test", help="Output directory")
    parser.add_argument("--num_cameras", type=int, default=1, help="Number of test cameras")
    parser.add_argument("--num_angles", type=int, default=200, help="Number of joint angles to test")
    parser.add_argument("--angle_range", nargs=2, type=float, help="Custom angle range [min, max]")
    parser.add_argument("--render_depth", action="store_true", help="Render depth images")
    parser.add_argument("--random_angles", action="store_true", help="Use random (instead of linear) joint angles")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    args = parser.parse_args()
    
    # Load model and cameras
    model, test_cameras = load_model_and_cameras(args.config_path, args.num_cameras)
    
    # Determine joint angle range
    if args.angle_range:
        joint_min, joint_max = args.angle_range
        print(f"Using custom angle range: [{joint_min:.3f}, {joint_max:.3f}]")
    else:
        joint_min, joint_max = model.joint_limits[0].item(), model.joint_limits[1].item()
        print(f"Using model's joint limits: [{joint_min:.3f}, {joint_max:.3f}]")
    
    # Optional reproducibility
    if args.seed is not None:
        torch.manual_seed(args.seed)
        print(f"Using random seed: {args.seed}")
    
    # Create test joint angles
    if args.random_angles:
        joint_angles = torch.empty(args.num_angles).uniform_(joint_min, joint_max)
        joint_angles = torch.sort(joint_angles).values  # sorted for smooth articulation
        print(f"\n🔀 Using random joint angles between [{joint_min:.3f}, {joint_max:.3f}]")
    else:
        joint_angles = torch.linspace(joint_min, joint_max, args.num_angles)
        print(f"\n📈 Using linearly spaced joint angles between [{joint_min:.3f}, {joint_max:.3f}]")

    print(f"Joint type: {model.joint_type}")
    print(f"Units: {'radians' if model.joint_type == 'revolute' else 'meters'}")
    print(f"Total test angles: {len(joint_angles)}")
    
    # Test rendering
    all_renders = test_arbitrary_joint_angles(model, test_cameras, joint_angles, args.output_dir)
    
    # Optional: Depth rendering
    if args.render_depth:
        depth_dir = args.output_dir / "depth"
        depth_dir.mkdir(parents=True, exist_ok=True)
        all_depths = test_arbitrary_joint_angles_depth(model, test_cameras, joint_angles, depth_dir)
        print(f"\n✅ Depth rendering complete! Saved to: {depth_dir}")
    
    print(f"\n✅ Joint angle testing complete!")
    print(f"Results saved to: {args.output_dir}")
    print(f"Total renders: {len(test_cameras) * len(joint_angles)}")


if __name__ == "__main__":
    main()