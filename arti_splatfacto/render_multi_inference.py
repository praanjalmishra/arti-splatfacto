#!/usr/bin/env python3
"""
Multi-Joint ArtiSplatfacto Inference Script
Tests arbitrary joint angle combinations for controllable multi-joint articulation rendering.
"""
import cv2
import torch
import numpy as np
from pathlib import Path
import argparse
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import imageio
import json
from itertools import product

from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.cameras.cameras import Cameras


def load_model_and_cameras(config_path: Path, num_test_cameras: int = 3):
    """Load trained multi-joint model and get test cameras."""
    print(f"\n[1/5] Loading checkpoint from {config_path}...")
    config, pipeline, checkpoint_path, step = eval_setup(config_path)
    model = pipeline.model
    model.eval()
    
    print(f"Loaded model at step {step}")
    print(f"Model type: {type(model).__name__}")
    
    # Verify multi-joint model
    if not hasattr(model, 'all_joint_params'):
        raise ValueError("Model does not have all_joint_params - not a multi-joint model!")
    
    # Get all joint IDs
    joint_ids = sorted(model.all_joint_params.keys())
    print(f"\n{'='*70}")
    print(f"Multi-Joint Model Summary")
    print(f"{'='*70}")
    print(f"Total Joints: {len(joint_ids)}")
    print(f"Joint IDs: {joint_ids}")
    print(f"Active Joint (from checkpoint): {model.config.active_joint_id}")
    
    # Print info for each joint
    joint_info = {}
    for joint_id in joint_ids:
        # Get joint limits from the joint parameters themselves
        if joint_id in model.all_joint_params:
            joint_params = model.all_joint_params[joint_id]
            
            if "min_angle" in joint_params and "max_angle" in joint_params:
                # Use the learned limits
                joint_min = joint_params["min_angle"].item()
                joint_max = joint_params["max_angle"].item()
                # Ensure min < max
                if joint_min > joint_max:
                    joint_min, joint_max = joint_max, joint_min
            else:
                # Fallback: try to get from stored buffer
                limits_attr = f'joint_limits_{joint_id}'
                if hasattr(model, limits_attr):
                    limits = getattr(model, limits_attr)
                    joint_min, joint_max = limits[0].item(), limits[1].item()
                else:
                    print(f"  ⚠️  No limits found for {joint_id}, using defaults")
        else:
            print(f"  ⚠️  No parameters found for {joint_id}")
        
        # Get joint type from metadata if available
        joint_type_attr = f'joint_type_{joint_id}'
        if hasattr(model, joint_type_attr):
            joint_type = getattr(model, joint_type_attr)
        else:
            # Fallback to default
            joint_type = getattr(model, 'joint_type', 'revolute')
        
        # Get number of Gaussians
        if joint_id in model.all_gauss_params_obj:
            n_gaussians = model.all_gauss_params_obj[joint_id]["means"].shape[0]
        else:
            n_gaussians = 0
        
        joint_info[joint_id] = {
            'limits': (joint_min, joint_max),
            'type': joint_type,
            'n_gaussians': n_gaussians
        }
        
        print(f"\n{joint_id}:")
        print(f"  Type: {joint_type}")
        print(f"  Limits: [{joint_min:.3f}, {joint_max:.3f}]")
        print(f"  Gaussians: {n_gaussians:,}")
        
        # Check for prior angles if available
        if joint_id in model.all_joint_params and "angle_deltas" in model.all_joint_params[joint_id]:
            prior_attr = f'joint_angles_prior_{joint_id}'
            if hasattr(model, prior_attr):
                prior = getattr(model, prior_attr)
                print(f"  Prior angles: {prior.shape[0]} frames, range [{prior.min():.3f}, {prior.max():.3f}]")
                deltas = model.all_joint_params[joint_id]["angle_deltas"]
                print(f"  Learned corrections: range [{deltas.min():.4f}, {deltas.max():.4f}]")
                # Show final angles
                final_angles = prior + deltas
                print(f"  Final angles: range [{final_angles.min():.3f}, {final_angles.max():.3f}]")
    
    print(f"{'='*70}\n")
    
    # Get test cameras
    print(f"[2/5] Setting up {num_test_cameras} test cameras...")
    datamanager = pipeline.datamanager
    test_cameras = []
    
    if hasattr(datamanager, "train_dataset"):
        dataset = datamanager.train_dataset
        metadata = dataset.metadata
        print(f"Dataset metadata keys: {list(metadata.keys())}")
        
        num_available = len(dataset)
        camera_indices = np.linspace(0, num_available - 1, num_test_cameras, dtype=int)
        
        for i, cam_idx in enumerate(camera_indices):
            # Slice 1 camera
            camera = dataset.cameras[cam_idx:cam_idx+1].to(model.device)
            
            # Attach metadata dictionary to this camera
            camera.metadata = {}
            
            for key, value in metadata.items():
                # Case 1: Per-frame tensor (e.g., times, joint_angles_joint_0)
                if torch.is_tensor(value) and value.ndim > 0:
                    camera.metadata[key] = value[cam_idx:cam_idx+1]
                # Case 2: Per-frame filename lists (depth, mask)
                elif isinstance(value, (list, tuple)) and len(value) > cam_idx:
                    camera.metadata[key] = value[cam_idx]
                # Case 3: Shared scalar / global info
                else:
                    camera.metadata[key] = value
            
            # Debug print - show angles for all joints
            t = camera.metadata.get("times", torch.tensor([0.0]))
            if torch.is_tensor(t):
                t = float(t.item())
            print(f"  Camera {i+1}: index {cam_idx}, time={t:.4f}")
            for joint_id in joint_ids:
                angle_key = f"joint_angles_{joint_id}"
                if angle_key in camera.metadata:
                    angle_val = camera.metadata[angle_key]
                    if torch.is_tensor(angle_val):
                        angle = float(angle_val.item())
                        print(f"    {joint_id}: {angle:.4f}")
            
            test_cameras.append(camera)
    else:
        raise AttributeError("No train dataset found in datamanager.")
    
    return model, test_cameras, joint_info

def create_joint_angle_combinations(
    joint_info: Dict,
    num_samples_per_joint: int = 5,
    mode: str = 'grid'
) -> List[Dict[str, float]]:
    """
    Create combinations of joint angles to test.
    
    Args:
        joint_info: Dictionary mapping joint_id to {limits, type, n_gaussians}
        num_samples_per_joint: Number of angle samples per joint
        mode: 'grid' (all combinations) or 'sweep' (one joint at a time)
    
    Returns:
        List of dictionaries mapping joint_id to angle value
    """
    joint_ids = sorted(joint_info.keys())
    
    # Create angle samples for each joint
    joint_samples = {}
    for joint_id in joint_ids:
        joint_min, joint_max = joint_info[joint_id]['limits']

        
        samples = torch.linspace(joint_min, joint_max, num_samples_per_joint)
        joint_samples[joint_id] = samples.tolist()
    print(f" joint limits for all teh joints: {joint_info}")
    import pdb; pdb.set_trace()


    if mode == 'grid':
        # Create all combinations (Cartesian product)
        all_values = [joint_samples[jid] for jid in joint_ids]
        combinations = list(product(*all_values))
        
        angle_configs = []
        for combo in combinations:
            config = {joint_ids[i]: combo[i] for i in range(len(joint_ids))}
            angle_configs.append(config)
        
        print(f"\n[3/5] Created {len(angle_configs)} joint angle combinations (grid mode)")
        print(f"  {num_samples_per_joint} samples per joint × {len(joint_ids)} joints")
        
    elif mode == 'sweep':
        # Sweep one joint at a time, others fixed at MIN
        angle_configs = []
        
        for active_joint_id in joint_ids:
            # All others at MIN
            base_config = {}
            for joint_id in joint_ids:
                joint_min, joint_max = joint_info[joint_id]['limits']
                base_config[joint_id] = joint_min
            
            # Sweep the active joint from MIN → MAX
            for angle in joint_samples[active_joint_id]:
                config = base_config.copy()
                config[active_joint_id] = angle
                angle_configs.append(config)

        print(f"\n[3/5] Created {len(angle_configs)} joint angle combinations (sweep mode)")
        print(f"  {num_samples_per_joint} samples × {len(joint_ids)} joints (MIN→MAX sweep)")

    elif mode == 'random':
        # Random sampling
        angle_configs = []
        num_random = num_samples_per_joint ** len(joint_ids)  # Same total as grid
        
        for _ in range(num_random):
            config = {}
            for joint_id in joint_ids:
                joint_min, joint_max = joint_info[joint_id]['limits']
                angle = np.random.uniform(joint_min, joint_max)
                config[joint_id] = angle
            angle_configs.append(config)
        
        print(f"\n[3/5] Created {len(angle_configs)} random joint angle combinations")
    
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    return angle_configs

def test_multi_joint_angles(
    model,
    test_cameras: List[Cameras],
    joint_info: Dict,
    angle_configs: List[Dict[str, float]],
    output_dir: Path,
    render_depth: bool = False
):
    """Test rendering at various multi-joint angle combinations with debug logging."""
    print(f"\n[4/5] Testing {len(angle_configs)} joint angle configurations...")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    all_renders = []
    all_depths = [] if render_depth else None
    
    joint_ids = sorted(joint_info.keys())
    
    with torch.no_grad():
        for cam_idx, camera in enumerate(test_cameras):
            view_renders = []
            view_depths = [] if render_depth else None
            print(f"\n  Camera {cam_idx+1}/{len(test_cameras)}:")
            
            for config_idx, angle_config in enumerate(angle_configs):
                # --- DEBUG: Start per-config logging ---
                print(f"\n  ── Config {config_idx+1}/{len(angle_configs)} ─────────────────────────────")
                
                # Ensure eval mode
                model.eval()

                # Set joint angles in camera metadata
                if not hasattr(camera, 'metadata'):
                    camera.metadata = {}
                
                for joint_id in joint_ids:
                    angle = angle_config[joint_id]
                    joint_min, joint_max = joint_info[joint_id]['limits']
                    clamped_angle = float(torch.clamp(
                        torch.tensor(angle), joint_min, joint_max
                    ))
                    angle_key = f"joint_angles_{joint_id}"
                    
                    # Store angle as consistent tensor format (1,1)
                    camera.metadata[angle_key] = torch.tensor(
                        [[clamped_angle]],
                        device=model.device,
                        dtype=torch.float32
                    )

                    # --- DEBUG: Log the metadata being written ---
                    print(f"    {angle_key} set to {clamped_angle:.4f} "
                          f"(limits [{joint_min:.3f}, {joint_max:.3f}]) "
                          f"device={model.device}, shape={camera.metadata[angle_key].shape}")
                
                # --- DEBUG: Verify what model sees before rendering ---
                for joint_id in joint_ids:
                    try:
                        angle_tensor = model.get_joint_angle_for_camera(camera, joint_id=joint_id)
                        print(f"    [MODEL CHECK] {joint_id} angle read as {angle_tensor.item():.4f}")
                    except Exception as e:
                        print(f"    [MODEL CHECK] Failed reading {joint_id}: {e}")

                # --- Render ---
                outputs = model.get_outputs(camera)
                rgb = outputs["rgb"].detach().cpu().numpy()
                
                # Convert to uint8 image
                img = (rgb * 255).astype(np.uint8)
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                
                # Overlay text of joint angles
                y_offset = 30
                for joint_id in joint_ids:
                    angle = angle_config[joint_id]
                    text = f"{joint_id}: {angle:.3f}"
                    cv2.putText(img_bgr, text, (20, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                    y_offset += 35

                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                
                # Save rendered image
                angle_str = "_".join([f"{jid}_{angle_config[jid]:.3f}" for jid in joint_ids])
                img_path = output_dir / f"cam_{cam_idx+1}_config_{config_idx:04d}_{angle_str}.png"
                imageio.imwrite(img_path, img_rgb)
                
                # --- DEBUG: confirm successful render output ---
                print(f"    Saved render: {img_path.name}, shape={rgb.shape}, "
                      f"mean RGB={rgb.mean():.4f}, min={rgb.min():.4f}, max={rgb.max():.4f}")
                
                view_renders.append(rgb)
                
                # Optional depth rendering
                if render_depth and "depth" in outputs:
                    depth = outputs["depth"].detach().cpu().numpy().squeeze()
                    view_depths.append(depth)
                    
                    depth_vis = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
                    depth_img = (depth_vis * 255).astype(np.uint8)
                    depth_color = cv2.applyColorMap(depth_img, cv2.COLORMAP_INFERNO)
                    
                    depth_path = output_dir / "depth" / f"depth_cam_{cam_idx+1}_config_{config_idx:04d}.png"
                    depth_path.parent.mkdir(exist_ok=True)
                    imageio.imwrite(depth_path, cv2.cvtColor(depth_color, cv2.COLOR_BGR2RGB))
                    print(f"    Saved depth map: {depth_path.name}")
            
            all_renders.append(view_renders)
            if render_depth:
                all_depths.append(view_depths)
    
    return all_renders, all_depths

def create_custom_camera(
    model,
    intrinsics: Dict[str, float],
    c2w: np.ndarray,
    image_width: int,
    image_height: int
) -> Cameras:
    """
    Create a custom camera for rendering.

    Args:
        intrinsics: dict with keys {fx, fy, cx, cy}
        c2w: (4x4) camera-to-world pose matrix (numpy array or torch tensor)
        image_width: output width in pixels
        image_height: output height in pixels

    Returns:
        Cameras object on the model device.
    """
    if isinstance(c2w, np.ndarray):
        c2w = torch.from_numpy(c2w).float()

    cam = Cameras(
        fx=torch.tensor([intrinsics["fx"]]),
        fy=torch.tensor([intrinsics["fy"]]),
        cx=torch.tensor([intrinsics["cx"]]),
        cy=torch.tensor([intrinsics["cy"]]),
        width=torch.tensor([image_width]),
        height=torch.tensor([image_height]),
        camera_to_worlds=c2w[None, ...],    # (1,4,4)
    )

    cam.metadata = {}
    return cam


def create_sweep_visualization(
    all_renders: List[List[np.ndarray]],
    angle_configs: List[Dict[str, float]],
    joint_info: Dict,
    output_dir: Path,
    mode: str
):
    """Create visualization for sweep mode (one joint moving at a time)."""
    print(f"\n[5/5] Creating sweep visualization...")
    
    joint_ids = sorted(joint_info.keys())
    num_cameras = len(all_renders)
    
    if mode == 'sweep':
        # For sweep mode: rows = cameras, cols = configs (grouped by joint)
        # Skip the first config (neutral pose)
        configs_per_joint = (len(angle_configs) - 1) // len(joint_ids)
        
        for cam_idx in range(num_cameras):
            fig, axes = plt.subplots(len(joint_ids), configs_per_joint, 
                                    figsize=(3*configs_per_joint, 3*len(joint_ids)))
            
            if len(joint_ids) == 1:
                axes = axes.reshape(1, -1)
            
            config_idx = 1  # Skip neutral pose
            for joint_idx, joint_id in enumerate(joint_ids):
                for sample_idx in range(configs_per_joint):
                    ax = axes[joint_idx, sample_idx]
                    
                    rgb = all_renders[cam_idx][config_idx]
                    ax.imshow(rgb)
                    ax.axis('off')
                    
                    # Title with joint angle
                    angle = angle_configs[config_idx][joint_id]
                    if joint_idx == 0:
                        ax.set_title(f'{angle:.3f}', fontsize=10)
                    
                    # Joint label on left
                    if sample_idx == 0:
                        ax.set_ylabel(f'{joint_id}', fontsize=10, rotation=90, labelpad=10)
                    
                    config_idx += 1
            
            fig.suptitle(f'Multi-Joint Sweep - Camera {cam_idx+1}', fontsize=12)
            plt.tight_layout()
            
            grid_path = output_dir / f"sweep_camera_{cam_idx+1}.png"
            plt.savefig(grid_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"  Saved sweep visualization for camera {cam_idx+1}: {grid_path}")
    
    elif mode == 'grid' and len(joint_ids) == 2:
        # For 2-joint grid: create 2D grid visualization
        for cam_idx in range(num_cameras):
            # Determine grid dimensions
            joint_0, joint_1 = joint_ids[0], joint_ids[1]
            
            # Count unique values for each joint
            angles_0 = sorted(set(config[joint_0] for config in angle_configs))
            angles_1 = sorted(set(config[joint_1] for config in angle_configs))
            
            n_rows, n_cols = len(angles_1), len(angles_0)
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(2*n_cols, 2*n_rows))
            
            if n_rows == 1:
                axes = axes.reshape(1, -1)
            if n_cols == 1:
                axes = axes.reshape(-1, 1)
            
            # Fill grid
            for config_idx, angle_config in enumerate(angle_configs):
                angle_0 = angle_config[joint_0]
                angle_1 = angle_config[joint_1]
                
                col_idx = angles_0.index(angle_0)
                row_idx = angles_1.index(angle_1)
                
                ax = axes[row_idx, col_idx]
                rgb = all_renders[cam_idx][config_idx]
                ax.imshow(rgb)
                ax.axis('off')
                
                # Labels
                if row_idx == 0:
                    ax.set_title(f'{angle_0:.2f}', fontsize=9)
                if col_idx == 0:
                    ax.set_ylabel(f'{angle_1:.2f}', fontsize=9, rotation=0, 
                                 labelpad=30, va='center')
            
            # Axis labels
            fig.text(0.5, 0.02, f'{joint_0} angle', ha='center', fontsize=11)
            fig.text(0.02, 0.5, f'{joint_1} angle', va='center', rotation='vertical', fontsize=11)
            fig.suptitle(f'2-Joint Grid - Camera {cam_idx+1}', fontsize=12)
            
            plt.tight_layout(rect=[0.03, 0.03, 1, 0.97])
            
            grid_path = output_dir / f"grid_2d_camera_{cam_idx+1}.png"
            plt.savefig(grid_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"  Saved 2D grid visualization for camera {cam_idx+1}: {grid_path}")


def save_animation(
    all_renders: List[List[np.ndarray]],
    angle_configs: List[Dict[str, float]],
    joint_info: Dict,
    output_dir: Path,
    fps: int = 10
):
    """Create animation videos for each camera view."""
    print(f"\n[Bonus] Creating animation videos (fps={fps})...")
    
    joint_ids = sorted(joint_info.keys())
    
    for cam_idx, renders in enumerate(all_renders):
        frames = []
        for config_idx, rgb in enumerate(renders):
            img = (rgb * 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            # Overlay joint angles
            y_offset = 30
            for joint_id in joint_ids:
                angle = angle_configs[config_idx][joint_id]
                text = f"{joint_id}: {angle:.3f}"
                cv2.putText(img_bgr, text, (20, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                y_offset += 35
            
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            frames.append(img_rgb)
        
        # Save video
        video_path = output_dir / f"animation_camera_{cam_idx+1}.mp4"
        imageio.mimsave(video_path, frames, fps=fps)
        print(f"  Saved animation for camera {cam_idx+1}: {video_path}")


def save_config_json(angle_configs: List[Dict[str, float]], output_dir: Path):
    """Save all joint angle configurations to JSON for reproducibility."""
    json_path = output_dir / "joint_angle_configs.json"
    
    with open(json_path, 'w') as f:
        json.dump(angle_configs, f, indent=2)
    
    print(f"\n  Saved joint angle configurations: {json_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Test multi-joint ArtiSplatfacto with arbitrary joint angle combinations"
    )
    parser.add_argument("config_path", type=Path, help="Path to config.yml")
    parser.add_argument("--output_dir", type=Path, default="./multi_joint_test", 
                       help="Output directory")
    parser.add_argument("--num_cameras", type=int, default=1, 
                       help="Number of test cameras")
    parser.add_argument("--samples_per_joint", type=int, default=5, 
                       help="Number of angle samples per joint")
    parser.add_argument("--mode", type=str, default='sweep', 
                       choices=['grid', 'sweep', 'random'],
                       help="Sampling mode: 'grid' (all combinations), 'sweep' (one at a time), 'random'")
    parser.add_argument("--render_depth", action="store_true", 
                       help="Render depth images")
    parser.add_argument("--create_animation", action="store_true", 
                       help="Create MP4 animations")
    parser.add_argument("--fps", type=int, default=10, 
                       help="FPS for animations")
    parser.add_argument("--seed", type=int, default=42, 
                       help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Set random seed
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        print(f"Using random seed: {args.seed}")
    
    # Load model and cameras
    model, test_cameras, joint_info = load_model_and_cameras(
        args.config_path, args.num_cameras
    )
    
    # Create joint angle combinations
    angle_configs = create_joint_angle_combinations(
        joint_info,
        num_samples_per_joint=args.samples_per_joint,
        mode=args.mode
    )

    # Example intrinsic parameters
    intrinsics = {
        "fx": 716.0789184570312,
        "fy": 716.0789184570312,
        "cx": 480.3292,
        "cy": 358.15444,
    }

    # Example pose (camera-to-world 4x4)
    c2w = np.array([
        [0.9196518980674783, -0.12286216219710004,  0.37302181636083687, -0.06704090782718791],
        [0.3853154654275932,  0.09852275247475487, -0.9175103592587474,  0.1686886227645357],
        [0.0759761704957655,  0.9875212181746028,   0.1379473275266568, -0.3409729404831472],
        [0.0,                 0.0,                  0.0,                  1.0]
    ])

    custom_cam = create_custom_camera(
        model,
        intrinsics,
        c2w,
        image_width=960,
        image_height=720
    ).to(model.device)

    # Add to the camera list
    test_cameras.append(custom_cam)

    
    #Test rendering
    all_renders, all_depths = test_multi_joint_angles(
        model,
        test_cameras,
        joint_info,
        angle_configs,
        args.output_dir,
        render_depth=args.render_depth
    )

    # all_renders, _ = test_multi_joint_angles(
    #     model,
    #     [custom_cam],
    #     joint_info,
    #     angle_configs,
    #     args.output_dir,
    #     render_depth=args.render_depth
    # )
    
    # Create visualizations
    create_sweep_visualization(
        all_renders,
        angle_configs,
        joint_info,
        args.output_dir,
        mode=args.mode
    )
    
    # Optional: Create animations
    if args.create_animation:
        save_animation(
            all_renders,
            angle_configs,
            joint_info,
            args.output_dir,
            fps=args.fps
        )
    
    # Save configuration for reproducibility
    save_config_json(angle_configs, args.output_dir)
    
    print(f"\n{'='*70}")
    print(f"✅ Multi-joint testing complete!")
    print(f"{'='*70}")
    print(f"Results saved to: {args.output_dir}")
    print(f"Total configurations tested: {len(angle_configs)}")
    print(f"Total renders: {len(test_cameras) * len(angle_configs)}")
    print(f"Mode: {args.mode}")
    
    if args.mode == 'grid':
        print(f"\n⚠️  Grid mode: {len(angle_configs)} combinations")
        if len(joint_info) > 2:
            print(f"   Warning: Visualization limited for {len(joint_info)} joints.")
            print(f"   Consider using 'sweep' mode for better visualization.")
    
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()