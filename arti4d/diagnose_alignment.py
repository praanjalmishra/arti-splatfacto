#!/usr/bin/env python3
import sys
import json
import torch
import numpy as np
import cv2
from pathlib import Path
from change_det.utils.io import read_transforms, params_to_cameras
from change_det.utils.render_utils import render_cameras
from nerfstudio.utils.eval_utils import eval_setup

CONFIG          = "/workspace/data/arti4d/nerf_output/din080_scene1/canonical/qed-splatter/config.yml"
CANON_TF        = "/workspace/data/arti4d/nerf_output/din080_scene1/canonical/transforms.json"
DATAPARSER_JSON = Path("/workspace/data/arti4d/nerf_output/din080_scene1/canonical/qed-splatter/dataparser_transforms.json")
OUT_DIR         = Path("sanity_check")
OUT_DIR.mkdir(exist_ok=True)


def apply_dataparser_transform(poses: torch.Tensor, dataparser_json: Path) -> torch.Tensor:
    with open(dataparser_json) as f:
        dp = json.load(f)
    T = torch.eye(4, dtype=torch.float32)
    T[:3, :] = torch.tensor(dp["transform"], dtype=torch.float32)
    scale = float(dp["scale"])
    out = poses.clone()
    out[:, :3, :3] = scale * (T[:3, :3] @ poses[:, :3, :3])
    out[:, :3,  3] = scale * (T[:3, :3] @ poses[:, :3, 3].unsqueeze(-1)).squeeze(-1) + scale * T[:3, 3]
    return out


_, pipeline, _, _ = eval_setup(Path(CONFIG), test_mode="inference")

result = read_transforms(CANON_TF, read_images=True, read_depth=False)
color_images, _, _, c2w, K, dist_params, _, _ = result

print(f"Before dataparser — c2w[0]:\n{c2w[0]}")
c2w_transformed = apply_dataparser_transform(c2w, DATAPARSER_JSON)
print(f"After dataparser  — c2w[0]:\n{c2w_transformed[0]}")

# Compare with what model actually has
train_cams = pipeline.datamanager.train_dataset.cameras
print(f"Train cam 0 (OpenCV):\n{train_cams.camera_to_worlds[0]}")

# Rebuild cameras with transformed poses
dist_zeros = torch.zeros(len(c2w), 4)
cameras = params_to_cameras(c2w_transformed, K.cpu(), dist_zeros, 1408, 1408)

indices = np.linspace(0, cameras.size - 1, 5, dtype=int)
for i, idx in enumerate(indices):
    rendered, _ = render_cameras(pipeline, cameras[idx:idx+1], device="cuda")

    rendered_np = (rendered[0].permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
    captured_np = (color_images[idx].permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)

    pair = np.concatenate([
        cv2.cvtColor(captured_np, cv2.COLOR_RGB2BGR),
        cv2.cvtColor(rendered_np, cv2.COLOR_RGB2BGR)
    ], axis=1)
    cv2.imwrite(str(OUT_DIR / f"pair_{i:02d}.png"), pair)
    print(f"Saved pair_{i:02d}.png  (frame idx={idx})")

print("Done.")