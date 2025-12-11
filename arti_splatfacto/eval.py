import json
from pathlib import Path
import torch
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from nerfstudio.utils.eval_utils import eval_setup
from arti_splatfacto.utils.img_utils import crop_imgs_w_masks


def ensure_rgb(pred):
    """
    Ensure render output is (H, W, 3).
    Handles cases where Splatfacto returns (H, W) or (H, W, 1).
    """
    if pred.ndim == 3 and pred.shape[-1] == 3:
        return pred
    if pred.ndim == 2:  # (H, W)
        pred = pred.unsqueeze(-1)
    if pred.shape[-1] == 1:
        pred = pred.repeat(1, 1, 3)
    return pred


def evaluate_model(config_path: str, split="eval"):
    config_path = Path(config_path)

    # Load model + pipeline
    _, pipeline, _, _ = eval_setup(
        config_path,
        eval_num_rays_per_chunk=None,
        test_mode="inference",
    )

    model = pipeline.model
    datamgr = pipeline.datamanager
    device = model.device

    # Select dataset
    dataset = datamgr.eval_dataset if split == "eval" else datamgr.train_dataset
    cameras = dataset.cameras.to(device)

    # Metrics on correct device
    psnr_fn = PeakSignalNoiseRatio(data_range=1.0).to(device)
    lpips_fn = LearnedPerceptualImagePatchSimilarity(normalize=True).to(device)

    results = []
    print(f"Evaluating {len(dataset)} images...")

    for i in range(len(dataset)):
        # ---- Load GT from dataset ----
        data = dataset[i]

        gt = data["image"].to(device)     # (H,W,3)
        if hasattr(model, "_downscale_if_required"):
            gt = model._downscale_if_required(gt)

        gt_t = gt.permute(2, 0, 1).unsqueeze(0)

        # ---- Load mask, if available ----
        mask = data.get("mask", None)
        if mask is not None:
            mask = mask.to(device)
            if hasattr(model, "_downscale_if_required"):
                mask = model._downscale_if_required(mask)

            # Normalize to (1,1,H,W)
            if mask.ndim == 3:  # (H,W,1)
                mask = mask.permute(2,0,1).unsqueeze(0).bool()
            elif mask.ndim == 2:
                mask = mask.unsqueeze(0).unsqueeze(0).bool()

            mask3 = mask.expand(-1, 3, -1, -1)
        else:
            mask = None

        # ---- Run model ----
        cam = cameras[i:i+1]
        with torch.no_grad():
            outputs = model.get_outputs_for_camera(cam)

        # ---- Predicted RGB (already composed with background) ----
        pred = outputs["rgb"]   # (H,W,3)
        pred = ensure_rgb(pred)

        pred_t = pred.permute(2, 0, 1).unsqueeze(0)

        # ---- RGB Metrics ----
        psnr_val = psnr_fn(gt_t, pred_t).item()
        ssim_val = structural_similarity_index_measure(gt_t, pred_t).item()
        lpips_val = lpips_fn(gt_t, pred_t).item()

        entry = {
            "idx": i,
            "psnr": psnr_val,
            "ssim": ssim_val,
            "lpips": lpips_val,
        }

        # ---- Masked metrics ----
        if mask is not None:
            gt_crop  = gt_t  * mask3
            pred_crop = pred_t * mask3

            entry["psnr_masked"] = psnr_fn(gt_crop, pred_crop).item()
            entry["ssim_masked"] = structural_similarity_index_measure(gt_crop, pred_crop).item()
            entry["lpips_masked"] = lpips_fn(gt_crop, pred_crop).item()

        # ---- Depth metrics (optional) ----
        if "depth_image" in data and outputs.get("depth") is not None:
            gt_depth = data["depth_image"].to(device).squeeze(-1)
            pred_depth = outputs["depth"].squeeze(-1)

            depth_abs = torch.abs(gt_depth - pred_depth).mean().item()
            entry["depth_mae"] = depth_abs

        results.append(entry)
        print(entry)

    # Save all results
    out_path = Path("arti_eval_results.json")
    json.dump(results, open(out_path, "w"), indent=2)
    print(f"\nSaved results → {out_path}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--split", type=str, default="eval")
    args = parser.parse_args()

    evaluate_model(args.config, args.split)
