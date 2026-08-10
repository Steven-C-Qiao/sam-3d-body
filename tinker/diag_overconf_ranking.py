"""Rank all flow-modelled dims by posterior overconfidence.

For each modelled dim (shape_indices + scale_indices), compute:
  * σ_pred (per-view std of NF samples, averaged over subjects×views)
  * σ_prior (population std from shape_scale_std.pt)
  * |z| = |β_gt - μ| / σ_pred  (median over subjects×views)

Rank ascending by σ_pred / σ_prior (most overconfident first).

Usage:
    python tinker/diag_overconf_ranking.py -E exp/exp_072_d20 \
        -L exp/exp_072_d20/saved_models/last.ckpt
"""
import os
import sys
import argparse
from pathlib import Path

import torch
from loguru import logger

sys.path.append(".")

from sam_3d_body.configs.config import get_config_defaults


def load_trainer(exp_dir, load_path, device):
    from sam_3d_body.trainer import Trainer
    cfg = get_config_defaults()
    cfg.merge_from_file(str(Path(exp_dir) / "config.yaml"))
    cfg.MODEL.MHR_HEAD.MHR_MODEL_PATH = (
        "checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt"
    )
    trainer = Trainer(cfg=cfg, vis_save_dir=str(Path(exp_dir) / "diag_tmp")).to(device)
    ckpt = torch.load(load_path, weights_only=False, map_location="cpu")
    sd = {k[6:] if k.startswith("model.") else k: v for k, v in ckpt["state_dict"].items()}
    trainer.model.load_state_dict(sd, strict=False)
    trainer.model.eval()
    return trainer


@torch.no_grad()
def collect(trainer, dataset, num_views, num_samples, max_batches, device):
    loader = trainer.multiview_eval_dataloader(
        num_view=num_views, batch_size=1, dataset_name=dataset,
    )
    sigmas_shape, resids_shape = [], []
    sigmas_scale, resids_scale = [], []
    n = 0
    for bidx, batch in enumerate(loader):
        if bidx >= max_batches:
            break
        for k, v in list(batch.items()):
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)
        bs, num_v = batch["img"].shape[:2]
        for k, v in list(batch.items()):
            if isinstance(v, torch.Tensor):
                if v.dim() >= 2 and v.shape[0] == bs and v.shape[1] == num_v:
                    batch[k] = v.flatten(0, 1)
        batch = trainer.preprocess(batch)
        outputs = trainer.model(batch, num_samples=num_samples)
        u = outputs["uncertainty_output"]
        mhr = outputs["mhr"]

        # Shape: 45-D.
        sigma_s = u["shape_samples"].std(dim=1)              # [B*V, 45]
        gt_shape = batch["shape_params"]                      # [B*V, 45]
        mu_shape = mhr["shape"]
        resid_s = (gt_shape - mu_shape).abs()
        sigmas_shape.append(sigma_s.cpu().float())
        resids_shape.append(resid_s.cpu().float())

        # Scale: 68-D, but flow only modulates the 10 scale_indices.
        sigma_sc = u["scale_samples"].std(dim=1)              # [B*V, 68]
        gt_scale = batch["model_params"][..., -68:]           # [B*V, 68]
        mu_scale = mhr["scale_68D"]
        resid_sc = (gt_scale - mu_scale).abs()
        sigmas_scale.append(sigma_sc.cpu().float())
        resids_scale.append(resid_sc.cpu().float())

        n += sigma_s.shape[0]
    return (torch.cat(sigmas_shape, 0), torch.cat(resids_shape, 0),
            torch.cat(sigmas_scale, 0), torch.cat(resids_scale, 0), n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-E", "--experiment_dir", default="exp/exp_072_d20")
    ap.add_argument(
        "-L", "--load_from_ckpt",
        default="exp/exp_072_d20/saved_models/last.ckpt",
    )
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--num_views", type=int, default=4)
    ap.add_argument("--num_samples", type=int, default=100)
    ap.add_argument("--max_batches", type=int, default=12)
    args = ap.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    os.environ["EGL_DEVICE_ID"] = args.gpus.split(",")[0]
    device = torch.device("cuda")

    trainer = load_trainer(args.experiment_dir, args.load_from_ckpt, device)
    nf_head = trainer.model.nf_head
    shape_indices = list(nf_head.shape_indices)
    scale_indices = list(nf_head.scale_indices)

    stds = torch.load("checkpoints/sam-3d-body-dinov3/shape_scale_std.pt",
                      weights_only=False, map_location="cpu")
    shape_std = stds["shape_std"].cpu().float()  # [45]
    scale_std = stds["scale_std"].cpu().float()  # [10]

    print(f"\n=== {args.experiment_dir} ===")
    print(f"shape_indices ({len(shape_indices)}): {shape_indices}")
    print(f"scale_indices ({len(scale_indices)}): {scale_indices}")

    # Use one dataset for ranking (4D-Dress, OOD).
    sigma_s, resid_s, sigma_sc, resid_sc, n = collect(
        trainer, "4d-dress", args.num_views, args.num_samples,
        args.max_batches, device,
    )
    print(f"Collected n={n} subject-view samples.\n")

    # Build full ranking: 10 shape modeled dims + 10 scale modeled dims.
    rows = []
    for d in shape_indices:
        sp = shape_std[d].item()
        s = sigma_s[:, d].mean().item()
        r = resid_s[:, d].median().item()
        zd = (resid_s[:, d] / sigma_s[:, d].clamp_min(1e-6)).median().item()
        rows.append(("shape", d, sp, s, r, zd))
    # Map scale_indices (which index into 68-D scale) to scale_std (which is the
    # 10-D stat aligned with scale_indices).
    for k, d68 in enumerate(scale_indices):
        sp = scale_std[k].item()
        s = sigma_sc[:, d68].mean().item()
        r = resid_sc[:, d68].median().item()
        zd = (resid_sc[:, d68] / sigma_sc[:, d68].clamp_min(1e-6)).median().item()
        rows.append(("scale", d68, sp, s, r, zd))

    # Sort by σ_pred / σ_prior ascending (most overconfident first).
    rows.sort(key=lambda x: x[3] / max(x[2], 1e-9))

    print(f"\n{'='*78}")
    print(" Posterior overconfidence ranking on modelled dims (4D-Dress, OOD) ")
    print(f"{'='*78}")
    print(f"{'rank':>4}  {'kind':>5} {'idx':>3} | {'σ_prior':>9} {'σ_pred':>9} "
          f"{'σ/σ_pri':>9} {'|res|':>9} {'|z|':>8}")
    print("-" * 78)
    for r, (kind, d, sp, s, res, zd) in enumerate(rows, start=1):
        marker = ""
        if s / max(sp, 1e-9) < 0.30: marker = "  <-- extreme"
        elif s / max(sp, 1e-9) < 0.50: marker = "  <-- severe"
        print(f"{r:>4}.  {kind:>5} d{d:<3} | {sp:>9.3f} {s:>9.3f} "
              f"{s/max(sp,1e-9):>9.3f} {res:>9.3f} {zd:>8.2f}{marker}")
    print()
    print("Calibrated reference: σ/σ_prior ≈ 1, |z| ≈ 0.67")


if __name__ == "__main__":
    main()
