"""
Probe the effective dimensionality of per-view shape uncertainty.

Hypothesis (from cross-view-logp diagnostic): the IS coverage gap is
~30 nats, not the ~10^3 predicted under a naive 22-D-unobserved model.
This is consistent with the per-view posterior being concentrated on
many fewer "uncertain" directions than naively expected.

Reports per batch:
  * Per-dim per-view sample std σ_{i,d}, normalized by the flow's
    training perturbation std σ_prior,d (calibrated reference).
  * Count of dims with σ/σ_prior > τ (default 0.5) — naive uncertain-dim count.
  * Participation-ratio effective dim:  d_eff = (Σσ²)² / Σσ⁴
  * Top-K dims ranked by per-view variance (large = uncertain) and by
    cross-view penalty contribution.

Usage:
    python tinker/diag_effective_dim.py \
        -E exp/exp_071_crop_shape \
        -L exp/exp_071_crop_shape/saved_models/last.ckpt \
        --max_batches 3
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
    cfg_yaml = Path(exp_dir) / "config.yaml"
    if cfg_yaml.exists():
        cfg.merge_from_file(str(cfg_yaml))
    cfg.MODEL.MHR_HEAD.MHR_MODEL_PATH = (
        "checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt"
    )

    trainer = Trainer(cfg=cfg, vis_save_dir=str(Path(exp_dir) / "diag_tmp")).to(device)

    logger.info(f"Loading checkpoint: {load_path}")
    ckpt = torch.load(load_path, weights_only=False, map_location="cpu")
    sd = {k[6:] if k.startswith("model.") else k: v for k, v in ckpt["state_dict"].items()}
    trainer.model.load_state_dict(sd, strict=False)
    trainer.model.eval()
    return trainer, cfg


def participation_ratio(sigma2):
    # sigma2: tensor of shape [..., D]
    s1 = sigma2.sum(dim=-1)
    s2 = sigma2.pow(2).sum(dim=-1)
    return (s1.pow(2) / (s2 + 1e-30))


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-E", "--experiment_dir", required=True)
    ap.add_argument("-L", "--load_from_ckpt", required=True)
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--dataset", default="4d-dress")
    ap.add_argument("--num_views", type=int, default=4)
    ap.add_argument("--num_samples", type=int, default=100)
    ap.add_argument("--max_batches", type=int, default=3)
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="σ/σ_prior threshold for uncertain-dim count.")
    args = ap.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    os.environ["EGL_DEVICE_ID"] = args.gpus.split(",")[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trainer, _ = load_trainer(args.experiment_dir, args.load_from_ckpt, device)
    nf = trainer.model.nf_head
    scale_indices = nf.scale_indices
    num_shape = nf.num_shape_comps
    D = nf.beta_dim
    logger.info(f"Flow beta_dim D={D} (num_shape={num_shape}, num_scale={D - num_shape})")

    # Reference scale: training perturbation stds (when present), else the
    # across-subject GT std collected on the fly.
    shape_std = getattr(nf, "_shape_perturb_std", None)
    scale_std = getattr(nf, "_scale_perturb_std", None)

    loader = trainer.multiview_eval_dataloader(
        num_view=args.num_views, batch_size=1, dataset_name=args.dataset,
    )

    if shape_std is None or scale_std is None:
        logger.warning(
            "nf._{shape,scale}_perturb_std missing — collecting GT std across "
            f"{args.max_batches} batches as the reference scale."
        )
        gt_betas = []
        peek_loader = list(loader)[: args.max_batches]
        for batch in peek_loader:
            gt_shape = batch["shape_params"].to(device)        # [B*V, 45] or [B, V, 45]
            gt_scale_68 = batch["model_params"][:, -68:].to(device)
            if gt_shape.dim() == 3:
                gt_shape = gt_shape.flatten(0, 1)
                gt_scale_68 = gt_scale_68.flatten(0, 1)
            gt_betas.append(
                torch.cat([gt_shape, gt_scale_68[..., scale_indices]], dim=-1)
            )
        gt_beta = torch.cat(gt_betas, dim=0).float()
        prior_std = gt_beta.std(dim=0).clamp_min(1e-3)         # [D]
        logger.info(
            f"GT-derived σ_prior over {gt_beta.shape[0]} samples: "
            f"min={prior_std.min().item():.3f}, "
            f"median={prior_std.median().item():.3f}, "
            f"max={prior_std.max().item():.3f}"
        )
        loader = peek_loader  # iterate the same batches now
    else:
        shape_std = shape_std.to(device)
        scale_std = scale_std.to(device)
        prior_std = torch.cat([shape_std, scale_std], dim=0).to(torch.float32)
        logger.info(
            f"Reference σ_prior (training perturb): "
            f"shape median={shape_std.median().item():.4f}, "
            f"scale median={scale_std.median().item():.4f}, "
            f"min={prior_std.min().item():.4f}, max={prior_std.max().item():.4f}"
        )

    for batch_idx, batch in enumerate(loader):
        if batch_idx >= args.max_batches:
            break

        for k, v in list(batch.items()):
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)
        bs, num_views = batch["img"].shape[:2]
        for k, v in list(batch.items()):
            if isinstance(v, torch.Tensor):
                if v.dim() >= 2 and v.shape[0] == bs and v.shape[1] == num_views:
                    batch[k] = v.flatten(0, 1)
        batch = trainer.preprocess(batch)

        outputs = trainer.model(batch, num_samples=args.num_samples)
        u = outputs["uncertainty_output"]
        mhr = outputs["mhr"]

        shape_s = u["shape_samples"].unflatten(0, (bs, num_views))   # [B, V, S, num_shape]
        scale_s = u["scale_samples"].unflatten(0, (bs, num_views))   # [B, V, S, 68]
        scale_s_sel = scale_s[..., scale_indices]                    # [B, V, S, num_scale]
        beta_s = torch.cat([shape_s, scale_s_sel], dim=-1).float()   # [B, V, S, D]

        # ---------------- per-view per-dim std vs prior ----------------
        # σ_{i,d} = std over S
        sigma_i_d = beta_s.std(dim=2)                                # [B, V, D]
        ratio_i_d = sigma_i_d / prior_std.view(1, 1, -1)             # [B, V, D]
        # Use sample variance for participation ratio.
        var_i_d = sigma_i_d.pow(2)                                   # [B, V, D]

        # ---------------- effective-dim metrics per view ----------------
        # Naive uncertain-dim count: # dims with σ/σ_prior > τ.
        n_uncertain = (ratio_i_d > args.threshold).sum(dim=-1).float()  # [B, V]
        # Participation ratio. d_eff is small if variance concentrates on few dims.
        d_eff = participation_ratio(var_i_d)                         # [B, V]
        # Same for *normalized* variance ratios (so prior-rescaled).
        d_eff_norm = participation_ratio(ratio_i_d.pow(2))           # [B, V]

        # ---------------- cross-view penalty per dim ----------------
        # Approximate per-dim contribution to cross-view log-prob *gap*.
        # For each (b, i, j) we compute, dim-by-dim, the expected squared
        # residual under view j's mean, scaled by view j's per-dim sample
        # precision (i.e. how view j's posterior penalises deviations).
        pred_shape = mhr["shape"].unflatten(0, (bs, num_views))      # [B, V, num_shape]
        pred_scale = mhr["scale_68D"].unflatten(0, (bs, num_views))[..., scale_indices]  # [B, V, num_scale]
        mean_beta_j = torch.cat([pred_shape, pred_scale], dim=-1).float()  # [B, V, D]

        # For each (b, i, k, j), residual = beta_i^k - mu_j.
        # beta_s: [B, V_i, S, D] -> [B, V_i, 1, S, D]
        # mean_beta_j: [B, V_j, D] -> [B, 1, V_j, 1, D]
        residual = beta_s.unsqueeze(2) - mean_beta_j.unsqueeze(1).unsqueeze(3)
        # shape [B, V_i, V_j, S, D]

        # View-j precision per dim: 1 / Var(beta_j^k).
        var_j_d = beta_s.var(dim=2).clamp_min(1e-8)                  # [B, V_j, D]
        prec_j_d = 1.0 / var_j_d                                     # [B, V_j, D]

        # Per-dim mean cross-view squared-residual penalty.
        # prec_j_d: [B, V_j, D] -> [B, 1, V_j, 1, D]
        sq = residual.pow(2) * prec_j_d.unsqueeze(1).unsqueeze(3)    # [B, V_i, V_j, S, D]
        # Mask self-pairs (i=j).
        diag = torch.eye(num_views, device=device).bool()             # [V, V]
        off_mask = (~diag).view(1, num_views, num_views, 1, 1)
        sq_off = sq * off_mask
        n_off = (~diag).sum().item()
        # mean over i, j (off-diag), k, b for per-dim ranking.
        per_dim_pen = sq_off.sum(dim=(0, 1, 2, 3)) / (bs * n_off * args.num_samples)
        # Take a Gaussian-style nat estimate: 0.5 * mean(squared / var_j).
        per_dim_pen_nats = 0.5 * per_dim_pen                          # [D]

        # Top-5 dims by penalty.
        top_k = 5
        top_vals, top_idx = per_dim_pen_nats.topk(top_k)
        top_str = ", ".join(
            f"d{int(i)}({v:.2f}n)" for i, v in zip(top_idx.tolist(), top_vals.tolist())
        )

        # Top-5 dims by per-view sigma ratio (averaged over B, V).
        ratio_avg = ratio_i_d.mean(dim=(0, 1))                       # [D]
        top_unc_vals, top_unc_idx = ratio_avg.topk(top_k)
        top_unc_str = ", ".join(
            f"d{int(i)}({v:.2f})" for i, v in zip(top_unc_idx.tolist(), top_unc_vals.tolist())
        )

        # ---------------- shape vs scale split ----------------
        # Effective dim split between shape (first num_shape) and scale (rest).
        var_shape = var_i_d[..., :num_shape]
        var_scale = var_i_d[..., num_shape:]
        d_eff_shape = participation_ratio(var_shape)                 # [B, V]
        d_eff_scale = participation_ratio(var_scale)                 # [B, V]
        n_uncertain_shape = (ratio_i_d[..., :num_shape] > args.threshold).sum(dim=-1).float()
        n_uncertain_scale = (ratio_i_d[..., num_shape:] > args.threshold).sum(dim=-1).float()

        logger.info(
            f"\n[batch {batch_idx}] B={bs} V={num_views} S={args.num_samples}, D={D}\n"
            f"  σ/σ_prior:   median={ratio_i_d.median().item():.3f}, "
            f"max={ratio_i_d.max().item():.3f}, "
            f"min={ratio_i_d.min().item():.3f}\n"
            f"  # dims σ/σ_prior > {args.threshold} (per view): "
            f"all={n_uncertain.mean().item():.1f}/{D} "
            f"(shape {n_uncertain_shape.mean().item():.1f}/{num_shape}, "
            f"scale {n_uncertain_scale.mean().item():.1f}/{D-num_shape})\n"
            f"  participation-ratio d_eff (raw var):    "
            f"all={d_eff.mean().item():.2f}/{D} "
            f"(shape {d_eff_shape.mean().item():.2f}/{num_shape}, "
            f"scale {d_eff_scale.mean().item():.2f}/{D-num_shape})\n"
            f"  participation-ratio d_eff (prior-norm): "
            f"all={d_eff_norm.mean().item():.2f}/{D}\n"
            f"  top dims by per-view σ/σ_prior:  {top_unc_str}\n"
            f"  top dims by cross-view penalty:  {top_str}\n"
            f"  total mean cross-view sq penalty (Gaussian nats): "
            f"{per_dim_pen_nats.sum().item():.2f}"
        )

    logger.info("Done.")


if __name__ == "__main__":
    main()
