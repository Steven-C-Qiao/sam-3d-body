"""
Diagnostic: cross-view vs self-view NF log-probabilities on multi-view eval batches.

Tests the IS-coverage-failure hypothesis from
docs/multiview_posterior_tightening.ipynb section 16+:

If per-view samples don't cover the joint posterior (extreme-cropping regime),
we expect cross-view log-probs to be hundreds of nats below self-view log-probs.

Reports per batch:
  * self-view log p(beta_i^k - mu_i | c_i): from beta_log_prob_ref
  * cross-view log p(beta_i^k - mu_j | c_j) for j != i: from cross_view_logp
  * Mean gap, spread across S samples, best-vs-mean cross
  * ESS of tempered IS weights (T from merge_params_nf_tempered)

Usage:
    python tinker/diag_cross_view_logp.py \
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
from sam_3d_body.models.meta_arch.nf_merging import merge_params_nf_tempered


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
    missing, unexpected = trainer.model.load_state_dict(sd, strict=False)
    logger.info(f"Loaded {len(sd)} params; missing={len(missing)}, unexpected={len(unexpected)}")
    trainer.model.eval()
    return trainer, cfg


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-E", "--experiment_dir", required=True)
    ap.add_argument("-L", "--load_from_ckpt", required=True)
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--num_views", type=int, default=4)
    ap.add_argument("--num_samples", type=int, default=100)
    ap.add_argument("--max_batches", type=int, default=3)
    args = ap.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    os.environ["EGL_DEVICE_ID"] = args.gpus.split(",")[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    trainer, _ = load_trainer(args.experiment_dir, args.load_from_ckpt, device)
    loader = trainer.multiview_eval_dataloader(
        num_view=args.num_views, batch_size=1,
        dataset_name=(args.dataset or "4d-dress"),
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
        param_dict = merge_params_nf_tempered(
            trainer.model.nf_head, outputs["mhr"], outputs["uncertainty_output"],
            bs, num_views, args.num_samples, batch=batch,
        )

        cv = param_dict["cross_view_log_prob_beta"]  # [B, V, V, S]
        wts = param_dict["is_weight_beta"]            # [B, V, S]
        B, V, _, S = cv.shape

        # Diagonal = self-view; off-diagonal = cross-view.
        diag_mask = torch.eye(V, dtype=torch.bool, device=cv.device)
        self_lp = cv[:, diag_mask]                           # [B, V, S]
        off_mask = ~diag_mask
        cross_lp = cv[:, off_mask].view(B, V, V - 1, S)      # [B, V, V-1, S]

        self_mean_per_view = self_lp.mean(dim=-1)            # [B, V]
        cross_mean_per_pair = cross_lp.mean(dim=-1)          # [B, V, V-1]
        cross_std_per_pair = cross_lp.std(dim=-1)            # [B, V, V-1]
        cross_max_per_pair = cross_lp.max(dim=-1).values     # [B, V, V-1]
        cross_min_per_pair = cross_lp.min(dim=-1).values     # [B, V, V-1]

        # Gap: self - cross averaged over (B, V) pairs.
        # cross is grouped by proposing-view i, so compare self_mean_per_view[b,i]
        # to cross_mean_per_pair[b,i,*].
        gap = self_mean_per_view.unsqueeze(-1) - cross_mean_per_pair  # [B, V, V-1]

        # ESS of tempered weights, pooled across V*S per subject.
        wts_pooled = wts.reshape(B, V * S)
        wts_pooled = wts_pooled / (wts_pooled.sum(dim=-1, keepdim=True) + 1e-30)
        ess = 1.0 / (wts_pooled.pow(2).sum(dim=-1) + 1e-30)            # [B]

        # Best vs mean cross (over the S samples, per (b, i, j)).
        best_vs_mean = cross_max_per_pair - cross_mean_per_pair        # [B, V, V-1]

        logger.info(
            f"\n[batch {batch_idx}] B={B} V={V} S={S}, total V*S={V*S}\n"
            f"  self-view  log-prob  : mean={self_mean_per_view.mean().item():+.2f}, "
            f"std-across-views={self_mean_per_view.std().item():.2f}\n"
            f"  cross-view log-prob  : mean={cross_mean_per_pair.mean().item():+.2f}, "
            f"sample-spread per pair={cross_std_per_pair.mean().item():.2f}\n"
            f"  cross min..max       : "
            f"min={cross_min_per_pair.mean().item():+.2f}, "
            f"max={cross_max_per_pair.mean().item():+.2f}\n"
            f"  GAP self-cross (mean): {gap.mean().item():+.2f} nats "
            f"(median={gap.median().item():+.2f}, max={gap.max().item():+.2f})\n"
            f"  best-vs-mean cross   : {best_vs_mean.mean().item():.2f} nats "
            f"(higher = some samples land near other view's mode)\n"
            f"  ESS / (V*S)          : {(ess / (V * S)).mean().item():.3f} "
            f"(1.0 = uniform, 1/(V*S) = one-hot)"
        )

    logger.info("Done.")


if __name__ == "__main__":
    main()
