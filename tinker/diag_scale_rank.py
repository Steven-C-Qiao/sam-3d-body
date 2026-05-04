"""Rank-of-disagreement diagnostic on the 10 flow-modulated scale dims (and 45 shape dims).

Tests the hypothesis from §38 follow-up: across views, the disagreement on the
10 body-scale dims is rank-1 along a single "overall body-magnitude" direction.
If true, multi-view merging on the scale block reduces to scalar averaging.

For each subject (B=1, V views, S samples):
  * Per-view regressor mean μ_j   [V, D]
  * Per-view sample mean β̄_j      [V, D]
  * Per-view sample residuals     [V, S, D]  (β_{j,k} − β̄_j)

Stack across subjects → centred matrices for SVD:
  * Across-view-of-μ:        [N·V, D] of  (μ_j − ⟨μ⟩_j_per_subject)
  * Across-view-of-β̄:        [N·V, D] of  (β̄_j − ⟨β̄⟩_j_per_subject)
  * Within-view-of-samples:  [N·V·S, D] of (β_{j,k} − β̄_j)

Run on 4D-Dress (OOD) and BEDLAM val (IID). Report variance-fraction per PC,
and dump the loading vector of PC1 in the scale subspace to inspect whether
it's "uniform across all 10 dims" (= overall magnitude) or sparse.

Usage:
    python tinker/diag_scale_rank.py --max_batches 25 --num_samples 100
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from loguru import logger

sys.path.append(".")

from sam_3d_body.configs.config import get_config_defaults


SCALE_INDICES = [3, 4, 5, 6, 7, 10, 11, 12, 13, 14]


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
def collect_per_subject(trainer, dataset, num_views, num_samples, max_batches, device):
    """Return three lists of arrays, one entry per subject."""
    loader = trainer.multiview_eval_dataloader(
        num_view=num_views, batch_size=1, dataset_name=dataset,
    )
    subjects = []
    for bidx, batch in enumerate(loader):
        if bidx >= max_batches:
            break
        for k, v in list(batch.items()):
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)
        bs, V = batch["img"].shape[:2]
        for k, v in list(batch.items()):
            if isinstance(v, torch.Tensor):
                if v.dim() >= 2 and v.shape[0] == bs and v.shape[1] == V:
                    batch[k] = v.flatten(0, 1)
        batch = trainer.preprocess(batch)
        outputs = trainer.model(batch, num_samples=num_samples)
        mhr = outputs["mhr"]
        u = outputs["uncertainty_output"]

        mu_shape = mhr["shape"].view(bs, V, -1)                          # [B, V, 45]
        mu_scale_full = mhr["scale_68D"].view(bs, V, -1)                 # [B, V, 68]
        mu_scale = mu_scale_full[..., SCALE_INDICES]                     # [B, V, 10]

        s_shape = u["shape_samples"].view(bs, V, num_samples, -1)        # [B, V, S, 45]
        s_scale_full = u["scale_samples"].view(bs, V, num_samples, -1)   # [B, V, S, 68]
        s_scale = s_scale_full[..., SCALE_INDICES]                       # [B, V, S, 10]

        bar_shape = s_shape.mean(dim=2)                                  # [B, V, 45]
        bar_scale = s_scale.mean(dim=2)                                  # [B, V, 10]
        resid_shape = s_shape - bar_shape.unsqueeze(2)                    # [B, V, S, 45]
        resid_scale = s_scale - bar_scale.unsqueeze(2)                    # [B, V, S, 10]

        for b in range(bs):
            subjects.append(dict(
                mu_shape=mu_shape[b].cpu().numpy(),       # [V, 45]
                mu_scale=mu_scale[b].cpu().numpy(),        # [V, 10]
                bar_shape=bar_shape[b].cpu().numpy(),      # [V, 45]
                bar_scale=bar_scale[b].cpu().numpy(),      # [V, 10]
                resid_shape=resid_shape[b].cpu().numpy(),  # [V, S, 45]
                resid_scale=resid_scale[b].cpu().numpy(),  # [V, S, 10]
            ))
    return subjects


def stack_across_view_residuals(subjects, key):
    """Centre each subject's per-view means by the subject mean; stack."""
    rows = []
    for s in subjects:
        m = s[key]                                      # [V, D]
        rows.append(m - m.mean(axis=0, keepdims=True))  # [V, D]
    return np.concatenate(rows, axis=0)                  # [N*V, D]


def stack_within_view_residuals(subjects, key):
    """Just concatenate per-(subject, view) sample residuals."""
    rows = []
    for s in subjects:
        r = s[key]                                       # [V, S, D]
        V, S, D = r.shape
        rows.append(r.reshape(V * S, D))
    return np.concatenate(rows, axis=0)                  # [N*V*S, D]


def report_pca(name, X, top_k=5, dim_labels=None):
    """X: [N, D]. Print variance fraction explained by top PCs and PC1 loading."""
    if X.shape[0] == 0:
        print(f"  {name}: empty")
        return
    # Centre (already centred for across-view; safe no-op).
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    var = (S ** 2) / max(Xc.shape[0] - 1, 1)
    frac = var / var.sum()
    cum = np.cumsum(frac)
    pc1 = Vt[0]                                         # [D]
    # Normalise loading sign so the largest |coef| is positive.
    if abs(pc1.min()) > pc1.max():
        pc1 = -pc1
    print(f"  {name}  shape={X.shape}")
    print(f"    var-frac top-{top_k}: " +
          " ".join(f"{f*100:5.1f}%" for f in frac[:top_k]) +
          f"   cum: {cum[top_k - 1] * 100:5.1f}%")
    if dim_labels is not None:
        # Show PC1 loading per dim (for the 10-D scale subspace).
        load_str = ", ".join(f"{lab}={pc1[i]:+.2f}" for i, lab in enumerate(dim_labels))
        print(f"    PC1 loading: {load_str}")
        # Uniformity score: if PC1 is roughly uniform sign+magnitude → "overall scale" direction.
        sign_concord = float(np.sign(pc1).sum() / len(pc1))   # 1 = all same sign
        rms = np.sqrt(np.mean(pc1 ** 2))
        max_over_rms = np.max(np.abs(pc1)) / max(rms, 1e-9)
        print(f"    PC1 sign-concordance={sign_concord:+.2f}  "
              f"(max|loading|/rms)={max_over_rms:.2f}  "
              f"(uniform-direction marker: |conc|≈1 and ratio≈1.0)")


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-E", "--experiment_dir", default="exp/exp_071_crop_shape")
    ap.add_argument(
        "-L", "--load_from_ckpt",
        default="exp/exp_071_crop_shape/saved_models/last.ckpt",
    )
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--num_views", type=int, default=4)
    ap.add_argument("--num_samples", type=int, default=100)
    ap.add_argument("--max_batches", type=int, default=25)
    args = ap.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    os.environ["EGL_DEVICE_ID"] = args.gpus.split(",")[0]
    device = torch.device("cuda")

    trainer = load_trainer(args.experiment_dir, args.load_from_ckpt, device)

    scale_labels = [f"s{i}" for i in SCALE_INDICES]

    for ds in ["4d-dress", "orbit-archviz-15-bbox44-smplx"]:
        logger.info(f"\n=== {ds} ===")
        subjects = collect_per_subject(
            trainer, ds, args.num_views, args.num_samples, args.max_batches, device,
        )
        if not subjects:
            continue

        print(f"\n{'#' * 90}\n {ds} (N subjects = {len(subjects)})\n{'#' * 90}")

        print("\n--- ACROSS-VIEW spread (μ_j − ⟨μ⟩_subject) ---")
        report_pca(
            "scale (10D, μ across V)",
            stack_across_view_residuals(subjects, "mu_scale"),
            top_k=5, dim_labels=scale_labels,
        )
        report_pca(
            "shape (45D, μ across V)",
            stack_across_view_residuals(subjects, "mu_shape"),
            top_k=5,
        )

        print("\n--- ACROSS-VIEW spread (β̄_j − ⟨β̄⟩_subject) ---")
        report_pca(
            "scale (10D, β̄ across V)",
            stack_across_view_residuals(subjects, "bar_scale"),
            top_k=5, dim_labels=scale_labels,
        )
        report_pca(
            "shape (45D, β̄ across V)",
            stack_across_view_residuals(subjects, "bar_shape"),
            top_k=5,
        )

        print("\n--- WITHIN-VIEW spread (per-view sample residuals) ---")
        report_pca(
            "scale (10D, samples within V)",
            stack_within_view_residuals(subjects, "resid_scale"),
            top_k=5, dim_labels=scale_labels,
        )
        report_pca(
            "shape (45D, samples within V)",
            stack_within_view_residuals(subjects, "resid_shape"),
            top_k=5,
        )

        # Concrete check: project mu_scale across-V residuals onto PC1 and onto a
        # uniform direction; see how much "overall magnitude" explains.
        across_mu_scale = stack_across_view_residuals(subjects, "mu_scale")
        unit_uniform = np.ones(across_mu_scale.shape[1])
        unit_uniform = unit_uniform / np.linalg.norm(unit_uniform)
        proj_uniform = across_mu_scale @ unit_uniform
        var_uniform = proj_uniform.var()
        var_total = across_mu_scale.var(axis=0).sum()
        print(f"\nVar along uniform-axis / total var (μ across V, scale 10D): "
              f"{var_uniform / max(var_total, 1e-12) * 100:.1f}%")


if __name__ == "__main__":
    main()
