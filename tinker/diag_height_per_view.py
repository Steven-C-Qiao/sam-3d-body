"""Per-view subject-height analysis.

Tests whether the trained model produces *different* subject heights from
different views of the same subject — i.e. whether scale ambiguity is
actually unresolved given the per-view image, or whether each view
collapses onto a sharply-confident (and view-specific) height.

For each subject (B=1, V views) we measure (in mm), with neutral pose:
  * GT subject height (single value, view-invariant by construction).
  * μ-prediction height per view (from mhr["shape"], mhr["scale_68D"]).
  * Bias-corrected oracle height per view (mean of S NF samples).
  * Per-view sample-height std (S samples per view).

Headline aggregates:
  * Across-view std of μ-height (per subject): how much views disagree on height.
  * Across-view std of oracle height: same, after sample averaging.
  * Within-view sample std: how broadly the flow itself spreads each view.
  * Ratio (across-view / within-view): >> 1 ⇒ A2 mode collapse on height.

Usage:
    python tinker/diag_height_per_view.py
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


def neutral_inputs_from_mhr(mhr, B, device):
    """Build zero pose/expression tensors of the right shapes."""
    return {
        "global_trans": torch.zeros(B, 3, device=device),
        "global_rot": torch.zeros(B, 3, device=device),
        "body_pose_params": torch.zeros(B, mhr["body_pose"].shape[-1], device=device),
        "hand_pose_params": torch.zeros(B, mhr["hand"].shape[-1], device=device),
        "expr_params": torch.zeros(B, mhr["face"].shape[-1], device=device),
    }


@torch.no_grad()
def heights_for(mhr_head, shape_params, scale_offsets, neutral_template):
    """Build neutral-pose meshes; return per-mesh height (m).

    height := largest extent along any of the 3 axes of the mesh AABB.
    For neutral standing pose, this is always head-to-foot.
    """
    B = shape_params.shape[0]
    inputs = {k: v.expand(B, *v.shape[1:]).contiguous() if v.shape[0] == 1
              else v.repeat(B // v.shape[0], *([1] * (v.dim() - 1)))
              for k, v in neutral_template.items()}
    out = mhr_head.mhr_forward(
        shape_params=shape_params,
        scale_offsets=scale_offsets,
        do_pcblend=True,
        **inputs,
    )
    verts = out[0] if isinstance(out, tuple) else out  # [B, V_mesh, 3]
    extents = verts.max(dim=1).values - verts.min(dim=1).values
    return extents.max(dim=1).values  # [B]


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
    ap.add_argument("--max_batches", type=int, default=8)
    args = ap.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    os.environ["EGL_DEVICE_ID"] = args.gpus.split(",")[0]
    device = torch.device("cuda")

    trainer = load_trainer(args.experiment_dir, args.load_from_ckpt, device)
    mhr_head = trainer.model.head_pose

    results_per_dataset = {}
    for ds in ["4d-dress", "orbit-archviz-15-bbox44-smplx"]:
        logger.info(f"\n=== {ds} ===")
        try:
            results_per_dataset[ds] = collect(
                trainer, mhr_head, ds, args.num_views, args.num_samples,
                args.max_batches, device,
            )
        except Exception as e:
            logger.warning(f"[{ds}] failed: {e}")
            raise

    print()
    print("#" * 90)
    print(" SUMMARY ")
    print("#" * 90)
    for ds, rows in results_per_dataset.items():
        print()
        report(ds, rows, args)


def collect(trainer, mhr_head, dataset, num_views, num_samples, max_batches, device):
    loader = trainer.multiview_eval_dataloader(
        num_view=num_views, batch_size=1, dataset_name=dataset,
    )
    rows = []
    neutral_template = None
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

        if neutral_template is None:
            neutral_template = neutral_inputs_from_mhr(mhr, B=1, device=device)

        gt_shape = batch["shape_params"].float()                      # [B*V, 45]
        if "scale_params" in batch and isinstance(batch["scale_params"], torch.Tensor):
            gt_scale = batch["scale_params"].float()                   # [B*V, 68]
        else:
            gt_scale = batch["model_params"][:, -68:].float()          # [B*V, 68]
        mu_shape = mhr["shape"].float()                                 # [B*V, 45]
        mu_scale = mhr["scale_68D"].float()                             # [B*V, 68]
        s_shape = u["shape_samples"].float()                            # [B*V, S, 45]
        s_scale = u["scale_samples"].float()                            # [B*V, S, 68]
        S = s_shape.shape[1]

        gt_h = heights_for(mhr_head, gt_shape, gt_scale, neutral_template).view(bs, V)
        mu_h = heights_for(mhr_head, mu_shape, mu_scale, neutral_template).view(bs, V)

        s_shape_flat = s_shape.reshape(-1, s_shape.shape[-1])           # [B*V*S, 45]
        s_scale_flat = s_scale.reshape(-1, s_scale.shape[-1])           # [B*V*S, 68]
        s_h_flat = heights_for(mhr_head, s_shape_flat, s_scale_flat, neutral_template)
        s_h = s_h_flat.view(bs, V, S)
        oracle_h = s_h.mean(dim=-1)                                     # [B, V]

        for b in range(bs):
            gt_subject = gt_h[b].mean().item()
            rows.append(dict(
                gt=gt_subject,
                gt_per_view=gt_h[b].cpu().numpy(),
                gt_per_view_std=gt_h[b].std().item(),
                mu_per_view=mu_h[b].cpu().numpy(),
                oracle_per_view=oracle_h[b].cpu().numpy(),
                sample_per_view_std=s_h[b].std(dim=-1).cpu().numpy(),
                mu_across_view_std=mu_h[b].std().item(),
                oracle_across_view_std=oracle_h[b].std().item(),
                mu_height_mean=mu_h[b].mean().item(),
                bias_per_view=(mu_h[b] - gt_subject).cpu().numpy(),
            ))

        logger.info(
            f"[batch {bidx}] GT={rows[-1]['gt']*1000:.0f}mm  "
            f"μ={rows[-1]['mu_height_mean']*1000:.0f}mm  "
            f"μ across-V std={rows[-1]['mu_across_view_std']*1000:.1f}mm  "
            f"oracle across-V std={rows[-1]['oracle_across_view_std']*1000:.1f}mm  "
            f"sample within-V σ (avg V)={rows[-1]['sample_per_view_std'].mean()*1000:.1f}mm"
        )
    return rows


def report(dataset, rows, args):
    if not rows:
        print(f"--- {dataset}: no rows ---")
        return
    gt_all = np.array([r["gt"] for r in rows]) * 1000
    mu_means = np.array([r["mu_height_mean"] for r in rows]) * 1000
    mu_across = np.array([r["mu_across_view_std"] for r in rows]) * 1000
    or_across = np.array([r["oracle_across_view_std"] for r in rows]) * 1000
    sample_within = np.array([r["sample_per_view_std"].mean() for r in rows]) * 1000
    bias = np.array([np.mean(r["bias_per_view"]) for r in rows]) * 1000
    bias_abs = np.array([np.mean(np.abs(r["bias_per_view"])) for r in rows]) * 1000

    print(f"=== {dataset}  (N subjects={len(rows)}, V={args.num_views}, S={args.num_samples}) ===")
    print()
    print(f"GT subject heights (mm):  mean={gt_all.mean():.0f}  pop-σ={gt_all.std():.1f}  "
          f"min={gt_all.min():.0f}  max={gt_all.max():.0f}")
    print(f"μ-pred mean heights (mm): mean={mu_means.mean():.0f}  pop-σ={mu_means.std():.1f}")
    print(f"Per-subject mean bias  (μ_v - GT, avg over V): "
          f"signed={bias.mean():+.1f} ± {bias.std():.1f}  |signed|={bias_abs.mean():.1f}")
    print()
    print(f"--- Across-view spread per subject (mm) ---")
    print(f"  μ-pred  σ across V:   mean={mu_across.mean():.1f}  med={np.median(mu_across):.1f}  "
          f"max={mu_across.max():.1f}")
    print(f"  oracle  σ across V:   mean={or_across.mean():.1f}  med={np.median(or_across):.1f}  "
          f"max={or_across.max():.1f}")
    print(f"--- Within-view sample spread per subject (mm) ---")
    print(f"  sample  σ within V:   mean={sample_within.mean():.1f}  "
          f"med={np.median(sample_within):.1f}  max={sample_within.max():.1f}")
    print()
    ratio = mu_across / np.maximum(sample_within, 1e-3)
    print(f"--- Ratio (μ across-V σ / sample within-V σ) ---")
    print(f"  mean={ratio.mean():.2f}  median={np.median(ratio):.2f}")
    print(f"  >> 1 ⇒ per-view modes disagree more than per-view flow uncertainty admits (A2 collapse)")
    print(f"  ≈  1 ⇒ flow's per-view spread roughly captures cross-view disagreement")
    print(f"  << 1 ⇒ flow is broader than disagreement (would be ideal for IS)")
    print()
    print(f"--- First 5 subjects (heights in mm) ---")
    print(f"{'#':>3} {'GT':>5} | {'μ per view':<28} | {'oracle per view':<28} | "
          f"{'sample σ per view':<24}")
    for i, r in enumerate(rows[:5]):
        muv = " ".join(f"{h*1000:.0f}" for h in r["mu_per_view"])
        orv = " ".join(f"{h*1000:.0f}" for h in r["oracle_per_view"])
        sv = " ".join(f"{h*1000:.1f}" for h in r["sample_per_view_std"])
        print(f"{i:>3} {r['gt']*1000:>5.0f} | {muv:<28} | {orv:<28} | {sv:<24}")


if __name__ == "__main__":
    main()
