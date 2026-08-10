"""Per-view β L2 versus crop intensity.

For each view in the multi-view eval set, record:
  * per-view β L2 = ||β_pred - β_gt||_2 over the 55 modelled dims
  * crop_ratio = scale_post / orig_scale (=1.0 if no extreme/random crop applied)

Then bin by crop_ratio and report mean L2 per bin, plus Spearman correlation.

Usage:
    python tinker/diag_crop_vs_beta.py -E exp/exp_071_crop_shape \
        -L exp/exp_071_crop_shape/saved_models/last.ckpt \
        --gpus 6 --max_batches 40
"""
import os
import sys
import argparse
from pathlib import Path

import torch
from loguru import logger

sys.path.append(".")

from sam_3d_body.configs.config import get_config_defaults


def load_trainer(exp_dir, load_path, device, config_path=None):
    from sam_3d_body.trainer import Trainer
    cfg = get_config_defaults()
    cfg_yaml = config_path or str(Path(exp_dir) / "config.yaml")
    cfg.merge_from_file(cfg_yaml)
    cfg.MODEL.MHR_HEAD.MHR_MODEL_PATH = (
        "checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt"
    )
    trainer = Trainer(cfg=cfg, vis_save_dir=str(Path(exp_dir) / "diag_crop_tmp")).to(device)
    ckpt = torch.load(load_path, weights_only=False, map_location="cpu")
    sd = {k[6:] if k.startswith("model.") else k: v for k, v in ckpt["state_dict"].items()}
    trainer.model.load_state_dict(sd, strict=False)
    trainer.model.eval()
    return trainer


@torch.no_grad()
def collect(trainer, dataset, num_views, max_batches, device, num_samples=100):
    loader = trainer.multiview_eval_dataloader(
        num_view=num_views, batch_size=1, dataset_name=dataset,
    )
    nf_head = trainer.model.nf_head
    scale_indices = nf_head.scale_indices

    l2_list, l2_std_list, ratio_list, ec_list, group_list = [], [], [], [], []
    spa_l2_list, spa_l2_std_list = [], []

    shape_std = getattr(nf_head, "_shape_perturb_std", None)
    scale_std_55 = getattr(nf_head, "_scale_perturb_std", None)
    perturb_std = None
    if shape_std is not None and scale_std_55 is not None:
        perturb_std = torch.cat([shape_std.cpu().float(), scale_std_55.cpu().float()])

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
        mhr = outputs["mhr"]
        u = outputs["uncertainty_output"]

        per_view_beta = torch.cat(
            [mhr["shape"], mhr["scale_68D"][..., scale_indices]], dim=-1
        )
        gt_shape = batch["shape_params"]
        gt_scale_68D = batch["model_params"][:, -68:]
        gt_beta = torch.cat([gt_shape, gt_scale_68D[..., scale_indices]], dim=-1)

        per_view_l2 = torch.sqrt(((per_view_beta - gt_beta) ** 2).sum(dim=-1)).cpu().float()
        l2_list.append(per_view_l2)

        # Sample-param-average β: mean of NF samples (per view), evaluated on the
        # 55 modelled dims (shape: all 45; scale: scale_indices).
        shape_s = u["shape_samples"]                                # (B*V, S, 45)
        scale_s = u["scale_samples"]                                # (B*V, S, 68)
        spa_beta = torch.cat(
            [shape_s.mean(dim=1), scale_s.mean(dim=1)[..., scale_indices]], dim=-1
        )                                                            # (B*V, 55)
        spa_l2 = torch.sqrt(((spa_beta - gt_beta) ** 2).sum(dim=-1)).cpu().float()
        spa_l2_list.append(spa_l2)

        if perturb_std is not None:
            std = perturb_std.to(per_view_beta.device, per_view_beta.dtype)
            l2_std = torch.sqrt((((per_view_beta - gt_beta) / std) ** 2).sum(dim=-1)).cpu().float()
            l2_std_list.append(l2_std)
            spa_l2_std = torch.sqrt((((spa_beta - gt_beta) / std) ** 2).sum(dim=-1)).cpu().float()
            spa_l2_std_list.append(spa_l2_std)

        scale_post = batch["scale_post_tensor"].view(-1).cpu().float()
        orig_scale = batch["orig_scale_tensor"].view(-1).cpu().float()
        ratio = (scale_post / orig_scale.clamp_min(1e-6)).cpu().float()
        ratio_list.append(ratio)

        ec = batch["did_extreme_crop"].view(-1).cpu().long()
        ec_list.append(ec)

        n_views = ec.numel()
        group_list.append(torch.full((n_views,), bidx, dtype=torch.long))

    l2 = torch.cat(l2_list)
    l2_std = torch.cat(l2_std_list) if l2_std_list else None
    spa_l2 = torch.cat(spa_l2_list)
    spa_l2_std = torch.cat(spa_l2_std_list) if spa_l2_std_list else None
    ratio = torch.cat(ratio_list)
    ec = torch.cat(ec_list)
    group = torch.cat(group_list)
    return l2, l2_std, spa_l2, spa_l2_std, ratio, ec, group


def spearman(a, b):
    ra = torch.argsort(torch.argsort(a)).float()
    rb = torch.argsort(torch.argsort(b)).float()
    ra = (ra - ra.mean()) / ra.std().clamp_min(1e-9)
    rb = (rb - rb.mean()) / rb.std().clamp_min(1e-9)
    return (ra * rb).mean().item()


def pearson(a, b):
    a = (a - a.mean()) / a.std().clamp_min(1e-9)
    b = (b - b.mean()) / b.std().clamp_min(1e-9)
    return (a * b).mean().item()


def report(l2, l2_std, ratio, ec, group, label="per_view β L2"):
    n = l2.numel()
    print(f"\n{'#' * 78}\n# Variant: {label}\n{'#' * 78}")
    print(f"\nCollected {n} per-view samples.")
    print(f"\ncrop_ratio min={ratio.min():.3f}  median={ratio.median():.3f}  "
          f"max={ratio.max():.3f}  mean={ratio.mean():.3f}")
    print(f"{label} min={l2.min():.3f}  median={l2.median():.3f}  "
          f"max={l2.max():.3f}  mean={l2.mean():.3f}")

    sp = spearman(ratio, l2)
    pe = pearson(ratio, l2)
    print(f"\nSpearman(crop_ratio, β L2) = {sp:+.3f}   (negative = more crop → larger error)")
    print(f"Pearson (crop_ratio, β L2) = {pe:+.3f}")

    print("\nMean β L2 by extreme-crop flag:")
    for flag, label in [(0, "no extreme crop"), (1, "extreme crop")]:
        mask = ec == flag
        if mask.any():
            print(f"  {label:<22} (n={int(mask.sum()):4d}):  "
                  f"L2 = {l2[mask].mean():.3f} ± {l2[mask].std():.3f}  "
                  f"ratio mean = {ratio[mask].mean():.3f}")

    print("\nMean β L2 by crop_ratio quartile (low ratio = heavy crop):")
    sorted_idx = torch.argsort(ratio)
    n = ratio.numel()
    quartiles = [
        ("Q1 (heaviest)", sorted_idx[: n // 4]),
        ("Q2", sorted_idx[n // 4 : n // 2]),
        ("Q3", sorted_idx[n // 2 : 3 * n // 4]),
        ("Q4 (lightest)", sorted_idx[3 * n // 4 :]),
    ]
    for label, idx in quartiles:
        if len(idx) == 0:
            continue
        print(f"  {label:<14} (n={len(idx):4d}):  "
              f"ratio ∈ [{ratio[idx].min():.3f}, {ratio[idx].max():.3f}]   "
              f"L2 = {l2[idx].mean():.3f} ± {l2[idx].std():.3f}")

    print("\nMean β L2 by fixed crop_ratio bins:")
    bins = [(0.0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.01)]
    for lo, hi in bins:
        mask = (ratio >= lo) & (ratio < hi)
        if mask.any():
            print(f"  ratio ∈ [{lo:.2f}, {hi:.2f}) (n={int(mask.sum()):4d}):  "
                  f"L2 = {l2[mask].mean():.3f} ± {l2[mask].std():.3f}")

    if l2_std is not None:
        print("\nSame analysis with std-normalised β L2:")
        sp = spearman(ratio, l2_std)
        print(f"Spearman(crop_ratio, β L2 / σ) = {sp:+.3f}")
        for label, idx in quartiles:
            if len(idx) == 0:
                continue
            print(f"  {label:<14} (n={len(idx):4d}):  "
                  f"L2/σ = {l2_std[idx].mean():.3f} ± {l2_std[idx].std():.3f}")

    print("\n--- Within-subject analysis (controls for subject-level β variation) ---")
    unique_groups = torch.unique(group)
    n_g = len(unique_groups)
    rho_within, n_used = [], 0
    diff_heaviest_lightest, diff_within_pairs = [], []
    rank_l2_when_heaviest, rank_l2_when_lightest = [], []
    for g in unique_groups:
        m = group == g
        if m.sum() < 2:
            continue
        r_g = ratio[m]
        l_g = l2[m]
        if r_g.std() < 1e-6:
            continue
        rho_within.append(spearman(r_g, l_g))
        n_used += 1
        order_r = torch.argsort(r_g)
        idx_heaviest = order_r[0].item()
        idx_lightest = order_r[-1].item()
        diff_heaviest_lightest.append(l_g[idx_heaviest].item() - l_g[idx_lightest].item())
        order_l = torch.argsort(l_g)
        rank_l_pos = torch.argsort(order_l).float()
        rank_l2_when_heaviest.append(rank_l_pos[idx_heaviest].item())
        rank_l2_when_lightest.append(rank_l_pos[idx_lightest].item())
    if rho_within:
        rho_within_t = torch.tensor(rho_within)
        print(f"Within-subject Spearman(crop_ratio, β L2):")
        print(f"  averaged over {n_used}/{n_g} subjects: mean={rho_within_t.mean():+.3f}, "
              f"median={rho_within_t.median():+.3f}, std={rho_within_t.std():.3f}")
        print(f"  fraction of subjects with negative ρ (more crop → larger error): "
              f"{(rho_within_t < 0).float().mean().item():.2%}")
        diff_t = torch.tensor(diff_heaviest_lightest)
        print(f"\nL2(heaviest crop) - L2(lightest crop) within each subject:")
        print(f"  mean={diff_t.mean():+.3f}, median={diff_t.median():+.3f}, "
              f"std={diff_t.std():.3f}")
        print(f"  fraction positive (heavier crop has larger L2): "
              f"{(diff_t > 0).float().mean().item():.2%}")
        rh = torch.tensor(rank_l2_when_heaviest)
        rl = torch.tensor(rank_l2_when_lightest)
        print(f"\nMean L2 rank-within-subject (0=lowest L2, V-1=highest):")
        print(f"  view with HEAVIEST crop:  rank = {rh.mean():.2f}")
        print(f"  view with LIGHTEST crop:  rank = {rl.mean():.2f}")
        print(f"  (if heavier crop → worse, expect heaviest > lightest)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-E", "--experiment_dir", required=True)
    ap.add_argument("-L", "--load_from_ckpt", required=True)
    ap.add_argument("-C", "--config", default=None)
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--num_views", type=int, default=4)
    ap.add_argument("--max_batches", type=int, default=40)
    ap.add_argument("--num_samples", type=int, default=100)
    ap.add_argument("--dataset", default=None)
    args = ap.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    os.environ["EGL_DEVICE_ID"] = args.gpus.split(",")[0]
    device = torch.device("cuda")

    trainer = load_trainer(args.experiment_dir, args.load_from_ckpt, device, args.config)

    dataset = args.dataset
    print(f"\n=== {args.experiment_dir} | dataset={dataset or 'default-bedlam'} ===")
    l2, l2_std, spa_l2, spa_l2_std, ratio, ec, group = collect(
        trainer, dataset, args.num_views, args.max_batches, device,
        num_samples=args.num_samples,
    )
    report(l2, l2_std, ratio, ec, group, label="per_view β L2 (head mean)")
    report(spa_l2, spa_l2_std, ratio, ec, group, label="sample_param_avg β L2 (mean of NF samples)")


if __name__ == "__main__":
    main()
