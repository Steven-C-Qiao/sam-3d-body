"""Per-PC geometric variance contribution.

For each shape PC d:
  * Perturb shape by +σ_prior_d on dim d only (others zero).
  * Build neutral-pose MHR mesh.
  * Measure per-vertex displacement v_d - v_0.
  * Geometric variance contribution := mean(||v_d - v_0||²)  [m²]
    (equivalent to RMS displacement squared, integrated over all verts)

This is the variance of the per-vertex 3-D coordinate due to a 1σ_prior
perturbation on PC d, averaged over vertices. Local linearisation
approximation; exact for linear models.

Reports per-PC contribution and cumulative variance retained by first K PCs.
"""
import os
import sys
import argparse
from pathlib import Path

import numpy as np
import torch
from loguru import logger

sys.path.append(".")


def load_trainer(exp_dir, load_path, device):
    from sam_3d_body.configs.config import get_config_defaults
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
def get_neutral_inputs_and_scale(trainer, device):
    loader = trainer.multiview_eval_dataloader(num_view=1, batch_size=1, dataset_name="4d-dress")
    batch = next(iter(loader))
    for k, v in list(batch.items()):
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)
    bs, num_views = batch["img"].shape[:2]
    for k, v in list(batch.items()):
        if isinstance(v, torch.Tensor):
            if v.dim() >= 2 and v.shape[0] == bs and v.shape[1] == num_views:
                batch[k] = v.flatten(0, 1)
    batch = trainer.preprocess(batch)
    outputs = trainer.model(batch, num_samples=1)
    mhr = outputs["mhr"]

    z = {
        "global_trans": torch.zeros_like(mhr["global_rot"][:1]),
        "global_rot": torch.zeros_like(mhr["global_rot"][:1]),
        "body_pose_params": torch.zeros_like(mhr["body_pose"][:1]),
        "hand_pose_params": torch.zeros_like(mhr["hand"][:1]),
        "expr_params": torch.zeros_like(mhr["face"][:1]),
    }
    base_scale = trainer.model.head_pose.scale_mean[None, :].to(device).float()
    return z, base_scale


@torch.no_grad()
def verts_for(mhr_head, zero_inputs, base_shape, base_scale):
    out = mhr_head.mhr_forward(
        shape_params=base_shape,
        scale_offsets=base_scale,
        do_pcblend=True,
        **zero_inputs,
    )
    verts = out[0] if isinstance(out, tuple) else out  # [1, V, 3]
    return verts[0]                                     # [V, 3]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-E", "--experiment_dir", default="exp/exp_071_crop_shape")
    ap.add_argument(
        "-L", "--load_from_ckpt",
        default="exp/exp_071_crop_shape/saved_models/last.ckpt",
    )
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--symmetric", action="store_true",
                    help="Average +σ and -σ deformation magnitudes (more robust).")
    args = ap.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    os.environ["EGL_DEVICE_ID"] = args.gpus.split(",")[0]
    device = torch.device("cuda")

    trainer = load_trainer(args.experiment_dir, args.load_from_ckpt, device)
    mhr_head = trainer.model.head_pose

    stds = torch.load("checkpoints/sam-3d-body-dinov3/shape_scale_std.pt",
                      weights_only=False, map_location=device)
    shape_std = stds["shape_std"].to(device).float()  # [45]

    zero_inputs, base_scale = get_neutral_inputs_and_scale(trainer, device)
    base_shape = torch.zeros(1, 45, device=device)
    v_neutral = verts_for(mhr_head, zero_inputs, base_shape, base_scale)  # [V, 3]
    n_verts = v_neutral.shape[0]

    # Per-PC geometric variance contribution.
    geom_var = np.zeros(45)            # mean ||v_d - v_0||²  (m²)
    rms_disp = np.zeros(45)             # sqrt(geom_var)        (m)
    max_disp = np.zeros(45)             # max ||v_d - v_0||    (m)

    for d in range(45):
        # Perturb +σ_prior_d.
        shape_p = base_shape.clone()
        shape_p[0, d] = shape_std[d]
        v_p = verts_for(mhr_head, zero_inputs, shape_p, base_scale)
        disp_p = (v_p - v_neutral).norm(dim=-1)             # [V]

        if args.symmetric:
            shape_n = base_shape.clone()
            shape_n[0, d] = -shape_std[d]
            v_n = verts_for(mhr_head, zero_inputs, shape_n, base_scale)
            disp_n = (v_n - v_neutral).norm(dim=-1)         # [V]
            disp = 0.5 * (disp_p + disp_n)
        else:
            disp = disp_p

        geom_var[d] = (disp ** 2).mean().item()
        rms_disp[d] = disp.pow(2).mean().sqrt().item()
        max_disp[d] = disp.max().item()

    # Compare to parameter variance.
    param_var = (shape_std.cpu().numpy()) ** 2

    # Sort by geometric variance contribution.
    geom_order = np.argsort(geom_var)[::-1]              # largest first
    cum_geom_in_index_order = np.cumsum(geom_var)        # for first-K analysis
    total_geom = geom_var.sum()

    print(f"\n{'='*80}")
    print(f"Per-PC geometric variance contribution (1σ_prior perturbation)")
    print(f"{'='*80}")
    print(f"\nTotal geometric variance Σ_d ⟨||v_d - v_0||²⟩  =  {total_geom*1e6:.2f} mm²")
    print(f"Total RMS displacement (1σ each PC, RSS)         =  {np.sqrt(total_geom)*1000:.2f} mm")

    # Per-dim table, in original index order.
    print(f"\n{'d':>3} {'σ_prior':>9} {'param-var%':>11} {'geom-RMS(mm)':>14} "
          f"{'max-disp(mm)':>14} {'geom-var%':>10} {'cum-geom%':>10}")
    print("-" * 80)
    for d in range(45):
        marker = ""
        if d == 9: marker = "  ← K=10"
        elif d == 19: marker = "  ← K=20"
        elif d == 29: marker = "  ← K=30"
        elif d == 44: marker = "  ← all"
        print(
            f"{d:>3} {shape_std[d].item():>9.4f} "
            f"{param_var[d]/param_var.sum()*100:>10.2f}% "
            f"{rms_disp[d]*1000:>13.2f} "
            f"{max_disp[d]*1000:>13.2f} "
            f"{geom_var[d]/total_geom*100:>9.2f}% "
            f"{cum_geom_in_index_order[d]/total_geom*100:>9.2f}%{marker}"
        )

    # First-K cumulative on both metrics.
    print(f"\n{'='*80}\nVariance retained by KEEPING FIRST K PCs (in index order)")
    print(f"{'='*80}")
    print(f"{'K':>3}  {'param-var%':>11}  {'geom-var%':>11}")
    for K in [5, 8, 10, 12, 15, 20, 25, 30, 35, 40, 45]:
        pv = param_var[:K].sum() / param_var.sum() * 100
        gv = geom_var[:K].sum() / total_geom * 100
        print(f"{K:>3}  {pv:>10.2f}%  {gv:>10.2f}%")

    # If you instead kept top-K by geometric variance.
    print(f"\n{'='*80}\nVariance retained by TOP-K geom-var PCs (re-ordered)")
    print(f"{'='*80}")
    print(f"{'K':>3}  {'PCs (idx)':<60}  {'cum geom%':>10}")
    cum_geom_sorted = np.cumsum(geom_var[geom_order])
    for K in [5, 8, 10, 12, 15, 20, 25, 30]:
        idx_str = ", ".join(f"d{d}" for d in geom_order[:K])
        print(f"{K:>3}  {idx_str[:58]:<60}  "
              f"{cum_geom_sorted[K-1] / total_geom * 100:>9.2f}%")

    # Compare top-by-param-var vs top-by-geom-var.
    print(f"\n{'='*80}\nTop-10 dims by each ranking")
    print(f"{'='*80}")
    print(f"  by param σ_prior²:   {[int(d) for d in np.argsort(param_var)[::-1][:10]]}")
    print(f"  by geometric var:    {[int(d) for d in geom_order[:10]]}")

    # Where the disagreement and uncertainty dims fall.
    print(f"\nDims of interest (rank in each ordering):")
    for d in [10, 11, 18, 25, 34, 43, 24, 7, 31, 28]:
        rank_param = int(np.where(np.argsort(param_var)[::-1] == d)[0][0]) + 1
        rank_geom = int(np.where(geom_order == d)[0][0]) + 1
        print(f"  d{d:>2}: param-var rank = {rank_param:>2},  "
              f"geom-var rank = {rank_geom:>2},  "
              f"geom-RMS = {rms_disp[d]*1000:.2f} mm,  "
              f"σ_prior = {shape_std[d].item():.3f}")


if __name__ == "__main__":
    main()
