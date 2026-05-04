"""Calibration check: predicted per-view σ vs actual |GT residual|.

For each shape PC d, computes per-subject:
  z_d = (β_gt[d] - μ_j[d]) / σ_predicted[d]

A well-calibrated flow has |z| ~ 1. Overconfidence is |z| >> 1.

Compares 4D-Dress (subjects unseen during training) vs a BEDLAM val
set (training-distribution). If σ/σ_prior is similar across both, the
overconfidence is structural — not subject-identity-leakage.

Usage:
    python tinker/diag_calibration_iid_ood.py
"""
import os
import sys
from pathlib import Path

import torch
from loguru import logger

sys.path.append(".")

from sam_3d_body.configs.config import get_config_defaults

UNC_DIMS = [43, 24, 7, 31, 28]
DIS_DIMS = [10, 34, 25, 18, 11]


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
def collect_calibration(trainer, dataset_name, num_views, num_samples, max_batches, device):
    """Returns:
        sigma_pred:   [N, 45]  per-(subject, view) predicted shape σ from NF samples
        abs_resid:    [N, 45]  |β_gt - μ_j| per (subject, view)
        z_score:      [N, 45]  signed z = (β_gt - μ_j) / σ_pred
    """
    loader = trainer.multiview_eval_dataloader(
        num_view=num_views, batch_size=1, dataset_name=dataset_name,
    )

    all_sigma, all_resid, all_z = [], [], []
    n_collected = 0
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

        # NF-sample-based predicted σ (per (b*v, d)).
        sigma = u["shape_samples"].std(dim=1)        # [B*V, 45]
        # GT residual.
        gt_shape = batch["shape_params"]              # [B*V, 45]
        mu_shape = mhr["shape"]                        # [B*V, 45]
        resid = gt_shape - mu_shape                    # [B*V, 45]
        z = resid / sigma.clamp_min(1e-6)
        all_sigma.append(sigma.cpu().float())
        all_resid.append(resid.abs().cpu().float())
        all_z.append(z.cpu().float())
        n_collected += sigma.shape[0]

    return (torch.cat(all_sigma, dim=0),
            torch.cat(all_resid, dim=0),
            torch.cat(all_z, dim=0),
            n_collected)


def main():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["EGL_DEVICE_ID"] = "0"
    device = torch.device("cuda")

    trainer = load_trainer("exp/exp_071_crop_shape",
                           "exp/exp_071_crop_shape/saved_models/last.ckpt", device)

    stds = torch.load("checkpoints/sam-3d-body-dinov3/shape_scale_std.pt",
                       weights_only=False, map_location=device)
    shape_std = stds["shape_std"].cpu().float()  # [45]

    NUM_BATCHES = 6
    NUM_SAMPLES = 100
    NUM_VIEWS = 4

    print(f"\nTraining datasets (BEDLAM): see config.DATASETS_AND_RATIOS — "
          f"all bedlam-* variants, no 4d-dress.")

    results = {}
    for ds in ["4d-dress", "orbit-archviz-15-bbox44-smplx"]:
        try:
            sigma, resid, z, n = collect_calibration(
                trainer, ds, NUM_VIEWS, NUM_SAMPLES, NUM_BATCHES, device,
            )
            results[ds] = (sigma, resid, z, n)
            logger.info(f"[{ds}] collected {n} samples")
        except Exception as e:
            logger.warning(f"[{ds}] failed: {e}")

    # ---- Report ----
    print(f"\n{'='*72}")
    print(f"Calibration: |β_gt - μ| / σ_predicted per dim")
    print(f"{'='*72}")

    print(f"\n{'dim':>5} {'set':>5} {'σ_prior':>9} | "
          f"{'4D-Dress (OOD)':>34} | {'BEDLAM (IID)':>34}")
    print(f"{'':>5} {'':>5} {'':>9} | "
          f"{'σ/σ_pri':>10} {'|res|':>8} {'med |z|':>10} | "
          f"{'σ/σ_pri':>10} {'|res|':>8} {'med |z|':>10}")
    print("-" * 110)

    def stats(d, sigma, resid, z):
        s = sigma[:, d].mean().item()
        r = resid[:, d].median().item()
        zd = z[:, d].abs().median().item()
        return s, r, zd

    for label, dims in [("UNC", UNC_DIMS), ("DIS", DIS_DIMS)]:
        for d in dims:
            sp = shape_std[d].item()
            row = f"d{d:>4} {label:>5} {sp:>9.3f} |"
            for ds in ["4d-dress", "orbit-archviz-15-bbox44-smplx"]:
                if ds in results:
                    sigma, resid, z, _ = results[ds]
                    s, r, zd = stats(d, sigma, resid, z)
                    row += f"  {s/sp:>9.3f} {r:>8.3f} {zd:>10.2f}  |"
                else:
                    row += f"  {'-':>9} {'-':>8} {'-':>10}  |"
            print(row)

    # Aggregate medians.
    print()
    for ds in ["4d-dress", "orbit-archviz-15-bbox44-smplx"]:
        if ds not in results:
            continue
        sigma, resid, z, n = results[ds]
        med_ratio = (sigma.mean(dim=0) / shape_std).median().item()
        med_z = z.abs().median().item()
        med_resid_per_sigma = (resid.median(dim=0).values / sigma.mean(dim=0)).median().item()
        print(f"[{ds}] N={n}, "
              f"median σ/σ_prior across 45 dims = {med_ratio:.3f}, "
              f"median |z| = {med_z:.2f}, "
              f"median (|res| / σ_pred) per dim = {med_resid_per_sigma:.2f}")
    print("Well-calibrated would give |z| ≈ 0.67 (median half-Gaussian) and "
          "(|res| / σ_pred) ≈ 0.67.")


if __name__ == "__main__":
    main()
