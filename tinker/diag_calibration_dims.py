"""Per-dim calibration check, parameterised by experiment dir.

Reports per-dim σ/σ_prior and |z|-score on whichever shape dims the head
actually models (head.shape_indices). Compares 4D-Dress (OOD) vs BEDLAM val (IID).

Usage:
    python tinker/diag_calibration_dims.py -E exp/exp_072_d20 \
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
def collect_calibration(trainer, dataset_name, num_views, num_samples, max_batches, device):
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

        sigma = u["shape_samples"].std(dim=1)            # [B*V, 45]
        gt_shape = batch["shape_params"]                  # [B*V, 45]
        mu_shape = mhr["shape"]                            # [B*V, 45]
        resid = gt_shape - mu_shape
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
    ap = argparse.ArgumentParser()
    ap.add_argument("-E", "--experiment_dir", default="exp/exp_072_d20")
    ap.add_argument(
        "-L", "--load_from_ckpt",
        default="exp/exp_072_d20/saved_models/last.ckpt",
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
    nf_head = trainer.model.nf_head
    shape_indices = list(nf_head.shape_indices)

    stds = torch.load("checkpoints/sam-3d-body-dinov3/shape_scale_std.pt",
                      weights_only=False, map_location="cpu")
    shape_std = stds["shape_std"].cpu().float()  # [45]

    print(f"\n=== {args.experiment_dir} ===")
    print(f"Modeled shape dims (head.shape_indices) — N={len(shape_indices)}: {shape_indices}")

    results = {}
    for ds in ["4d-dress", "orbit-archviz-15-bbox44-smplx"]:
        try:
            sigma, resid, z, n = collect_calibration(
                trainer, ds, args.num_views, args.num_samples, args.max_batches, device,
            )
            results[ds] = (sigma, resid, z, n)
            logger.info(f"[{ds}] collected {n} samples")
        except Exception as e:
            logger.warning(f"[{ds}] failed: {e}")
            raise

    print(f"\n{'='*84}")
    print("Per-modeled-dim calibration (σ_pred from NF samples vs σ_prior, "
          "and |z| against GT)")
    print(f"{'='*84}")
    print(f"\n{'dim':>5} {'σ_prior':>9} | "
          f"{'4D-Dress (OOD)':>30} | {'BEDLAM val (IID)':>30}")
    print(f"{'':>5} {'':>9} | "
          f"{'σ/σ_pri':>10} {'|res|':>8} {'|z|':>8} | "
          f"{'σ/σ_pri':>10} {'|res|':>8} {'|z|':>8}")
    print("-" * 84)

    def stats(d, sigma, resid, z):
        s = sigma[:, d].mean().item()
        r = resid[:, d].median().item()
        zd = z[:, d].abs().median().item()
        return s, r, zd

    for d in shape_indices:
        sp = shape_std[d].item()
        row = f"d{d:>4} {sp:>9.3f} |"
        for ds in ["4d-dress", "orbit-archviz-15-bbox44-smplx"]:
            sigma, resid, z, _ = results[ds]
            s, r, zd = stats(d, sigma, resid, z)
            row += f"  {s/sp:>9.3f} {r:>8.3f} {zd:>8.2f}  |"
        print(row)

    # Aggregate medians on modeled dims only.
    print()
    for ds in ["4d-dress", "orbit-archviz-15-bbox44-smplx"]:
        sigma, resid, z, n = results[ds]
        sigma_modeled = sigma[:, shape_indices]
        z_modeled = z[:, shape_indices]
        sp_modeled = shape_std[shape_indices]
        med_ratio = (sigma_modeled.mean(dim=0) / sp_modeled).median().item()
        med_z = z_modeled.abs().median().item()
        print(f"[{ds}] N={n}, modeled-dims-only: "
              f"median σ/σ_prior = {med_ratio:.3f}, median |z| = {med_z:.2f}")
    print("Calibrated reference: σ/σ_prior ≈ 1, |z| ≈ 0.67 (median half-Gaussian).")


if __name__ == "__main__":
    main()
