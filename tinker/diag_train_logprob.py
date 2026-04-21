"""
Check GT residual log-prob using the ACTUAL training dataloader + training mode.

The claim: train_loss_param_nll tensorboard value says log p(GT residual) ~ +160
during training. My eval-mode measurement on BEDLAM val-split multiview data
gave -43. This script loads the same checkpoint, runs forward through the
training loss on the training dataloader, and reports raw log-probs to resolve
the discrepancy.
"""
import os
import sys
import argparse
from pathlib import Path

import torch
from loguru import logger

sys.path.append(".")
from sam_3d_body.configs.config import get_config_defaults


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-E", "--experiment_dir", required=True)
    ap.add_argument("-L", "--load_from_ckpt", required=True)
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--max_batches", type=int, default=5)
    ap.add_argument("--model_mode", type=str, default="train", choices=["train", "eval"])
    args = ap.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    os.environ["EGL_DEVICE_ID"] = args.gpus.split(",")[0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from sam_3d_body.trainer import Trainer

    cfg = get_config_defaults()
    cfg.merge_from_file(str(Path(args.experiment_dir) / "config.yaml"))
    cfg.MODEL.MHR_HEAD.MHR_MODEL_PATH = "checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt"
    cfg.DATASET.NUM_WORKERS = 2  # small for fast startup

    trainer = Trainer(cfg=cfg, vis_save_dir=str(Path(args.experiment_dir) / "diag_tmp")).to(device)

    ckpt = torch.load(args.load_from_ckpt, weights_only=False, map_location="cpu")
    raw_sd = ckpt["state_dict"]
    model_sd = {k[6:] if k.startswith("model.") else k: v for k, v in raw_sd.items()}
    missing, unexpected = trainer.model.load_state_dict(model_sd, strict=False)
    logger.info(f"Loaded: missing={len(missing)}, unexpected={len(unexpected)}")

    if args.model_mode == "train":
        trainer.model.train()
        logger.warning("MODEL.TRAIN() — dropout + BN batch-stats ACTIVE everywhere.")
    else:
        trainer.model.eval()

    loader = trainer.train_dataloader()

    nf = trainer.model.nf_head
    criterion = trainer.criterion

    logger.info(f"Reading {args.max_batches} training batches with batch_size={cfg.DATASET.BATCH_SIZE} ...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= args.max_batches:
                break

            for k, v in list(batch.items()):
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            batch = trainer.preprocess(batch)
            predictions = trainer.model(batch, num_samples=cfg.MODEL.NUM_SAMPLES)

            # Use the loss module's GT residual pathway so numbers match training exactly.
            loss_dict = criterion(predictions, batch)

            # Raw log-prob (not scaled by PARAM_NLL_WEIGHT): criterion stores these in loss_dict.
            gt_lp = loss_dict.get("gt_residual_log_prob", None)
            sample_lp = predictions["uncertainty_output"]["log_prob"]  # [B*V, S]
            lp_beta = predictions["uncertainty_output"]["log_prob_beta"]
            lp_theta = predictions["uncertainty_output"]["log_prob_theta"]

            print(f"\n[batch {batch_idx}]  mode={args.model_mode}  B={batch['img'].shape[0]}")
            if gt_lp is not None:
                print(f"  GT residual log-prob: mean={gt_lp.mean().item():+.2f} "
                      f"std={gt_lp.std().item():.2f}  min={gt_lp.min().item():+.2f} "
                      f"max={gt_lp.max().item():+.2f}")
            print(f"  Sample  log-prob    : mean={sample_lp.mean().item():+.2f} "
                  f"std={sample_lp.std().item():.2f}")
            print(f"  Sample  log p(β)    : mean={lp_beta.mean().item():+.2f} "
                  f"std={lp_beta.std().item():.2f}")
            print(f"  Sample  log p(θ|β)  : mean={lp_theta.mean().item():+.2f} "
                  f"std={lp_theta.std().item():.2f}")

            if "loss_param_nll" in loss_dict:
                scaled = loss_dict["loss_param_nll"].item()
                raw_nll = scaled / cfg.LOSS.PARAM_NLL_WEIGHT if cfg.LOSS.PARAM_NLL_WEIGHT > 0 else float("nan")
                print(f"  loss_param_nll (scaled) = {scaled:+.3f}   implied mean log-prob = {-raw_nll:+.2f}")


if __name__ == "__main__":
    main()
