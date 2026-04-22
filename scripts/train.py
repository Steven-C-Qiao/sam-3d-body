import os
import shutil
import torch
import argparse
import glob
from pathlib import Path

from loguru import logger

import pytorch_lightning as pl

from pytorch_lightning.strategies import DDPStrategy
# Set PyTorch multiprocessing sharing strategy to file_system to avoid "Too many open files" error
torch.multiprocessing.set_sharing_strategy("file_system")

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

import sys
sys.path.append(".")

from sam_3d_body.trainer import Trainer
from sam_3d_body.configs.config import get_config_defaults


def run_train(exp_dir, resume_path=None, load_path=None, seed=42, dev=False, config_path=None, lr=None):
    pl.seed_everything(seed)

    cfg = get_config_defaults()

    if config_path is not None:
        cfg.merge_from_file(config_path)

    if lr is not None:
        cfg.TRAIN.LR = lr
        logger.info(f"Overriding TRAIN.LR with CLI value: {lr}")

    # if load_path is not None or resume_path is not None:
    if resume_path is not None:
        config_yaml_path = Path(exp_dir) / "config.yaml"
        if config_yaml_path.exists():
            logger.info(f"Loading config overrides from {config_yaml_path}")
            cfg.merge_from_file(str(config_yaml_path))


    torch.set_float32_matmul_precision(cfg.TRAIN.FP16_TYPE)

    if dev:
        cfg.DATASET.BATCH_SIZE = 2
        cfg.DATASET.DATASETS_AND_RATIOS = "static-hdri-bbox44-smplx"
        cfg.DATASET.NUM_WORKERS = 4
        exp_dir = "exp/exp_test"
        num_sanity_val_steps = 0
    else:
        num_sanity_val_steps = 2

    # Create directories
    model_save_dir = os.path.join(exp_dir, "saved_models")
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    vis_save_dir = os.path.join(exp_dir, "vis")
    if not os.path.exists(vis_save_dir):
        os.makedirs(vis_save_dir)


    config_yaml_out = Path(exp_dir) / "config.yaml"
    with open(config_yaml_out, "w") as f:
        f.write(cfg.dump())
    logger.info(f"Saved YAML config to {config_yaml_out}")

    model = Trainer(
        cfg=cfg,
        vis_save_dir=vis_save_dir,
        always_visualise=args.plot,
    )

    # Lightning only appends "/dataloader_idx_N" when >1 val dataloaders exist.
    # With the 4D-DRESS loader skipped (path missing), only one remains and the
    # logged key is plain "val_total_loss".
    monitor_key = "val_total_loss/dataloader_idx_0" if os.path.isdir("/scratches/kyuban/share/4DDress") else "val_total_loss"
    checkpoint_kwargs = {
        "dirpath": model_save_dir,
        "filename": "val_loss_{epoch:03d}",
        "save_top_k": 1,
        "every_n_epochs": 1,
        "save_last": False,
        "verbose": True,
        "monitor": monitor_key,
        "mode": "min",
    }
    checkpoint_callbacks = [ModelCheckpoint(**checkpoint_kwargs)]

    checkpoint_callbacks.append(ModelCheckpoint(
        dirpath=model_save_dir,
        filename="last",
        every_n_epochs=1,
        save_top_k=1,
        save_last=False,
        monitor=None,
        verbose=False,
    ))

    tensorboard_logger = TensorBoardLogger(exp_dir, name="lightning_logs")

    if load_path is not None:
        logger.info(f"Loading checkpoint: {load_path}")
        ckpt = torch.load(load_path, weights_only=False, map_location="cpu")

        # Extract state_dict from checkpoint
        state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

        model_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("model."):
                # Remove 'model.' prefix to match the actual model structure
                param_name = key[6:]  # Remove 'model.' prefix
                model_state_dict[param_name] = value
            elif not any(
                k in key
                for k in [
                    "optimizer",
                    "lr_scheduler",
                    "epoch",
                    "global_step",
                    "callbacks",
                ]
            ):
                # Try loading directly (might be a raw model checkpoint)
                model_state_dict[key] = value

        if model_state_dict:
            missing_keys, unexpected_keys = model.model.load_state_dict(
                model_state_dict, strict=False
            )
            logger.info(f"Loaded {len(model_state_dict)} parameters from checkpoint")
            if missing_keys:
                logger.warning(f"Missing keys (not loaded): {len(missing_keys)} keys")
            if unexpected_keys:
                print(unexpected_keys)
                logger.warning(
                    f"Unexpected keys (ignored): {len(unexpected_keys)} keys"
                )
        else:
            logger.warning("No model parameters found in checkpoint state_dict!")


    trainer = pl.Trainer(
        max_epochs=cfg.TRAIN.NUM_EPOCHS,
        max_steps=getattr(cfg.TRAIN, "MAX_STEPS", -1),
        devices="auto",
        # strategy=DDPStrategy(find_unused_parameters=True),
        strategy="auto",
        callbacks=checkpoint_callbacks,
        logger=tensorboard_logger,
        num_sanity_val_steps=0,
        gradient_clip_val=1.0,
        precision="16-mixed" if cfg.TRAIN.USE_FP16 else "32-true",
        profiler=(os.environ.get("PL_PROFILER") or None),
    )
    trainer.fit(model, ckpt_path=resume_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment_dir",
        "-E",
        type=str,
        help="Path to directory where logs and checkpoints are saved.",
    )
    parser.add_argument(
        "--resume_training_states",
        "-R",
        type=str,
        default=None,
        help="Load training state. For resuming.",
    )
    parser.add_argument(
        "--load_from_ckpt",
        "-L",
        type=str,
        default=None,
        help="Path to checkpoint. Load for finetuning",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="Comma-separated list of GPU indices to use. E.g., '0,1,2'",
    )
    parser.add_argument("--dev", action="store_true")
    parser.add_argument("--config", "-C", type=str, default=None,
                        help="YAML config override file (merged on top of defaults).")
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override TRAIN.LR from the config.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="If set, always generate visualisations each step.",
    )
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    if args.plot:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        os.environ["EGL_DEVICE_ID"] = "0"
    if args.gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        device_ids = list(map(int, args.gpus.split(",")))
        # EGL (used by pyrender) selects its GPU via EGL_DEVICE_ID, which is
        # independent of CUDA_VISIBLE_DEVICES and defaults to physical device 0.
        # Set it explicitly to keep the renderer on the same GPU as training.
        os.environ["EGL_DEVICE_ID"] = str(device_ids[0])
        logger.info(f"Using GPUs: {args.gpus} (Device IDs: {device_ids})")

    run_train(
        exp_dir=args.experiment_dir,
        resume_path=args.resume_training_states,
        load_path=args.load_from_ckpt,
        dev=args.dev,
        config_path=args.config,
        lr=args.lr,
    )
