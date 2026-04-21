"""
Compare sample log-prob computed via the INVERSE path (sample_and_log_prob,
current code) vs the FORWARD path (log_prob(x)) on the same sampled x.

Hypothesis (user's): eval() mode evaluation "in the other direction" differs,
train() mode matches. Reason: the forward path goes through BatchNorm1d inside
each coupling's ResNet; eval uses running stats (calibrated to GT x
distribution seen during training), train uses current-batch stats.

What to expect:
  - For any mode where the flow is a true bijection at fixed BN state:
       log_prob_forward(x_sampled) == sample_log_prob
    Because forward(inverse(z)) == z exactly, and
       lp_inverse = log p_Z(z) - const,  lp_forward = log p_Z(f(x)) + const
    round-trip → same.
  - In train mode, BN batch stats can differ between the inverse pass
    (batch = z-derived tensors) and the forward pass (batch = x-derived
    tensors), breaking the exact inversion — so a gap is possible.
"""
import os
import sys
import argparse
from pathlib import Path

import torch
from loguru import logger

sys.path.append(".")
from sam_3d_body.configs.config import get_config_defaults


def sample_and_recompute(flow, num_samples, context, device):
    """Draw N samples, return both the inverse-path log-prob and the forward-path log-prob."""
    import math
    B = context.shape[0]
    D = flow._distribution._shape[0]

    # Draw z, expand context, go through inverse manually so we share z explicitly.
    z = torch.randn(B, num_samples, D, device=device)
    z_flat = z.reshape(B * num_samples, D)
    ctx_flat = context.unsqueeze(1).expand(-1, num_samples, -1).reshape(B * num_samples, -1)

    # Inverse path: x = f_inv(z, c), returns logabsdet for z → x.
    x_flat, logabsdet_inv = flow._transform.inverse(z_flat, context=ctx_flat)
    log_pz = -0.5 * (z_flat ** 2).sum(-1) - 0.5 * D * math.log(2 * math.pi)
    lp_inverse = log_pz - logabsdet_inv  # matches the sample_and_log_prob convention

    # Forward path: evaluate log-prob of x under the same flow.
    lp_forward, _ = flow.log_prob(inputs=x_flat, context=ctx_flat)

    return lp_inverse.reshape(B, num_samples), lp_forward.reshape(B, num_samples)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-E", "--experiment_dir", required=True)
    ap.add_argument("-L", "--load_from_ckpt", required=True)
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--max_batches", type=int, default=3)
    ap.add_argument("--num_samples", type=int, default=64)
    args = ap.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    os.environ["EGL_DEVICE_ID"] = args.gpus.split(",")[0]

    device = torch.device("cuda")
    from sam_3d_body.trainer import Trainer

    cfg = get_config_defaults()
    cfg.merge_from_file(str(Path(args.experiment_dir) / "config.yaml"))
    cfg.MODEL.MHR_HEAD.MHR_MODEL_PATH = "checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt"
    cfg.DATASET.NUM_WORKERS = 2

    trainer = Trainer(cfg=cfg, vis_save_dir=str(Path(args.experiment_dir) / "diag_tmp")).to(device)
    ckpt = torch.load(args.load_from_ckpt, weights_only=False, map_location="cpu")
    raw_sd = ckpt["state_dict"]
    model_sd = {k[6:] if k.startswith("model.") else k: v for k, v in raw_sd.items()}
    trainer.model.load_state_dict(model_sd, strict=False)

    loader = trainer.train_dataloader()

    for mode_name in ["eval", "flow_train"]:
        trainer.model.eval()
        if mode_name == "flow_train":
            trainer.model.nf_head.flow_beta.train()
            trainer.model.nf_head.flow_theta.train()

        logger.info(f"\n===== mode = {mode_name} =====")
        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
                if batch_idx >= args.max_batches:
                    break
                for k, v in list(batch.items()):
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(device)
                batch = trainer.preprocess(batch)

                # Forward through the model to get contexts + means.
                predictions = trainer.model(batch, num_samples=cfg.MODEL.NUM_SAMPLES)
                u = predictions["uncertainty_output"]

                nf = trainer.model.nf_head
                ctx_beta = u["flow_context_beta"]
                # flow_context_theta is [B, S, C] post-projection with shape_sample_from_beta — we need
                # a single context per batch-element for a fresh sample draw. Reuse the beta flow
                # context to build a stage-2 context at the mean (shape_mean, scale_mean, pose_mean).
                mhr = predictions["mhr"]
                from sam_3d_body.models.modules.mhr_utils import (
                    convert_pose_cont_to_flow_context, scale_indices,
                )
                pose_params = convert_pose_cont_to_flow_context(mhr["pred_pose_raw"][:, 6:])
                ctx_theta_parts = [
                    u["flow_context_raw"],
                    mhr["shape"],                         # condition on mean shape (Δβ=0)
                    mhr["scale_68D"][..., scale_indices], # condition on mean scale (Δscale=0)
                    pose_params["aa_3dofs"],
                    pose_params["params_1dofs"],
                ]
                if nf.model_cam:
                    ctx_theta_parts.append(mhr["pred_cam"])
                ctx_theta = nf.theta_context_proj(torch.cat(ctx_theta_parts, dim=-1))

                lp_inv_beta, lp_fwd_beta = sample_and_recompute(
                    nf.flow_beta, args.num_samples, ctx_beta, device,
                )
                lp_inv_theta, lp_fwd_theta = sample_and_recompute(
                    nf.flow_theta, args.num_samples, ctx_theta, device,
                )

                diff_beta = lp_fwd_beta - lp_inv_beta
                diff_theta = lp_fwd_theta - lp_inv_theta

                print(
                    f"[batch {batch_idx}] mode={mode_name} "
                    f"B={ctx_beta.shape[0]} S={args.num_samples}"
                )
                print(
                    f"  β: inverse_lp mean={lp_inv_beta.mean().item():+.2f}  "
                    f"forward_lp mean={lp_fwd_beta.mean().item():+.2f}  "
                    f"Δ(fwd-inv) mean={diff_beta.mean().item():+.3f} ± {diff_beta.std().item():.3f} "
                    f"max|Δ|={diff_beta.abs().max().item():.3f}"
                )
                print(
                    f"  θ: inverse_lp mean={lp_inv_theta.mean().item():+.2f}  "
                    f"forward_lp mean={lp_fwd_theta.mean().item():+.2f}  "
                    f"Δ(fwd-inv) mean={diff_theta.mean().item():+.3f} ± {diff_theta.std().item():.3f} "
                    f"max|Δ|={diff_theta.abs().max().item():.3f}"
                )


if __name__ == "__main__":
    main()
