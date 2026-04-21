"""
Diagnostic: measure the per-sample spread of NF log-likelihoods from a trained
NFARHead, comparing stage-1 (beta) only, stage-2 (theta|beta) only, and the
joint sum.

Tests the claim:
    "The overall log-likelihood (beta + theta) is stable (range a few nats),
     but the stage-wise log-likelihoods individually have a much larger range."

Usage:
    python scripts/diag_logprob_spread.py \
        -E exp/exp_058_so3_c4 \
        -L exp/exp_058_so3_c4/saved_models/last.ckpt \
        --dataset 4d-dress --num_views 4 --num_samples 100 --max_batches 3
"""
import os
import sys
import argparse
from pathlib import Path

import numpy as np
import torch
from loguru import logger

sys.path.append(".")

from sam_3d_body.configs.config import get_config_defaults


def load_trainer(exp_dir, load_path, device):
    from sam_3d_body.trainer import Trainer

    cfg = get_config_defaults()
    config_yaml = Path(exp_dir) / "config.yaml"
    if config_yaml.exists():
        cfg.merge_from_file(str(config_yaml))
    cfg.MODEL.MHR_HEAD.MHR_MODEL_PATH = (
        "checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt"
    )

    trainer = Trainer(cfg=cfg, vis_save_dir=str(Path(exp_dir) / "diag_tmp")).to(device)

    logger.info(f"Loading checkpoint: {load_path}")
    ckpt = torch.load(load_path, weights_only=False, map_location="cpu")
    raw_sd = ckpt["state_dict"]
    model_sd = {}
    for k, v in raw_sd.items():
        model_sd[k[6:] if k.startswith("model.") else k] = v
    missing, unexpected = trainer.model.load_state_dict(model_sd, strict=False)
    logger.info(f"Loaded {len(model_sd)} params; missing={len(missing)}, unexpected={len(unexpected)}")

    trainer.model.eval()
    return trainer, cfg


def compute_mode_logprob(nf, u, mhr):
    """Compute log p(x_star | c) for x_star = f_inv(z=0; c), for both stages.

    For additive-coupling ConditionalGlow the total log|det J| is constant,
    so this should equal the sample-set max log-prob (up to float noise).
    """
    from sam_3d_body.models.modules.mhr_utils import (
        convert_pose_cont_to_flow_context, scale_indices,
    )
    device = u["flow_context_beta"].device
    BV = u["flow_context_beta"].shape[0]

    # Stage 1: mode via z=0.
    ctx_beta = u["flow_context_beta"]                     # [BV, 2048]
    z_beta = torch.zeros(BV, 1, nf.beta_dim, device=device)
    x_beta_star, lp_beta_star, _ = nf.flow_beta.sample_and_log_prob(
        num_samples=1, noise=z_beta, context=ctx_beta,
    )
    x_beta_star = x_beta_star.squeeze(1)                  # [BV, beta_dim]
    lp_beta_star = lp_beta_star.squeeze(1)                # [BV]

    # Stage 2: context built at stage-1 mode (autoregressive joint mode).
    shape_res = x_beta_star[..., : nf.num_shape_comps]
    scale_res = x_beta_star[..., nf.num_shape_comps:]
    shape_at_mode = mhr["shape"] + shape_res
    scale_selected_at_mode = mhr["scale_68D"][..., scale_indices] + scale_res

    pose_cont = mhr["pred_pose_raw"][:, 6:]
    pose_params = convert_pose_cont_to_flow_context(pose_cont)
    ctx_theta_parts = [
        u["flow_context_raw"],
        shape_at_mode,
        scale_selected_at_mode,
        pose_params["aa_3dofs"],
        pose_params["params_1dofs"],
    ]
    if nf.model_cam:
        ctx_theta_parts.append(mhr["pred_cam"])
    ctx_theta = nf.theta_context_proj(torch.cat(ctx_theta_parts, dim=-1))

    z_theta = torch.zeros(BV, 1, nf.theta_dim, device=device)
    _, lp_theta_star, _ = nf.flow_theta.sample_and_log_prob(
        num_samples=1, noise=z_theta, context=ctx_theta,
    )
    lp_theta_star = lp_theta_star.squeeze(1)              # [BV]

    return lp_beta_star, lp_theta_star


def compute_gt_residual_logprob(nf, u, mhr, batch):
    """log p(Δβ_gt | c) and log p(Δθ_gt | c, Δβ_gt), matching loss logic."""
    from sam_3d_body.models.modules.mhr_utils import (
        convert_mhr_params_to_flow_params,
        convert_pose_cont_to_flow_context,
        so3_residual_aa,
        batch9Dfrom6D,
        scale_indices,
    )

    gt_flow_params, gt_rotmats = convert_mhr_params_to_flow_params(
        batch["model_params"], batch["shape_params"],
        include_global_rot=nf.model_glob_rot,
        include_shape=nf.model_shape,
        include_scale=nf.model_scale,
        flip_global_rot=True,
        return_rotmats=True,
    )
    pose_params = convert_pose_cont_to_flow_context(mhr["pred_pose_raw"][:, 6:])

    mean_beta = torch.cat(
        [mhr["shape"], mhr["scale_68D"][..., scale_indices]], dim=-1
    )
    beta_residual = gt_flow_params[..., : nf.beta_dim] - mean_beta

    pose_3dof_residual = so3_residual_aa(
        pose_params["rotmat_3dofs"], gt_rotmats["pose_3dof_rotmat"]
    )
    offset_1dof = nf.beta_dim + 39
    pose_1dof_residual = (
        gt_flow_params[..., offset_1dof : offset_1dof + 34]
        - pose_params["params_1dofs"]
    )
    theta_parts = [pose_3dof_residual, pose_1dof_residual]
    if nf.model_glob_rot:
        mean_glob_rotmat = batch9Dfrom6D(mhr["pred_pose_raw"][:, :6]).unflatten(-1, (3, 3))
        glob_rot_residual = so3_residual_aa(
            mean_glob_rotmat.unsqueeze(-3),
            gt_rotmats["glob_rotmat"].unsqueeze(-3),
        )
        theta_parts.append(glob_rot_residual)
    theta_residual_no_cam = torch.cat(theta_parts, dim=-1)

    if nf.model_cam:
        cam_residual = batch["gt_pred_cam"] - mhr["pred_cam"]
        theta_residual = torch.cat([theta_residual_no_cam, cam_residual], dim=-1)
    else:
        theta_residual = theta_residual_no_cam

    # Stage-1 at GT.
    ctx_beta = u["flow_context_beta"]
    lp_beta_gt, _ = nf.flow_beta.log_prob(beta_residual, ctx_beta)

    # Stage-2 at GT (teacher-forced shape).
    shape_gt = mhr["shape"] + beta_residual[..., : nf.num_shape_comps]
    scale_selected_gt = (
        mhr["scale_68D"][..., scale_indices] + beta_residual[..., nf.num_shape_comps:]
    )
    ctx_theta_parts = [
        u["flow_context_raw"], shape_gt, scale_selected_gt,
        pose_params["aa_3dofs"], pose_params["params_1dofs"],
    ]
    if nf.model_cam:
        ctx_theta_parts.append(mhr["pred_cam"])
    ctx_theta = nf.theta_context_proj(torch.cat(ctx_theta_parts, dim=-1))
    lp_theta_gt, _ = nf.flow_theta.log_prob(theta_residual, ctx_theta)

    return lp_beta_gt, lp_theta_gt


def summarize(vals, label):
    """vals: tensor of shape [N_samples] or [B, V, S]"""
    v = vals.detach().float().flatten(-1)  # flatten last dim only if 1-D
    # Support both 1-D and higher-D: always compute per-last-dim stats.
    if v.dim() == 0:
        return
    if vals.dim() == 1:
        spread = (vals.max() - vals.min()).item()
        std = vals.std().item()
        mean = vals.mean().item()
        logger.info(f"  {label}: mean={mean:+.2f} std={std:.2f} range(max-min)={spread:.2f}")
    else:
        # Compute spread per (B, V) across S samples.
        spread = (vals.max(dim=-1).values - vals.min(dim=-1).values)  # [B, V]
        std = vals.std(dim=-1)  # [B, V]
        mean = vals.mean(dim=-1)  # [B, V]
        logger.info(
            f"  {label}: "
            f"mean={mean.mean().item():+.2f} ± {mean.std().item():.2f}, "
            f"per-view std={std.mean().item():.2f} ± {std.std().item():.2f}, "
            f"per-view range={spread.mean().item():.2f} ± {spread.std().item():.2f} "
            f"(max={spread.max().item():.2f})"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-E", "--experiment_dir", required=True)
    ap.add_argument("-L", "--load_from_ckpt", required=True)
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--dataset", default="4d-dress")
    ap.add_argument("--num_views", type=int, default=4)
    ap.add_argument("--num_samples", type=int, default=100)
    ap.add_argument("--max_batches", type=int, default=3)
    ap.add_argument(
        "--train_mode", type=str, default="eval", choices=["eval", "full_train", "flow_only"],
        help="eval: all modules in eval. full_train: model.train(). "
             "flow_only: flow_beta/flow_theta in train mode (dropout + BN batch-stats), rest in eval."
    )
    args = ap.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    device_id0 = int(args.gpus.split(",")[0])
    os.environ["EGL_DEVICE_ID"] = str(device_id0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    trainer, cfg = load_trainer(args.experiment_dir, args.load_from_ckpt, device)

    trainer.model.eval()
    if args.train_mode == "full_train":
        trainer.model.train()
        logger.warning("Running model.train() — dropout/BN active everywhere.")
    elif args.train_mode == "flow_only":
        trainer.model.nf_head.flow_beta.train()
        trainer.model.nf_head.flow_theta.train()
        logger.warning("Flow-only TRAIN mode: dropout + BN batch-stats active in NF only.")

    loader = trainer.multiview_eval_dataloader(
        num_view=args.num_views, batch_size=1, dataset_name=args.dataset,
    )

    all_logp_beta = []
    all_logp_theta = []
    all_logp_total = []
    all_cross_logp_beta = []  # shape [B, V_i, V_j, S] with diagonal masked
    all_mode_beta = []
    all_mode_theta = []
    all_gt_beta = []
    all_gt_theta = []

    with torch.no_grad():
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
            u = outputs["uncertainty_output"]
            mhr = outputs["mhr"]

            # Reshape to [B, V, S]
            lp_beta = u["log_prob_beta"].unflatten(0, (bs, num_views))   # [B, V, S]
            lp_theta = u["log_prob_theta"].unflatten(0, (bs, num_views)) # [B, V, S]
            lp_total = u["log_prob"].unflatten(0, (bs, num_views))       # [B, V, S]

            all_logp_beta.append(lp_beta.cpu())
            all_logp_theta.append(lp_theta.cpu())
            all_logp_total.append(lp_total.cpu())

            # Cross-view beta log-prob: evaluate sampled beta residual from view i under view j's flow.
            # Reconstruct stage-1 residual for view i: uses beta_residual_samples = samples[..., :beta_dim].
            nf = trainer.model.nf_head
            beta_dim = nf.beta_dim
            beta_res = u["samples"][..., :beta_dim].unflatten(0, (bs, num_views))  # [B, V, S, 55]

            # For cross-view: beta sample from view i is an ABSOLUTE beta.
            #   abs_beta_i = shape_mean_i + shape_res_i  (concat with scale).
            # Under view j's flow, residual is abs_beta_i - mean_beta_j.
            shape_mean = mhr["shape"].unflatten(0, (bs, num_views))
            scale68_mean = mhr["scale_68D"].unflatten(0, (bs, num_views))
            from sam_3d_body.models.modules.mhr_utils import scale_indices
            mean_beta = torch.cat(
                [shape_mean, scale68_mean[..., scale_indices]], dim=-1
            )  # [B, V, 55]
            # Absolute beta samples from view i: mean_beta_i + beta_res_i.
            abs_beta = mean_beta.unsqueeze(2) + beta_res  # [B, V, S, 55]

            beta_context = u["flow_context_beta"].unflatten(0, (bs, num_views))  # [B, V, 2048]

            S = args.num_samples
            cross = torch.zeros(bs, num_views, num_views, S, device=device)
            for b in range(bs):
                for i in range(num_views):
                    for j in range(num_views):
                        if i == j:
                            # Self-view log-prob (should equal lp_beta[b,i] within float error).
                            cross[b, i, j] = lp_beta[b, i].to(device)
                            continue
                        res_ij = abs_beta[b, i] - mean_beta[b, j].unsqueeze(0)  # [S, 55]
                        ctx_j = beta_context[b, j].unsqueeze(0).expand(S, -1)  # [S, 2048]
                        lp_ij, _ = nf.flow_beta.log_prob(inputs=res_ij, context=ctx_j)
                        cross[b, i, j] = lp_ij
            all_cross_logp_beta.append(cross.cpu())

            # ----- Mode (z=0) log-prob for both stages -----
            mode_beta, mode_theta = compute_mode_logprob(nf, u, mhr)
            all_mode_beta.append(mode_beta.unflatten(0, (bs, num_views)).cpu())
            all_mode_theta.append(mode_theta.unflatten(0, (bs, num_views)).cpu())

            # ----- GT residual log-prob for both stages -----
            gt_beta, gt_theta = compute_gt_residual_logprob(nf, u, mhr, batch)
            all_gt_beta.append(gt_beta.unflatten(0, (bs, num_views)).cpu())
            all_gt_theta.append(gt_theta.unflatten(0, (bs, num_views)).cpu())

            logger.info(f"[batch {batch_idx}] bs={bs} V={num_views} S={args.num_samples}")

    lp_beta = torch.cat(all_logp_beta, dim=0)      # [B_total, V, S]
    lp_theta = torch.cat(all_logp_theta, dim=0)
    lp_total = torch.cat(all_logp_total, dim=0)
    cross = torch.cat(all_cross_logp_beta, dim=0)  # [B_total, V, V, S]

    logger.info("=" * 78)
    logger.info(f"AGGREGATE over {lp_beta.shape[0]} sernos × {lp_beta.shape[1]} views × {lp_beta.shape[2]} samples")
    logger.info("-" * 78)
    logger.info("Self-view log-probs (samples drawn from p(·|I_i), evaluated under p(·|I_i)):")
    summarize(lp_beta, "log p(Δβ | c)          ")
    summarize(lp_theta, "log p(Δθ | c, Δβ)      ")
    summarize(lp_total, "log p(Δβ,Δθ) = β + θ   ")
    logger.info("-" * 78)
    # Per-sample decomposition: does (β + θ) cancel sample-to-sample variation?
    rng_beta = (lp_beta.max(dim=-1).values - lp_beta.min(dim=-1).values)
    rng_theta = (lp_theta.max(dim=-1).values - lp_theta.min(dim=-1).values)
    rng_total = (lp_total.max(dim=-1).values - lp_total.min(dim=-1).values)
    logger.info(
        f"Per-view range ratio: total / β = {(rng_total / rng_beta).mean().item():.3f}, "
        f"total / θ = {(rng_total / rng_theta).mean().item():.3f}"
    )
    corr_bt = []
    for b in range(lp_beta.shape[0]):
        for v in range(lp_beta.shape[1]):
            b_ = lp_beta[b, v]
            t_ = lp_theta[b, v]
            c = torch.corrcoef(torch.stack([b_, t_]))[0, 1]
            corr_bt.append(c.item())
    logger.info(f"Per-view corr(log p(β), log p(θ|β)) across samples: mean={np.mean(corr_bt):+.3f} ± {np.std(corr_bt):.3f}")

    logger.info("-" * 78)
    logger.info("Cross-view stage-1: log p(β_i^k | I_j) for i != j  (proposal-under-other-view):")
    B, V, _, S = cross.shape
    off = []
    for i in range(V):
        for j in range(V):
            if i != j:
                off.append(cross[:, i, j])
    off = torch.stack(off, dim=1)  # [B, (V*(V-1)), S]
    summarize(off.reshape(B, -1), "log p(β_i^k | I_{j!=i})")

    # Log-weight range per batch (what the IS softmax sees).
    # log w_i^k = Σ_{j != i} log p(β_i^k | I_j). Pool across all i -> V*S candidates.
    logw_all = []
    for b in range(B):
        pieces = []
        for i in range(V):
            lw = torch.zeros(S)
            for j in range(V):
                if j == i:
                    continue
                lw = lw + cross[b, i, j]
            pieces.append(lw)
        logw_all.append(torch.cat(pieces))
    logw_all = torch.stack(logw_all)  # [B, V*S]
    rng_logw = (logw_all.max(dim=-1).values - logw_all.min(dim=-1).values)
    logger.info(
        f"IS log-weight (Σ_{{j!=i}} log p) range across V*S candidates: "
        f"mean={rng_logw.mean().item():.1f}  median={rng_logw.median().item():.1f}  "
        f"max={rng_logw.max().item():.1f}"
    )
    logger.info("=" * 78)

    # --------- Mode (z=0) vs sample max, and GT residual log-prob ---------
    mode_beta = torch.cat(all_mode_beta, dim=0)    # [B_total, V]
    mode_theta = torch.cat(all_mode_theta, dim=0)
    gt_beta = torch.cat(all_gt_beta, dim=0)
    gt_theta = torch.cat(all_gt_theta, dim=0)

    sample_max_beta = lp_beta.max(dim=-1).values    # [B_total, V]
    sample_max_theta = lp_theta.max(dim=-1).values
    sample_mean_beta = lp_beta.mean(dim=-1)
    sample_mean_theta = lp_theta.mean(dim=-1)

    logger.info("MODE (z=0) vs SAMPLE MAX vs GT residual log-prob (per stage, per view):")
    logger.info("-" * 78)
    logger.info(
        f"  Stage 1 β   mode={mode_beta.mean().item():+.2f} ± {mode_beta.std().item():.2f} | "
        f"sample_max={sample_max_beta.mean().item():+.2f} ± {sample_max_beta.std().item():.2f} | "
        f"sample_mean={sample_mean_beta.mean().item():+.2f} ± {sample_mean_beta.std().item():.2f} | "
        f"gt={gt_beta.mean().item():+.2f} ± {gt_beta.std().item():.2f}"
    )
    logger.info(
        f"  Stage 2 θ|β mode={mode_theta.mean().item():+.2f} ± {mode_theta.std().item():.2f} | "
        f"sample_max={sample_max_theta.mean().item():+.2f} ± {sample_max_theta.std().item():.2f} | "
        f"sample_mean={sample_mean_theta.mean().item():+.2f} ± {sample_mean_theta.std().item():.2f} | "
        f"gt={gt_theta.mean().item():+.2f} ± {gt_theta.std().item():.2f}"
    )
    diff_beta = mode_beta - sample_max_beta
    diff_theta = mode_theta - sample_max_theta
    logger.info(
        f"  Mode - sample_max β: min={diff_beta.min().item():+.3f} max={diff_beta.max().item():+.3f} "
        f"(positive => mode exceeds empirical max ✓)"
    )
    logger.info(
        f"  Mode - sample_max θ: min={diff_theta.min().item():+.3f} max={diff_theta.max().item():+.3f}"
    )
    gt_gap_beta = mode_beta - gt_beta
    gt_gap_theta = mode_theta - gt_theta
    logger.info(
        f"  Mode - GT         β: mean={gt_gap_beta.mean().item():+.2f} ± {gt_gap_beta.std().item():.2f}  "
        f"θ: mean={gt_gap_theta.mean().item():+.2f} ± {gt_gap_theta.std().item():.2f}"
    )
    logger.info("=" * 78)

    # --------- Cross-view JOINT log-prob: does θ cancel β's cross-view spread? ---------
    cross_view_joint(trainer, args.num_views, args.num_samples, args.max_batches, device, args.dataset)


def cross_view_joint(trainer, num_views, num_samples, max_batches, device, dataset_name):
    loader = trainer.multiview_eval_dataloader(
        num_view=num_views, batch_size=1, dataset_name=dataset_name,
    )
    return _cross_view_joint_impl(trainer, loader, num_samples, max_batches, device)


def _cross_view_joint_impl(trainer, loader, num_samples, max_batches, device):
    """Evaluate cross-view JOINT log-prob log p(β_i^k, θ_i^k | I_j) and compare
    to β-only cross-view spread. Tests whether β+θ is tighter than β alone
    across cross-view samples (the setting relevant to IS merging)."""
    from sam_3d_body.models.modules.mhr_utils import (
        scale_indices, convert_pose_cont_to_flow_context,
    )
    nf = trainer.model.nf_head
    beta_dim = nf.beta_dim

    cross_beta_all = []
    cross_theta_all = []
    cross_total_all = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= max_batches:
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
            out = trainer.model(batch, num_samples=num_samples)
            u = out["uncertainty_output"]
            mhr = out["mhr"]

            # Per-view absolute beta samples (combined shape+scale).
            shape_mean = mhr["shape"].unflatten(0, (bs, V))              # [B,V,45]
            scale68_mean = mhr["scale_68D"].unflatten(0, (bs, V))        # [B,V,68]
            mean_beta = torch.cat(
                [shape_mean, scale68_mean[..., scale_indices]], dim=-1
            )                                                             # [B,V,55]
            beta_res = u["samples"][..., :beta_dim].unflatten(0, (bs, V)) # [B,V,S,55]
            abs_beta = mean_beta.unsqueeze(2) + beta_res                  # [B,V,S,55]
            # Absolute shape sample (for θ context) and scale 68-D (with selected slice overwritten).
            shape_samples_abs = abs_beta[..., :nf.num_shape_comps]        # [B,V,S,45]
            scale68_abs = scale68_mean.unsqueeze(2).expand(-1, -1, num_samples, -1).clone()  # [B,V,S,68]
            scale68_abs[..., scale_indices] = abs_beta[..., nf.num_shape_comps:]

            # Absolute θ samples from stage 2 of view i (residual ordering: [3dof,1dof,glob_rot?,cam?]).
            theta_res_i = u["samples"][..., beta_dim:].unflatten(0, (bs, V))  # [B,V,S,theta_dim]
            # For cross-view θ evaluation, we need θ in view j's residual space.
            # Absolute θ is view-indep; view j's mean is μθ_j, so residual_{under j} = θ_abs - μθ_j.
            # But θ residual is already encoded in SO(3) Lie algebra for 3dof/glob_rot and additive for 1dof/cam.
            # Recomputing with SO(3) composition is involved; here we approximate using additive residuals
            # for 1dof+cam and recompose 3dof using SO(3). For simplicity & directional answer, we use
            # the ADDITIVE approximation: θ_abs - μθ_j in the stored residual space. This is an upper
            # bound on how well cross-view log-probs can be, but correlation/range trends should still
            # be informative.
            # Easier route: use view i's own residual (no re-basing). This evaluates the stage-2 flow
            # with view j's CONTEXT but view i's RESIDUALS — equivalent to asking "how likely is the
            # sampled residual under view j's posterior shape/feature conditioning?". It does NOT
            # double-count the mean shift, so spread here reflects flow-sharpness-under-context.
            # We'll use this interpretation as a pragmatic proxy.
            theta_residual_j_space = theta_res_i  # [B,V,S,theta_dim]  (interpreted as above)

            # Build stage-2 context under view j using abs shape from view i's sample.
            pose_raw = mhr["pred_pose_raw"].unflatten(0, (bs, V))  # [B,V, 6+cont]
            flow_ctx = u["flow_context_raw"].unflatten(0, (bs, V)) # [B,V,1024]
            if nf.model_cam:
                pred_cam_mean = mhr["pred_cam"].unflatten(0, (bs, V))  # [B,V,3]
            else:
                pred_cam_mean = None

            S = num_samples
            cross_beta = torch.zeros(bs, V, V, S, device=device)
            cross_theta = torch.zeros(bs, V, V, S, device=device)
            beta_context = u["flow_context_beta"].unflatten(0, (bs, V))

            for b in range(bs):
                # Pre-compute view j's pose-context pieces.
                pose_cont_j = []
                for j in range(V):
                    pose_params_j = convert_pose_cont_to_flow_context(pose_raw[b, j, 6:].unsqueeze(0))
                    pose_cont_j.append(pose_params_j)

                for i in range(V):
                    for j in range(V):
                        # beta: residual under view j using abs_beta_i.
                        res_beta_ij = abs_beta[b, i] - mean_beta[b, j].unsqueeze(0)  # [S,55]
                        ctx_beta_j = beta_context[b, j].unsqueeze(0).expand(S, -1)  # [S,2048]
                        lp_b, _ = nf.flow_beta.log_prob(inputs=res_beta_ij, context=ctx_beta_j)
                        cross_beta[b, i, j] = lp_b

                        # theta: use view j's pose mean & flow context + view i's shape sample.
                        aa_3dofs_j = pose_cont_j[j]["aa_3dofs"].expand(S, -1)
                        params_1dofs_j = pose_cont_j[j]["params_1dofs"].expand(S, -1)
                        ctx_parts = [
                            flow_ctx[b, j].unsqueeze(0).expand(S, -1),
                            shape_samples_abs[b, i],                 # [S,45]
                            scale68_abs[b, i, :, scale_indices],     # [S,10]
                            aa_3dofs_j,
                            params_1dofs_j,
                        ]
                        if nf.model_cam:
                            ctx_parts.append(pred_cam_mean[b, j].unsqueeze(0).expand(S, -1))
                        ctx_theta_j = nf.theta_context_proj(torch.cat(ctx_parts, dim=-1))
                        # Residual: additive proxy θ_i (stored residual) — see comment above.
                        lp_t, _ = nf.flow_theta.log_prob(
                            inputs=theta_residual_j_space[b, i], context=ctx_theta_j
                        )
                        cross_theta[b, i, j] = lp_t

            cross_total = cross_beta + cross_theta
            cross_beta_all.append(cross_beta.cpu())
            cross_theta_all.append(cross_theta.cpu())
            cross_total_all.append(cross_total.cpu())

    cb = torch.cat(cross_beta_all, dim=0)
    ct = torch.cat(cross_theta_all, dim=0)
    cT = torch.cat(cross_total_all, dim=0)

    # Off-diagonal only.
    B, V, _, S = cb.shape
    def pool_off(x):
        pcs = []
        for i in range(V):
            for j in range(V):
                if i != j:
                    pcs.append(x[:, i, j])
        return torch.stack(pcs, dim=1)  # [B, V*(V-1), S]

    cb_off = pool_off(cb).reshape(B, -1)
    ct_off = pool_off(ct).reshape(B, -1)
    cT_off = pool_off(cT).reshape(B, -1)

    logger.info("=" * 78)
    logger.info("CROSS-VIEW (i != j): sample from view i, evaluate under view j")
    logger.info("-" * 78)
    summarize(cb_off, "log p(β_i^k | I_j)                 ")
    summarize(ct_off, "log p(θ_i^k | I_j, β_i^k)          ")
    summarize(cT_off, "log p(β,θ)_i^k under I_j = β + θ   ")
    rng_b = (cb_off.max(dim=-1).values - cb_off.min(dim=-1).values)
    rng_t = (ct_off.max(dim=-1).values - ct_off.min(dim=-1).values)
    rng_T = (cT_off.max(dim=-1).values - cT_off.min(dim=-1).values)
    logger.info(
        f"Cross-view range ratio: total / β = {(rng_T / rng_b).mean().item():.3f}, "
        f"total / θ = {(rng_T / rng_t).mean().item():.3f}"
    )
    corrs = []
    for b in range(B):
        c = torch.corrcoef(torch.stack([cb_off[b], ct_off[b]]))[0, 1]
        corrs.append(c.item())
    logger.info(f"Cross-view corr(log p(β|I_j), log p(θ|I_j,β)): mean={np.mean(corrs):+.3f} ± {np.std(corrs):.3f}")
    logger.info("=" * 78)


if __name__ == "__main__":
    main()
