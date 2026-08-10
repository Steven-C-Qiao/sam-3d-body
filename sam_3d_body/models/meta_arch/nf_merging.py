import math
from typing import Optional, Dict

import torch



def _psis_smooth_log_weights(log_w: torch.Tensor):
    """
    Pareto-Smoothed Importance Sampling (Vehtari et al. 2017).

    Fits a Generalised Pareto Distribution (GPD) to the upper tail of the
    log-weight distribution, then replaces the tail values with smoothed GPD
    quantile estimates.  This reduces the variance of IS without changing the
    point the weights are centred on.

    Args:
        log_w: 1-D tensor of unnormalised log importance weights [N].

    Returns:
        log_w_smooth: Smoothed log-weights, same shape as log_w.  Safe to pass
                      directly to softmax (max-shifted for numerical stability).
        k: Pareto shape diagnostic (ξ).
           k < 0.5  → IS reliable
           0.5–0.7  → marginal; results usable but interpret with care
           k > 0.7  → IS unreliable; switch to merge_params_nf_gaussian
    """
    S = log_w.shape[0]
    log_w = log_w - log_w.max()  # shift; doesn't affect softmax but aids float32

    # Number of tail samples to fit (Vehtari's recommended heuristic)
    M = min(int(math.ceil(0.2 * S)), int(math.ceil(3.0 * math.sqrt(S))))
    M = max(M, 5)  # need enough points for moment matching

    sorted_idx = torch.argsort(log_w)           # ascending
    u = log_w[sorted_idx[-M]].item()            # threshold = min of top-M
    z = log_w[sorted_idx[-M:]] - u             # exceedances ≥ 0, shape [M]

    z_bar = z.mean().item()
    s2 = z.var().item()

    if s2 < 1e-12 or z_bar < 1e-12:
        # Degenerate tail — cannot fit GPD; return unchanged, signal via k=inf
        return log_w, float("inf")

    # GPD parameter estimation via method of moments
    k = 0.5 * (1.0 - z_bar ** 2 / s2)          # shape ξ (= Pareto k diagnostic)
    sigma = z_bar * (1.0 - k)                   # scale σ

    # Evaluate GPD quantile function at plotting positions p_r = (r-0.5)/M
    r = torch.arange(1, M + 1, dtype=log_w.dtype, device=log_w.device)
    p_r = (r - 0.5) / M

    if abs(k) < 1e-6:
        quantiles = u + sigma * (-torch.log1p(-p_r))          # exponential limit
    else:
        quantiles = u + (sigma / k) * (torch.pow(1.0 - p_r, -k) - 1.0)

    # Cap: no smoothed weight should be able to dominate alone (log(S) cap)
    quantiles = quantiles.clamp(max=math.log(S))

    # Write smoothed values back in the same ascending sorted order
    log_w_smooth = log_w.clone()
    log_w_smooth[sorted_idx[-M:]] = quantiles

    return log_w_smooth, k


def merge_params_nf_psis(
    nf_head,
    mhr_out,
    uncertainty_out,
    bs,
    num_views,
    num_samples,
):
    """
    IS merge with Pareto-Smoothed Importance Sampling (PSIS; Vehtari et al. 2017).

    Candidate generation is identical to merge_params_nf_is: for each view i,
    draw samples beta_i^k ~ p(beta|I_i) and compute log-weights
        log w_i^k = Σ_{j≠i} log p(beta_i^k | I_j).

    Before softmax, the upper tail of the pooled log-weight distribution is
    replaced by smoothed GPD quantile estimates via _psis_smooth_log_weights.
    This reduces the variance that causes weight collapse in high dimensions
    while preserving the IS point estimate.

    A Pareto k diagnostic is computed per batch element and printed when k > 0.5.
    If k > 0.7 consistently, consider switching to merge_params_nf_gaussian.
    """
    S = num_samples

    pred_shape = mhr_out["shape"].unflatten(0, (bs, num_views))
    pred_scale68 = mhr_out["scale_68D"].unflatten(0, (bs, num_views))

    beta_log_prob_ref = uncertainty_out["log_prob_beta"].unflatten(0, (bs, num_views))
    shape_samples = uncertainty_out["shape_samples"].unflatten(0, (bs, num_views))
    scale68_samples = uncertainty_out["scale_samples"].unflatten(0, (bs, num_views))
    beta_context = uncertainty_out["flow_context_beta"].unflatten(0, (bs, num_views))

    merged_shape = []
    merged_scale68 = []

    cross_view_logp = torch.full(
        (bs, num_views, num_views, S),
        float("nan"),
        device=shape_samples.device,
        dtype=shape_samples.dtype,
    )
    is_weights = torch.zeros(
        (bs, num_views, S), device=shape_samples.device, dtype=shape_samples.dtype,
    )

    for b in range(bs):
        candidate_beta = []
        candidate_logw = []

        for i in range(num_views):
            beta_i_parts = []
            if nf_head.num_shape_comps > 0:
                beta_i_parts.append(shape_samples[b, i, :, nf_head.shape_indices])
            if nf_head.num_scale_comps > 0:
                beta_i_parts.append(scale68_samples[b, i, :, nf_head.scale_indices])
            beta_i = torch.cat(beta_i_parts, dim=-1)

            logw_i = torch.zeros(S, device=beta_i.device, dtype=beta_i.dtype)
            for j in range(num_views):
                if j == i:
                    continue
                mean_beta_parts = []
                if nf_head.num_shape_comps > 0:
                    mean_beta_parts.append(pred_shape[b, j, nf_head.shape_indices])
                if nf_head.num_scale_comps > 0:
                    mean_beta_parts.append(pred_scale68[b, j, nf_head.scale_indices])
                mean_beta_j = torch.cat(mean_beta_parts, dim=-1)
                residual_j = beta_i - mean_beta_j.unsqueeze(0)
                context_j = beta_context[b, j].unsqueeze(0).expand(S, -1)
                logp_j, _ = nf_head.flow_beta.log_prob(inputs=residual_j, context=context_j)
                cross_view_logp[b, i, j] = logp_j
                logw_i = logw_i + logp_j

            candidate_beta.append(beta_i)
            candidate_logw.append(logw_i)

        candidate_beta = torch.cat(candidate_beta, dim=0)
        candidate_logw = torch.cat(candidate_logw, dim=0)

        # PSIS: smooth the tail before normalising
        candidate_logw_smooth, k = _psis_smooth_log_weights(candidate_logw)
        if k > 0.7:
            print(f"[PSIS] b={b}: k={k:.3f} > 0.7 — IS unreliable; consider merge_params_nf_gaussian")
        elif k > 0.5:
            print(f"[PSIS] b={b}: k={k:.3f} > 0.5 — IS marginal")

        candidate_w = torch.softmax(candidate_logw_smooth, dim=0)
        is_weights[b] = candidate_w.reshape(num_views, S)
        merged_beta = (candidate_w.unsqueeze(-1) * candidate_beta).sum(dim=0)

        shape_merged = pred_shape[b].mean(dim=0).clone()
        if nf_head.num_shape_comps > 0:
            shape_merged[nf_head.shape_indices] = merged_beta[: nf_head.num_shape_comps]
        merged_shape.append(shape_merged)
        scale68_merged = pred_scale68[b].mean(dim=0).clone()
        if nf_head.num_scale_comps > 0:
            scale68_merged[nf_head.scale_indices] = merged_beta[nf_head.num_shape_comps :]
        merged_scale68.append(scale68_merged)

    # Diagonal of cross_view_logp = self-view stage-1 log-prob (no need to recompute).
    diag = torch.arange(num_views, device=cross_view_logp.device)
    cross_view_logp[:, diag, diag, :] = beta_log_prob_ref

    shape_mu_star = torch.stack(merged_shape, dim=0)     # [B, 45]
    scale_mu_star_full = torch.stack(merged_scale68, dim=0)  # [B, 68]

    shape_avg = pred_shape.mean(dim=1)
    scale_avg = pred_scale68.mean(dim=1)

    # ------------- per-view best sample (for visualization) -------------
    best_sample_idx = beta_log_prob_ref.argmax(dim=-1)   # [B, V]
    idx_shape = best_sample_idx.unsqueeze(2).unsqueeze(-1).expand(
        bs, num_views, 1, shape_samples.shape[-1]
    )
    idx_scale = best_sample_idx.unsqueeze(2).unsqueeze(-1).expand(
        bs, num_views, 1, scale68_samples.shape[-1]
    )
    best_shape_per_view = torch.gather(shape_samples, 2, idx_shape).squeeze(2)
    best_scale_per_view_68D = torch.gather(scale68_samples, 2, idx_scale).squeeze(2)

    return {
        "avg_shape": shape_avg,
        "avg_scale": scale_avg,
        "merged_shape": shape_mu_star,
        "merged_scale": scale_mu_star_full,
        "best_logprob_sample_shape": best_shape_per_view,
        "best_logprob_sample_scale_68D": best_scale_per_view_68D,
        "cross_view_log_prob_beta": cross_view_logp,
        "is_weight_beta": is_weights,
    }


def merge_params_nf_tempered(
    nf_head,
    mhr_out,
    uncertainty_out,
    bs,
    num_views,
    num_samples,
    temperature=None,
    batch=None,
    gt_hand_scale_override=True,
):
    """
    IS merge with temperature-scaled log-weights.

    Divides log-weights by `temperature` (default: beta_dim = 55) before
    softmax.  This is equivalent to weighting by p(beta|I)^{1/T} rather than
    p(beta|I), which flattens the weight distribution and prevents collapse in
    high-dimensional spaces.  The per-dimension geometric-mean interpretation
    (T = D) is the most natural choice.

    A temperature of 1.0 reduces to standard IS (merge_params_nf_is).

    When ``gt_hand_scale_override`` is True, the merged scale_68D has its hand
    bone scales (indices 8, 9 + 18..67) replaced with GT (from
    ``batch["model_params"][:, -68:]``) for any hand dim that is not modelled
    by the flow (i.e. not in ``nf_head.scale_indices``). Per-finger dims
    (18..67) are never flow-modelled and so are always overridden; the global
    hand scales (8, 9) are overridden only when not in scale_indices.
    """
    T = temperature if temperature is not None else float(nf_head.beta_dim)
    S = num_samples

    T = 1.0

    if gt_hand_scale_override:
        modelled = set(int(i) for i in nf_head.scale_indices)
        override_indices = [i for i in range(68) if i not in modelled]
        if override_indices:
            gt_scale_68D = batch["model_params"][:, -68:]   # (B*V, 68)
            mhr_out["scale_68D"][:, override_indices] = gt_scale_68D[:, override_indices]
            uncertainty_out["scale_samples"][..., override_indices] = (
                gt_scale_68D[:, override_indices].unsqueeze(1).expand(-1, S, -1)
            )
    else:
        override_indices = []

    pred_shape = mhr_out["shape"].unflatten(0, (bs, num_views))
    pred_scale68 = mhr_out["scale_68D"].unflatten(0, (bs, num_views))

    beta_log_prob_ref = uncertainty_out["log_prob_beta"].unflatten(0, (bs, num_views))
    shape_samples = uncertainty_out["shape_samples"].unflatten(0, (bs, num_views))
    scale68_samples = uncertainty_out["scale_samples"].unflatten(0, (bs, num_views))
    beta_context = uncertainty_out["flow_context_beta"].unflatten(0, (bs, num_views))

    merged_shape = []
    merged_scale68 = []

    cross_view_logp = torch.full(
        (bs, num_views, num_views, S),
        float("nan"),
        device=shape_samples.device,
        dtype=shape_samples.dtype,
    )
    is_weights = torch.zeros(
        (bs, num_views, S), device=shape_samples.device, dtype=shape_samples.dtype,
    )

    for b in range(bs):
        candidate_beta = []
        candidate_logw = []

        for i in range(num_views):
            beta_i_parts = []
            if nf_head.num_shape_comps > 0:
                beta_i_parts.append(shape_samples[b, i, :, nf_head.shape_indices])
            if nf_head.num_scale_comps > 0:
                beta_i_parts.append(scale68_samples[b, i, :, nf_head.scale_indices])
            beta_i = torch.cat(beta_i_parts, dim=-1)

            logw_i = torch.zeros(S, device=beta_i.device, dtype=beta_i.dtype)
            for j in range(num_views):
                if j == i:
                    continue
                mean_beta_parts = []
                if nf_head.num_shape_comps > 0:
                    mean_beta_parts.append(pred_shape[b, j, nf_head.shape_indices])
                if nf_head.num_scale_comps > 0:
                    mean_beta_parts.append(pred_scale68[b, j, nf_head.scale_indices])
                mean_beta_j = torch.cat(mean_beta_parts, dim=-1)
                residual_j = beta_i - mean_beta_j.unsqueeze(0)
                context_j = beta_context[b, j].unsqueeze(0).expand(S, -1)
                logp_j, _ = nf_head.flow_beta.log_prob(inputs=residual_j, context=context_j)
                cross_view_logp[b, i, j] = logp_j

                logw_i = logw_i + logp_j

            candidate_beta.append(beta_i)
            candidate_logw.append(logw_i / T)  # temperature scaling

        candidate_beta = torch.cat(candidate_beta, dim=0)
        candidate_logw = torch.cat(candidate_logw, dim=0)

        candidate_w = torch.softmax(candidate_logw, dim=0)
        is_weights[b] = candidate_w.reshape(num_views, S)
        merged_beta = (candidate_w.unsqueeze(-1) * candidate_beta).sum(dim=0)


        shape_merged = pred_shape[b].mean(dim=0).clone()
        if nf_head.num_shape_comps > 0:
            shape_merged[nf_head.shape_indices] = merged_beta[: nf_head.num_shape_comps]
        merged_shape.append(shape_merged)
        scale68_merged = pred_scale68[b].mean(dim=0).clone()
        if nf_head.num_scale_comps > 0:
            scale68_merged[nf_head.scale_indices] = merged_beta[nf_head.num_shape_comps :]
        merged_scale68.append(scale68_merged)

    diag = torch.arange(num_views, device=cross_view_logp.device)
    cross_view_logp[:, diag, diag, :] = beta_log_prob_ref

    shape_mu_star = torch.stack(merged_shape, dim=0)
    scale_mu_star_full = torch.stack(merged_scale68, dim=0)

    shape_avg = pred_shape.mean(dim=1)
    scale_avg = pred_scale68.mean(dim=1)

    # ------------- per-view best sample (for visualization) -------------
    best_sample_idx = beta_log_prob_ref.argmax(dim=-1)
    idx_shape = best_sample_idx.unsqueeze(2).unsqueeze(-1).expand(
        bs, num_views, 1, shape_samples.shape[-1]
    )
    idx_scale = best_sample_idx.unsqueeze(2).unsqueeze(-1).expand(
        bs, num_views, 1, scale68_samples.shape[-1]
    )
    best_shape_per_view = torch.gather(shape_samples, 2, idx_shape).squeeze(2)
    best_scale_per_view_68D = torch.gather(scale68_samples, 2, idx_scale).squeeze(2)

    return {
        "avg_shape": shape_avg,
        "avg_scale": scale_avg,
        "merged_shape": shape_mu_star,
        "merged_scale": scale_mu_star_full,
        "best_logprob_sample_shape": best_shape_per_view,
        "best_logprob_sample_scale_68D": best_scale_per_view_68D,
        "cross_view_log_prob_beta": cross_view_logp,
        "is_weight_beta": is_weights,
    }


def merge_params_nf_is(
    nf_head,
    mhr_out,
    uncertainty_out,
    bs,
    num_views,
    num_samples,
):
    """
    Importance-sampling merge of multiview shape/scale predictions using NF stage-1 likelihoods.

    For proposal view i with samples beta_i^k ~ p(beta|I_i), weight each sample by
    product_{j != i} p(beta_i^k | I_j), and pool over all proposals.

    NOTE: This approach suffers from weight collapse in practice. In the 55D
    shape+scale space, NF log-probs vary by 100+ nats across samples, so
    softmax degenerates to one-hot regardless of the number of samples.
    See merge_params_nf_gaussian for a robust alternative.
    """
    S = num_samples

    pred_shape = mhr_out["shape"].unflatten(0, (bs, num_views))
    pred_scale68 = mhr_out["scale_68D"].unflatten(0, (bs, num_views))
    pred_scale_params = mhr_out["scale"].unflatten(0, (bs, num_views))

    beta_context = uncertainty_out["flow_context_beta"].unflatten(0, (bs, num_views))
    beta_log_prob_ref = uncertainty_out["log_prob_beta"].unflatten(0, (bs, num_views))
    flow_samples = uncertainty_out["samples"].unflatten(0, (bs, num_views))
    beta_residual_samples = flow_samples[..., : nf_head.beta_dim]
    shape_samples = uncertainty_out["shape_samples"].unflatten(0, (bs, num_views))
    scale68_samples = uncertainty_out["scale_samples"].unflatten(0, (bs, num_views))

    merged_shape = []
    merged_scale68 = []

    cross_view_logp = torch.full(
        (bs, num_views, num_views, S),
        float("nan"),
        device=shape_samples.device,
        dtype=shape_samples.dtype,
    )
    is_weights = torch.zeros(
        (bs, num_views, S), device=shape_samples.device, dtype=shape_samples.dtype,
    )

    for b in range(bs):
        candidate_beta = []
        candidate_logw = []

        for i in range(num_views):
            # Proposal samples from view i: beta = [selected_shape, selected_scale].
            beta_i_parts = []
            if nf_head.num_shape_comps > 0:
                beta_i_parts.append(shape_samples[b, i, :, nf_head.shape_indices])
            if nf_head.num_scale_comps > 0:
                beta_i_parts.append(scale68_samples[b, i, :, nf_head.scale_indices])
            beta_i = torch.cat(beta_i_parts, dim=-1)

            # # Debug self-consistency:
            # # Use the exact sampled stage-1 residuals from uncertainty_out["samples"].
            # # This avoids reconstruction mismatch from absolute samples and means.
            # residual_i = beta_residual_samples[b, i]  # [S, 55]
            # context_i = beta_context[b, i].unsqueeze(0).expand(S, -1)  # [S, 2048]
            # logp_i_recomputed, _ = nf_head.flow_beta.log_prob(
            #     inputs=residual_i, context=context_i
            # )
            # logp_i_ref = beta_log_prob_ref[b, i]  # [S]
            # max_abs_diff = (logp_i_recomputed - logp_i_ref).abs().max()
            # if max_abs_diff > 1e-5:
            #     diff = logp_i_recomputed - logp_i_ref
            #     print(
            #         "NF stage-1 log-prob mismatch for self-view. "
            #         f"b={b}, i={i}, S={S}, "
            #         f"max_abs_diff={float(max_abs_diff):.6e}, "
            #         f"mean_abs_diff={float(diff.abs().mean()):.6e}, "
            #         f"recomputed[min,max]=({float(logp_i_recomputed.min()):.6e}, {float(logp_i_recomputed.max()):.6e}), "
            #         f"ref[min,max]=({float(logp_i_ref.min()):.6e}, {float(logp_i_ref.max()):.6e})"
            #     )
            #     import ipdb; ipdb.set_trace()

            # Eq. 18 generalised to multi-view: w ~ Π_{j != i} p(beta | I_j).
            logw_i = torch.zeros(S, device=beta_i.device, dtype=beta_i.dtype)
            for j in range(num_views):
                if j == i:
                    continue
                mean_beta_parts = []
                if nf_head.num_shape_comps > 0:
                    mean_beta_parts.append(pred_shape[b, j, nf_head.shape_indices])
                if nf_head.num_scale_comps > 0:
                    mean_beta_parts.append(pred_scale68[b, j, nf_head.scale_indices])
                mean_beta_j = torch.cat(mean_beta_parts, dim=-1)
                residual_j = beta_i - mean_beta_j.unsqueeze(0)
                context_j = beta_context[b, j].unsqueeze(0).expand(S, -1)  # [S, 2048]
                logp_j, _ = nf_head.flow_beta.log_prob(
                    inputs=residual_j, context=context_j
                )
                cross_view_logp[b, i, j] = logp_j

                logw_i = logw_i + logp_j

            candidate_beta.append(beta_i)
            candidate_logw.append(logw_i)


        candidate_beta = torch.cat(candidate_beta, dim=0)  # [V*S, 55]
        candidate_logw = torch.cat(candidate_logw, dim=0)  # [V*S]

        candidate_w = torch.softmax(candidate_logw, dim=0)  # normalized importance weights
        is_weights[b] = candidate_w.reshape(num_views, S)

        merged_beta = (candidate_w.unsqueeze(-1) * candidate_beta).sum(dim=0)
        shape_merged = pred_shape[b].mean(dim=0).clone()
        if nf_head.num_shape_comps > 0:
            shape_merged[nf_head.shape_indices] = merged_beta[: nf_head.num_shape_comps]
        merged_shape.append(shape_merged)

        scale68_merged = pred_scale68[b].mean(dim=0).clone()
        if nf_head.num_scale_comps > 0:
            scale68_merged[nf_head.scale_indices] = merged_beta[nf_head.num_shape_comps :]
        merged_scale68.append(scale68_merged)

    diag = torch.arange(num_views, device=cross_view_logp.device)
    cross_view_logp[:, diag, diag, :] = beta_log_prob_ref

    shape_mu_star = torch.stack(merged_shape, dim=0)  # [B, 45]
    scale_mu_star_full = torch.stack(merged_scale68, dim=0)  # [B, 68]

    shape_avg = pred_shape.mean(dim=1)
    scale_avg = pred_scale68.mean(dim=1)

    # ------------- per-view best sample (for visualization) -------------
    best_sample_idx = beta_log_prob_ref.argmax(dim=-1)  # [B, N_view]
    idx_shape = best_sample_idx.unsqueeze(2).unsqueeze(-1).expand(
        bs, num_views, 1, shape_samples.shape[-1]
    )  # [B, N_view, 1, 45]
    idx_scale = best_sample_idx.unsqueeze(2).unsqueeze(-1).expand(
        bs, num_views, 1, scale68_samples.shape[-1]
    )  # [B, N_view, 1, 68]
    best_shape_per_view = torch.gather(shape_samples, 2, idx_shape).squeeze(2)  # [B, N_view, 45]
    best_scale_per_view_68D = (
        torch.gather(scale68_samples, 2, idx_scale).squeeze(2)
    )  # [B, N_view, 68]

    return {
        "avg_shape": shape_avg,
        "avg_scale": scale_avg,
        "merged_shape": shape_mu_star,
        "merged_scale": scale_mu_star_full,
        "best_logprob_sample_shape": best_shape_per_view,
        "best_logprob_sample_scale_68D": best_scale_per_view_68D,
        "cross_view_log_prob_beta": cross_view_logp,
        "is_weight_beta": is_weights,
    }


def merge_params_nf_gaussian(
    nf_head,
    mhr_out,
    uncertainty_out,
    bs,
    num_views,
    num_samples,
):
    """
    Multi-view shape/scale fusion via precision-weighted Gaussian product.

    For each view i, the NF samples give an empirical estimate of per-dimension
    uncertainty (sample variance σ²_ij along each shape/scale dim j).  The fused
    estimate is the precision-weighted mean:

        μ*_j = (Σ_i σ⁻²_ij · μ_ij) / (Σ_i σ⁻²_ij)

    Views with low variance in a dimension (high certainty about it) contribute
    more to that dimension's merged value.  This captures the intuition that a
    frontal image is certain about width but uncertain about depth-related shape
    parameters, while a side image is complementary.

    This is exact for Gaussian posteriors and avoids the IS weight collapse that
    occurs in the 55D shape+scale space (see merge_params_nf_is).
    """
    S = num_samples

    pred_shape = mhr_out["shape"].unflatten(0, (bs, num_views))           # [B, V, 45]
    pred_scale68 = mhr_out["scale_68D"].unflatten(0, (bs, num_views))     # [B, V, 68]

    beta_log_prob_ref = uncertainty_out["log_prob_beta"].unflatten(0, (bs, num_views))  # [B, V, S]
    shape_samples = uncertainty_out["shape_samples"].unflatten(0, (bs, num_views))             # [B, V, S, 45]
    scale68_samples = uncertainty_out["scale_samples"].unflatten(0, (bs, num_views))           # [B, V, S, 68]

    # ---- Precision-weighted Gaussian product ----
    shape_mu = shape_samples.mean(dim=2)                              # [B, V, 45]
    shape_var = shape_samples.var(dim=2)                              # [B, V, 45]

    scale_selected_samples = scale68_samples[..., nf_head.scale_indices]     # [B, V, S, 10]
    scale_mu = scale_selected_samples.mean(dim=2)                    # [B, V, 10]
    scale_var = scale_selected_samples.var(dim=2)                    # [B, V, 10]

    shape_prec = 1.0 / (shape_var + 1e-6)                           # [B, V, 45]
    scale_prec = 1.0 / (scale_var + 1e-6)                           # [B, V, 10]

    shape_mu_star = (shape_prec * shape_mu).sum(dim=1) / shape_prec.sum(dim=1)   # [B, 45]
    scale_mu_star = (scale_prec * scale_mu).sum(dim=1) / scale_prec.sum(dim=1)  # [B, 10]

    scale_mu_star_full = pred_scale68.mean(dim=1).clone()            # [B, 68]
    scale_mu_star_full[:, nf_head.scale_indices] = scale_mu_star

    shape_avg = pred_shape.mean(dim=1)
    scale_avg = pred_scale68.mean(dim=1)

    # ------------- per-view best sample (for visualization) -------------
    best_sample_idx = beta_log_prob_ref.argmax(dim=-1)  # [B, N_view]
    idx_shape = best_sample_idx.unsqueeze(2).unsqueeze(-1).expand(
        bs, num_views, 1, shape_samples.shape[-1]
    )  # [B, N_view, 1, 45]
    idx_scale = best_sample_idx.unsqueeze(2).unsqueeze(-1).expand(
        bs, num_views, 1, scale68_samples.shape[-1]
    )  # [B, N_view, 1, 68]
    best_shape_per_view = torch.gather(shape_samples, 2, idx_shape).squeeze(2)  # [B, N_view, 45]
    best_scale_per_view_68D = (
        torch.gather(scale68_samples, 2, idx_scale).squeeze(2)
    )  # [B, N_view, 68]

    return {
        "avg_shape": shape_avg,
        "avg_scale": scale_avg,
        "merged_shape": shape_mu_star,
        "merged_scale": scale_mu_star_full,
        "best_logprob_sample_shape": best_shape_per_view,
        "best_logprob_sample_scale_68D": best_scale_per_view_68D,
    }


def _langevin_log_pi_and_grad(state, view_means, view_context, nf_head):
    """
    Evaluate log π(β) = Σ_j log p(β - μ_j | c_j) and its gradient w.r.t. β.

    state:        [B, V_chain, D]
    view_means:   [B, V_view,  D]
    view_context: [B, V_view,  C]
    Returns:
        log_pi: [B, V_chain]
        grad:   [B, V_chain, D]
    """
    B, V_c, D = state.shape
    V_v = view_means.shape[1]
    state_req = state.detach().clone().requires_grad_(True)

    # Cross: every chain's state evaluated under every view.
    residual = state_req.unsqueeze(2) - view_means.unsqueeze(1)           # [B, V_c, V_v, D]
    context_expanded = view_context.unsqueeze(1).expand(B, V_c, V_v, -1)  # [B, V_c, V_v, C]

    logp_flat, _ = nf_head.flow_beta.log_prob(
        inputs=residual.reshape(B * V_c * V_v, D),
        context=context_expanded.reshape(B * V_c * V_v, -1),
    )
    log_pi = logp_flat.view(B, V_c, V_v).sum(dim=2)                       # [B, V_c]

    (grad,) = torch.autograd.grad(
        log_pi.sum(), state_req, create_graph=False, retain_graph=False
    )
    grad = grad.clamp(-100.0, 100.0)
    return log_pi.detach(), grad.detach()


def merge_params_nf_langevin(
    nf_head,
    mhr_out,
    uncertainty_out,
    bs,
    num_views,
    num_samples,
    num_steps: int = 200,
    burn_in_frac: float = 0.5,
    tail_frac: float = 0.4,
    step_size: float = 0.01,
    adapt_step: bool = True,
    target_accept: float = 0.574,
    variant: str = "mala",
    variance_floor_frac: float = 0.1,
    init_strategy: str = "gaussian_warm",  # "gaussian_warm" | "view_mean"
    init_noise_scale: float = 0.5,
    verbose: bool = False,
):
    """
    Multi-view shape/scale fusion via V-chain preconditioned MALA.

    Target posterior (flat prior, views conditionally independent given β):
        log π(β) = Σ_j log p(β - μ_j | c_j)       (via nf_head.flow_beta)

    Runs V parallel chains per batch element, each initialised at a view mean
    μ_i. Preconditioner M_diag is the Gaussian-product posterior variance
    (per-dim). Step size adapts via Robbins-Monro during burn-in to the 0.574
    optimal-MALA target. Final estimate is the mean over the last tail_frac of
    steps, pooled across chains.

    Falls back to merge_params_nf_gaussian when num_views <= 1.
    """
    if num_views <= 1:
        return merge_params_nf_gaussian(
            nf_head, mhr_out, uncertainty_out, bs, num_views, num_samples
        )

    assert not nf_head.training, "NFARHead must be in eval mode for Langevin merge."
    assert variant in ("mala", "ula"), f"Unknown variant {variant!r}"

    D = nf_head.beta_dim
    num_shape = nf_head.num_shape_comps
    num_scale = D - num_shape

    # ---- Setup ----
    pred_shape = mhr_out["shape"].unflatten(0, (bs, num_views))                 # [B, V, 45]
    pred_scale68 = mhr_out["scale_68D"].unflatten(0, (bs, num_views))           # [B, V, 68]
    view_means = torch.cat(
        [pred_shape, pred_scale68[..., nf_head.scale_indices]], dim=-1
    ).detach()                                                                   # [B, V, 55]
    view_context = (
        uncertainty_out["flow_context_beta"]
        .unflatten(0, (bs, num_views))
        .detach()
    )                                                                            # [B, V, C]

    shape_samples = uncertainty_out["shape_samples"].unflatten(0, (bs, num_views))
    scale68_samples = uncertainty_out["scale_samples"].unflatten(0, (bs, num_views))
    per_view_samples = torch.cat(
        [shape_samples, scale68_samples[..., nf_head.scale_indices]], dim=-1
    )                                                                            # [B, V, S, 55]
    var_j = per_view_samples.var(dim=2)                                          # [B, V, 55]

    # Variance floor from NF-trained per-dim GT std (guards against collapsed dims).
    device = var_j.device
    dtype = var_j.dtype
    shape_std = getattr(nf_head, "_shape_perturb_std", None)
    scale_std = getattr(nf_head, "_scale_perturb_std", None)
    if shape_std is not None:
        floor_shape = (shape_std.to(device=device, dtype=dtype) * variance_floor_frac) ** 2
    else:
        floor_shape = torch.full((num_shape,), 1e-4, device=device, dtype=dtype)
    if scale_std is not None:
        floor_scale = (scale_std.to(device=device, dtype=dtype) * variance_floor_frac) ** 2
    else:
        floor_scale = torch.full((num_scale,), 1e-4, device=device, dtype=dtype)
    floor = torch.cat([floor_shape, floor_scale]).view(1, 1, D)                 # [1, 1, 55]
    var_j = torch.clamp(var_j, min=floor)

    precision_j = 1.0 / var_j                                                    # [B, V, 55]
    M_diag = 1.0 / precision_j.sum(dim=1)                                        # [B, 55]

    # Freeze flow params so autograd doesn't build a graph into them.
    flow_params = list(nf_head.flow_beta.parameters())
    prev_requires = [p.requires_grad for p in flow_params]
    for p in flow_params:
        p.requires_grad_(False)

    try:
        # ---- V-chain MALA ----
        if init_strategy == "gaussian_warm":
            # All V chains init at precision-weighted Gaussian merge estimate,
            # perturbed per-chain by init_noise_scale * √M_diag so chains explore
            # different directions around the mode.
            gauss_mean = (precision_j * per_view_samples.mean(dim=2)).sum(dim=1) / precision_j.sum(dim=1)
            # gauss_mean: [B, 55] — joint MAP under Gaussian approx.
            noise = torch.randn_like(view_means) * M_diag.unsqueeze(1).sqrt() * init_noise_scale
            state = gauss_mean.unsqueeze(1) + noise                              # [B, V, 55]
        elif init_strategy == "view_mean":
            state = view_means.clone()                                           # [B, V, 55]
        else:
            raise ValueError(f"Unknown init_strategy {init_strategy!r}")
        log_eps = torch.full(
            (bs, num_views), math.log(step_size), device=device, dtype=dtype
        )
        ema_accept = torch.full(
            (bs, num_views), target_accept, device=device, dtype=dtype
        )

        tail_start = int(num_steps * (1.0 - tail_frac))
        burn_in_steps = int(num_steps * burn_in_frac)
        trajectory = []
        accept_history = []

        with torch.enable_grad():
            for t in range(num_steps):
                eps = log_eps.exp().unsqueeze(-1)                                # [B, V, 1]
                M_bcast = M_diag.unsqueeze(1)                                    # [B, 1, 55]
                var_q = eps * M_bcast                                            # [B, V, 55]

                log_pi_cur, grad_cur = _langevin_log_pi_and_grad(
                    state, view_means, view_context, nf_head
                )

                drift = 0.5 * var_q * grad_cur
                noise = torch.randn_like(state)
                proposal = state + drift + var_q.sqrt() * noise

                if variant == "mala":
                    log_pi_prop, grad_prop = _langevin_log_pi_and_grad(
                        proposal, view_means, view_context, nf_head
                    )
                    drift_back = 0.5 * var_q * grad_prop

                    var_q_safe = var_q + 1e-8
                    log_q_fwd = -0.5 * ((proposal - state - drift) ** 2 / var_q_safe).sum(-1)
                    log_q_bwd = -0.5 * ((state - proposal - drift_back) ** 2 / var_q_safe).sum(-1)

                    log_alpha = (log_pi_prop - log_pi_cur) + (log_q_bwd - log_q_fwd)
                    u = torch.rand_like(log_alpha).log()
                    accept = u < log_alpha                                       # [B, V]

                    state = torch.where(accept.unsqueeze(-1), proposal, state)

                    if adapt_step and t < burn_in_steps:
                        # Faster adaptation: effective LR = 5/(t+10). Over ~100 steps this
                        # shifts log_eps by up to ~12 log-units, enough to find the right
                        # eps magnitude from any initial guess.
                        gamma = 5.0 / (t + 10.0)
                        ema_accept = 0.9 * ema_accept + 0.1 * accept.to(dtype)
                        log_eps = log_eps + gamma * (ema_accept - target_accept)

                    accept_history.append(accept.to(dtype).mean().item())
                else:  # ULA
                    state = proposal
                    accept_history.append(1.0)

                state = state.detach()

                if t >= tail_start:
                    trajectory.append(state.clone())

        tail = torch.stack(trajectory, dim=0)                                    # [L, B, V, 55]
        merged_beta = tail.mean(dim=(0, 2))                                       # [B, 55]

        if verbose:
            mean_accept = sum(accept_history) / max(1, len(accept_history))
            post_burn = accept_history[burn_in_steps:] or [float("nan")]
            post_burn_accept = sum(post_burn) / len(post_burn)
            print(
                f"[Langevin/{variant}] T={num_steps} burn_in={burn_in_steps} "
                f"mean_accept={mean_accept:.3f} post_burn_accept={post_burn_accept:.3f} "
                f"final_eps={math.exp(log_eps.mean().item()):.4f}"
            )

    finally:
        for p, prev in zip(flow_params, prev_requires):
            p.requires_grad_(prev)

    # ---- Assemble outputs (match existing merge API) ----
    shape_mu_star = merged_beta[:, :num_shape]                                   # [B, 45]
    scale_mu_star_full = pred_scale68.mean(dim=1).clone()                         # [B, 68]
    scale_mu_star_full[:, nf_head.scale_indices] = merged_beta[:, num_shape:]

    shape_avg = pred_shape.mean(dim=1)
    scale_avg = pred_scale68.mean(dim=1)

    beta_log_prob_ref = uncertainty_out["log_prob_beta"].unflatten(0, (bs, num_views))
    best_sample_idx = beta_log_prob_ref.argmax(dim=-1)
    idx_shape = best_sample_idx.unsqueeze(2).unsqueeze(-1).expand(
        bs, num_views, 1, shape_samples.shape[-1]
    )
    idx_scale = best_sample_idx.unsqueeze(2).unsqueeze(-1).expand(
        bs, num_views, 1, scale68_samples.shape[-1]
    )
    best_shape_per_view = torch.gather(shape_samples, 2, idx_shape).squeeze(2)
    best_scale_per_view_68D = torch.gather(scale68_samples, 2, idx_scale).squeeze(2)

    return {
        "avg_shape": shape_avg,
        "avg_scale": scale_avg,
        "merged_shape": shape_mu_star,
        "merged_scale": scale_mu_star_full,
        "best_logprob_sample_shape": best_shape_per_view,
        "best_logprob_sample_scale_68D": best_scale_per_view_68D,
    }


_MERGE_METHODS = ("psis", "tempered", "is", "gaussian", "langevin")


def merge_params_nf(
    nf_head,
    mhr_out,
    uncertainty_out,
    bs,
    num_views,
    num_samples,
    method: str = "psis",
    langevin_kwargs: Optional[Dict] = None,
    batch: Optional[Dict] = None,
    gt_hand_scale_override: bool = True,
):
    """
    Multi-view shape/scale fusion dispatcher.

    Args:
        method: One of:
            "psis"     — Pareto-Smoothed IS (default; self-diagnosing, robust)
            "tempered" — Temperature-scaled IS (T = beta_dim; simple, stable)
            "gaussian" — Precision-weighted Gaussian product (fastest, no NF calls)
            "is"       — Raw IS (reference; collapses in practice for D=55)
            "langevin" — V-chain MALA over the joint NF posterior
        langevin_kwargs: Optional kwargs forwarded to merge_params_nf_langevin.
        batch / gt_hand_scale_override: forwarded to ``merge_params_nf_tempered``;
            ignored by other methods.
    """
    if method == "psis":
        return merge_params_nf_psis(nf_head, mhr_out, uncertainty_out, bs, num_views, num_samples)
    elif method == "tempered":
        return merge_params_nf_tempered(
            nf_head, mhr_out, uncertainty_out, bs, num_views, num_samples,
            batch=batch, gt_hand_scale_override=gt_hand_scale_override,
        )
    elif method == "gaussian":
        return merge_params_nf_gaussian(nf_head, mhr_out, uncertainty_out, bs, num_views, num_samples)
    elif method == "is":
        return merge_params_nf_is(nf_head, mhr_out, uncertainty_out, bs, num_views, num_samples)
    elif method == "langevin":
        return merge_params_nf_langevin(
            nf_head, mhr_out, uncertainty_out, bs, num_views, num_samples,
            **(langevin_kwargs or {}),
        )
    else:
        raise ValueError(f"Unknown merge method {method!r}. Choose from {_MERGE_METHODS}.")


@torch.no_grad()
def resample_cam_for_merged_shape(
    model,
    mhr_out,
    uncertainty_out,
    param_dict,
    batch,
    bs,
    num_views,
    num_cam_samples: int = 32,
):
    """Draw stage-2 NF samples per view conditioned on the *merged* shape/scale,
    average them, and return the resulting per-view ``pred_cam_t`` so
    merged-mesh reprojection is consistent with the merged body size.

    Stage 2 is shape-aware: it models ``p(Δθ, Δcam | c, μθ, μcam, β)``, so
    feeding it the merged β (instead of the per-view β sample) yields a camera
    translation matched to the merged mesh. A single NF sample is noisy, so
    we draw ``num_cam_samples`` and average the resulting ``pred_cam`` before
    the nonlinear conversion to ``pred_cam_t``.

    Args:
        model:            SAM3DBody model (needs ``.nf_head`` and ``.head_camera``).
        mhr_out:          per-view MHR mean predictions (shape_params[B*V, ...]).
        uncertainty_out:  NF output dict — requires ``flow_context_raw`` (B*V, 1024).
        param_dict:       output of ``merge_params_nf`` — requires ``merged_shape``
                          (B, 45) and ``merged_scale`` (B, 68).
        batch:            multi-view-flattened batch (shape keys ``(B*V, 1, …)``).
        num_cam_samples:  number of stage-2 camera samples to average.

    Returns:
        merged_pred_cam_t:  (B*V, 3) camera translation in full-image coordinates.
    """
    nf_head = model.nf_head
    head_camera = model.head_camera

    N = int(num_cam_samples)
    merged_shape = param_dict["merged_shape"].repeat_interleave(num_views, dim=0)
    merged_scale = param_dict["merged_scale"].repeat_interleave(num_views, dim=0)
    # Broadcast to N samples along the stage-2 sample dim.
    merged_shape = merged_shape.unsqueeze(1).expand(-1, N, -1).contiguous()
    merged_scale = merged_scale.unsqueeze(1).expand(-1, N, -1).contiguous()

    flow_context_raw = uncertainty_out["flow_context_raw"]  # (B*V, 1024)

    stage2 = nf_head.sample_theta_given_beta(
        flow_context=flow_context_raw,
        mean_pred=mhr_out,
        shape_samples=merged_shape,
        scale_samples_68D=merged_scale,
        gt_height=batch.get("gt_height") if nf_head.height_condition else None,
    )
    cam_samples = stage2["cam_samples"]  # (B*V, N, 3) if model_cam else None
    if cam_samples is None:
        # No camera head modelled — fall back to mean camera translation.
        return mhr_out["pred_cam_t"]

    # Average pred_cam across stage-2 samples. Averaging in pred_cam space
    # (equivalently, in residual space) is more principled than averaging the
    # non-linearly-derived ``pred_cam_t``.
    merged_pred_cam = cam_samples.mean(dim=1)  # (B*V, 3)

    bbox_center = batch["bbox_center"][:, 0]            # (B*V, 2)
    bbox_size = batch["bbox_scale"][:, 0, 0]            # (B*V,)
    ori_img_size = batch["ori_img_size"][:, 0]          # (B*V, 2)
    cam_int = batch["cam_int"]                          # (B*V, 3, 3)

    zeros_pts = torch.zeros(
        merged_pred_cam.shape[0], 1, 3,
        device=merged_pred_cam.device, dtype=merged_pred_cam.dtype,
    )
    proj = head_camera.perspective_projection(
        zeros_pts,
        merged_pred_cam,
        bbox_center,
        bbox_size,
        ori_img_size,
        cam_int,
        use_intrin_center=model.cfg.MODEL.DECODER.get("USE_INTRIN_CENTER", False),
    )
    return proj["pred_cam_t"]  # (B*V, 3)


def get_mhr_outputs(
    mhr_head,
    batch,
    mhr_out,
    param_dict,
    bs,
    num_views,
    uncertainty_out=None,
    nf_head=None,
):
    ret = {}

    # ------------- β-space tensors for L2-in-parameter-space metrics -------------
    # Authoritative reference for what merging is actually optimising for: the
    # posterior mean β. Vertex-space metrics like PVE-T-SC quotient out global
    # scale and translation, hiding fusion gains in those directions.
    if nf_head is not None:
        scale_indices = nf_head.scale_indices

        per_view_beta = torch.cat(
            [mhr_out["shape"], mhr_out["scale_68D"][..., scale_indices]], dim=-1
        )                                                                         # (B*V, 55)
        gt_shape = batch["shape_params"]
        gt_scale_68D = batch["model_params"][:, -68:]
        gt_beta = torch.cat([gt_shape, gt_scale_68D[..., scale_indices]], dim=-1)  # (B*V, 55)

        avg_beta_per_subj = torch.cat(
            [param_dict["avg_shape"], param_dict["avg_scale"][..., scale_indices]], dim=-1
        )                                                                         # (B, 55)
        avg_beta = avg_beta_per_subj.repeat_interleave(num_views, dim=0)          # (B*V, 55)

        ret["per_view_beta"] = per_view_beta
        ret["gt_beta"] = gt_beta
        ret["avg_beta"] = avg_beta

        if "merged_shape" in param_dict and "merged_scale" in param_dict:
            merged_beta_per_subj = torch.cat(
                [param_dict["merged_shape"], param_dict["merged_scale"][..., scale_indices]], dim=-1
            )                                                                     # (B, 55)
            ret["merged_beta"] = merged_beta_per_subj.repeat_interleave(num_views, dim=0)

        if uncertainty_out is not None:
            shape_s = uncertainty_out["shape_samples"]                            # (B*V, S, 45)
            scale_s = uncertainty_out["scale_samples"]                            # (B*V, S, 68)
            ret["sample_beta"] = torch.cat(
                [shape_s, scale_s[..., scale_indices]], dim=-1
            )                                                                     # (B*V, S, 55)
            ret["sample_param_avg_beta"] = torch.cat(
                [shape_s.mean(dim=1), scale_s.mean(dim=1)[..., scale_indices]], dim=-1
            )                                                                     # (B*V, 55)

        shape_std = getattr(nf_head, "_shape_perturb_std", None)
        scale_std = getattr(nf_head, "_scale_perturb_std", None)
        if shape_std is not None and scale_std is not None:
            ret["beta_perturb_std"] = torch.cat(
                [shape_std.to(per_view_beta.device, per_view_beta.dtype),
                 scale_std.to(per_view_beta.device, per_view_beta.dtype)],
                dim=0,
            )                                                                     # (55,)


    mhr_zero_inputs = {
        "global_trans": torch.zeros_like(mhr_out["global_rot"]),
        "global_rot": torch.zeros_like(mhr_out["global_rot"]),
        "body_pose_params": torch.zeros_like(mhr_out["body_pose"]),
        "hand_pose_params": torch.zeros_like(mhr_out["hand"]),
        "expr_params": torch.zeros_like(mhr_out["face"]),
    }
    mhr_output_config = {
        "return_keypoints": True,
        "return_joint_coords": True,
        "return_model_params": True,
        "return_joint_rotations": True,
        "do_pcblend": True,
    }

    
    # ------------- average pred -------------
    # B, C -> B * num_views, C
    avg_shape = param_dict["avg_shape"].repeat_interleave(num_views, dim=0)
    avg_scale = param_dict["avg_scale"].repeat_interleave(num_views, dim=0)

    # posed 
    avg_mhr_output = mhr_head.mhr_forward(
        shape_params=avg_shape,
        scale_offsets=avg_scale,
        global_trans=torch.zeros_like(mhr_out["global_rot"]),
        global_rot=mhr_out["global_rot"],
        body_pose_params=mhr_out["body_pose"],
        hand_pose_params=mhr_out["hand"],
        expr_params=mhr_out["face"],
        **mhr_output_config,
    )
    avg_verts, avg_kp3d, avg_jcoords, _, _ = avg_mhr_output
    avg_verts[..., [1, 2]] *= -1
    avg_kp3d[..., [1, 2]] *= -1
    avg_jcoords[..., [1, 2]] *= -1

    ret["avg_verts"] = avg_verts
    ret["avg_kp3d"] = avg_kp3d
    ret["avg_jcoords"] = avg_jcoords

    # neutral
    mean_neutral_mhr_output = mhr_head.mhr_forward(
        shape_params=avg_shape,
        scale_offsets=avg_scale,
        **mhr_zero_inputs,
        **mhr_output_config,
    )
    mean_neutral_verts, mean_neutral_kp3d, mean_neutral_jcoords, _, _ = mean_neutral_mhr_output

    ret["avg_neutral_verts"] = mean_neutral_verts
    ret["avg_neutral_kp3d"] = mean_neutral_kp3d
    ret["avg_neutral_jcoords"] = mean_neutral_jcoords

    # ------------- merged -------------
    if "merged_shape" in param_dict and "merged_scale" in param_dict:
        merged_shape = param_dict["merged_shape"].repeat_interleave(num_views, dim=0)
        merged_scale = param_dict["merged_scale"].repeat_interleave(num_views, dim=0)

        merged_mhr_output = mhr_head.mhr_forward(
            shape_params=merged_shape,
            scale_offsets=merged_scale,
            global_trans=torch.zeros_like(mhr_out["global_rot"]),
            global_rot=mhr_out["global_rot"],
            body_pose_params=mhr_out["body_pose"],
            hand_pose_params=mhr_out["hand"],
            expr_params=mhr_out["face"],
            **mhr_output_config,
        )
        merged_verts, merged_kp3d, merged_jcoords, _, _ = merged_mhr_output
        merged_verts[..., [1, 2]] *= -1
        merged_kp3d[..., [1, 2]] *= -1
        merged_jcoords[..., [1, 2]] *= -1
        ret["merged_verts"] = merged_verts
        ret["merged_kp3d"] = merged_kp3d
        ret["merged_jcoords"] = merged_jcoords

        merged_neutral_mhr_output = mhr_head.mhr_forward(
            shape_params=merged_shape,
            scale_offsets=merged_scale,
            **mhr_zero_inputs,
            **mhr_output_config,
        )
        merged_neutral_verts, merged_neutral_kp3d, merged_neutral_jcoords, _, _ = (
            merged_neutral_mhr_output
        )
        ret["merged_neutral_verts"] = merged_neutral_verts
        ret["merged_neutral_kp3d"] = merged_neutral_kp3d
        ret["merged_neutral_jcoords"] = merged_neutral_jcoords


    # ------------- neutral gt -------------
    gt_shape = batch["shape_params"]
    gt_model_params = batch["model_params"]
    gt_face_params = batch["face_expr_coeffs"]
    gt_model_params[:, :-68] = torch.zeros_like(gt_model_params[:, :-68])
    gt_face_params = torch.zeros_like(gt_face_params)
    # gt_neutral_mhr_output = mhr_head.mhr(
    #     gt_shape, gt_model_params, gt_face_params
    # )
    # gt_neutral_verts, gt_neutral_skeleton_state = gt_neutral_mhr_output

    # gt_neutral_jcoords, _, _ = torch.split(
    #     gt_neutral_skeleton_state, [3, 4, 1], dim=2
    # )
    # gt_neutral_verts = gt_neutral_verts / 100
    # gt_neutral_jcoords = gt_neutral_jcoords / 100

    gt_neutral_mhr_output = mhr_head.mhr_forward(
        shape_params=gt_shape,
        scale_offsets=gt_model_params[:, -68:],
        **mhr_zero_inputs,
        **mhr_output_config,
    )
    gt_neutral_verts, gt_neutral_kp3d, gt_neutral_jcoords, _, _ = gt_neutral_mhr_output

    ret["gt_neutral_verts"] = gt_neutral_verts
    ret["gt_neutral_kp3d"] = gt_neutral_kp3d
    ret["gt_neutral_jcoords"] = gt_neutral_jcoords

    # ------------- neutral pred (per-view) -------------
    mhr_shape = mhr_out["shape"]
    mhr_scale = mhr_out["scale_68D"]

    per_view_neutral_mhr_output = mhr_head.mhr_forward(
        shape_params=mhr_shape,
        scale_offsets=mhr_scale,
        **mhr_zero_inputs,
        **mhr_output_config,
    )
    per_view_neutral_verts, per_view_neutral_kp3d, per_view_neutral_jcoords, _, _ = (
        per_view_neutral_mhr_output
    )

    ret["per_view_neutral_verts"] = per_view_neutral_verts
    ret["per_view_neutral_kp3d"] = per_view_neutral_kp3d
    ret["per_view_neutral_jcoords"] = per_view_neutral_jcoords

    # Also compute per-view "best" neutral prediction based on NF log-prob argmax.
    # This is meant for visualization only; metrics should still use the mean/regular outputs above.
    best_logprob_sample_shape = param_dict.get("best_logprob_sample_shape", None)
    best_logprob_sample_scale_68D = param_dict.get("best_logprob_sample_scale_68D", None)
    if best_logprob_sample_shape is not None and best_logprob_sample_scale_68D is not None:
        best_shape_flat = best_logprob_sample_shape.reshape(bs * num_views, -1)
        best_scale_flat = best_logprob_sample_scale_68D.reshape(bs * num_views, -1)
        best_logprob_neutral_out = mhr_head.mhr_forward(
            shape_params=best_shape_flat,
            scale_offsets=best_scale_flat,
            **mhr_zero_inputs,
            **mhr_output_config,
        )
        best_logprob_neutral_verts, best_logprob_neutral_kp3d, best_logprob_neutral_jcoords, _, _ = (
            best_logprob_neutral_out
        )
        ret["best_logprob_sample_neutral_verts"] = best_logprob_neutral_verts
        ret["best_logprob_sample_neutral_kp3d"] = best_logprob_neutral_kp3d
        ret["best_logprob_sample_neutral_jcoords"] = best_logprob_neutral_jcoords

    # ------------- neutral samples (all S samples per view) -------------
    if uncertainty_out is not None:
        shape_s = uncertainty_out["shape_samples"]   # [B*V, S, 45]
        scale_s = uncertainty_out["scale_samples"]   # [B*V, S, 68]
        BV, S, _ = shape_s.shape

        shape_s_flat = shape_s.flatten(0, 1)         # [B*V*S, 45]
        scale_s_flat = scale_s.flatten(0, 1)         # [B*V*S, 68]
        mhr_zero_inputs_s = {k: v.repeat_interleave(S, dim=0) for k, v in mhr_zero_inputs.items()}

        sample_neutral_out = mhr_head.mhr_forward(
            shape_params=shape_s_flat,
            scale_offsets=scale_s_flat,
            **mhr_zero_inputs_s,
            **mhr_output_config,
        )
        sample_neutral_verts, _, sample_neutral_jcoords, _, _ = sample_neutral_out
        ret["sample_neutral_verts"] = sample_neutral_verts.reshape(BV, S, *sample_neutral_verts.shape[1:])
        ret["sample_neutral_jcoords"] = sample_neutral_jcoords.reshape(BV, S, *sample_neutral_jcoords.shape[1:])

        # ------------- sample-param-average neutral (average residual samples, then MHR once) -------------
        # shape_samples = shape_mean + shape_residual_samples, so mean over S equals
        # shape_mean + mean(shape_residual_samples). Distinct from sample_neutral_verts,
        # which runs MHR per sample and averages *in vertex space*.
        sample_param_avg_shape = shape_s.mean(dim=1)  # [B*V, 45]
        sample_param_avg_scale = scale_s.mean(dim=1)  # [B*V, 68]

        sample_param_avg_neutral_out = mhr_head.mhr_forward(
            shape_params=sample_param_avg_shape,
            scale_offsets=sample_param_avg_scale,
            **mhr_zero_inputs,
            **mhr_output_config,
        )
        (
            sample_param_avg_neutral_verts,
            sample_param_avg_neutral_kp3d,
            sample_param_avg_neutral_jcoords,
            _,
            _,
        ) = sample_param_avg_neutral_out
        ret["sample_param_avg_neutral_verts"] = sample_param_avg_neutral_verts
        ret["sample_param_avg_neutral_kp3d"] = sample_param_avg_neutral_kp3d
        ret["sample_param_avg_neutral_jcoords"] = sample_param_avg_neutral_jcoords

    if "cross_view_log_prob_beta" in param_dict:
        ret["cross_view_log_prob_beta"] = param_dict["cross_view_log_prob_beta"]
    if "is_weight_beta" in param_dict:
        ret["is_weight_beta"] = param_dict["is_weight_beta"]

    # for k, v in ret.items():
    #     print(k, v.shape)
    # import ipdb; ipdb.set_trace()

    return ret