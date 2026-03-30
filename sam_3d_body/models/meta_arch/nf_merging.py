import torch

from sam_3d_body.models.modules.mhr_utils import scale_indices


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

    beta_context = uncertainty_out["flow_context_shape_scale"].unflatten(0, (bs, num_views))
    beta_log_prob_ref = uncertainty_out["log_prob_shape_scale"].unflatten(0, (bs, num_views))
    flow_samples = uncertainty_out["samples"].unflatten(0, (bs, num_views))
    beta_residual_samples = flow_samples[..., -nf_head.shape_scale_dim :]
    shape_samples = uncertainty_out["shape_samples"].unflatten(0, (bs, num_views))
    scale68_samples = uncertainty_out["scale_samples"].unflatten(0, (bs, num_views))

    merged_shape = []
    merged_scale68 = []

    for b in range(bs):
        candidate_beta = []
        candidate_logw = []

        for i in range(num_views):
            # Proposal samples from view i: beta = [shape(45), selected_scale(10)].
            beta_i = torch.cat(
                [
                    shape_samples[b, i],  # [S, 45]
                    scale68_samples[b, i, :, scale_indices],  # [S, 10]
                ],
                dim=-1,
            )  # [S, 55]

            # # Debug self-consistency:
            # # Use the exact sampled stage-1 residuals from uncertainty_out["samples"].
            # # This avoids reconstruction mismatch from absolute samples and means.
            # residual_i = beta_residual_samples[b, i]  # [S, 55]
            # context_i = beta_context[b, i].unsqueeze(0).expand(S, -1)  # [S, 2048]
            # logp_i_recomputed, _ = nf_head.flow_shape_scale.log_prob(
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
                mean_beta_j = torch.cat(
                    [
                        pred_shape[b, j],  # [45]
                        pred_scale68[b, j, scale_indices],  # [10]
                    ],
                    dim=-1,
                )  # [55]
                residual_j = beta_i - mean_beta_j.unsqueeze(0)  # [S, 55]
                context_j = beta_context[b, j].unsqueeze(0).expand(S, -1)  # [S, 2048]
                logp_j, _ = nf_head.flow_shape_scale.log_prob(
                    inputs=residual_j, context=context_j
                )

                logw_i = logw_i + logp_j

            candidate_beta.append(beta_i)
            candidate_logw.append(logw_i)

        candidate_beta = torch.cat(candidate_beta, dim=0)  # [V*S, 55]
        candidate_logw = torch.cat(candidate_logw, dim=0)  # [V*S]

        candidate_w = torch.softmax(candidate_logw, dim=0)  # normalized importance weights

        merged_beta = (candidate_w.unsqueeze(-1) * candidate_beta).sum(dim=0)  # [55]
        merged_shape.append(merged_beta[: nf_head.num_shape_comps])

        scale68_merged = pred_scale68[b].mean(dim=0)
        scale68_merged[scale_indices] = merged_beta[nf_head.num_shape_comps :]
        merged_scale68.append(scale68_merged)

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

    beta_log_prob_ref = uncertainty_out["log_prob_shape_scale"].unflatten(0, (bs, num_views))  # [B, V, S]
    shape_samples = uncertainty_out["shape_samples"].unflatten(0, (bs, num_views))             # [B, V, S, 45]
    scale68_samples = uncertainty_out["scale_samples"].unflatten(0, (bs, num_views))           # [B, V, S, 68]

    # ---- Precision-weighted Gaussian product ----
    shape_mu = shape_samples.mean(dim=2)                              # [B, V, 45]
    shape_var = shape_samples.var(dim=2)                              # [B, V, 45]

    scale_selected_samples = scale68_samples[..., scale_indices]     # [B, V, S, 10]
    scale_mu = scale_selected_samples.mean(dim=2)                    # [B, V, 10]
    scale_var = scale_selected_samples.var(dim=2)                    # [B, V, 10]

    shape_prec = 1.0 / (shape_var + 1e-6)                           # [B, V, 45]
    scale_prec = 1.0 / (scale_var + 1e-6)                           # [B, V, 10]

    shape_mu_star = (shape_prec * shape_mu).sum(dim=1) / shape_prec.sum(dim=1)   # [B, 45]
    scale_mu_star = (scale_prec * scale_mu).sum(dim=1) / scale_prec.sum(dim=1)  # [B, 10]

    scale_mu_star_full = pred_scale68.mean(dim=1).clone()            # [B, 68]
    scale_mu_star_full[:, scale_indices] = scale_mu_star

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


def merge_params_nf(
    nf_head,
    mhr_out,
    uncertainty_out,
    bs,
    num_views,
    num_samples,
):
    """Default merge strategy: precision-weighted Gaussian product (see merge_params_nf_gaussian)."""
    # return merge_params_nf_gaussian(
    #     nf_head, mhr_out, uncertainty_out, bs, num_views, num_samples
    # )

    return merge_params_nf_is(
        nf_head, mhr_out, uncertainty_out, bs, num_views, num_samples
    )


def get_mhr_outputs(
    mhr_head,
    batch,
    mhr_out,
    param_dict,
    bs,
    num_views,
    uncertainty_out=None,
):
    ret = {}


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

    # for k, v in ret.items():
    #     print(k, v.shape)
    # import ipdb; ipdb.set_trace()

    return ret