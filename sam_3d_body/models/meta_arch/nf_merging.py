import torch

from sam_3d_body.models.modules.mhr_utils import scale_indices

def merge_params_nf(
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
    """
    S = num_samples

    pred_shape = mhr_out["shape"].unflatten(0, (bs, num_views))
    pred_scale68 = mhr_out["scale_68D"].unflatten(0, (bs, num_views))
    pred_scale_params = mhr_out["scale"].unflatten(0, (bs, num_views))

    beta_context = uncertainty_out["flow_context_shape_scale"].unflatten(0, (bs, num_views))
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

    return {
        "avg_shape": shape_avg,
        "avg_scale": scale_avg,
        "merged_shape": shape_mu_star,
        "merged_scale": scale_mu_star_full,
    }




def get_mhr_outputs(
    mhr_head,
    batch,
    mhr_out,
    param_dict,
    bs,
    num_views,
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

    # for k, v in ret.items():
    #     print(k, v.shape)
    # import ipdb; ipdb.set_trace()

    return ret