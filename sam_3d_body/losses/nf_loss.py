import torch
import torch.nn as nn
import pytorch_lightning as pl

from sam_3d_body.models.modules.mhr_utils import (
    batchXYZfrom6D,
    convert_mhr_params_to_flow_params,
    convert_pose_cont_to_flow_context,
    scale_indices,
)


class Loss(pl.LightningModule):
    def __init__(self, cfg, scale_mean, scale_comps, nf_head=None):
        super().__init__()

        self.cfg = cfg
        self.register_buffer("scale_mean", scale_mean, persistent=False)
        self.register_buffer("scale_comps", scale_comps, persistent=False)

        object.__setattr__(self, "nf_head", nf_head)

        self.mse_loss = nn.MSELoss(reduction="none")
        self.kp2d_loss = nn.L1Loss(reduction="none")

        self.hand_keypoint_indices = list(range(21, 63))  # 21–62 inclusive
        hand_weight = getattr(self.cfg.LOSS, "HAND_WEIGHT", 0.1)
        self.hand_weight = hand_weight

    def _compute_gt_residual_log_prob(self, true_residual, predictions):
        head_type = self.cfg.MODEL.HEAD_TYPE
        uncertainty_output = predictions["uncertainty_output"]

        if head_type == "nf":
            flow_context = uncertainty_output["flow_context"]
            flow_log_prob, _ = self.nf_head.log_prob(true_residual, flow_context)
            return flow_log_prob

        if head_type == "nf_ar":
            flow_context_raw = uncertainty_output["flow_context_raw"]
            mean_pred = predictions["mhr"]

            flow_context_beta = self.nf_head.beta_context_proj(
                torch.cat(
                    [
                        flow_context_raw,
                        mean_pred["shape"],
                        mean_pred["scale_68D"][..., scale_indices],
                    ],
                    dim=-1,
                )
            )

            pose_mean_cont = mean_pred["pred_pose_raw"][:, 6:]
            pose_params = convert_pose_cont_to_flow_context(pose_mean_cont)
            aa_3dofs = pose_params["aa_3dofs"]  # B, 39
            params_1dofs = pose_params["params_1dofs"]  # B, 34

            beta_residual_true = true_residual[..., self.nf_head.theta_dim :]
            shape_residual_true = beta_residual_true[
                ...,
                : self.nf_head.num_shape_comps,
            ]
            scale_residual_true = beta_residual_true[
                ...,
                self.nf_head.num_shape_comps :,
            ]

            shape_sample_true = mean_pred["shape"] + shape_residual_true
            scale_sample_selected_true = (
                mean_pred["scale_68D"][..., scale_indices] + scale_residual_true
            )

            context_theta_parts = [
                flow_context_raw,
                shape_sample_true,
                scale_sample_selected_true,
                aa_3dofs,
                params_1dofs,
            ]
            if self.nf_head.model_cam:
                context_theta_parts.append(mean_pred["pred_cam"])
            flow_context_theta = self.nf_head.theta_context_proj(
                torch.cat(context_theta_parts, dim=-1)
            )

            flow_log_prob, _ = self.nf_head.log_prob(
                true_residual, flow_context_beta, flow_context_theta
            )
            return flow_log_prob

        raise ValueError(f"Unsupported MODEL.HEAD_TYPE: {head_type}")

    def forward(self, predictions, batch):
        loss_dict = {}

        B, N = batch["img"].shape[:2]

        if self.cfg.LOSS.KP2D_WEIGHT > 0:
            pred_kp2d_samples = predictions["kp2d_samples_cropped"]
            num_samples = pred_kp2d_samples.shape[1]

            visibility = batch["visibility"]
            visibility = visibility.unsqueeze(1).expand(-1, num_samples, -1)

            gt_kp2d = batch["keypoints_2d"]
            gt_kp2d = gt_kp2d.unsqueeze(1).expand(-1, num_samples, -1, -1)

            kp2d_loss = self.kp2d_loss(pred_kp2d_samples, gt_kp2d)
            kp2d_loss = kp2d_loss.mean(dim=-1)

            if not getattr(self.cfg.LOSS, "BYPASS_VISIBILITY", False):
                kp2d_loss = kp2d_loss * visibility
            # kp2d_loss[..., self.hand_keypoint_indices] *= self.hand_weight

            best_of_n = getattr(self.cfg.LOSS, "KP2D_BEST_OF_N", 0)
            if best_of_n > 0:
                # Penalise only the N closest samples to GT; allow others to be diverse
                per_sample_loss = kp2d_loss.mean(dim=-1)  # (B, num_samples)
                k = min(best_of_n, num_samples)
                loss_kp2d_samples = torch.topk(per_sample_loss, k=k, dim=1, largest=False).values.mean()
            else:
                loss_kp2d_samples = kp2d_loss.mean()

            loss_dict["loss_kp2d_samples"] = (
                self.cfg.LOSS.KP2D_WEIGHT * loss_kp2d_samples
            )

        if self.cfg.LOSS.KP3D_WEIGHT > 0 and getattr(self.cfg.LOSS, "KP3D_ON_SAMPLES", True):
            pred_kp3d_samples = predictions["kp3d_samples"]
            num_samples = pred_kp3d_samples.shape[1]

            visibility = batch["visibility"]
            visibility = visibility.unsqueeze(1).expand(-1, num_samples, -1)

            gt_kp3d = batch["keypoints_3d"][..., :3]
            gt_kp3d = gt_kp3d.unsqueeze(1).expand(-1, num_samples, -1, -1)

            kp3d_loss = self.mse_loss(pred_kp3d_samples, gt_kp3d)
            kp3d_loss = kp3d_loss.mean(dim=-1)
            
            if not getattr(self.cfg.LOSS, "BYPASS_VISIBILITY", False):  
                kp3d_loss = kp3d_loss * visibility
            # kp3d_loss[..., self.hand_keypoint_indices] *= self.hand_weight

            loss_kp3d_samples = kp3d_loss.mean()
            loss_dict["loss_kp3d_samples"] = (
                self.cfg.LOSS.KP3D_WEIGHT * loss_kp3d_samples
            )

        if self.cfg.LOSS.PARAM_NLL_WEIGHT > 0 or self.cfg.LOSS.PARAM_L2_WEIGHT > 0:
            """
            Evaluate the gt residual NLL
            """
            gt_model_params = batch["model_params"]
            gt_shape = batch["shape_params"]

            gt_flow_params = convert_mhr_params_to_flow_params(
                gt_model_params,
                gt_shape,
                include_global_rot=getattr(self.cfg.MODEL, "MODEL_GLOB_ROT", False),
                include_shape=getattr(self.cfg.MODEL, "MODEL_SHAPE", True),
                include_scale=getattr(self.cfg.MODEL, "MODEL_SCALE", True),
            )

            mean_pred = predictions["mhr"]
            # When MODEL_GLOB_ROT, include the mean prediction's global rotation
            # so that the flow learns a residual, not the absolute rotation.
            if getattr(self.cfg.MODEL, "MODEL_GLOB_ROT", False):
                glob_rot_euler_mean = batchXYZfrom6D(mean_pred["pred_pose_raw"][:, :6])
                mean_global = torch.cat([
                    torch.zeros_like(glob_rot_euler_mean),  # global_trans (unused)
                    glob_rot_euler_mean,
                ], dim=-1)
            else:
                mean_global = torch.zeros_like(mean_pred["body_pose"][..., :6])
            mean_pred_flow_params = convert_mhr_params_to_flow_params(
                torch.cat(
                    [
                        mean_global,
                        mean_pred["body_pose"][..., :130],  # gets rid of jaw
                        mean_pred["scale_68D"],
                    ],
                    dim=-1,
                ),
                mean_pred["shape"],
                include_global_rot=getattr(self.cfg.MODEL, "MODEL_GLOB_ROT", False),
                include_shape=getattr(self.cfg.MODEL, "MODEL_SHAPE", True),
                include_scale=getattr(self.cfg.MODEL, "MODEL_SCALE", True),
            )

            true_residual = gt_flow_params - mean_pred_flow_params

            if getattr(self.cfg.MODEL, "MODEL_CAM", False):
                theta_dim_no_cam = self.nf_head.theta_dim - self.nf_head.num_cam_comps
                cam_zeros = torch.zeros(
                    true_residual.shape[0], self.nf_head.num_cam_comps,
                    device=true_residual.device, dtype=true_residual.dtype,
                )
                true_residual = torch.cat([
                    true_residual[..., :theta_dim_no_cam],
                    cam_zeros,
                    true_residual[..., theta_dim_no_cam:],
                ], dim=-1)

        if self.cfg.LOSS.PARAM_NLL_WEIGHT > 0:
            uncertainty_output = predictions["uncertainty_output"]
            num_samples = uncertainty_output["samples"].shape[1]

            flow_log_prob = self._compute_gt_residual_log_prob(
                true_residual, predictions
            )
            nll_loss = -flow_log_prob.mean()

            # Keep the GT-residual log probability for visualization/analysis.
            # This is the log p(gt_residual | context) before weighting.
            loss_dict["loss_param_nll"] = self.cfg.LOSS.PARAM_NLL_WEIGHT * nll_loss

        # DEPRECATED: raw parameter-space variance. Can be gamed by random noise.
        # Prefer KP3D_INVISIBLE_SPREAD_WEIGHT instead.
        entropy_weight = getattr(self.cfg.LOSS, "ENTROPY_WEIGHT", 0.0)
        if entropy_weight > 0:
            samples = predictions["uncertainty_output"]["samples"]  # [B, N, D]
            entropy_bonus = samples.var(dim=1).mean()
            loss_dict["loss_entropy"] = -entropy_weight * entropy_bonus

        # Principled diversity: maximise 3D keypoint spread over invisible/occluded joints
        # only. Cannot be gamed by random noise — visible joints are already constrained
        # by the 2D keypoint loss; only genuinely ambiguous joints are rewarded for spread.
        kp3d_spread_weight = getattr(self.cfg.LOSS, "KP3D_INVISIBLE_SPREAD_WEIGHT", 0.0)
        if kp3d_spread_weight > 0:
            kp3d_samples = predictions["kp3d_samples"]   # (B, N, J, 3)
            invisible_mask = (~batch["visibility"].bool()).float()  # (B, J)

            if invisible_mask.sum() > 0:
                _N = kp3d_samples.shape[1]
                sample_mean = kp3d_samples.mean(dim=1, keepdim=True)     # (B, 1, J, 3)
                centered = kp3d_samples - sample_mean                     # (B, N, J, 3)
                # Squared L2 distances — no sqrt avoids gradient instability at zero
                dists = (centered ** 2).sum(dim=-1)                       # (B, N, J)
                invis_n = invisible_mask.unsqueeze(1).expand(-1, _N, -1)  # (B, N, J)
                spread = (
                    (dists * invis_n).sum(dim=-1)
                    / invis_n.sum(dim=-1).clamp(min=1)
                )  # (B, N)
                loss_dict["loss_kp3d_invisible_spread"] = -kp3d_spread_weight * spread.mean()

        # Ray-decomposed diversity for visible joints (mode-2 / single-view ambiguity).
        # Samples are rewarded for spreading along the camera ray (depth direction) and
        # penalised for spreading perpendicular to the ray (which violates 2D consistency).
        along_ray_weight = getattr(self.cfg.LOSS, "KP3D_ALONG_RAY_WEIGHT", 0.0)
        perp_ray_weight  = getattr(self.cfg.LOSS, "KP3D_PERP_RAY_WEIGHT",  0.0)
        if along_ray_weight > 0 or perp_ray_weight > 0:
            kp3d_samples = predictions["kp3d_samples"]   # (B, N, J, 3)
            visible_mask = batch["visibility"].float()   # (B, J)

            if visible_mask.sum() > 0:
                # Camera translation in body frame — used to compute ray directions.
                # P_cam = P_body + trans_cam, so the ray from camera to joint is
                # normalize(gt_kp3d_body + trans_cam).
                if "cam_ext" in batch:
                    trans_cam = batch["cam_ext"][:, :3, 3]   # (B, 3) — BEDLAM / 4D-dress
                else:
                    trans_cam = batch["trans_cam"]            # (B, 3) — SSP-3D
                gt_kp3d_cam = batch["keypoints_3d"] + trans_cam.unsqueeze(1)  # (B, J, 3)
                ray = gt_kp3d_cam / (gt_kp3d_cam.norm(dim=-1, keepdim=True) + 1e-8)  # (B, J, 3)

                _N = kp3d_samples.shape[1]
                sample_mean = kp3d_samples.mean(dim=1, keepdim=True)   # (B, 1, J, 3)
                centered = kp3d_samples - sample_mean                   # (B, N, J, 3)

                ray_exp = ray.unsqueeze(1)                              # (B, 1, J, 3)
                scalar_proj = (centered * ray_exp).sum(dim=-1)          # (B, N, J)
                perp = centered - scalar_proj.unsqueeze(-1) * ray_exp  # (B, N, J, 3)

                dist_along = scalar_proj ** 2                           # (B, N, J)
                dist_perp  = (perp ** 2).sum(dim=-1)                    # (B, N, J)

                vis_n = visible_mask.unsqueeze(1).expand(-1, _N, -1)    # (B, N, J)
                denom = vis_n.sum(dim=-1).clamp(min=1)                  # (B, N)

                if along_ray_weight > 0:
                    spread_along = ((dist_along * vis_n).sum(dim=-1) / denom).mean()
                    loss_dict["loss_kp3d_along_ray"] = -along_ray_weight * spread_along

                if perp_ray_weight > 0:
                    spread_perp = ((dist_perp * vis_n).sum(dim=-1) / denom).mean()
                    loss_dict["loss_kp3d_perp_ray"] = perp_ray_weight * spread_perp

        if self.cfg.LOSS.PARAM_L2_WEIGHT > 0:
            param_l2_loss = self.mse_loss(
                true_residual.unsqueeze(1).expand(-1, num_samples, -1),
                predictions["uncertainty_output"]["samples"]
            )
            param_l2_loss = param_l2_loss.mean()
            loss_dict["loss_param_l2"] = self.cfg.LOSS.PARAM_L2_WEIGHT * param_l2_loss

        assert "total_loss" not in loss_dict
        loss_dict["total_loss"] = sum(
            v for k, v in loss_dict.items() if k != "total_loss"
        )
        
        # loss_dict["gt_residual_log_prob"] = flow_log_prob.detach()

        if torch.isnan(loss_dict["total_loss"]):
            loss_dict["total_loss"] = torch.zeros_like(loss_dict["total_loss"])

        # for k, v in loss_dict.items():
        #     print(f"{k}: {v.item():.3f}", end=" ")
        # print('')
        # import ipdb; ipdb.set_trace()

        return loss_dict
