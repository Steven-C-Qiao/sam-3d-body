import torch
import torch.nn as nn
import pytorch_lightning as pl

from sam_3d_body.models.modules.mhr_utils import (
    convert_mhr_params_to_flow_params,
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
            kp2d_loss = kp2d_loss * visibility
            # kp2d_loss[..., self.hand_keypoint_indices] *= self.hand_weight

            loss_kp2d_samples = kp2d_loss.mean()

            loss_dict["loss_kp2d_samples"] = (
                self.cfg.LOSS.KP2D_WEIGHT * loss_kp2d_samples
            )

        if self.cfg.LOSS.KP3D_WEIGHT > 0:
            pred_kp3d_samples = predictions["kp3d_samples"]
            num_samples = pred_kp3d_samples.shape[1]

            visibility = batch["visibility"]
            visibility = visibility.unsqueeze(1).expand(-1, num_samples, -1)

            # pred_kp3d is in the wrong way up in 3D space, and projects correctly onto the image.
            # Thus, flip gt_kp3d for loss. Both pred and gt are upside down
            gt_kp3d = batch["keypoints_3d"][..., :3]
            gt_kp3d[..., [1, 2]] *= -1
            gt_kp3d = gt_kp3d.unsqueeze(1).expand(-1, num_samples, -1, -1)

            kp3d_loss = self.mse_loss(pred_kp3d_samples, gt_kp3d)
            kp3d_loss = kp3d_loss.mean(dim=-1)
            kp3d_loss = kp3d_loss * visibility
            # kp3d_loss[..., self.hand_keypoint_indices] *= self.hand_weight

            loss_kp3d_samples = kp3d_loss.mean()
            loss_dict["loss_kp3d_samples"] = (
                self.cfg.LOSS.KP3D_WEIGHT * loss_kp3d_samples
            )

        if self.cfg.LOSS.PARAM_NLL_WEIGHT > 0:
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
            mean_pred_flow_params = convert_mhr_params_to_flow_params(
                torch.cat(
                    [
                        torch.zeros_like(
                            mean_pred["body_pose"][..., :6]
                        ),  # Adds global, which is not used
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

            uncertainty_output = predictions["uncertainty_output"]
            num_samples = uncertainty_output["samples"].shape[1]

            # Stage-1 context depends only on (c, μβ).
            flow_context_raw = uncertainty_output["flow_context_raw"]
            mean_pred = predictions["mhr"]
            flow_context_shape_scale = self.nf_head.context_shape_scale_proj(
                torch.cat(
                    [
                        flow_context_raw,
                        mean_pred["shape"],
                        mean_pred["scale_68D"][..., scale_indices],
                    ],
                    dim=-1,
                )
            )

            # Stage-2 context depends on Δβ (i.e. the shape+scale residual).
            pose_mean_cont = mean_pred["pred_pose_raw"][:, 6:]
            pose_params = self.nf_head.convert_pose_cont_to_params_for_context(
                pose_mean_cont
            )
            aa_3dofs = pose_params["aa_3dofs"]  # B, 39
            params_1dofs = pose_params["params_1dofs"]  # B, 34

            shape_scale_residual_true = true_residual[..., self.nf_head.pose_dim :]
            shape_residual_true = shape_scale_residual_true[
                ...,
                : self.nf_head.num_shape_comps,
            ]
            scale_residual_true = shape_scale_residual_true[
                ...,
                self.nf_head.num_shape_comps :,
            ]

            shape_sample_true = mean_pred["shape"] + shape_residual_true
            scale_sample_selected_true = (
                mean_pred["scale_68D"][..., scale_indices] + scale_residual_true
            )

            flow_context_pose = self.nf_head.context_pose_proj(
                torch.cat(
                    [
                        flow_context_raw,
                        shape_sample_true,
                        scale_sample_selected_true,
                        aa_3dofs,
                        params_1dofs,
                    ],
                    dim=-1,
                )
            )

            flow_log_prob, z = self.nf_head.log_prob(
                true_residual, flow_context_shape_scale, flow_context_pose
            )
            nll_loss = -flow_log_prob.mean()

            loss_dict["loss_param_nll"] = self.cfg.LOSS.PARAM_NLL_WEIGHT * nll_loss

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
        if torch.isnan(loss_dict["total_loss"]):
            loss_dict["total_loss"] = torch.zeros_like(loss_dict["total_loss"])

        # for k, v in loss_dict.items():
        #     print(f"{k}: {v.item():.3f}", end=" ")
        # print('')
        # import ipdb; ipdb.set_trace()

        return loss_dict
