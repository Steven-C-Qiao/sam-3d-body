import torch
import torch.nn as nn
import pytorch_lightning as pl

from sam_3d_body.models.modules.mhr_utils import convert_mhr_params_to_flow_params


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

            # pred_kp3d is in the wrong way up in 3D space, and projects correctly onto the image.
            # Thus, flip gt_kp3d for loss. Both pred and gt are upside down
            gt_kp3d = batch["keypoints_3d"][..., :3]
            gt_kp3d[..., [1, 2]] *= -1
            gt_kp3d = gt_kp3d.unsqueeze(1).expand(
                -1, pred_kp3d_samples.shape[1], -1, -1
            )

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
            gt_face = batch["face_expr_coeffs"]


            gt_scale_68D = gt_model_params[:, -68:]
            gt_pose = gt_model_params[:, 6:-68]


            gt_flow_params = convert_mhr_params_to_flow_params(gt_model_params, gt_shape)
            
            mean_pred = predictions["mhr"]
            mean_pred_flow_params = convert_mhr_params_to_flow_params(
                torch.cat([
                    torch.zeros_like(mean_pred["body_pose"][..., :6]), # Adds global, which is not used
                    mean_pred["body_pose"][..., :130], # gets rid of jaw
                    mean_pred["scale_68D"],
                ], dim=-1), 
                mean_pred["shape"]
            )

            # from sam_3d_body.models.modules.mhr_utils import (
            #     all_param_3dof_rot_idxs, 
            #     all_param_1dof_rot_idxs, 
            #     indices_3dof, 
            #     indices_1dof, 
            #     scale_indices
            # )
            # from sam_3d_body.models.modules.mhr_utils import batch6DFromXYZ

            # gt_pose_3dof_euler = gt_pose[..., all_param_3dof_rot_idxs[:-1].flatten()]
            # gt_pose_3dof_euler = torch.cat([gt_pose_3dof_euler, torch.zeros_like(gt_pose_3dof_euler[..., :3])], dim=-1)
            # gt_pose_3dof_euler = gt_pose_3dof_euler.unflatten(-1, (-1, 3))
            # gt_pose_1dof_angle = gt_pose[..., indices_1dof]
            # gt_pose_3dof_rotmat = batch6DFromXYZ(gt_pose_3dof_euler, return_9D=True)
            # gt_pose_3dof_aa = matrix_to_axis_angle(gt_pose_3dof_rotmat)
            # gt_pose_3dof_aa_selected = gt_pose_3dof_aa[..., indices_3dof, :].flatten(-2, -1)
            # gt_scale_selected = gt_scale_68D[..., scale_indices]
            # gt_flow_params  = torch.cat([
            #     gt_pose_3dof_aa_selected, gt_pose_1dof_angle, gt_shape, gt_scale_selected
            # ], dim=-1)

            # mean_pred = predictions["mhr"]
            # mean_pred_shape = mean_pred["shape"]
            # mean_pred_scale_68D = mean_pred["scale_68D"]
            # mean_pred_pose = mean_pred["body_pose"][..., 6:]
            # mean_pred_pose_3dof_euler = mean_pred_pose[..., all_param_3dof_rot_idxs[:-1].flatten()]
            # mean_pred_pose_3dof_euler = torch.cat([mean_pred_pose_3dof_euler, torch.zeros_like(mean_pred_pose_3dof_euler[..., :3])], dim=-1)
            # mean_pred_pose_3dof_euler = mean_pred_pose_3dof_euler.unflatten(-1, (-1, 3))
            # mean_pred_pose_1dof_angle = mean_pred_pose[..., indices_1dof]
            # mean_pred_pose_3dof_rotmat = batch6DFromXYZ(mean_pred_pose_3dof_euler, return_9D=True)
            # mean_pred_pose_3dof_aa = matrix_to_axis_angle(mean_pred_pose_3dof_rotmat)
            # mean_pred_pose_3dof_aa_selected = mean_pred_pose_3dof_aa[..., indices_3dof, :].flatten(-2, -1)
            # mean_pred_scale_selected = mean_pred_scale_68D[..., scale_indices]
            # mean_pred_flow_params  = torch.cat([
            #     mean_pred_pose_3dof_aa_selected, mean_pred_pose_1dof_angle, mean_pred_shape, mean_pred_scale_selected
            # ], dim=-1)

            true_residual = gt_flow_params - mean_pred_flow_params
            # evaluate the true_residual nll 
            flow_context = predictions["uncertainty_output"]["flow_context"]

            self.nf_head.eval()
            flow_log_prob, z = self.nf_head.log_prob(
                true_residual, 
                flow_context
            )

            nll_loss = - flow_log_prob.mean()

            auto_sample_loglik = predictions['uncertainty_output']['log_prob']
            samples = predictions["uncertainty_output"]["samples"]            
            sample_log_prob, z = self.nf_head.log_prob(samples.flatten(0, 1), flow_context.repeat_interleave(5, dim=0))
            sample_log_prob = sample_log_prob.unflatten(0, (B, -1))
            print(flow_log_prob[:5])
            print(auto_sample_loglik[0, :5])
            print(sample_log_prob[0, :5])
    
            self.nf_head.train()

            loss_dict["loss_param_nll"] = (self.cfg.LOSS.PARAM_NLL_WEIGHT * nll_loss)

        assert "total_loss" not in loss_dict
        loss_dict["total_loss"] = sum(
            v for k, v in loss_dict.items() if k != "total_loss"
        )
        if torch.isnan(loss_dict["total_loss"]):
            loss_dict["total_loss"] = torch.zeros_like(loss_dict["total_loss"])

        for k, v in loss_dict.items():
            print(f"{k}: {v.item():.3f}", end=" ")
        print(flow_log_prob[:5])
        print('')
        import ipdb; ipdb.set_trace()

        return loss_dict
