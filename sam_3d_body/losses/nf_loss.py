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

            true_residual = gt_flow_params - mean_pred_flow_params
            
            flow_context = predictions["uncertainty_output"]["flow_context"]

            self.nf_head.eval()
            flow_log_prob, z = self.nf_head.log_prob(
                true_residual, 
                flow_context
            )

            nll_loss = - flow_log_prob.mean()

            # auto_sample_loglik = predictions['uncertainty_output']['log_prob']
            # samples = predictions["uncertainty_output"]["samples"]            
            # sample_log_prob, z = self.nf_head.log_prob(samples.flatten(0, 1), flow_context.repeat_interleave(5, dim=0))
            # sample_log_prob = sample_log_prob.unflatten(0, (B, -1))
            # print(flow_log_prob[:5])
            # print(auto_sample_loglik[0, :5])
            # print(sample_log_prob[0, :5])
    
            self.nf_head.train()

            loss_dict["loss_param_nll"] = (self.cfg.LOSS.PARAM_NLL_WEIGHT * nll_loss)

        assert "total_loss" not in loss_dict
        loss_dict["total_loss"] = sum(
            v for k, v in loss_dict.items() if k != "total_loss"
        )
        if torch.isnan(loss_dict["total_loss"]):
            loss_dict["total_loss"] = torch.zeros_like(loss_dict["total_loss"])

        # for k, v in loss_dict.items():
        #     print(f"{k}: {v.item():.3f}", end=" ")
        # print('')
        # print(flow_log_prob[:5])
        # import ipdb; ipdb.set_trace()

        return loss_dict
