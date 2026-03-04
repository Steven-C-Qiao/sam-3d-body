import os
import torch
import roma
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from einops import rearrange
from torch.distributions import MultivariateNormal
from pytorch3d.transforms import matrix_to_axis_angle, axis_angle_to_matrix
from sam_3d_body.models.modules.mhr_utils import (
    mhr_param_hand_mask,
    mhr_cont_hand_idxs,
    mhr_param_hand_idxs,
)
from sam_3d_body.models.sampling import build_tril


class Loss(pl.LightningModule):
    def __init__(self, cfg, scale_mean, scale_comps):
        super().__init__()

        self.cfg = cfg
        self.register_buffer("scale_mean", scale_mean, persistent=False)
        self.register_buffer("scale_comps", scale_comps, persistent=False)

        self.mse_loss = nn.MSELoss(reduction="none")
        self.kp2d_loss = nn.L1Loss(reduction="none")
        self.gaussian_nll_loss = nn.GaussianNLLLoss(reduction="mean")


        self.hand_keypoint_indices = list(range(21, 63))  # 21–62 inclusive
        hand_weight = getattr(self.cfg.LOSS, "HAND_WEIGHT", 0.1)
        self.hand_weight = hand_weight

        # Debug visualization directory
        self.debug_vis_dir = None

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


        assert "total_loss" not in loss_dict
        loss_dict["total_loss"] = sum(
            v for k, v in loss_dict.items() if k != "total_loss"
        )

        if torch.isnan(loss_dict["total_loss"]):
            loss_dict["total_loss"] = torch.zeros_like(loss_dict["total_loss"])

        # for k, v in loss_dict.items():
        #     print(f"{k}: {v.item():.3f}", end=" ")
        # import ipdb; ipdb.set_trace()

        return loss_dict
