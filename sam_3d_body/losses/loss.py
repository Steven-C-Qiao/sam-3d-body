import torch
import torch.nn as nn
import pytorch_lightning as pl
import roma
from pytorch3d.transforms import matrix_to_axis_angle
from sam_3d_body.models.modules import rot6d_to_rotmat
from sam_3d_body.models.modules.mhr_utils import (
    NUM_BODY_3DOF_JOINTS,
    BODY_1DOF_ROT_IDXS,
    NUM_BODY_1DOF_ANGLES,
)


class Loss(pl.LightningModule):
    def __init__(self, cfg, scale_mean, scale_comps):
        super().__init__()

        self.cfg = cfg
        self.register_buffer("scale_mean", scale_mean, persistent=False)
        self.register_buffer("scale_comps", scale_comps, persistent=False)

        self.mse_loss = nn.MSELoss(reduction="none")
        self.kp2d_loss = nn.L1Loss(reduction="none")
        self.gaussian_nll_loss = nn.GaussianNLLLoss(reduction="mean")

        # Hand keypoint indices in MHR70: right hand 21–41, left hand 42–62
        # Total 70 keypoints; hands are indices 21–62 (42 keypoints)
        self.hand_keypoint_indices = list(range(21, 63))  # 21–62 inclusive

        # Get hand weight from config if present, otherwise default to 0.1
        hand_weight = getattr(self.cfg.LOSS, "HAND_WEIGHT", 0.1)

        # Store hand weight; full keypoint count (70 + dense) will be inferred
        # dynamically from tensors during training so we don't assume a fixed
        # dense keypoint length.
        self.hand_weight = hand_weight

    def forward(self, predictions, batch):
        loss_dict = {}

        B, N = batch["img"].shape[:2]

        pred_mhr = predictions["mhr"]

        if self.cfg.LOSS.KP2D_WEIGHT > 0:
            pred_kp2d_samples = predictions["mhr_samples_keypoints_2d_cropped"]
            num_samples = pred_kp2d_samples.shape[1]

            gt_kp2d = batch["keypoints_2d"]
            gt_kp2d = gt_kp2d.unsqueeze(1).expand(-1, num_samples, -1, -1)

            kp2d_loss = self.kp2d_loss(pred_kp2d_samples, gt_kp2d)
            kp2d_loss = kp2d_loss.mean(dim=-1)
            kp2d_loss[..., self.hand_keypoint_indices] *= self.hand_weight

            loss_kp2d_samples = kp2d_loss.mean()

            loss_dict["loss_kp2d_samples"] = (
                self.cfg.LOSS.KP2D_WEIGHT * loss_kp2d_samples
            )

        if self.cfg.LOSS.KP3D_WEIGHT > 0:
            pred_kp3d_samples = predictions["mhr_samples_keypoints_3d"]

            # pred_kp3d is in the wrong way up in 3D space, and projects correctly onto the image.
            # Thus, flip gt_kp3d for loss. Both pred and gt are upside down
            gt_kp3d = batch["keypoints_3d"][..., :3]
            gt_kp3d[..., [1, 2]] *= -1
            gt_kp3d = gt_kp3d.unsqueeze(1).expand(
                -1, pred_kp3d_samples.shape[1], -1, -1
            )

            kp3d_loss = self.mse_loss(pred_kp3d_samples, gt_kp3d)
            kp3d_loss = kp3d_loss.mean(dim=-1)

            kp3d_loss[..., self.hand_keypoint_indices] *= self.hand_weight

            loss_kp3d_samples = kp3d_loss.mean()
            loss_dict["loss_kp3d_samples"] = (
                self.cfg.LOSS.KP3D_WEIGHT * loss_kp3d_samples
            )

        if self.cfg.LOSS.SHAPE_PARAM_WEIGHT > 0:
            gt_shape_params = batch["shape_params"]
            pred_shape_params = pred_mhr["shape"]
            pred_shape_uncertainty = pred_mhr["shape_uncertainty"]

            loss_shape_params = self.gaussian_nll_loss(
                pred_shape_params, gt_shape_params, pred_shape_uncertainty
            )
            loss_dict["loss_shape_params"] = (
                self.cfg.LOSS.SHAPE_PARAM_WEIGHT * loss_shape_params
            )

        if self.cfg.LOSS.SCALE_PARAM_WEIGHT > 0:
            indices = [3, 4, 5, 6, 7, 10, 11, 12, 13, 14]

            gt_scale = batch["scale_params"]
            pred_scale = pred_mhr["scale"]
            pred_scale_var = pred_mhr["scale_uncertainty"]

            pred_scale = self.scale_mean[None, :] + pred_scale @ self.scale_comps

            loss_scale_params = self.gaussian_nll_loss(
                pred_scale[:, indices], gt_scale[:, indices], pred_scale_var
            )
            loss_dict["loss_scale_params"] = (
                self.cfg.LOSS.SCALE_PARAM_WEIGHT * loss_scale_params
            )

        # Gaussian NLL loss on pose parameters in axis-angle space:
        # - Global rotation (3D axis-angle)
        # - 3-DoF body joints (3*NUM_BODY_3DOF_JOINTS axis-angle)
        # - 1-DoF body joint angles (NUM_BODY_1DOF_ANGLES scalars)
        if getattr(self.cfg.LOSS, "POSE_PARAM_WEIGHT", 0.0) > 0:
            gt_model_params = batch["model_params"]
            pred_body_pose = pred_mhr["body_pose"]  # [B*N, 133]
            pred_pose_uncertainty = pred_mhr[
                "pose_uncertainty"
            ]  # [B*N, pose_uncertainty_dim]

            # Flatten batch/person dims if needed to match prediction layout.
            if gt_model_params.dim() == 3:  # [B, N, D]
                B, N = gt_model_params.shape[:2]
                gt_model_params = gt_model_params.view(B * N, -1)
            else:
                gt_model_params = gt_model_params.view(-1, gt_model_params.shape[-1])

            pred_body_pose = pred_body_pose.view(-1, pred_body_pose.shape[-1])
            pred_pose_uncertainty = pred_pose_uncertainty.view(
                -1, pred_pose_uncertainty.shape[-1]
            )

            # Body pose parameters occupy 133 dims starting after 3 trans + 3 global rot.
            gt_body_pose = gt_model_params[:, 6 : 6 + pred_body_pose.shape[-1]]
            gt_global_rot_euler = gt_model_params[
                :, 3:6
            ]  # Global rotation in Euler (ZYX)

            # Convert global rotation from Euler to axis-angle
            # roma expects signature: euler_to_rotmat(convention, angles)
            gt_global_rot_mat = roma.euler_to_rotmat("ZYX", gt_global_rot_euler)
            gt_global_rot_axis_angle = matrix_to_axis_angle(
                gt_global_rot_mat
            )  # [B*N, 3]

            # Get predicted global rotation from pred_pose_raw (6D) and convert to axis-angle
            pred_pose_raw = pred_mhr["pred_pose_raw"]  # [B*N, 6 + body_cont_dim]
            pred_global_rot_6d = pred_pose_raw[:, :6]
            # Convert 6D to rotation matrix, then to axis-angle
            pred_global_rot_mat = rot6d_to_rotmat(pred_global_rot_6d)
            pred_global_rot_axis_angle = matrix_to_axis_angle(
                pred_global_rot_mat
            )  # [B*N, 3]

            # Extract 3-DoF body joint rotations and convert to axis-angle
            # all_param_3dof_rot_idxs defines which indices in body_pose correspond to 3-DoF joints
            all_param_3dof_rot_idxs = torch.LongTensor(
                [
                    (0, 2, 4),
                    (6, 8, 10),
                    (12, 13, 14),
                    (15, 16, 17),
                    (18, 19, 20),
                    (21, 22, 23),
                    (24, 25, 26),
                    (27, 28, 29),
                    (34, 35, 36),
                    (37, 38, 39),
                    (44, 45, 46),
                    (53, 54, 55),
                    (64, 65, 66),
                    (85, 69, 73),
                    (86, 70, 79),
                    (87, 71, 82),
                    (88, 72, 76),
                    (91, 92, 93),
                    (112, 96, 100),
                    (113, 97, 106),
                    (114, 98, 109),
                    (115, 99, 103),
                    (130, 131, 132),
                ]
            ).to(gt_body_pose.device)

            # Extract 3-DoF Euler angles from body pose
            gt_body_3dof_euler = gt_body_pose[
                :, all_param_3dof_rot_idxs.flatten()
            ]  # [B*N, 69]
            pred_body_3dof_euler = pred_body_pose[
                :, all_param_3dof_rot_idxs.flatten()
            ]  # [B*N, 69]

            # Reshape to [B*N, NUM_BODY_3DOF_JOINTS, 3] and convert to axis-angle
            gt_body_3dof_euler_reshaped = gt_body_3dof_euler.view(
                -1, NUM_BODY_3DOF_JOINTS, 3
            )
            pred_body_3dof_euler_reshaped = pred_body_3dof_euler.view(
                -1, NUM_BODY_3DOF_JOINTS, 3
            )

            # Convert each 3-DoF joint from Euler to axis-angle
            gt_body_3dof_rot_mat = roma.euler_to_rotmat(
                "ZYX", gt_body_3dof_euler_reshaped.view(-1, 3)
            ).view(-1, NUM_BODY_3DOF_JOINTS, 3, 3)
            pred_body_3dof_rot_mat = roma.euler_to_rotmat(
                "ZYX", pred_body_3dof_euler_reshaped.view(-1, 3)
            ).view(-1, NUM_BODY_3DOF_JOINTS, 3, 3)

            gt_body_3dof_axis_angle = matrix_to_axis_angle(
                gt_body_3dof_rot_mat.view(-1, 3, 3)
            ).view(
                -1, NUM_BODY_3DOF_JOINTS, 3
            )  # [B*N, NUM_BODY_3DOF_JOINTS, 3]
            pred_body_3dof_axis_angle = matrix_to_axis_angle(
                pred_body_3dof_rot_mat.view(-1, 3, 3)
            ).view(
                -1, NUM_BODY_3DOF_JOINTS, 3
            )  # [B*N, NUM_BODY_3DOF_JOINTS, 3]

            # Flatten 3-DoF axis-angle to [B*N, 3*NUM_BODY_3DOF_JOINTS]
            gt_body_3dof_axis_angle_flat = gt_body_3dof_axis_angle.view(
                -1, 3 * NUM_BODY_3DOF_JOINTS
            )
            pred_body_3dof_axis_angle_flat = pred_body_3dof_axis_angle.view(
                -1, 3 * NUM_BODY_3DOF_JOINTS
            )

            # Extract 1-DoF angles (already in angle space, no conversion needed)
            BODY_1DOF_ROT_IDXS_device = BODY_1DOF_ROT_IDXS.to(gt_body_pose.device)
            gt_body_1dof_angles = gt_body_pose[
                :, BODY_1DOF_ROT_IDXS_device
            ]  # [B*N, NUM_BODY_1DOF_ANGLES]
            pred_body_1dof_angles = pred_body_pose[
                :, BODY_1DOF_ROT_IDXS_device
            ]  # [B*N, NUM_BODY_1DOF_ANGLES]

            # Concatenate: [global (3), body_3dof (3*23), body_1dof (NUM_BODY_1DOF_ANGLES)]
            gt_pose_axis_angle = torch.cat(
                [
                    gt_global_rot_axis_angle,
                    gt_body_3dof_axis_angle_flat,
                    gt_body_1dof_angles,
                ],
                dim=-1,
            )  # [B*N, 3 + 3*NUM_BODY_3DOF_JOINTS + NUM_BODY_1DOF_ANGLES]

            pred_pose_axis_angle = torch.cat(
                [
                    pred_global_rot_axis_angle,
                    pred_body_3dof_axis_angle_flat,
                    pred_body_1dof_angles,
                ],
                dim=-1,
            )  # [B*N, 3 + 3*NUM_BODY_3DOF_JOINTS + NUM_BODY_1DOF_ANGLES]

            # Use Gaussian NLL loss: assumes target ~ N(pred, uncertainty)
            # uncertainty is already positive exp(...) and represents variance
            loss_pose_params = self.gaussian_nll_loss(
                pred_pose_axis_angle, gt_pose_axis_angle, pred_pose_uncertainty
            )
            loss_dict["loss_pose_params"] = (
                self.cfg.LOSS.POSE_PARAM_WEIGHT * loss_pose_params
            )

        assert "total_loss" not in loss_dict
        loss_dict["total_loss"] = sum(
            v for k, v in loss_dict.items() if k != "total_loss"
        )

        # for k, v in loss_dict.items():
        #     print(f'{k}: {v.item():.3f}', end=' ')
        # import ipdb; ipdb.set_trace()

        return loss_dict
