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
from sam_3d_body.models.modules.mhr_utils import mhr_param_hand_mask, mhr_cont_hand_idxs, mhr_param_hand_idxs
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

        # Hand keypoint indices in MHR70: right hand 21–41, left hand 42–62
        # Total 70 keypoints; hands are indices 21–62 (42 keypoints)
        self.hand_keypoint_indices = list(range(21, 63))  # 21–62 inclusive

        # Get hand weight from config if present, otherwise default to 0.1
        hand_weight = getattr(self.cfg.LOSS, "HAND_WEIGHT", 0.1)

        # Store hand weight; full keypoint count (70 + dense) will be inferred
        # dynamically from tensors during training so we don't assume a fixed
        # dense keypoint length.
        self.hand_weight = hand_weight

        # Debug visualization directory
        self.debug_vis_dir = None

    def forward(self, predictions, batch):
        loss_dict = {}

        B, N = batch["img"].shape[:2]

        pred_mhr = predictions["mhr"]

        if self.cfg.LOSS.JOINTS_3D_WEIGHT > 0:
            pred_joints_3d = predictions["mhr_samples_joints_3d"]
            gt_joints_3d = batch["joints_3d"]
            visibility = batch["visibility"]
            visibility = visibility.unsqueeze(1).expand(-1, pred_joints_3d.shape[1], -1)
            gt_joints_3d = gt_joints_3d.unsqueeze(1).expand(-1, pred_joints_3d.shape[1], -1, -1)

            joints_3d_loss = self.mse_loss(pred_joints_3d, gt_joints_3d)
            joints_3d_loss = joints_3d_loss.mean(dim=-1)
            joints_3d_loss = joints_3d_loss * visibility
            joints_3d_loss = joints_3d_loss.mean()
            loss_dict["loss_joints_3d"] = (self.cfg.LOSS.JOINTS_3D_WEIGHT * joints_3d_loss)

        if self.cfg.LOSS.JOINTS_2D_WEIGHT > 0:
            pred_joints_2d = predictions["mhr_samples_joints_2d_cropped"]
            gt_joints_2d = batch["joints_2d"]
            visibility = batch["visibility"]
            visibility = visibility.unsqueeze(1).expand(-1, pred_joints_2d.shape[1], -1)
            gt_joints_2d = gt_joints_2d.unsqueeze(1).expand(-1, pred_joints_2d.shape[1], -1, -1)
            joints_2d_loss = self.kp2d_loss(pred_joints_2d, gt_joints_2d)
            joints_2d_loss = joints_2d_loss.mean(dim=-1)
            joints_2d_loss = joints_2d_loss * visibility
            joints_2d_loss = joints_2d_loss.mean()
            loss_dict["loss_joints_2d"] = (self.cfg.LOSS.JOINTS_2D_WEIGHT * joints_2d_loss)


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

        if self.cfg.LOSS.POSE_PARAM_WEIGHT > 0:
            # fmt: off
            all_param_3dof_rot_idxs = torch.LongTensor([(0, 2, 4), (6, 8, 10), (12, 13, 14), (15, 16, 17), (18, 19, 20), (21, 22, 23), (24, 25, 26), (27, 28, 29), (34, 35, 36), (37, 38, 39), (44, 45, 46), (53, 54, 55), (64, 65, 66), (85, 69, 73), (86, 70, 79), (87, 71, 82), (88, 72, 76), (91, 92, 93), (112, 96, 100), (113, 97, 106), (114, 98, 109), (115, 99, 103), (130, 131, 132)])
            all_param_1dof_rot_idxs = torch.LongTensor([1, 3, 5, 7, 9, 11, 30, 31, 32, 33, 40, 41, 42, 43, 47, 48, 49, 50, 51, 52, 56, 57, 58, 59, 60, 61, 62, 63, 67, 68, 74, 75, 77, 78, 80, 81, 83, 84, 89, 90, 94, 95, 101, 102, 104, 105, 107, 108, 110, 111, 116, 117, 118, 119, 120, 121, 122, 123])
            # fmt: on
            num_3dof_angles = len(all_param_3dof_rot_idxs) * 3  # 69
            num_1dof_angles = len(all_param_1dof_rot_idxs)  # 58
            body_3dof_idxs = torch.tensor([v for v in all_param_3dof_rot_idxs.flatten() if v not in mhr_param_hand_idxs])
            body_1dof_idxs = torch.tensor([v for v in all_param_1dof_rot_idxs if v not in mhr_param_hand_idxs])

            gt_pose = batch["model_params"][:, 6 : 6 + 133]
            gt_pose[..., mhr_param_hand_mask] = 0
            gt_pose[..., -3:] = 0

            gt_3dof_euler = gt_pose[:, all_param_3dof_rot_idxs.flatten()].unflatten(
                -1, (-1, 3)
            )  # B 23 3

            # Find, amongst the 23 3dof joints, which ones are all zeroed out
            zero_hand_indices_3dof = (gt_3dof_euler.abs().sum(dim=-1) == 0).all(dim=0).nonzero(as_tuple=True)[0]

            gt_3dof_rotmat = roma.euler_to_rotmat("XYZ", gt_3dof_euler)  # B 23 3 3

            gt_3dof_aa = matrix_to_axis_angle(gt_3dof_rotmat)  # B 23 3

            gt_1dof_angles = gt_pose[..., all_param_1dof_rot_idxs]  # B 58

            nonzero_hand_indices_1dof = (gt_1dof_angles.abs() != 0).any(dim=0).nonzero(as_tuple=True)[0]

            pred_pose_euler = pred_mhr["body_pose"]
            pred_3dof_euler = pred_pose_euler[
                :, all_param_3dof_rot_idxs.flatten()
            ].unflatten(
                -1, (-1, 3)
            )  # B 23 3
            
            pred_3dof_rotmat = roma.euler_to_rotmat("XYZ", pred_3dof_euler)  # B 23 3 3

            pred_3dof_aa = matrix_to_axis_angle(pred_3dof_rotmat)  # B 23 3

            pred_1dof_angles = pred_pose_euler[..., all_param_1dof_rot_idxs]  # B 58
            

            pred_var = pred_mhr["pose_uncertainty"]
            if self.cfg.MODEL.FULL_COV == True:
                assert pred_var.shape[-1] == (2 * num_3dof_angles + num_1dof_angles)
                pred_3dof_aa = pred_3dof_aa.flatten(0, 1)
                gt_3dof_aa = gt_3dof_aa.flatten(0, 1)
                cholesky_flat_3dofs = pred_var[:, : 2 * num_3dof_angles]

                cholesky_3dofs = build_tril(
                    rearrange(cholesky_flat_3dofs, "b (j c) -> (b j) c", c=6)
                )

                var_1dofs = pred_var[:, 2 * num_3dof_angles :]
                var_1dofs = var_1dofs[:, nonzero_hand_indices_1dof]
                cholesky_1dofs = torch.sqrt(torch.diag_embed(var_1dofs))


                dist_3dof = MultivariateNormal(
                    pred_3dof_aa, 
                    scale_tril=cholesky_3dofs
                )
                dist_1dof = MultivariateNormal(
                    pred_1dof_angles[:, nonzero_hand_indices_1dof], 
                    scale_tril=cholesky_1dofs
                )
                loss_3dof = -dist_3dof.log_prob(gt_3dof_aa)
                loss_3dof.unflatten(0, (B, 23))[:, zero_hand_indices_3dof] = 0
                loss_1dof = -dist_1dof.log_prob(gt_1dof_angles[:, nonzero_hand_indices_1dof])

                # If any element of loss_3dof or loss_1dof is nan, set it to zero
                if torch.isnan(loss_3dof).any():
                    loss_3dof = torch.where(torch.isnan(loss_3dof), torch.zeros_like(loss_3dof), loss_3dof)
                if torch.isnan(loss_1dof).any():
                    loss_1dof = torch.where(torch.isnan(loss_1dof), torch.zeros_like(loss_1dof), loss_1dof)


            else:
                assert pred_var.shape[-1] == (num_3dof_angles + num_1dof_angles)
                var_3dofs = pred_var[:, :num_3dof_angles].unflatten(-1, (-1, 3))
                var_1dofs = pred_var[:, num_3dof_angles:]

                loss_3dof = self.gaussian_nll_loss(
                    pred_3dof_aa, gt_3dof_aa, var_3dofs
                )
                loss_1dof = self.gaussian_nll_loss(
                    pred_1dof_angles, gt_1dof_angles, var_1dofs
                )   

            loss_dict["loss_pose_3dof"] = (
                self.cfg.LOSS.POSE_PARAM_WEIGHT * loss_3dof.mean()
            )
            loss_dict["loss_pose_1dof"] = (
                self.cfg.LOSS.POSE_PARAM_WEIGHT * loss_1dof.mean()
            )
            loss_dict["loss_pose_params"] = (
                loss_dict["loss_pose_3dof"] + loss_dict["loss_pose_1dof"]
            )

            # # Debug visualization: compare GT and predicted axis-angle rotations
            # if self.training and hasattr(self, 'global_step') and self.global_step % 100 == 0:
            #     self._visualize_axis_angle_comparison(
            #         gt_3dof_aa[0].detach().cpu(),
            #         pred_3dof_aa[0].detach().cpu(),
            #         step=self.global_step if hasattr(self, 'global_step') else 0
            #     )

            # import matplotlib.pyplot as plt

            # fig, axs = plt.subplots(3, 1, figsize=(12, 15))

            # # Subplot 1: Plot the commented block (raw params in Euler)
            # gt_euler_to_plot = batch['model_params'].cpu().detach().numpy()[0, 6: 6+133]
            # pred_euler_to_plot = pred_pose_euler.cpu().detach().numpy()[0]
            # axs[0].scatter(np.arange(len(gt_euler_to_plot)), gt_euler_to_plot, label='GT Euler', s=2)
            # axs[0].scatter(np.arange(len(pred_euler_to_plot)), pred_euler_to_plot, label='Pred Euler', s=2)
            # for i in range(len(gt_euler_to_plot)):
            #     axs[0].plot([i, i], [gt_euler_to_plot[i], pred_euler_to_plot[i]], color='gray', alpha=0.4, linewidth=1)
            # axs[0].set_title('Raw body_pose (Euler params)')
            # axs[0].legend()

            # # Subplot 2: Axis-angle comparison
            # gt_aa_to_plot = gt_3dof_aa[0].flatten().cpu().detach().numpy()
            # pred_aa_to_plot = pred_3dof_aa[0].flatten().cpu().detach().numpy()
            # axs[1].scatter(np.arange(len(gt_aa_to_plot)), gt_aa_to_plot, label='GT Axis-Angle', s=2)
            # axs[1].scatter(np.arange(len(pred_aa_to_plot)), pred_aa_to_plot, label='Pred Axis-Angle', s=2)
            # for i in range(len(gt_aa_to_plot)):
            #     axs[1].plot([i, i], [gt_aa_to_plot[i], pred_aa_to_plot[i]], color='gray', alpha=0.4, linewidth=1)
            # axs[1].set_title('Body pose: Axis-angle (from rotmat)')
            # axs[1].legend()

            # # Subplot 3: 1DoF joint angles comparison
            # gt_1dof_to_plot = gt_1dof_angles[0].cpu().detach().numpy()
            # pred_1dof_to_plot = pred_1dof_angles[0].cpu().detach().numpy()
            # axs[2].scatter(np.arange(len(gt_1dof_to_plot)), gt_1dof_to_plot, label='GT 1DoF Angles', s=2)
            # axs[2].scatter(np.arange(len(pred_1dof_to_plot)), pred_1dof_to_plot, label='Pred 1DoF Angles', s=2)
            # for i in range(len(gt_1dof_to_plot)):
            #     axs[2].plot([i, i], [gt_1dof_to_plot[i], pred_1dof_to_plot[i]], color='gray', alpha=0.4, linewidth=1)
            # axs[2].set_title('1DoF Body Joint Angles')
            # axs[2].legend()

            # fig.tight_layout()
            # plt.savefig('axis_angle_comparison.png')
            # plt.close(fig)
            # import ipdb; ipdb.set_trace()

        assert "total_loss" not in loss_dict
        loss_dict["total_loss"] = sum(
            v for k, v in loss_dict.items() if k != "total_loss"
        )

        if torch.isnan(loss_dict['total_loss']):
            loss_dict['total_loss'] = torch.zeros_like(loss_dict['total_loss'])

        # for k, v in loss_dict.items():
        #     print(f"{k}: {v.item():.3f}", end=" ")
        # import ipdb; ipdb.set_trace()

        return loss_dict

    def _visualize_axis_angle_comparison(self, gt_aa, pred_aa, step=0):
        """
        Visualize comparison between GT and predicted axis-angle rotations.
        Shows all angles, pairing GT and Pred at the same center for each joint.

        Args:
            gt_aa: [N, 3] tensor of GT axis-angle rotations
            pred_aa: [N, 3] tensor of predicted axis-angle rotations
            step: current training step
        """
        # Set up debug visualization directory
        if self.debug_vis_dir is None:
            if hasattr(self, "logger") and self.logger is not None:
                if hasattr(self.logger, "log_dir") and self.logger.log_dir:
                    self.debug_vis_dir = os.path.join(
                        self.logger.log_dir, "debug_rotations"
                    )
                else:
                    self.debug_vis_dir = "./debug_rotations"
            else:
                self.debug_vis_dir = "./debug_rotations"
            os.makedirs(self.debug_vis_dir, exist_ok=True)

        # Convert to numpy
        gt_aa_np = gt_aa.numpy() if isinstance(gt_aa, torch.Tensor) else gt_aa
        pred_aa_np = pred_aa.numpy() if isinstance(pred_aa, torch.Tensor) else pred_aa

        num_joints = gt_aa_np.shape[0]

        # Convert axis-angle to rotation matrices for visualization
        gt_aa_t = torch.from_numpy(gt_aa_np).float()
        pred_aa_t = torch.from_numpy(pred_aa_np).float()

        gt_rotmats = axis_angle_to_matrix(gt_aa_t)  # [N, 3, 3]
        pred_rotmats = axis_angle_to_matrix(pred_aa_t)  # [N, 3, 3]

        # Compute rotation axes (normalized axis-angle vectors) and angles (magnitudes)
        gt_axes = gt_aa_np.copy()
        pred_axes = pred_aa_np.copy()
        gt_angles = np.linalg.norm(gt_aa_np, axis=1, keepdims=True)
        pred_angles = np.linalg.norm(pred_aa_np, axis=1, keepdims=True)

        # Normalize to get unit rotation axes (handle zero rotations)
        gt_axes_norm = np.where(gt_angles > 1e-6, gt_axes / gt_angles, gt_axes)
        pred_axes_norm = np.where(
            pred_angles > 1e-6, pred_axes / pred_angles, pred_axes
        )

        # Scale by rotation angle for visualization (so longer arrows = larger rotations)
        scale_factor = 0.5  # Scale factor for better visualization
        gt_angles_1d = gt_angles.squeeze()  # (N,)
        pred_angles_1d = pred_angles.squeeze()  # (N,)
        gt_axes_scaled = gt_axes_norm * (
            gt_angles_1d[:, np.newaxis] * scale_factor + 0.1
        )
        pred_axes_scaled = pred_axes_norm * (
            pred_angles_1d[:, np.newaxis] * scale_factor + 0.1
        )

        # Calculate grid dimensions for subplots
        cols = min(5, num_joints)  # Max 5 columns
        rows = (num_joints + cols - 1) // cols  # Ceiling division

        # Create figure with subplots - one 3D plot per joint
        fig = plt.figure(figsize=(4 * cols, 4 * rows))

        colors = plt.cm.tab10(np.linspace(0, 1, num_joints))

        # Find max range for consistent scaling
        max_range = (
            max(np.abs(gt_axes_scaled).max(), np.abs(pred_axes_scaled).max()) * 1.2
        )

        for i in range(num_joints):
            ax = fig.add_subplot(rows, cols, i + 1, projection="3d")

            # GT rotation axis (solid) - same center as pred
            gt_label = f"GT: {gt_angles[i, 0]:.3f} rad"
            ax.quiver(
                0,
                0,
                0,
                gt_axes_scaled[i, 0],
                gt_axes_scaled[i, 1],
                gt_axes_scaled[i, 2],
                color=colors[i],
                arrow_length_ratio=0.2,
                linewidth=3,
                alpha=0.8,
                label=gt_label,
            )

            # Pred rotation axis (dashed) - same center as GT
            pred_label = f"Pred: {pred_angles[i, 0]:.3f} rad"
            ax.quiver(
                0,
                0,
                0,
                pred_axes_scaled[i, 0],
                pred_axes_scaled[i, 1],
                pred_axes_scaled[i, 2],
                color=colors[i],
                arrow_length_ratio=0.2,
                linewidth=3,
                linestyle="--",
                alpha=0.8,
                label=pred_label,
            )

            # Set equal aspect ratio and limits
            ax.set_xlim([-max_range, max_range])
            ax.set_ylim([-max_range, max_range])
            ax.set_zlim([-max_range, max_range])
            ax.set_xlabel("X", fontsize=8)
            ax.set_ylabel("Y", fontsize=8)
            ax.set_zlabel("Z", fontsize=8)
            ax.set_title(f"Joint {i}", fontsize=10)
            ax.legend(fontsize=8, loc="upper right")

        plt.suptitle(
            f"Axis-Angle Rotation Comparison (Step {step})\nArrow length ∝ rotation angle",
            fontsize=14,
        )
        plt.tight_layout()

        # Save figure
        save_path = os.path.join(
            self.debug_vis_dir, f"rotation_comparison_step_{step:06d}.png"
        )
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

        # Also print numerical comparison
        gt_mags = np.linalg.norm(gt_aa_np, axis=1)
        pred_mags = np.linalg.norm(pred_aa_np, axis=1)
        print(f"\n[Debug] Rotation comparison at step {step}:")
        print(f"GT axis-angle magnitudes: {gt_mags}")
        print(f"Pred axis-angle magnitudes: {pred_mags}")
        print(f"Magnitude differences: {np.abs(gt_mags - pred_mags)}")
        print(f"Mean magnitude error: {np.mean(np.abs(gt_mags - pred_mags)):.4f}")
        print(f"Saved visualization to: {save_path}")
