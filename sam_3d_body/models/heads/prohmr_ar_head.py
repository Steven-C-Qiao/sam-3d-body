import torch
import torch.nn as nn
from torch.amp import autocast
from yacs.config import CfgNode
from typing import Optional, Dict, Tuple
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle

from sam_3d_body.models.modules.mhr_utils import (
    batch9Dfrom6D,
    batchXYZfrom6D,
    compact_cont_to_model_params_body,
    convert_mhr_params_to_flow_params,
    convert_flow_samples_to_mhr_params,
    convert_pose_cont_to_flow_context,
    so3_compose_aa,
    so3_residual_aa,
    scale_indices,
)

from nflows.flows import ConditionalGlow

from sam_3d_body.models.modules.coupling_layers import (
    ConditionalGlowAffine,
    ConditionalGlowUnclampedAffine,
    ConditionalGlowSpline,
)


class NFARHead(nn.Module):
    def __init__(self, cfg: CfgNode):
        super(NFARHead, self).__init__()

        self.model_glob_rot = getattr(cfg.MODEL, "MODEL_GLOB_ROT", False)
        self.model_shape = getattr(cfg.MODEL, "MODEL_SHAPE", True)
        self.model_scale = getattr(cfg.MODEL, "MODEL_SCALE", True)
        self.model_cam = getattr(cfg.MODEL, "MODEL_CAM", False)

        self.num_3dof_comps = 39
        self.num_1dof_comps = 34
        self.num_shape_comps = 45 if self.model_shape else 0
        self.num_scale_comps = 10 if self.model_scale else 0
        self.num_glob_rot_comps = 3 if self.model_glob_rot else 0
        self.num_cam_comps = 3 if self.model_cam else 0

        # Factorised v2 autoregressive model:
        #   Stage 1: p(Δβ | c, μβ) over shape+scale residuals.
        #   Stage 2: p(Δθ, Δcam? | c, μθ, μcam?, Δβ) over pose (+camera) residuals.
        self.beta_dim = self.num_shape_comps + self.num_scale_comps
        # theta ordering: [3dof(39) | 1dof(34) | glob_rot?(3) | cam?(3)]
        self.theta_dim = self.num_3dof_comps + self.num_1dof_comps + self.num_glob_rot_comps + self.num_cam_comps
        self.flow_dim = self.theta_dim + self.beta_dim
        flow_num_layers = cfg.MODEL.FLOW_NUM_LAYERS
        flow_dropout = cfg.MODEL.FLOW_DROPOUT

        flow_config_beta = {
            "flow_dim": self.beta_dim,
            "num_layers": flow_num_layers,
            "context_features": 2048,
            "layer_hidden_features": 1024,
            "layer_depth": 2,
            "dropout_probability": flow_dropout,
        }
        flow_config_theta = {
            "flow_dim": self.theta_dim,
            "num_layers": flow_num_layers,
            "context_features": 2048,
            "layer_hidden_features": 1024,
            "layer_depth": 2,
            "dropout_probability": flow_dropout,
        }

        flow_coupling = getattr(cfg.MODEL, "FLOW_COUPLING", "additive").lower()
        flow_extra_kwargs = {}
        if flow_coupling == "additive":
            flow_cls = ConditionalGlow
        elif flow_coupling == "affine":
            flow_cls = ConditionalGlowUnclampedAffine
        elif flow_coupling == "clamped_affine":
            flow_cls = ConditionalGlowAffine
        elif flow_coupling == "rq_spline":
            flow_cls = ConditionalGlowSpline
            flow_extra_kwargs = {
                "num_bins": cfg.MODEL.FLOW_SPLINE_NUM_BINS,
                "tails": cfg.MODEL.FLOW_SPLINE_TAILS,
                "tail_bound": cfg.MODEL.FLOW_SPLINE_TAIL_BOUND,
            }
        else:
            raise ValueError(
                f"Unsupported MODEL.FLOW_COUPLING='{flow_coupling}'. "
                "Expected one of: ['additive', 'affine', 'clamped_affine', 'rq_spline']."
            )
        self.flow_coupling = flow_coupling

        # BatchNorm inside each coupling's ResidualNet makes the flow's forward
        # bijection differ between train (batch stats) and eval (running stats),
        # which breaks log_prob calibration at inference. Disable it — ActNorm
        # already handles per-channel normalisation and is train/eval-consistent.
        self.flow_beta = flow_cls(
            flow_config_beta["flow_dim"],
            flow_config_beta["layer_hidden_features"],
            flow_config_beta["num_layers"],
            flow_config_beta["layer_depth"],
            dropout_probability=flow_config_beta["dropout_probability"],
            context_features=flow_config_beta["context_features"],
            batch_norm_within_layers=False,
            **flow_extra_kwargs,
        )
        self.flow_theta = flow_cls(
            flow_config_theta["flow_dim"],
            flow_config_theta["layer_hidden_features"],
            flow_config_theta["num_layers"],
            flow_config_theta["layer_depth"],
            dropout_probability=flow_config_theta["dropout_probability"],
            context_features=flow_config_theta["context_features"],
            batch_norm_within_layers=False,
            **flow_extra_kwargs,
        )
        self.num_samples = cfg.MODEL.NUM_SAMPLES
        self.shape_perturb_scale = cfg.MODEL.SHAPE_PERTURB_SCALE
        self.scale_perturb_scale = cfg.MODEL.SCALE_PERTURB_SCALE
        self.beta_perturb_detach = cfg.MODEL.BETA_PERTURB_DETACH

        # Per-dimension GT std for mode-2 perturbation (shape: 45D, scale: 10D selected).
        # Loaded once at init; used as noise scale multiplied by the respective perturb scale.
        if self.shape_perturb_scale > 0 or self.scale_perturb_scale > 0:
            stats = torch.load(cfg.MODEL.BETA_PERTURB_STATS_PATH, map_location="cpu", weights_only=True)
            self.register_buffer("_shape_perturb_std", stats["shape_std"].float())  # (45,)
            self.register_buffer("_scale_perturb_std", stats["scale_std"].float())  # (10,)
        else:
            self._shape_perturb_std = None
            self._scale_perturb_std = None

        # Stage 1 context: [flow_context, shape_mean, scale_mean_selected]
        context_beta_dim = 1024 + 45 + 10
        self.beta_context_proj = nn.Linear(context_beta_dim, 2048)

        # Stage 2 context: [flow_context, shape_sample, scale_sample_selected, aa_3dofs, params_1dofs, (pred_cam?)]
        context_theta_dim = 1024 + 45 + 10 + 39 + 34 + (3 if self.model_cam else 0)
        self.theta_context_proj = nn.Linear(context_theta_dim, 2048)

        self.register_buffer("initialized_beta", torch.tensor(False))
        self.register_buffer("initialized_theta", torch.tensor(False))

    def initialize_actnorm(self, batch: Dict, mean_pred: Dict, flow_context: torch.Tensor):
        # Compute GT flow params and rotation matrices.
        gt_flow_params, gt_rotmats = convert_mhr_params_to_flow_params(
            batch["model_params"],
            batch["shape_params"],
            include_global_rot=self.model_glob_rot,
            include_shape=self.model_shape,
            include_scale=self.model_scale,
            flip_global_rot=True,
            return_rotmats=True,
        )

        # Mean prediction via direct 6D→AA path (no euler roundtrip bias).
        pose_params_mean = convert_pose_cont_to_flow_context(
            mean_pred["pred_pose_raw"][:, 6:]
        )
        beta_parts = []
        if self.model_shape:
            beta_parts.append(mean_pred["shape"])
        if self.model_scale:
            beta_parts.append(mean_pred["scale_68D"][..., scale_indices])
        mean_beta = torch.cat(beta_parts, dim=-1) if beta_parts else None

        # Piecewise residual: SO(3) for 3DOF + glob_rot, additive for beta + 1DOF.
        # Beta residual (additive)
        beta_residual = gt_flow_params[..., : self.beta_dim] - mean_beta

        # Theta residual: [3dof(39), 1dof(34), glob_rot?(3)]
        pose_3dof_residual = so3_residual_aa(
            pose_params_mean["rotmat_3dofs"], gt_rotmats["pose_3dof_rotmat"]
        )
        offset_1dof = self.beta_dim + 39
        pose_1dof_residual = (
            gt_flow_params[..., offset_1dof : offset_1dof + 34]
            - pose_params_mean["params_1dofs"]
        )
        theta_parts = [pose_3dof_residual, pose_1dof_residual]
        if self.model_glob_rot:
            mean_glob_rotmat = batch9Dfrom6D(
                mean_pred["pred_pose_raw"][:, :6]
            ).unflatten(-1, (3, 3))
            glob_rot_residual = so3_residual_aa(
                mean_glob_rotmat.unsqueeze(-3),
                gt_rotmats["glob_rotmat"].unsqueeze(-3),
            )
            theta_parts.append(glob_rot_residual)
        theta_residual_no_cam = torch.cat(theta_parts, dim=-1)

        if self.model_cam:
            cam_residual = batch["gt_pred_cam"] - mean_pred["pred_cam"]
            theta_residual = torch.cat([theta_residual_no_cam, cam_residual], dim=-1)
        else:
            theta_residual = theta_residual_no_cam

        shape_mean = mean_pred["shape"]
        scale_mean = mean_pred["scale_68D"]
        pose_params = convert_pose_cont_to_flow_context(
            mean_pred["pred_pose_raw"][:, 6:]
        )
        aa_3dofs = pose_params["aa_3dofs"]
        params_1dofs = pose_params["params_1dofs"]

        context_beta = self.beta_context_proj(
            torch.cat(
                [
                    flow_context,
                    shape_mean,
                    scale_mean[..., scale_indices],
                ],
                dim=-1,
            )
        )

        # Pose context uses GT shape (teacher forcing), mirroring nf_loss.py.
        shape_residual_true = beta_residual[..., : self.num_shape_comps]
        scale_residual_true = beta_residual[..., self.num_shape_comps :]
        shape_sample_true = shape_mean + shape_residual_true
        scale_sample_selected_true = scale_mean[..., scale_indices] + scale_residual_true

        context_theta_parts = [
            flow_context,
            shape_sample_true,
            scale_sample_selected_true,
            aa_3dofs,
            params_1dofs,
        ]
        if self.model_cam:
            context_theta_parts.append(mean_pred["pred_cam"])
        context_theta= self.theta_context_proj(torch.cat(context_theta_parts, dim=-1))

        with torch.no_grad():
            _, _ = self.flow_beta.log_prob(beta_residual, context_beta)
            _, _ = self.flow_theta.log_prob(theta_residual, context_theta)
            self.initialized_beta |= True
            self.initialized_theta |= True

        # nflows ActNorm._initialize sets self.initialized.data = torch.tensor(True)
        # which creates a CPU tensor, breaking DDP buffer sync. Fix by moving any
        # CPU buffers back to the correct device.
        device = beta_residual.device
        for module in [self.flow_beta, self.flow_theta]:
            for name, buf in module.named_buffers():
                if buf.device != device:
                    buf.data = buf.data.to(device)

        print('initialized actnorm')

    @autocast("cuda", enabled=False)
    def log_prob(
        self,
        params: torch.Tensor,
        flow_context_beta: torch.Tensor,
        flow_context_theta: torch.Tensor,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Compute summed conditional log-probability for the factorised model.

        Args:
            params: Residual parameters in flow ordering:
                [beta (shape(45) + scale(10)), theta (3dof(39) + 1dof(34) + glob_rot?(3) + cam?(3))]
                with shapes matching `self.flow_dim`.
            flow_context_beta: [B, 2048] context for stage 1.
            flow_context_theta: [B, 2048] context for stage 2.
        """
        if params.shape[-1] != self.flow_dim:
            raise ValueError(
                f"Expected params last dim {self.flow_dim}, got {params.shape[-1]}"
            )

        # Split: [beta (shape+scale), theta (3dof+1dof+glob_rot?+cam?)].
        beta_params = params[..., : self.beta_dim]
        theta_params = params[..., self.beta_dim :]

        if theta_params.shape[-1] != self.theta_dim:
            raise ValueError(
                "Internal split mismatch: "
                f"expected theta_dim={self.theta_dim}, got {theta_params.shape[-1]}"
            )

        log_prob_beta, z_beta = self.flow_beta.log_prob(
            inputs=beta_params, context=flow_context_beta
        )
        log_prob_theta, z_theta = self.flow_theta.log_prob(
            inputs=theta_params, context=flow_context_theta
        )

        log_prob_total = log_prob_beta + log_prob_theta
        return log_prob_total, (z_beta, z_theta)


    @autocast("cuda", enabled=False)
    def sample_theta_given_beta(
        self,
        flow_context: torch.Tensor,
        mean_pred: Dict,
        shape_samples: torch.Tensor,
        scale_samples_68D: torch.Tensor,
    ) -> Dict:
        """Stage 2 of the autoregressive NF: sample pose/glob_rot/cam residuals
        conditioned on a given ``(shape_samples, scale_samples_68D)``.

        Separated out so the same stage-2 context + flow path can be reused
        with the *merged* shape/scale (from multi-view merging) in place of
        the stage-1 sample — yielding a merged-consistent camera.

        Args:
            flow_context:       (B, 1024) raw flow context, before projection
            mean_pred:          dict with ``pred_pose_raw`` (B, 6+pose_cont)
                                 and (if ``self.model_cam``) ``pred_cam`` (B, 3)
            shape_samples:      (B, N, 45) shape samples to condition on
            scale_samples_68D:  (B, N, 68) scale samples to condition on

        Returns dict with:
            ``theta_samples_residual``: (B, N, theta_dim)
            ``theta_log_prob``:         (B, N)
            ``pose_samples``:           MHR-compatible pose params (B, N, …)
            ``global_rot_samples``:     (B, N, 3) Euler XYZ or ``None``
            ``cam_samples``:            (B, N, 3) ``pred_cam`` or ``None``
            ``context_theta``:          (B*N, 2048) post-projection context
        """
        B, N = shape_samples.shape[0], shape_samples.shape[1]

        pose_mean_cont = mean_pred["pred_pose_raw"][:, 6:]
        pose_params_mhr = compact_cont_to_model_params_body(pose_mean_cont)
        pose_params = convert_pose_cont_to_flow_context(pose_mean_cont)
        aa_3dofs = pose_params["aa_3dofs"]
        params_1dofs = pose_params["params_1dofs"]

        flow_context_expanded = flow_context.unsqueeze(1).repeat(1, N, 1)
        aa_3dofs_expanded = aa_3dofs.unsqueeze(1).repeat(1, N, 1)
        params_1dofs_expanded = params_1dofs.unsqueeze(1).repeat(1, N, 1)

        context_theta_parts = [
            flow_context_expanded,
            shape_samples,
            scale_samples_68D[..., scale_indices],
            aa_3dofs_expanded,
            params_1dofs_expanded,
        ]
        if self.model_cam:
            pred_cam_expanded = mean_pred["pred_cam"].unsqueeze(1).repeat(1, N, 1)
            context_theta_parts.append(pred_cam_expanded)
        context_theta = self.theta_context_proj(
            torch.cat(context_theta_parts, dim=-1).reshape(B * N, -1)
        )

        theta_residual_flat, theta_log_prob_flat, _ = self.flow_theta.sample_and_log_prob(
            1, context=context_theta,
        )
        theta_samples_residual = theta_residual_flat.squeeze(1).reshape(B, N, self.theta_dim)
        theta_log_prob = theta_log_prob_flat.squeeze(1).reshape(B, N)

        pose_3dof_residual_samples = theta_samples_residual[..., : self.num_3dof_comps]
        pose_1dof_residual_samples = theta_samples_residual[
            ..., self.num_3dof_comps : self.num_3dof_comps + self.num_1dof_comps
        ]
        aa_3dof_samples = so3_compose_aa(
            aa_3dofs.unsqueeze(1), pose_3dof_residual_samples
        )
        params_1dofs_samples = (
            params_1dofs.unsqueeze(1).repeat(1, N, 1) + pose_1dof_residual_samples
        )
        pose_samples = convert_flow_samples_to_mhr_params(
            aa_3dof_samples, params_1dofs_samples, pose_params_mhr
        )

        if self.model_glob_rot:
            gr_offset = self.num_3dof_comps + self.num_1dof_comps
            glob_rot_aa_residual = theta_samples_residual[
                ..., gr_offset : gr_offset + self.num_glob_rot_comps
            ]
            glob_rot_6d_mean = mean_pred["pred_pose_raw"][:, :6]
            glob_rot_mean_mat = batch9Dfrom6D(glob_rot_6d_mean).unflatten(-1, (3, 3))
            glob_rot_delta_mat = axis_angle_to_matrix(glob_rot_aa_residual)
            glob_rot_mat_samples = glob_rot_mean_mat.unsqueeze(1) @ glob_rot_delta_mat
            glob_rot_6d_samples = torch.cat(
                [glob_rot_mat_samples[..., :, 0], glob_rot_mat_samples[..., :, 1]],
                dim=-1,
            )
            glob_rot_euler_samples = batchXYZfrom6D(glob_rot_6d_samples)
        else:
            glob_rot_euler_samples = None

        if self.model_cam:
            cam_offset = (
                self.num_3dof_comps + self.num_1dof_comps + self.num_glob_rot_comps
            )
            cam_residual_samples = theta_samples_residual[
                ..., cam_offset : cam_offset + 3
            ]
            cam_samples = mean_pred["pred_cam"].unsqueeze(1) + cam_residual_samples
        else:
            cam_samples = None

        return {
            "theta_samples_residual": theta_samples_residual,
            "theta_log_prob": theta_log_prob,
            "pose_samples": pose_samples,
            "global_rot_samples": glob_rot_euler_samples,
            "cam_samples": cam_samples,
            "context_theta": context_theta,
        }

    def forward(
        self,
        flow_context: torch.Tensor,
        mean_pred: Dict,
        num_samples: int = 0,
        batch: Dict = None,
    ) -> Dict:
        """
        Given context and mean predictions, compute residual uncertainty by NF
        sampling needs to be handled here, instead of in model forward
        """
        if num_samples <= 0:
            num_samples = self.num_samples

        B, N = flow_context.shape[0], num_samples

        shape_mean = mean_pred["shape"]  # B, 45
        scale_mean = mean_pred["scale_68D"]  # B, 68


        # ----------------------------------------------------------------------
        # Stage 1: p(Δβ | c, μβ) — shape+scale residuals.
        # ----------------------------------------------------------------------
        beta_context = self.beta_context_proj(
            torch.cat(
                [
                    flow_context,
                    shape_mean,
                    scale_mean[..., scale_indices],
                ],
                dim=-1,
            )
        )

        if (not self.initialized_beta.item()) or (not self.initialized_theta.item()):
            self.initialize_actnorm(batch, mean_pred=mean_pred, flow_context=flow_context)
            print("Initialised ActNorm")

        beta_residual_samples, beta_log_prob, beta_z = self.flow_beta.sample_and_log_prob(
            N,
            context=beta_context,
        )


        # Mode-2: perturb stage-1 beta samples before conditioning stage-2.
        if self.training and (self.shape_perturb_scale > 0 or self.scale_perturb_scale > 0):
            shape_part = beta_residual_samples[..., : self.num_shape_comps]
            scale_part = beta_residual_samples[..., self.num_shape_comps :]
            if self.shape_perturb_scale > 0:
                shape_part = shape_part + torch.randn_like(shape_part) * self._shape_perturb_std * self.shape_perturb_scale
            if self.scale_perturb_scale > 0:
                scale_part = scale_part + torch.randn_like(scale_part) * self._scale_perturb_std * self.scale_perturb_scale
            beta_residual_for_stage2 = torch.cat([shape_part, scale_part], dim=-1)
            if self.beta_perturb_detach:
                beta_residual_for_stage2 = beta_residual_for_stage2.detach()
        else:
            beta_residual_for_stage2 = beta_residual_samples

        # beta_residual_for_stage2: [B, N, beta_dim]
        shape_residual_samples = beta_residual_for_stage2[..., : self.num_shape_comps]
        scale_residual_samples = beta_residual_for_stage2[..., self.num_shape_comps :]

        shape_samples = shape_mean.unsqueeze(1).repeat(1, N, 1)
        if self.num_shape_comps > 0:
            shape_samples = shape_samples + shape_residual_samples

        scale_samples_68D = scale_mean.unsqueeze(1).repeat(1, N, 1)
        if self.model_scale and self.num_scale_comps > 0:
            scale_samples_68D[..., scale_indices] = (
                scale_samples_68D[..., scale_indices] + scale_residual_samples
            )

        # ----------------------------------------------------------------------
        #  Stage 2: p(Δθ | c, μθ, Δβ) — pose residuals conditioned on sampled β.
        # ----------------------------------------------------------------------
        stage2 = self.sample_theta_given_beta(
            flow_context=flow_context,
            mean_pred=mean_pred,
            shape_samples=shape_samples,
            scale_samples_68D=scale_samples_68D,
        )
        theta_samples_residual = stage2["theta_samples_residual"]
        theta_log_prob = stage2["theta_log_prob"]
        pose_samples = stage2["pose_samples"]
        glob_rot_euler_samples = stage2["global_rot_samples"]
        cam_samples = stage2["cam_samples"]
        context_theta = stage2["context_theta"]

        
        # # DEBUG: override samples with GT residual (SO(3) Lie algebra composition).
        # if batch is not None and "model_params" in batch:
        #     gt_flow_params, gt_rotmats = convert_mhr_params_to_flow_params(
        #         batch["model_params"], batch["shape_params"],
        #         include_global_rot=self.model_glob_rot,
        #         include_shape=self.model_shape,
        #         include_scale=self.model_scale,
        #         flip_global_rot=True,
        #         return_rotmats=True,
        #     )
        #     # Piecewise GT residual: SO(3) for 3DOF+glob_rot, additive for beta+1DOF.
        #     _beta = []
        #     if self.model_shape:
        #         _beta.append(mean_pred["shape"])
        #     if self.model_scale:
        #         _beta.append(mean_pred["scale_68D"][..., scale_indices])
        #     mean_beta = torch.cat(_beta, dim=-1)
        #     gt_beta_residual = gt_flow_params[..., : self.beta_dim] - mean_beta
        #     gt_pose_3dof_residual = so3_residual_aa(
        #         pose_params["rotmat_3dofs"], gt_rotmats["pose_3dof_rotmat"]
        #     )  # pose_params from convert_pose_cont_to_flow_context above
        #     offset_1dof = self.beta_dim + 39
        #     gt_pose_1dof_residual = (
        #         gt_flow_params[..., offset_1dof : offset_1dof + 34]
        #         - pose_params["params_1dofs"]
        #     )
        #     gt_residual = torch.cat([gt_beta_residual, gt_pose_3dof_residual, gt_pose_1dof_residual], dim=-1)

        #     # Override shape samples with GT
        #     gt_shape_residual = gt_beta_residual[..., : self.num_shape_comps]
        #     shape_samples = (shape_mean + gt_shape_residual).unsqueeze(1).expand(-1, N, -1)

        #     # Override scale samples with GT
        #     gt_scale_residual = gt_beta_residual[..., self.num_shape_comps :]
        #     scale_samples_68D = scale_mean.unsqueeze(1).repeat(1, N, 1)
        #     scale_samples_68D[..., scale_indices] = (
        #         scale_mean[..., scale_indices] + gt_scale_residual
        #     ).unsqueeze(1).expand(-1, N, -1)

        #     # Override pose samples with GT (SO(3) composition)
        #     gt_aa_3dof_samples = so3_compose_aa(aa_3dofs, gt_pose_3dof_residual).unsqueeze(1).expand(-1, N, -1)
        #     gt_params_1dofs_samples = (params_1dofs + gt_pose_1dof_residual).unsqueeze(1).expand(-1, N, -1)
        #     gt_pose_130D = batch["model_params"][:, 6:-68]
        #     gt_pose_133D = torch.cat([gt_pose_130D, torch.zeros_like(gt_pose_130D[:, :3])], dim=-1)
        #     pose_samples = convert_flow_samples_to_mhr_params(gt_aa_3dof_samples, gt_params_1dofs_samples, gt_pose_133D)

        #     # Override global rotation samples with GT (SO(3) composition)
        #     if self.model_glob_rot:
        #         gt_glob_rot_residual = so3_residual_aa(
        #             batch9Dfrom6D(mean_pred["pred_pose_raw"][:, :6]).unflatten(-1, (3, 3)).unsqueeze(-3),
        #             gt_rotmats["glob_rotmat"].unsqueeze(-3),
        #         )
        #         gt_glob_rot_aa = so3_compose_aa(
        #             matrix_to_axis_angle(batch9Dfrom6D(mean_pred["pred_pose_raw"][:, :6]).unflatten(-1, (3, 3))),
        #             gt_glob_rot_residual,
        #         )
        #         gt_glob_rot_mat = axis_angle_to_matrix(gt_glob_rot_aa)
        #         gt_glob_6d = torch.cat([gt_glob_rot_mat[..., :, 0], gt_glob_rot_mat[..., :, 1]], dim=-1)
        #         glob_rot_euler_samples = batchXYZfrom6D(gt_glob_6d).unsqueeze(1).expand(-1, N, -1)
        #         gt_residual = torch.cat([gt_residual, gt_glob_rot_residual], dim=-1)

        #     # Override camera samples with GT
        #     if self.model_cam and "gt_pred_cam" in batch:
        #         gt_cam_residual = batch["gt_pred_cam"] - mean_pred["pred_cam"]
        #         cam_samples = batch["gt_pred_cam"].unsqueeze(1).expand(-1, N, -1)
        #     elif self.model_cam:
        #         cam_samples = mean_pred["pred_cam"].unsqueeze(1).expand(-1, N, -1)
        #     else:
        #         cam_samples = None

        #     # Rebuild full residual for the samples tensor
        #     if self.model_cam and "gt_pred_cam" in batch:
        #         samples = torch.cat([gt_residual, gt_cam_residual], dim=-1)
        #     else:
        #         samples = gt_residual
        #     samples = samples.unsqueeze(1).expand(-1, N, -1)

        beta_log_prob = beta_log_prob.reshape(B, N)
        log_prob = beta_log_prob + theta_log_prob

        # Full residual vector.
        # Ordering: [beta (shape(45) + scale(10)), theta (3dof(39) + 1dof(34) + glob_rot?(3) + cam?(3))].
        samples = torch.cat([beta_residual_samples, theta_samples_residual], dim=-1)

        ret = {
            "log_prob": log_prob,
            "log_prob_beta": beta_log_prob,
            "log_prob_theta": theta_log_prob,
            "samples": samples,
            "shape_samples": shape_samples,
            "scale_samples": scale_samples_68D,
            "pose_samples": pose_samples,
            "global_rot_samples": glob_rot_euler_samples,
            "cam_samples": cam_samples,
            "flow_context_beta": beta_context,
            "flow_context_theta": context_theta.reshape(B, N, -1),
            "flow_context_raw": flow_context,
        }

        return ret

