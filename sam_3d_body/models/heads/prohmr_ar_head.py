import roma
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.amp import autocast
from yacs.config import CfgNode
from typing import Optional, Dict, Tuple
from pytorch3d.transforms import (
    axis_angle_to_matrix,
    matrix_to_axis_angle,
)

from sam_3d_body.models.modules.mhr_utils import (
    batch9Dfrom6D,
    compact_cont_to_model_params_body,
    convert_mhr_params_to_flow_params,
    mhr_param_hand_mask,
    all_param_3dof_rot_idxs,
    all_param_1dof_rot_idxs,
    all_param_1dof_trans_idxs,
    all_param_3dof_rot_idxs_except_hands,
    all_param_1dof_rot_idxs_except_hands,
    indices_3dof,
    indices_1dof,
    scale_indices,
)

from nflows.distributions.normal import StandardNormal
from nflows.flows import ConditionalGlow
from nflows.flows.base import Flow
from nflows.nn import nets as nets
from nflows.transforms.base import CompositeTransform
from nflows.transforms.coupling import AffineCouplingTransform
from nflows.transforms.lu import LULinear
from nflows.transforms.normalization import ActNorm


class ClampedAffineCouplingTransform(AffineCouplingTransform):
    """Affine coupling with explicit log-scale clamp."""

    def _scale_and_shift(self, transform_params):
        shift = transform_params[:, : self.num_transform_features, ...]
        raw_log_scale = transform_params[:, self.num_transform_features :, ...]
        log_scale = torch.tanh(raw_log_scale) * 2.0
        scale = torch.exp(log_scale)
        return scale, shift


class ConditionalGlowAffine(Flow):
    """Conditional Glow variant with affine coupling layers."""

    def __init__(
        self,
        features,
        hidden_features,
        num_layers,
        num_blocks_per_layer,
        activation=F.relu,
        dropout_probability=0.5,
        context_features=None,
        batch_norm_within_layers=True,
    ):
        coupling_constructor = ClampedAffineCouplingTransform

        mask = torch.ones(features)
        mask[::2] = -1

        def create_resnet(in_features, out_features):
            return nets.ResidualNet(
                in_features,
                out_features,
                hidden_features=hidden_features,
                num_blocks=num_blocks_per_layer,
                activation=activation,
                context_features=context_features,
                dropout_probability=dropout_probability,
                use_batch_norm=batch_norm_within_layers,
            )

        layers = []
        for _ in range(num_layers):
            layers.append(ActNorm(features=features))
            layers.append(LULinear(features=features))
            transform = coupling_constructor(
                mask=mask, transform_net_create_fn=create_resnet
            )
            mask *= -1
            layers.append(transform)

        super().__init__(
            transform=CompositeTransform(layers),
            distribution=StandardNormal([features]),
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
        self.shape_scale_dim = self.num_shape_comps + self.num_scale_comps
        self.pose_dim = self.num_glob_rot_comps + self.num_3dof_comps + self.num_1dof_comps + self.num_cam_comps
        self.flow_dim = self.pose_dim + self.shape_scale_dim
        flow_num_layers = cfg.MODEL.FLOW_NUM_LAYERS
        flow_dropout = cfg.MODEL.FLOW_DROPOUT

        flow_config_shape_scale = {
            "flow_dim": self.shape_scale_dim,
            "num_layers": flow_num_layers,
            "context_features": 2048,
            "layer_hidden_features": 1024,
            "layer_depth": 2,
            "dropout_probability": flow_dropout,
        }
        flow_config_pose = {
            "flow_dim": self.pose_dim,
            "num_layers": flow_num_layers,
            "context_features": 2048,
            "layer_hidden_features": 1024,
            "layer_depth": 2,
            "dropout_probability": flow_dropout,
        }

        flow_coupling = getattr(cfg.MODEL, "FLOW_COUPLING", "additive").lower()
        if flow_coupling == "additive":
            flow_cls = ConditionalGlow
        elif flow_coupling == "affine":
            flow_cls = ConditionalGlowAffine
        else:
            raise ValueError(
                f"Unsupported MODEL.FLOW_COUPLING='{flow_coupling}'. "
                "Expected one of: ['additive', 'affine']."
            )
        self.flow_coupling = flow_coupling

        self.flow_shape_scale = flow_cls(
            flow_config_shape_scale["flow_dim"],
            flow_config_shape_scale["layer_hidden_features"],
            flow_config_shape_scale["num_layers"],
            flow_config_shape_scale["layer_depth"],
            dropout_probability=flow_config_shape_scale["dropout_probability"],
            context_features=flow_config_shape_scale["context_features"],
        )
        self.flow_pose = flow_cls(
            flow_config_pose["flow_dim"],
            flow_config_pose["layer_hidden_features"],
            flow_config_pose["num_layers"],
            flow_config_pose["layer_depth"],
            dropout_probability=flow_config_pose["dropout_probability"],
            context_features=flow_config_pose["context_features"],
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
        context_shape_scale_dim = 1024 + 45 + 10
        self.beta_context_proj = nn.Linear(context_shape_scale_dim, 2048)

        # Stage 2 context: [flow_context, shape_sample, scale_sample_selected, aa_3dofs, params_1dofs, (pred_cam?)]
        context_pose_dim = 1024 + 45 + 10 + 39 + 34 + (3 if self.model_cam else 0)
        self.pose_context_proj = nn.Linear(context_pose_dim, 2048)

        self.register_buffer("initialized_shape_scale", torch.tensor(False))
        self.register_buffer("initialized_pose", torch.tensor(False))

    def initialize_actnorm(self, batch: Dict, mean_pred: Dict, flow_context: torch.Tensor):
        # Compute GT flow params.
        gt_flow_params = convert_mhr_params_to_flow_params(
            batch["model_params"],
            batch["shape_params"],
            include_global_rot=self.model_glob_rot,
            include_shape=self.model_shape,
            include_scale=self.model_scale,
        )

        # Compute mean-prediction flow params (mirrors nf_loss.py).
        mean_pred_flow_params = convert_mhr_params_to_flow_params(
            torch.cat(
                [
                    torch.zeros_like(mean_pred["body_pose"][..., :6]),  # dummy global
                    mean_pred["body_pose"][..., :130],  # body pose without jaw
                    mean_pred["scale_68D"],
                ],
                dim=-1,
            ),
            mean_pred["shape"],
            include_global_rot=self.model_glob_rot,
            include_shape=self.model_shape,
            include_scale=self.model_scale,
        )

        # Flows are trained on residuals, so initialise with residuals.
        true_residual = gt_flow_params - mean_pred_flow_params
        # convert_mhr_params_to_flow_params doesn't include camera, so split at
        # the pose dim *without* camera, then splice camera residual in.
        pose_dim_no_cam = self.pose_dim - self.num_cam_comps
        pose_residual_no_cam = true_residual[..., :pose_dim_no_cam]
        shape_scale_residual = true_residual[..., pose_dim_no_cam:]

        if self.model_cam:
            cam_residual = batch["gt_pred_cam"] - mean_pred["pred_cam"]
            pose_residual = torch.cat([pose_residual_no_cam, cam_residual], dim=-1)
        else:
            pose_residual = pose_residual_no_cam

        shape_mean = mean_pred["shape"]
        scale_mean = mean_pred["scale_68D"]
        pose_mean_cont = mean_pred["pred_pose_raw"][:, 6:]
        pose_params = self.convert_pose_cont_to_params_for_context(pose_mean_cont)
        aa_3dofs = pose_params["aa_3dofs"]
        params_1dofs = pose_params["params_1dofs"]

        context_shape_scale = self.beta_context_proj(
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
        shape_residual_true = shape_scale_residual[..., : self.num_shape_comps]
        scale_residual_true = shape_scale_residual[..., self.num_shape_comps :]
        shape_sample_true = shape_mean + shape_residual_true
        scale_sample_selected_true = scale_mean[..., scale_indices] + scale_residual_true

        context_pose_parts = [
            flow_context,
            shape_sample_true,
            scale_sample_selected_true,
            aa_3dofs,
            params_1dofs,
        ]
        if self.model_cam:
            context_pose_parts.append(mean_pred["pred_cam"])
        context_pose = self.pose_context_proj(torch.cat(context_pose_parts, dim=-1))

        with torch.no_grad():
            _, _ = self.flow_shape_scale.log_prob(shape_scale_residual, context_shape_scale)
            _, _ = self.flow_pose.log_prob(pose_residual, context_pose)
            self.initialized_shape_scale |= True
            self.initialized_pose |= True

        print('initialized actnorm')

    @autocast("cuda", enabled=False)
    def log_prob(
        self,
        params: torch.Tensor,
        flow_context_shape_scale: torch.Tensor,
        flow_context_pose: torch.Tensor,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Compute summed conditional log-probability for the factorised model.

        Args:
            params: Residual parameters in the same ordering as
                `convert_mhr_params_to_flow_params`, i.e.
                [pose_part (= glob rot aa (optional), 3dof, 1dof), shape_params, scale_params]
                with shapes matching `self.flow_dim`.
            flow_context_shape_scale: [B, 2048] context for stage 1.
            flow_context_pose: [B, 2048] context for stage 2.
        """
        if params.shape[-1] != self.flow_dim:
            raise ValueError(
                f"Expected params last dim {self.flow_dim}, got {params.shape[-1]}"
            )

        # Split residual parameters into pose part and (shape+scale) part.
        pose_params = params[..., : self.pose_dim]
        shape_scale_params = params[..., self.pose_dim :]

        if shape_scale_params.shape[-1] != self.shape_scale_dim:
            raise ValueError(
                "Internal split mismatch: "
                f"expected shape_scale_dim={self.shape_scale_dim}, got {shape_scale_params.shape[-1]}"
            )

        log_prob_shape_scale, z_shape_scale = self.flow_shape_scale.log_prob(
            inputs=shape_scale_params, context=flow_context_shape_scale
        )
        log_prob_pose, z_pose = self.flow_pose.log_prob(
            inputs=pose_params, context=flow_context_pose
        )

        log_prob_total = log_prob_shape_scale + log_prob_pose
        return log_prob_total, (z_shape_scale, z_pose)


    @autocast("cuda", enabled=False)
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

        pose_mean_cont = mean_pred["pred_pose_raw"][:, 6:] # glob 

        pose_params_mhr = compact_cont_to_model_params_body(pose_mean_cont)

        pose_params = self.convert_pose_cont_to_params_for_context(pose_mean_cont)
        aa_3dofs = pose_params["aa_3dofs"]  # B, 39
        params_1dofs = pose_params["params_1dofs"]  # B, 34

        # Stage 1: p(Δβ | c, μβ) — shape+scale residuals.
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

        if (not self.initialized_shape_scale.item()) or (not self.initialized_pose.item()):
            self.initialize_actnorm(batch, mean_pred=mean_pred, flow_context=flow_context)
            print("Initialised ActNorm")

        beta_samples, beta_log_prob, beta_z = self.flow_shape_scale.sample_and_log_prob(
            N,
            context=beta_context,
        )


        # Mode-2: perturb stage-1 beta samples before conditioning stage-2.
        # Shape noise: all 45 components. Scale noise: only the selected 10 indices
        # (beta_samples[..., num_shape_comps:] already contains only those 10).
        # Noise std = per-dim GT std * perturb_scale, so scale=1.0 matches data spread.
        # Detach by default so KP2D gradients do not corrupt the stage-1 distribution.
        if self.training and (self.shape_perturb_scale > 0 or self.scale_perturb_scale > 0):
            shape_part = beta_samples[..., : self.num_shape_comps]
            scale_part = beta_samples[..., self.num_shape_comps :]
            if self.shape_perturb_scale > 0:
                shape_part = shape_part + torch.randn_like(shape_part) * self._shape_perturb_std * self.shape_perturb_scale
            if self.scale_perturb_scale > 0:
                scale_part = scale_part + torch.randn_like(scale_part) * self._scale_perturb_std * self.scale_perturb_scale
            beta_samples_for_stage2 = torch.cat([shape_part, scale_part], dim=-1)
            if self.beta_perturb_detach:
                beta_samples_for_stage2 = beta_samples_for_stage2.detach()
        else:
            beta_samples_for_stage2 = beta_samples

        # shape_scale_samples: [B, N, shape_scale_dim]
        shape_residual_samples = beta_samples_for_stage2[..., : self.num_shape_comps]
        scale_residual_samples = beta_samples_for_stage2[..., self.num_shape_comps :]

        shape_samples = shape_mean.unsqueeze(1).repeat(1, N, 1)
        if self.num_shape_comps > 0:
            shape_samples = shape_samples + shape_residual_samples

        scale_samples_68D = scale_mean.unsqueeze(1).repeat(1, N, 1)
        if self.model_scale and self.num_scale_comps > 0:
            scale_samples_68D[..., scale_indices] = (
                scale_samples_68D[..., scale_indices] + scale_residual_samples
            )

        # Stage 2: p(Δθ | c, μθ, Δβ) — pose residuals conditioned on sampled β.
        flow_context_expanded = flow_context.unsqueeze(1).repeat(1, N, 1)
        aa_3dofs_expanded = aa_3dofs.unsqueeze(1).repeat(1, N, 1)
        params_1dofs_expanded = params_1dofs.unsqueeze(1).repeat(1, N, 1)

        context_pose_parts = [
            flow_context_expanded,
            shape_samples,
            scale_samples_68D[..., scale_indices],
            aa_3dofs_expanded,
            params_1dofs_expanded,
        ]
        if self.model_cam:
            pred_cam_expanded = mean_pred["pred_cam"].unsqueeze(1).repeat(1, N, 1)
            context_pose_parts.append(pred_cam_expanded)
        context_pose = self.pose_context_proj(
            torch.cat(context_pose_parts, dim=-1).reshape(B * N, -1)
        )

        pose_samples_flat, pose_log_prob_flat, pose_z_flat = self.flow_pose.sample_and_log_prob(
            1,
            context=context_pose,
        )

        # pose_samples_flat: [B * N, 1, pose_dim]
        pose_samples_residual = pose_samples_flat.squeeze(1).reshape(B, N, self.pose_dim)
        pose_log_prob = pose_log_prob_flat.squeeze(1).reshape(B, N)

        offset = self.num_glob_rot_comps
        pose_3dof_residual_samples = pose_samples_residual[
            ..., offset : offset + self.num_3dof_comps
        ]
        pose_1dof_residual_samples = pose_samples_residual[
            ...,
            offset
            + self.num_3dof_comps : offset
            + self.num_3dof_comps
            + self.num_1dof_comps,
        ]

        aa_3dof_samples = (
            aa_3dofs.unsqueeze(1).repeat(1, N, 1) + pose_3dof_residual_samples
        )
        params_1dofs_samples = (
            params_1dofs.unsqueeze(1).repeat(1, N, 1) + pose_1dof_residual_samples
        )

        pose_samples = self.convert_samples_to_params(
            aa_3dof_samples, params_1dofs_samples, pose_params_mhr
        )

        # Global rotation samples: convert sampled axis-angle to XYZ Euler.
        if self.model_glob_rot:
            glob_rot_aa_samples = pose_samples_residual[..., : self.num_glob_rot_comps]  # (B, N, 3)
            glob_rot_mat_samples = axis_angle_to_matrix(glob_rot_aa_samples)  # (B, N, 3, 3)
            glob_rot_euler_samples = roma.rotmat_to_euler("ZYX", glob_rot_mat_samples).flip(-1)  # (B, N, 3) ZYX→XYZ
        else:
            glob_rot_euler_samples = None

        # Camera samples: add residual to mean pred_cam.
        if self.model_cam:
            cam_offset = self.num_glob_rot_comps + self.num_3dof_comps + self.num_1dof_comps
            cam_residual_samples = pose_samples_residual[..., cam_offset : cam_offset + 3]  # (B, N, 3)
            cam_samples = mean_pred["pred_cam"].unsqueeze(1) + cam_residual_samples  # (B, N, 3)
        else:
            cam_samples = None

        
        # DEBUG: override samples with GT residual
        if batch is not None and "model_params" in batch:
            gt_flow_params = convert_mhr_params_to_flow_params(
                batch["model_params"],
                batch["shape_params"],
                include_global_rot=self.model_glob_rot,
                include_shape=self.model_shape,
                include_scale=self.model_scale,
            )
            mean_pred_flow_params = convert_mhr_params_to_flow_params(
                torch.cat(
                    [
                        torch.zeros_like(mean_pred["body_pose"][..., :6]),
                        mean_pred["body_pose"][..., :130],
                        mean_pred["scale_68D"],
                    ],
                    dim=-1,
                ),
                mean_pred["shape"],
                include_global_rot=self.model_glob_rot,
                include_shape=self.model_shape,
                include_scale=self.model_scale,
            )
            # gt_residual from convert_mhr_params_to_flow_params does NOT include camera.
            # Ordering: [glob_rot?(3) | 3dof(39) | 1dof(34) | shape(45) | scale(10)]
            gt_residual = gt_flow_params - mean_pred_flow_params  # (B, flow_dim_no_cam)
            pose_dim_no_cam = self.pose_dim - self.num_cam_comps

            # Override shape samples with GT
            gt_shape_residual = gt_residual[..., pose_dim_no_cam : pose_dim_no_cam + self.num_shape_comps]
            shape_samples = (shape_mean + gt_shape_residual).unsqueeze(1).expand(-1, N, -1)

            # Override scale samples with GT
            gt_scale_residual = gt_residual[..., pose_dim_no_cam + self.num_shape_comps :]
            gt_scale_68D = batch["model_params"][:, -68:]  # (B, 68)
            scale_samples_68D = gt_scale_68D.unsqueeze(1).expand(-1, N, -1).clone()
            scale_samples_68D[..., scale_indices] = (
                scale_mean[..., scale_indices].unsqueeze(1) + gt_scale_residual.unsqueeze(1)
            ).expand(-1, N, -1)

            # Override pose samples with GT
            offset = self.num_glob_rot_comps
            gt_pose_3dof_residual = gt_residual[..., offset : offset + self.num_3dof_comps]
            gt_pose_1dof_residual = gt_residual[..., offset + self.num_3dof_comps : offset + self.num_3dof_comps + self.num_1dof_comps]
            gt_aa_3dof_samples = (aa_3dofs + gt_pose_3dof_residual).unsqueeze(1).expand(-1, N, -1)
            gt_params_1dofs_samples = (params_1dofs + gt_pose_1dof_residual).unsqueeze(1).expand(-1, N, -1)
            # Use GT pose params (130D → 133D with jaw zeros) as base so non-modelled
            # joints (hands, jaw, etc.) come from GT instead of mean prediction.
            gt_pose_130D = batch["model_params"][:, 6:-68]  # (B, 130)
            gt_pose_133D = torch.cat([gt_pose_130D, torch.zeros_like(gt_pose_130D[:, :3])], dim=-1)  # (B, 133)
            pose_samples = self.convert_samples_to_params(gt_aa_3dof_samples, gt_params_1dofs_samples, gt_pose_133D)

            # Override global rotation samples with GT
            if self.model_glob_rot:
                gt_glob_rot_aa = gt_residual[..., :self.num_glob_rot_comps]  # (B, 3)
                gt_glob_rot_mat = axis_angle_to_matrix(gt_glob_rot_aa)  # (B, 3, 3)
                gt_glob_rot_euler = roma.rotmat_to_euler("ZYX", gt_glob_rot_mat).flip(-1)  # (B, 3) ZYX→XYZ
                glob_rot_euler_samples = gt_glob_rot_euler.unsqueeze(1).expand(-1, N, -1)

            # Override camera samples with GT
            if self.model_cam and "gt_pred_cam" in batch:
                gt_cam_residual = batch["gt_pred_cam"] - mean_pred["pred_cam"]  # (B, 3)
                cam_samples = batch["gt_pred_cam"].unsqueeze(1).expand(-1, N, -1)
            elif self.model_cam:
                cam_samples = mean_pred["pred_cam"].unsqueeze(1).expand(-1, N, -1)
            else:
                cam_samples = None

            # Rebuild full residual: splice camera residual between pose and shape/scale
            gt_pose_residual = gt_residual[..., :pose_dim_no_cam]
            gt_shape_scale_residual = gt_residual[..., pose_dim_no_cam:]
            if self.model_cam and "gt_pred_cam" in batch:
                samples = torch.cat([gt_pose_residual, gt_cam_residual, gt_shape_scale_residual], dim=-1)
            else:
                samples = gt_residual
            samples = samples.unsqueeze(1).expand(-1, N, -1)


        beta_log_prob = beta_log_prob.reshape(B, N)
        log_prob = beta_log_prob + pose_log_prob

        # Full residual vector in the same ordering as `convert_mhr_params_to_flow_params`.
        # Ordering: [pose_part (glob? + 3dof + 1dof + cam?), shape(45), scale(10)].
        samples = torch.cat([pose_samples_residual, beta_samples], dim=-1)

        ret = {
            "log_prob": log_prob,
            "log_prob_shape_scale": beta_log_prob,
            "log_prob_pose": pose_log_prob,
            "samples": samples,
            "shape_samples": shape_samples,
            "scale_samples": scale_samples_68D,
            "pose_samples": pose_samples,
            "global_rot_samples": glob_rot_euler_samples,
            "cam_samples": cam_samples,
            "flow_context_shape_scale": beta_context,
            "flow_context_pose": context_pose.reshape(B, N, -1),
            "flow_context_raw": flow_context,
        }

        return ret

    def convert_samples_to_params(
        self,
        aa_3dof_samples: torch.Tensor,
        params_1dofs_samples: torch.Tensor,
        pose_mean: torch.Tensor,
    ):
        B, N, D = aa_3dof_samples.shape
        pose_mean = pose_mean.unsqueeze(1).repeat(1, N, 1)

        aa_3dof_samples = aa_3dof_samples.unflatten(-1, (-1, 3))
        rotmat_3dof_samples = axis_angle_to_matrix(aa_3dof_samples)

        x_raw = rotmat_3dof_samples[..., :, 0]
        y_raw = rotmat_3dof_samples[..., :, 1]

        x = F.normalize(x_raw, dim=-1)
        z = torch.cross(x, y_raw, dim=-1)
        z = F.normalize(z, dim=-1)
        y = torch.cross(z, x, dim=-1)

        matrix = torch.stack([x, y, z], dim=-1)

        sy = torch.sqrt(
            matrix[..., 0, 0] * matrix[..., 0, 0]
            + matrix[..., 1, 0] * matrix[..., 1, 0]
        )
        singular = sy < 1e-6
        singular = singular.float()

        x = torch.atan2(matrix[..., 2, 1], matrix[..., 2, 2])
        y = torch.atan2(-matrix[..., 2, 0], sy)
        z = torch.atan2(matrix[..., 1, 0], matrix[..., 0, 0])

        xs = torch.atan2(-matrix[..., 1, 2], matrix[..., 1, 1])
        ys = torch.atan2(-matrix[..., 2, 0], sy)
        zs = matrix[..., 1, 0] * 0

        euler_3dof_samples = torch.zeros_like(matrix[..., 0])
        euler_3dof_samples[..., 0] = x * (1 - singular) + xs * singular
        euler_3dof_samples[..., 1] = y * (1 - singular) + ys * singular
        euler_3dof_samples[..., 2] = z * (1 - singular) + zs * singular

        # euler_3dof_samples = aa_to_euler(aa_3dof_samples, "XYZ")
        euler_3dof_samples = euler_3dof_samples.flatten(-2, -1)

        pose_mean[..., all_param_3dof_rot_idxs_except_hands.flatten()] = (
            euler_3dof_samples
        )
        pose_mean[..., all_param_1dof_rot_idxs_except_hands] = params_1dofs_samples
        pose_mean[..., mhr_param_hand_mask] = 0
        pose_mean[..., -3:] = 0
        return pose_mean

    def convert_pose_cont_to_params_for_context(self, pose_cont: torch.Tensor):
        num_3dof_angles = len(all_param_3dof_rot_idxs) * 3  # 69
        num_1dof_angles = len(all_param_1dof_rot_idxs)  # 58
        num_1dof_trans = len(all_param_1dof_trans_idxs)  # 6
        assert pose_cont.shape[-1] == (
            2 * num_3dof_angles + 2 * num_1dof_angles + num_1dof_trans
        )
        # Get subsets
        cont_3dofs = pose_cont[..., : 2 * num_3dof_angles]
        cont_1dofs = pose_cont[
            ..., 2 * num_3dof_angles : 2 * num_3dof_angles + 2 * num_1dof_angles
        ]
        cont_trans = pose_cont[..., 2 * num_3dof_angles + 2 * num_1dof_angles :]

        cont_3dofs = cont_3dofs.unflatten(-1, (-1, 6))
        rotmat_3dofs = batch9Dfrom6D(cont_3dofs).unflatten(-1, (3, 3))

        aa_3dofs = matrix_to_axis_angle(rotmat_3dofs)[:, indices_3dof, ...].flatten(
            -2, -1
        )

        cont_1dofs = cont_1dofs.unflatten(-1, (-1, 2))  # (sincos)
        params_1dofs = torch.atan2(cont_1dofs[..., -2], cont_1dofs[..., -1])
        params_1dofs = params_1dofs[:, indices_1dof]

        ret = {
            "aa_3dofs": aa_3dofs,
            "params_1dofs": params_1dofs,
        }
        return ret

