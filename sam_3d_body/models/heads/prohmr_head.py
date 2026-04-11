import torch
import torch.nn as nn
from torch.amp import autocast
from typing import Optional, Dict, Tuple
from nflows.flows import ConditionalGlow
from yacs.config import CfgNode
from sam_3d_body.models.modules.mhr_utils import (
    compact_cont_to_model_params_body,
    convert_mhr_params_to_flow_params,
    convert_flow_samples_to_mhr_params,
    convert_pose_cont_to_flow_context,
)


class NFHead(nn.Module):
    def __init__(self, cfg: CfgNode):
        super(NFHead, self).__init__()

        self.model_glob_rot = getattr(cfg.MODEL, "MODEL_GLOB_ROT", False)
        self.model_shape = getattr(cfg.MODEL, "MODEL_SHAPE", True)
        self.model_scale = getattr(cfg.MODEL, "MODEL_SCALE", True)

        self.num_3dof_comps = 39
        self.num_1dof_comps = 34
        self.num_shape_comps = 45 if self.model_shape else 0
        self.num_scale_comps = 10 if self.model_scale else 0
        self.num_glob_rot_comps = 3 if self.model_glob_rot else 0

        self.flow_dim = (
            self.num_scale_comps
            + self.num_3dof_comps
            + self.num_1dof_comps
            + self.num_shape_comps
            + self.num_glob_rot_comps
        )
        config = {
            "flow_dim": self.flow_dim,
            "num_layers": 4,
            "context_features": 2048,
            "layer_hidden_features": 1024,
            "layer_depth": 2,
        }
        self.flow = ConditionalGlow(
            config["flow_dim"],
            config["layer_hidden_features"],
            config["num_layers"],
            config["layer_depth"],
            context_features=config["context_features"],
        )
        self.num_samples = cfg.MODEL.NUM_SAMPLES

        context_dim = 1024 + 45 + 10 + 39 + 34
        self.context_proj = nn.Linear(context_dim, 2048)

        self.register_buffer("initialized", torch.tensor(False))

    def initialize_actnorm(self, batch: Dict, flow_context: torch.Tensor):
        model_params = batch["model_params"]
        shape_params = batch["shape_params"]
        flow_params = convert_mhr_params_to_flow_params(
            model_params,
            shape_params,
            include_global_rot=self.model_glob_rot,
            include_shape=self.model_shape,
            include_scale=self.model_scale,
            flip_global_rot=True,
        )

        with torch.no_grad():
            _, _ = self.flow.log_prob(flow_params, flow_context)
            self.initialized |= True

        # nflows ActNorm._initialize creates CPU tensors — fix for DDP.
        device = flow_params.device
        for name, buf in self.flow.named_buffers():
            if buf.device != device:
                buf.data = buf.data.to(device)

    @autocast("cuda", enabled=False)
    def log_prob(self, params: torch.Tensor, flow_context: torch.Tensor) -> Tuple:
        log_prob, z = self.flow.log_prob(
            inputs=params,
            context=flow_context,
        )
        return log_prob, z

    @autocast("cuda", enabled=False)
    def flow_forward(
        self,
        flow_context: torch.Tensor,
        num_samples: int,
    ) -> Dict:
        """
        Args:
            flow_context: [B, C], from LoRA token
            num_samples:  int, number of NF samples per batch element
        """

        samples, log_prob, z = self.flow.sample_and_log_prob(
            num_samples,
            context=flow_context,
        )

        ret = {
            "samples": samples,
            "log_prob": log_prob,
            "z": z,
        }
        return ret

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

        Args:
            mean_pred:

        """
        if num_samples <= 0:
            num_samples = self.num_samples

        B, N = flow_context.shape[0], num_samples

        shape_mean = mean_pred["shape"]  # B, 45
        scale_mean = mean_pred["scale_68D"]  # B, 68
        scale_indices = [3, 4, 5, 6, 7, 10, 11, 12, 13, 14]

        pose_mean_cont = mean_pred["pred_pose_raw"][
            :, 6:
        ]  # first 6 are global transl and rot

        pose_params_mhr = compact_cont_to_model_params_body(pose_mean_cont)

        pose_params = self.convert_pose_cont_to_params_for_context(pose_mean_cont)
        aa_3dofs = pose_params["aa_3dofs"]  # B, 39
        params_1dofs = pose_params["params_1dofs"]  # B, 34

        flow_context = self.context_proj(
            torch.cat(
                [
                    flow_context,
                    shape_mean,
                    scale_mean[..., scale_indices],
                    aa_3dofs,
                    params_1dofs,
                ],
                dim=-1,
            )
        )

        if not self.initialized.item():
            self.initialize_actnorm(batch, flow_context=flow_context)
            print("Initialised ActNorm")

        flow_output = self.flow_forward(
            flow_context,
            num_samples=N,
        )
        samples = flow_output["samples"]
        log_prob = flow_output["log_prob"]
        z = flow_output["z"]

        # Flow ordering: [shape?(45) | scale?(10) | 3dof(39) | 1dof(34) | glob_rot?(3)]
        theta_offset = self.num_shape_comps + self.num_scale_comps
        pose_3dof_residual_samples = samples[..., theta_offset : theta_offset + self.num_3dof_comps]
        pose_1dof_residual_samples = samples[
            ...,
            theta_offset
            + self.num_3dof_comps : theta_offset
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

        # shape_residual_samples = samples[..., 39+34:39+34+45]
        shape_samples = shape_mean.unsqueeze(1).repeat(
            1, N, 1
        )  # + shape_residual_samples

        scale_samples_68D = scale_mean.unsqueeze(1).repeat(1, N, 1)
        if self.model_scale and self.num_scale_comps > 0:
            scale_offset = self.num_shape_comps
            scale_residual_samples = samples[..., scale_offset : scale_offset + self.num_scale_comps]
            scale_samples_68D[..., scale_indices] += scale_residual_samples

        ret = {
            "samples": samples,
            "log_prob": log_prob,
            "z": z,
            "shape_samples": shape_samples,
            "scale_samples": scale_samples_68D,
            "pose_samples": pose_samples,
            # "shape_residual_samples": shape_residual_samples,
            # "scale_residual_samples": scale_residual_samples,
            # "pose_3dof_residual_samples_aa": pose_3dof_residual_samples,
            # "pose_1dof_residual_samples": pose_1dof_residual_samples,
            "flow_context": flow_context,
        }

        return ret

    def convert_samples_to_params(self, aa_3dof_samples, params_1dofs_samples, pose_mean):
        return convert_flow_samples_to_mhr_params(aa_3dof_samples, params_1dofs_samples, pose_mean)

    def convert_pose_cont_to_params_for_context(self, pose_cont):
        return convert_pose_cont_to_flow_context(pose_cont)

        # with torch.cuda.amp.autocast(enabled=False):

        #     body_cont_3dofs = body_pose_cont[..., : 2 * num_3dof_angles]
        #     body_cont_3dofs = body_cont_3dofs.unflatten(-1, (-1, 6))

        #     # --------------------- path 1 ---------------------
        #     selected_euler = batchXYZfrom6D(body_cont_3dofs)[:, indices_3dof, ...].flatten(-2, -1)

        #     x_raw = body_cont_3dofs[..., :3]
        #     y_raw = body_cont_3dofs[..., 3:]
        #     import torch.nn.functional as F

        #     x = F.normalize(x_raw, dim=-1)
        #     z = torch.cross(x, y_raw, dim=-1)
        #     z = F.normalize(z, dim=-1)
        #     y = torch.cross(z, x, dim=-1)

        #     matrix = torch.stack([x, y, z], dim=-1)  # ... x 3 x 3

        #     # --------------------- path 2 ---------------------

        #     rotmat_3dofs = batch9Dfrom6D(body_cont_3dofs).unflatten(-1, (3, 3))

        #     aa_3dofs = matrix_to_axis_angle(rotmat_3dofs)[:, indices_3dof, ...].flatten(-2, -1)

        #     matrix = axis_angle_to_matrix(aa_3dofs.unflatten(-1, (-1, 3)))

        #     # Now get it into euler
        #     # https://github.com/papagina/RotationContinuity/blob/758b0ce551c06372cab7022d4c0bdf331c89c696/shapenet/code/tools.py#L412
        #     sy = torch.sqrt(
        #         matrix[..., 0, 0] * matrix[..., 0, 0] + matrix[..., 1, 0] * matrix[..., 1, 0]
        #     )
        #     singular = sy < 1e-6
        #     singular = singular.float()

        #     x = torch.atan2(matrix[..., 2, 1], matrix[..., 2, 2])
        #     y = torch.atan2(-matrix[..., 2, 0], sy)
        #     z = torch.atan2(matrix[..., 1, 0], matrix[..., 0, 0])

        #     xs = torch.atan2(-matrix[..., 1, 2], matrix[..., 1, 1])
        #     ys = torch.atan2(-matrix[..., 2, 0], sy)
        #     zs = matrix[..., 1, 0] * 0

        #     out_euler = torch.zeros_like(matrix[..., 0])
        #     out_euler[..., 0] = x * (1 - singular) + xs * singular
        #     out_euler[..., 1] = y * (1 - singular) + ys * singular
        #     out_euler[..., 2] = z * (1 - singular) + zs * singular

        #     euler = out_euler.flatten(-2, -1)

        #     # euler = matrix_to_euler_angles(rotmat, "XYZ").flatten(-2, -1)

        #     x = selected_euler - euler
        #     print(x.mean(), x.max())

        #     import ipdb; ipdb.set_trace()
