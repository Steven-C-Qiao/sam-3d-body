import torch
import torch.nn as nn
from torch.amp import autocast
from typing import Optional, Dict, Tuple
from nflows.flows import ConditionalGlow
from yacs.config import CfgNode

from sam_3d_body.models.modules.mhr_utils import (
    batch9Dfrom6D,
    compact_cont_to_model_params_body,
    mhr_param_hand_mask,
)

from pytorch3d.transforms import (
    axis_angle_to_matrix,
    matrix_to_axis_angle,
    matrix_to_euler_angles,
)
def aa_to_euler(aa, euler_convention="XYZ"):
    rotmat = axis_angle_to_matrix(aa)
    euler = matrix_to_euler_angles(rotmat, euler_convention)
    return euler


class NFHead(nn.Module):
    def __init__(self):
        super(NFHead, self).__init__()

        self.num_shape_comps = 45
        self.num_scale_comps = 10
        self.num_pose_comps = 39
        self.num_1dof_comps = 34

        self.flow_dim = self.num_shape_comps + self.num_scale_comps + self.num_pose_comps + self.num_1dof_comps
        config = {
            "flow_dim": self.flow_dim,
            "num_layers": 4,
            "context_features": 2048,
            "layer_hidden_features": 1024,
            "layer_depth": 2
        }
        # self.flow = ConditionalGlow(cfg.MODEL.FLOW.DIM, cfg.MODEL.FLOW.LAYER_HIDDEN_FEATURES,
        #                             cfg.MODEL.FLOW.NUM_LAYERS, cfg.MODEL.FLOW.LAYER_DEPTH,
        #                             context_features=cfg.MODEL.FLOW.CONTEXT_FEATURES)
        self.flow = ConditionalGlow(
            config["flow_dim"],
            config["layer_hidden_features"],
            config["num_layers"],
            config["layer_depth"],
            context_features=config["context_features"]
        )

        self.context_proj = nn.Linear(self.flow_dim + 1024, 2048)


    def initialze_actnorm(self):
        pass 
        


    @autocast("cuda", enabled=False)
    def log_prob(self, params: Dict, flow_context: torch.Tensor) -> Tuple:
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

        Returns:
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

    

    def forward(
        self,
        flow_context: torch.Tensor,
        mean_pred: Dict,
        num_samples: int = 5,
    ) -> Dict:
        """
            Given context and mean predictions, compute residual uncertainty by NF
            sampling needs to be handled here, instead of in model forward

            Args:
                mean_pred:

        """
        B, N = flow_context.shape[0], num_samples

        
        shape_mean = mean_pred["shape"] # B, 45
        scale_mean = mean_pred["scale_68D"] # B, 68 
        indices = [3, 4, 5, 6, 7, 10, 11, 12, 13, 14]

        pose_mean_cont = mean_pred["pred_pose_raw"][:, 6:] # first 6 are global transl and rot 

        pose_params_mhr = compact_cont_to_model_params_body(pose_mean_cont)

        pose_params = self.convert_pose_cont_to_params_for_context(pose_mean_cont)
        aa_3dofs = pose_params["aa_3dofs"] # B, 39 
        params_1dofs = pose_params["params_1dofs"] # B, 34 

        flow_context = self.context_proj(
            torch.cat([flow_context, shape_mean, scale_mean[..., indices], aa_3dofs, params_1dofs], dim=-1)
        )

        flow_output = self.flow_forward(
            flow_context,
            num_samples=N,
        )
        samples = flow_output["samples"]
        log_prob = flow_output["log_prob"]
        z = flow_output["z"]

        pose_3dof_residual_samples = samples[..., :39]
        pose_1dof_residual_samples = samples[..., 39:39+34]

        aa_3dof_samples = aa_3dofs.unsqueeze(1).repeat(1, N, 1) + pose_3dof_residual_samples
        params_1dofs_samples = params_1dofs.unsqueeze(1).repeat(1, N, 1) + pose_1dof_residual_samples

        pose_samples = self.convert_samples_to_params(aa_3dof_samples, params_1dofs_samples, pose_params_mhr)
        
        
        shape_residual_samples = samples[..., 39+34:39+34+45]
        shape_samples = shape_mean.unsqueeze(1).repeat(1, N, 1) + shape_residual_samples
        
        scale_residual_samples = samples[..., 39+34+45:39+34+45+10]
        scale_samples_68D = scale_mean.unsqueeze(1).repeat(1, N, 1)
        scale_samples_68D[..., indices] += scale_residual_samples

        ret = {
            "samples": samples,
            "log_prob": log_prob,
            "z": z,
            "shape_samples": shape_samples,
            "scale_samples": scale_samples_68D,
            "pose_samples": pose_samples,
            "flow_context": flow_context,
        }        
        return ret 
    


    def convert_samples_to_params(self, aa_3dof_samples: torch.Tensor, params_1dofs_samples: torch.Tensor, pose_mean: torch.Tensor):
         # fmt: off
        all_param_3dof_rot_idxs_except_hands = torch.LongTensor([(0, 2, 4), (6, 8, 10), (12, 13, 14), (15, 16, 17), (18, 19, 20), (21, 22, 23), (24, 25, 26), (27, 28, 29), (34, 35, 36), (37, 38, 39), (44, 45, 46), (53, 54, 55), (130, 131, 132)])
        all_param_1dof_rot_idxs_except_hands = torch.LongTensor([1, 3, 5, 7, 9, 11, 30, 31, 32, 33, 40, 41, 42, 43, 47, 48, 49, 50, 51, 52, 56, 57, 58, 59, 60, 61, 116, 117, 118, 119, 120, 121, 122, 123])
        # fmt: on
        B, N, D = aa_3dof_samples.shape
        pose_mean = pose_mean.unsqueeze(1).repeat(1, N, 1)

        aa_3dof_samples = aa_3dof_samples.unflatten(-1, (-1, 3))
        euler_3dof_samples = aa_to_euler(aa_3dof_samples, "XYZ")
        euler_3dof_samples = euler_3dof_samples.flatten(-2, -1)

        pose_mean[..., all_param_3dof_rot_idxs_except_hands.flatten()] = (
            euler_3dof_samples
        )
        pose_mean[..., all_param_1dof_rot_idxs_except_hands] = (
            params_1dofs_samples
        )
        pose_mean[..., mhr_param_hand_mask] = 0
        pose_mean[..., -3:] = 0
        return pose_mean
    




    def convert_pose_cont_to_params_for_context(self, pose_cont: torch.Tensor):


        # fmt: off
        all_param_3dof_rot_idxs = torch.LongTensor([(0, 2, 4), (6, 8, 10), (12, 13, 14), (15, 16, 17), (18, 19, 20), (21, 22, 23), (24, 25, 26), (27, 28, 29), (34, 35, 36), (37, 38, 39), (44, 45, 46), (53, 54, 55), (64, 65, 66), (85, 69, 73), (86, 70, 79), (87, 71, 82), (88, 72, 76), (91, 92, 93), (112, 96, 100), (113, 97, 106), (114, 98, 109), (115, 99, 103), (130, 131, 132)])
        all_param_1dof_rot_idxs = torch.LongTensor([1, 3, 5, 7, 9, 11, 30, 31, 32, 33, 40, 41, 42, 43, 47, 48, 49, 50, 51, 52, 56, 57, 58, 59, 60, 61, 62, 63, 67, 68, 74, 75, 77, 78, 80, 81, 83, 84, 89, 90, 94, 95, 101, 102, 104, 105, 107, 108, 110, 111, 116, 117, 118, 119, 120, 121, 122, 123])
        all_param_1dof_trans_idxs = torch.LongTensor([124, 125, 126, 127, 128, 129])
        all_param_3dof_rot_idxs_except_hands = torch.LongTensor([(0, 2, 4), (6, 8, 10), (12, 13, 14), (15, 16, 17), (18, 19, 20), (21, 22, 23), (24, 25, 26), (27, 28, 29), (34, 35, 36), (37, 38, 39), (44, 45, 46), (53, 54, 55), (130, 131, 132)])
        all_param_1dof_rot_idxs_except_hands = torch.LongTensor([1, 3, 5, 7, 9, 11, 30, 31, 32, 33, 40, 41, 42, 43, 47, 48, 49, 50, 51, 52, 56, 57, 58, 59, 60, 61, 116, 117, 118, 119, 120, 121, 122, 123])
        indices_3dof = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 22]
        indices_1dof = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 50, 51, 52, 53, 54, 55, 56, 57]
        # fmt: on
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

        aa_3dofs = matrix_to_axis_angle(rotmat_3dofs)[
            :, indices_3dof, ...
        ].flatten(-2, -1)

        cont_1dofs = cont_1dofs.unflatten(-1, (-1, 2))  # (sincos)
        params_1dofs = torch.atan2(cont_1dofs[..., -2], cont_1dofs[..., -1])
        params_1dofs = params_1dofs[:, indices_1dof]

        ret = {
            "aa_3dofs": aa_3dofs,
            "params_1dofs": params_1dofs,   
        }
        return ret
