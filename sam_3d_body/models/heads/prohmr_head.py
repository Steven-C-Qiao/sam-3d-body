import torch
import torch.nn as nn
from torch.amp import autocast
from typing import Optional, Dict, Tuple
from nflows.flows import ConditionalGlow
from yacs.config import CfgNode
import torch.nn.functional as F

from sam_3d_body.models.modules.mhr_utils import (
    batch9Dfrom6D,
    compact_cont_to_model_params_body,
    mhr_param_hand_mask,
    convert_mhr_params_to_flow_params,
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
    def __init__(self, cfg: CfgNode):
        super(NFHead, self).__init__()

        self.num_shape_comps = 45
        self.num_scale_comps = 10
        self.num_pose_comps = 39
        self.num_1dof_comps = 34

        self.flow_dim = (
            self.num_scale_comps 
            + self.num_pose_comps 
            + self.num_1dof_comps 
            + self.num_shape_comps 
        )
        config = {
            "flow_dim": self.flow_dim,
            "num_layers": 4,
            "context_features": 2048,
            "layer_hidden_features": 1024,
            "layer_depth": 2
        }
        self.flow = ConditionalGlow(
            config["flow_dim"],
            config["layer_hidden_features"],
            config["num_layers"],
            config["layer_depth"],
            context_features=config["context_features"],
            # dropout_probability=0.0
        )

        # Default number of NF samples per instance (can be overridden at call time)
        self.num_samples = cfg.MODEL.NUM_SAMPLES

        self.context_proj = nn.Linear(self.flow_dim + 1024, 2048)

        self.register_buffer('initialized', torch.tensor(False))



    def initialize_actnorm(self, batch: Dict, flow_context: torch.Tensor):
        """
        Initialize ActNorm buffers by running a dummy forward step
        Args:
            batch (Dict): Dictionary containing batch data
            conditioning_feats (torch.Tensor): Tensor of shape (N, C) containing the conditioning features extracted using thee backbonee
        """

        model_params = batch['model_params']
        shape_params = batch['shape_params']
        flow_params = convert_mhr_params_to_flow_params(model_params, shape_params)

        with torch.no_grad():
            _, _ = self.flow.log_prob(flow_params, flow_context)
            self.initialized |= True

    @autocast("cuda", enabled=False)
    def log_prob(self, params: torch.Tensor, flow_context: torch.Tensor) -> Tuple:

        """
        Compute log probability of parameters given flow context.
        Args:
            params: torch.Tensor of shape (B, flow_dim) containing flow parameters.
                   Flow samples are of dim B, 128, where:
                   - :39 are 3dofs
                   - 39:73 are 1dofs  
                   - 73:73+45 are shape
                   - 73+45:73+45+10 are selected scale
            flow_context: torch.Tensor of shape (B, context_features) containing conditioning features
            
        Returns:
            log_prob: torch.Tensor of log probabilities
            z: torch.Tensor of latent representations
        """
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
        
        shape_mean = mean_pred["shape"] # B, 45
        scale_mean = mean_pred["scale_68D"] # B, 68 
        scale_indices = [3, 4, 5, 6, 7, 10, 11, 12, 13, 14]

        pose_mean_cont = mean_pred["pred_pose_raw"][:, 6:] # first 6 are global transl and rot 

        pose_params_mhr = compact_cont_to_model_params_body(pose_mean_cont)

        pose_params = self.convert_pose_cont_to_params_for_context(pose_mean_cont)
        aa_3dofs = pose_params["aa_3dofs"] # B, 39 
        params_1dofs = pose_params["params_1dofs"] # B, 34 

        flow_context = self.context_proj(
            torch.cat([
                flow_context, 
                shape_mean, 
                scale_mean[..., scale_indices], 
                aa_3dofs, 
                params_1dofs
            ], dim=-1)
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

        


        samples[..., :73] *= 0.2
        samples[..., 73:] *= 0.5


        pose_3dof_residual_samples = samples[..., :39]
        pose_1dof_residual_samples = samples[..., 39:39+34]

        aa_3dof_samples = aa_3dofs.unsqueeze(1).repeat(1, N, 1) + pose_3dof_residual_samples
        params_1dofs_samples = params_1dofs.unsqueeze(1).repeat(1, N, 1) + pose_1dof_residual_samples

        pose_samples = self.convert_samples_to_params(aa_3dof_samples, params_1dofs_samples, pose_params_mhr)

        
        # shape_residual_samples = samples[..., 39+34:39+34+45]
        shape_samples = shape_mean.unsqueeze(1).repeat(1, N, 1) # + shape_residual_samples

        # scale_residual_samples = samples[..., 39+34+45:39+34+45+10]
        scale_residual_samples = samples[..., -10:]
        scale_samples_68D = scale_mean.unsqueeze(1).repeat(1, N, 1)
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
        
        # gt_model_params = batch["model_params"]
        # gt_shape = batch["shape_params"]

        # gt_flow_params = convert_mhr_params_to_flow_params(gt_model_params, gt_shape)
        
        # mean_pred_flow_params = convert_mhr_params_to_flow_params(
        #     torch.cat([
        #         torch.zeros_like(mean_pred["body_pose"][..., :6]), # Adds global, which is not used
        #         mean_pred["body_pose"][..., :130], # gets rid of jaw
        #         mean_pred["scale_68D"],
        #     ], dim=-1), 
        #     mean_pred["shape"]
        # )

        # true_residual = gt_flow_params - mean_pred_flow_params
        # samples = true_residual.unsqueeze(1).repeat(1, N, 1)

        # aa_3dof_samples = gt_flow_params[..., :39].unsqueeze(1).repeat(1, N, 1) 
        # params_1dofs_samples = gt_flow_params[..., 39:39+34].unsqueeze(1).repeat(1, N, 1)

        
        # log_prob_using_func, z = self.log_prob(
        #     samples.flatten(0, 1), 
        #     flow_context.repeat_interleave(N, dim=0)
        # )
        # log_prob_using_func = log_prob_using_func.unflatten(0, (B, N))
        # print(log_prob_using_func[0])
        # print(log_prob[0])
                
        # self.flow.eval()

        # flow_output = self.flow_forward(
        #     flow_context,
        #     num_samples=N,
        # )
        # samples = flow_output["samples"]
        # log_prob = flow_output["log_prob"]
        # z = flow_output["z"]

        
        # log_prob_using_func, z = self.log_prob(
        #     samples.flatten(0, 1), 
        #     flow_context.repeat_interleave(N, dim=0)
        # )
        # log_prob_using_func = log_prob_using_func.unflatten(0, (B, N))
        # print(log_prob_using_func[0])
        # print(log_prob[0])
        return ret 
    


    def convert_samples_to_params(self, aa_3dof_samples: torch.Tensor, params_1dofs_samples: torch.Tensor, pose_mean: torch.Tensor):
         # fmt: off
        all_param_3dof_rot_idxs_except_hands = torch.LongTensor([(0, 2, 4), (6, 8, 10), (12, 13, 14), (15, 16, 17), (18, 19, 20), (21, 22, 23), (24, 25, 26), (27, 28, 29), (34, 35, 36), (37, 38, 39), (44, 45, 46), (53, 54, 55), (130, 131, 132)])
        all_param_1dof_rot_idxs_except_hands = torch.LongTensor([1, 3, 5, 7, 9, 11, 30, 31, 32, 33, 40, 41, 42, 43, 47, 48, 49, 50, 51, 52, 56, 57, 58, 59, 60, 61, 116, 117, 118, 119, 120, 121, 122, 123])
        # fmt: on
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
            matrix[..., 0, 0] * matrix[..., 0, 0] + matrix[..., 1, 0] * matrix[..., 1, 0]
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


