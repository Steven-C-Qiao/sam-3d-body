import torch
import numpy as np

import os

from sam_3d_body.models.modules.mhr_utils import (
    batch9Dfrom6D,
    batch4Dfrom2D,
    compact_cont_to_model_params_body,
    compact_cont_to_rotmat_body,
    mhr_param_hand_mask,
)

from pytorch3d.transforms import (
    axis_angle_to_matrix,
    matrix_to_axis_angle,
    matrix_to_euler_angles,
)

path = "/scratches/juban/cq244/sam-3d-body/checkpoints/sam-3d-body-dinov3/"

scale_bias = np.load(os.path.join(path, "scale_mean.npy"))
scale_comps = np.load(os.path.join(path, "scale_comps.npy"))


def _sample(x, var, num_samples=4):
    """
    Args:
        x: [B, D]
        var: [B, D]
        num_samples: int

    Returns:
        [B, num_samples, D]
    """
    assert x.shape[-1] == var.shape[-1]

    x = x.unsqueeze(1).repeat(1, num_samples, 1)
    std = torch.sqrt(var)[:, None].repeat(1, num_samples, 1)
    noise = torch.randn_like(x) * std
    return x + noise


def sample_shape(shape_mean, shape_var, num_samples=4):
    return _sample(shape_mean, shape_var, num_samples)


def sample_scale(scale_mean, scale_var, num_samples=4):
    indices = [3, 4, 5, 6, 7, 10, 11, 12, 13, 14]
    assert scale_mean.shape[-1] == 28, f"scale_mean.shape: {scale_mean.shape}"
    assert len(scale_mean.shape) == 2, f"scale_mean.shape: {scale_mean.shape}"
    assert scale_var.shape[-1] == len(indices), f"scale_var.shape: {scale_var.shape}"

    scale_bias_t = torch.from_numpy(scale_bias).float().to(scale_mean.device)
    scale_comps_t = torch.from_numpy(scale_comps).float().to(scale_mean.device)

    scale_68D = scale_bias_t[None, :] + scale_mean @ scale_comps_t

    samples = _sample(scale_68D[..., indices], scale_var, num_samples)

    scale_68D = scale_68D.unsqueeze(1).repeat(1, num_samples, 1)

    scale_68D[:, :, indices] = samples

    return scale_68D


def gen_pose_samples(body_pose_cont, var, num_samples=4):
    # fmt: off
    all_param_3dof_rot_idxs = torch.LongTensor([(0, 2, 4), (6, 8, 10), (12, 13, 14), (15, 16, 17), (18, 19, 20), (21, 22, 23), (24, 25, 26), (27, 28, 29), (34, 35, 36), (37, 38, 39), (44, 45, 46), (53, 54, 55), (64, 65, 66), (85, 69, 73), (86, 70, 79), (87, 71, 82), (88, 72, 76), (91, 92, 93), (112, 96, 100), (113, 97, 106), (114, 98, 109), (115, 99, 103), (130, 131, 132)])
    all_param_1dof_rot_idxs = torch.LongTensor([1, 3, 5, 7, 9, 11, 30, 31, 32, 33, 40, 41, 42, 43, 47, 48, 49, 50, 51, 52, 56, 57, 58, 59, 60, 61, 62, 63, 67, 68, 74, 75, 77, 78, 80, 81, 83, 84, 89, 90, 94, 95, 101, 102, 104, 105, 107, 108, 110, 111, 116, 117, 118, 119, 120, 121, 122, 123])
    all_param_1dof_trans_idxs = torch.LongTensor([124, 125, 126, 127, 128, 129])
    # fmt: on
    num_3dof_angles = len(all_param_3dof_rot_idxs) * 3  # 69
    num_1dof_angles = len(all_param_1dof_rot_idxs)  # 58
    num_1dof_trans = len(all_param_1dof_trans_idxs)  # 6
    assert body_pose_cont.shape[-1] == (
        2 * num_3dof_angles + 2 * num_1dof_angles + num_1dof_trans
    )
    # Get subsets
    body_cont_3dofs = body_pose_cont[..., : 2 * num_3dof_angles]
    body_cont_1dofs = body_pose_cont[
        ..., 2 * num_3dof_angles : 2 * num_3dof_angles + 2 * num_1dof_angles
    ]
    body_cont_trans = body_pose_cont[..., 2 * num_3dof_angles + 2 * num_1dof_angles :]

    # ------ 3dofs ------
    body_cont_3dofs = body_cont_3dofs.unflatten(-1, (-1, 6))
    body_rotmat_3dofs = batch9Dfrom6D(body_cont_3dofs).unflatten(-1, (3, 3))

    body_aa_3dofs = matrix_to_axis_angle(body_rotmat_3dofs).flatten(-2, -1)

    body_aa_3dofs_sample = _sample(
        body_aa_3dofs, var[:, :num_3dof_angles], num_samples
    ).unflatten(-1, (-1, 3))

    body_rotmat_3dofs_sample = axis_angle_to_matrix(body_aa_3dofs_sample)
    body_euler_3dofs_sample = matrix_to_euler_angles(body_rotmat_3dofs_sample, "XYZ")
    body_euler_3dofs_sample = body_euler_3dofs_sample.flatten(-2, -1)

    # ------ 1dofs ------
    body_cont_1dofs = body_cont_1dofs.unflatten(-1, (-1, 2))  # (sincos)
    body_params_1dofs = torch.atan2(body_cont_1dofs[..., -2], body_cont_1dofs[..., -1])

    body_params_1dofs_sample = _sample(
        body_params_1dofs, var[:, num_3dof_angles:], num_samples
    )

    # ------ trans ------
    body_trans_sample = body_cont_trans.unsqueeze(1).repeat(1, num_samples, 1)

    # ------ assemble ------
    body_pose_params = (
        torch.zeros(*body_pose_cont.shape[:-1], 133, device=body_pose_cont.device)
        .unsqueeze(1)
        .repeat(1, num_samples, 1)
    )

    body_pose_params[..., all_param_3dof_rot_idxs.flatten()] = body_euler_3dofs_sample
    body_pose_params[..., all_param_1dof_rot_idxs] = body_params_1dofs_sample
    body_pose_params[..., all_param_1dof_trans_idxs] = body_trans_sample

    body_pose_params[..., mhr_param_hand_mask] = 0
    body_pose_params[..., -3:] = 0

    return body_pose_params


def gen_samples(
    output,
    num_samples=5,
    sample_pose=True,
):
    """
    Args:
        output: dict with shape, scale, pred_pose_raw, pose_uncertainty, etc.
        num_samples: int
        sample_pose: if True sample pose; if False use mean pose repeated for each sample.

    Returns:
        shape_samples: [B, N, ...]
        scale_samples: [B, N, 68]
        pose_samples: [B, N, 133] body pose params (Euler 133-dim)
    """
    shape_mean = output["shape"]
    shape_var = output["shape_uncertainty"]

    scale_mean = output["scale"]
    scale_var = output["scale_uncertainty"]

    pose_mean = output["pred_pose_raw"][:, 6:]
    pose_var = output["pose_uncertainty"]

    shape_samples = sample_shape(shape_mean, shape_var, num_samples)
    scale_samples = sample_scale(scale_mean, scale_var, num_samples)
    if pose_mean is not None and pose_var is not None and sample_pose:
        pose_samples = gen_pose_samples(pose_mean, pose_var, num_samples)
    elif pose_mean is not None:
        # Use mean pose repeated for each sample (no pose sampling)
        body_pose_mean_133 = compact_cont_to_model_params_body(pose_mean)
        body_pose_mean_133 = body_pose_mean_133.clone()
        hand_mask = mhr_param_hand_mask.to(body_pose_mean_133.device)
        body_pose_mean_133[:, hand_mask] = 0
        body_pose_mean_133[:, -3:] = 0
        pose_samples = (
            body_pose_mean_133.unsqueeze(1).expand(-1, num_samples, -1).contiguous()
        )
    else:
        pose_samples = None

    return shape_samples, scale_samples, pose_samples
