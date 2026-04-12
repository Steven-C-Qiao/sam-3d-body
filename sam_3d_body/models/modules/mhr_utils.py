# Copyright (c) Meta Platforms, Inc. and affiliates.

import json
import math
import os.path as osp
import pickle

import cv2

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle


# fmt: off
all_param_3dof_rot_idxs = torch.LongTensor([(0, 2, 4), (6, 8, 10), (12, 13, 14), (15, 16, 17), (18, 19, 20), (21, 22, 23), (24, 25, 26), (27, 28, 29), (34, 35, 36), (37, 38, 39), (44, 45, 46), (53, 54, 55), (64, 65, 66), (85, 69, 73), (86, 70, 79), (87, 71, 82), (88, 72, 76), (91, 92, 93), (112, 96, 100), (113, 97, 106), (114, 98, 109), (115, 99, 103), (130, 131, 132)])
all_param_1dof_rot_idxs = torch.LongTensor([1, 3, 5, 7, 9, 11, 30, 31, 32, 33, 40, 41, 42, 43, 47, 48, 49, 50, 51, 52, 56, 57, 58, 59, 60, 61, 62, 63, 67, 68, 74, 75, 77, 78, 80, 81, 83, 84, 89, 90, 94, 95, 101, 102, 104, 105, 107, 108, 110, 111, 116, 117, 118, 119, 120, 121, 122, 123])
all_param_1dof_trans_idxs = torch.LongTensor([124, 125, 126, 127, 128, 129])
all_param_3dof_rot_idxs_except_hands = torch.LongTensor([(0, 2, 4), (6, 8, 10), (12, 13, 14), (15, 16, 17), (18, 19, 20), (21, 22, 23), (24, 25, 26), (27, 28, 29), (34, 35, 36), (37, 38, 39), (44, 45, 46), (53, 54, 55), (130, 131, 132)])
all_param_1dof_rot_idxs_except_hands = torch.LongTensor([1, 3, 5, 7, 9, 11, 30, 31, 32, 33, 40, 41, 42, 43, 47, 48, 49, 50, 51, 52, 56, 57, 58, 59, 60, 61, 116, 117, 118, 119, 120, 121, 122, 123])
indices_3dof = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 22]
indices_1dof = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 50, 51, 52, 53, 54, 55, 56, 57]
num_3dof_angles = len(all_param_3dof_rot_idxs) * 3  # 69
num_1dof_angles = len(all_param_1dof_rot_idxs)  # 58
num_1dof_trans = len(all_param_1dof_trans_idxs)  # 6
scale_indices = [3, 4, 5, 6, 7, 10, 11, 12, 13, 14]
# fmt: on




def convert_mhr_params_to_flow_params(
    model_params: torch.Tensor,
    shape_params: torch.Tensor,
    include_global_rot: bool = False,
    include_shape: bool = True,
    include_scale: bool = True,
    flip_global_rot: bool = False,
    return_rotmats: bool = False,
) -> torch.Tensor:
    assert model_params.shape[-1] == 204
    assert shape_params.shape[-1] == 45
    
    scale = model_params[:, -68:]
    pose = model_params[:, 6:-68]

    pose_3dof_euler = pose[..., all_param_3dof_rot_idxs[:-1].flatten()]
    pose_3dof_euler = torch.cat([pose_3dof_euler, torch.zeros_like(pose_3dof_euler[..., :3])], dim=-1)
    pose_3dof_euler = pose_3dof_euler.unflatten(-1, (-1, 3))
    pose_1dof_angle = pose[..., all_param_1dof_rot_idxs_except_hands]

    pose_3dof_rotmat = batch6DFromXYZ(pose_3dof_euler, return_9D=True)
    pose_3dof_rotmat_selected = pose_3dof_rotmat[..., indices_3dof, :, :]  # (B, 13, 3, 3)
    pose_3dof_aa = matrix_to_axis_angle(pose_3dof_rotmat_selected).flatten(-2, -1)

    scale = scale[..., scale_indices]

    # Output ordering: [beta (shape? + scale?), theta (3dof + 1dof + glob_rot?)]
    beta_parts = []
    if include_shape:
        beta_parts.append(shape_params)
    if include_scale:
        beta_parts.append(scale)

    theta_parts = [pose_3dof_aa, pose_1dof_angle]
    if include_global_rot:
        # model_params[..., :6] == [global_trans(3), global_rot(3)] in XYZ Euler.
        glob_euler = model_params[:, 3:6]  # (B, 3)
        glob_rotmat = batch6DFromXYZ(glob_euler, return_9D=True)  # (B, 3, 3)
        if flip_global_rot:
            # GT global rotation is in camera frame (mesh already oriented).
            # Predictions are in "upright" frame (mesh flipped by verts[..., [1,2]] *= -1).
            # Compose with YZ-flip so the residual reconciles these conventions.
            yz_flip = torch.tensor([[1., 0., 0.], [0., -1., 0.], [0., 0., -1.]],
                                   device=glob_rotmat.device, dtype=glob_rotmat.dtype)
            glob_rotmat = yz_flip @ glob_rotmat
        glob_aa = matrix_to_axis_angle(glob_rotmat)  # (B, 3)
        theta_parts.append(glob_aa)

    flow_params = torch.cat(beta_parts + theta_parts, dim=-1)

    if return_rotmats:
        rotmats = {"pose_3dof_rotmat": pose_3dof_rotmat_selected}
        if include_global_rot:
            rotmats["glob_rotmat"] = glob_rotmat  # (B, 3, 3), after optional yz_flip
        return flow_params, rotmats
    return flow_params


def so3_compose_aa(mean_aa_flat: torch.Tensor, residual_flat: torch.Tensor) -> torch.Tensor:
    """Compose rotations via right-perturbation: R_sample = R_mean @ Exp(delta).

    Works for any flat axis-angle vector whose last dim is a multiple of 3
    (e.g., 39 for 13 joints, 3 for global rotation).

    Args:
        mean_aa_flat:    (..., J*3) mean axis-angle (broadcasts with residual)
        residual_flat:   (..., J*3) residual in body-local tangent space
    Returns:
        (..., J*3) composed axis-angle
    """
    mean_unflat = mean_aa_flat.unflatten(-1, (-1, 3))       # (..., J, 3)
    res_unflat = residual_flat.unflatten(-1, (-1, 3))        # (..., J, 3)
    R_mean = axis_angle_to_matrix(mean_unflat)               # (..., J, 3, 3)
    R_delta = axis_angle_to_matrix(res_unflat)               # (..., J, 3, 3)
    R_composed = R_mean @ R_delta                             # (..., J, 3, 3)
    return matrix_to_axis_angle(R_composed).flatten(-2, -1)  # (..., J*3)


def so3_residual_aa(mean_rotmat: torch.Tensor, gt_rotmat: torch.Tensor) -> torch.Tensor:
    """Compute right-perturbation residual: delta = Log(R_mean^T @ R_gt).

    Args:
        mean_rotmat: (..., J, 3, 3)
        gt_rotmat:   (..., J, 3, 3)
    Returns:
        (..., J*3) residual axis-angle
    """
    R_rel = mean_rotmat.transpose(-1, -2) @ gt_rotmat  # (..., J, 3, 3)
    return matrix_to_axis_angle(R_rel).flatten(-2, -1)  # (..., J*3)


def rotation_angle_difference(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Compute the angle difference (magnitude) between two batches of SO(3) rotation matrices.
    Args:
        A: Tensor of shape (*, 3, 3), batch of rotation matrices.
        B: Tensor of shape (*, 3, 3), batch of rotation matrices.
    Returns:
        Tensor of shape (*,), angle differences in radians.
    """
    # Compute relative rotation matrix
    R_rel = torch.matmul(A, B.transpose(-2, -1))  # (B, 3, 3)
    # Compute trace of relative rotation
    trace = R_rel[..., 0, 0] + R_rel[..., 1, 1] + R_rel[..., 2, 2]  # (B,)
    # Compute angle using the trace formula
    cos_theta = (trace - 1) / 2
    # Clamp for numerical stability
    cos_theta_clamped = torch.clamp(cos_theta, -1.0, 1.0)
    # Compute angle difference
    angle = torch.acos(cos_theta_clamped)
    return angle


def fix_wrist_euler(
    wrist_xzy, limits_x=(-2.2, 1.0), limits_z=(-2.2, 1.5), limits_y=(-1.2, 1.5)
):
    """
    wrist_xzy: B x 2 x 3 (X, Z, Y angles)
    Returns: Fixed angles within joint limits
    """
    x, z, y = wrist_xzy[..., 0], wrist_xzy[..., 1], wrist_xzy[..., 2]

    x_alt = torch.atan2(torch.sin(x + torch.pi), torch.cos(x + torch.pi))
    z_alt = torch.atan2(torch.sin(-(z + torch.pi)), torch.cos(-(z + torch.pi)))
    y_alt = torch.atan2(torch.sin(y + torch.pi), torch.cos(y + torch.pi))

    # Calculate L2 violation distance
    def calc_violation(val, limits):
        below = torch.clamp(limits[0] - val, min=0.0)
        above = torch.clamp(val - limits[1], min=0.0)
        return below**2 + above**2

    violation_orig = (
        calc_violation(x, limits_x)
        + calc_violation(z, limits_z)
        + calc_violation(y, limits_y)
    )

    violation_alt = (
        calc_violation(x_alt, limits_x)
        + calc_violation(z_alt, limits_z)
        + calc_violation(y_alt, limits_y)
    )

    # Use alternative where it has lower L2 violation
    use_alt = violation_alt < violation_orig

    # Stack alternative and apply mask
    wrist_xzy_alt = torch.stack([x_alt, z_alt, y_alt], dim=-1)
    result = torch.where(use_alt.unsqueeze(-1), wrist_xzy_alt, wrist_xzy)

    return result


def batch6DFromXYZ(r, return_9D=False):
    """
    Generate a matrix representing a rotation defined by a XYZ-Euler
    rotation.

    Args:
        r: ... x 3 rotation vectors

    Returns:
        ... x 6
    """
    rc = torch.cos(r)
    rs = torch.sin(r)
    cx = rc[..., 0]
    cy = rc[..., 1]
    cz = rc[..., 2]
    sx = rs[..., 0]
    sy = rs[..., 1]
    sz = rs[..., 2]

    result = torch.empty(list(r.shape[:-1]) + [3, 3], dtype=r.dtype).to(r.device)

    result[..., 0, 0] = cy * cz
    result[..., 0, 1] = -cx * sz + sx * sy * cz
    result[..., 0, 2] = sx * sz + cx * sy * cz
    result[..., 1, 0] = cy * sz
    result[..., 1, 1] = cx * cz + sx * sy * sz
    result[..., 1, 2] = -sx * cz + cx * sy * sz
    result[..., 2, 0] = -sy
    result[..., 2, 1] = sx * cy
    result[..., 2, 2] = cx * cy

    if not return_9D:
        return torch.cat([result[..., :, 0], result[..., :, 1]], dim=-1)
    else:
        return result


# https://github.com/papagina/RotationContinuity/blob/758b0ce551c06372cab7022d4c0bdf331c89c696/shapenet/code/tools.py#L82
def batchXYZfrom6D(poses):
    # Args: poses: ... x 6, where "6" is the combined first and second columns
    # First, get the rotaiton matrix
    x_raw = poses[..., :3]
    y_raw = poses[..., 3:]

    x = F.normalize(x_raw, dim=-1)
    z = torch.cross(x, y_raw, dim=-1)
    z = F.normalize(z, dim=-1)
    y = torch.cross(z, x, dim=-1)

    matrix = torch.stack([x, y, z], dim=-1)  # ... x 3 x 3

    # Now get it into euler
    # https://github.com/papagina/RotationContinuity/blob/758b0ce551c06372cab7022d4c0bdf331c89c696/shapenet/code/tools.py#L412
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

    out_euler = torch.zeros_like(matrix[..., 0])
    out_euler[..., 0] = x * (1 - singular) + xs * singular
    out_euler[..., 1] = y * (1 - singular) + ys * singular
    out_euler[..., 2] = z * (1 - singular) + zs * singular

    return out_euler


def resize_image(image_array, scale_factor, interpolation=cv2.INTER_LINEAR):
    new_height = int(image_array.shape[0] // scale_factor)
    new_width = int(image_array.shape[1] // scale_factor)
    resized_image = cv2.resize(
        image_array, (new_width, new_height), interpolation=interpolation
    )

    return resized_image


def compact_cont_to_model_params_hand(hand_cont):
    # These are ordered by joint, not model params ^^
    assert hand_cont.shape[-1] == 54
    hand_dofs_in_order = torch.tensor([3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 2, 3, 1, 1])
    assert sum(hand_dofs_in_order) == 27
    # Mask of 3DoFs into hand_cont
    mask_cont_threedofs = torch.cat(
        [torch.ones(2 * k).bool() * (k in [3]) for k in hand_dofs_in_order]
    )
    # Mask of 1DoFs (including 2DoF) into hand_cont
    mask_cont_onedofs = torch.cat(
        [torch.ones(2 * k).bool() * (k in [1, 2]) for k in hand_dofs_in_order]
    )
    # Mask of 3DoFs into hand_model_params
    mask_model_params_threedofs = torch.cat(
        [torch.ones(k).bool() * (k in [3]) for k in hand_dofs_in_order]
    )
    # Mask of 1DoFs (including 2DoF) into hand_model_params
    mask_model_params_onedofs = torch.cat(
        [torch.ones(k).bool() * (k in [1, 2]) for k in hand_dofs_in_order]
    )

    # Convert hand_cont to eulers
    ## First for 3DoFs
    hand_cont_threedofs = hand_cont[..., mask_cont_threedofs].unflatten(-1, (-1, 6))
    hand_model_params_threedofs = batchXYZfrom6D(hand_cont_threedofs).flatten(-2, -1)
    ## Next for 1DoFs
    hand_cont_onedofs = hand_cont[..., mask_cont_onedofs].unflatten(
        -1, (-1, 2)
    )  # (sincos)
    hand_model_params_onedofs = torch.atan2(
        hand_cont_onedofs[..., -2], hand_cont_onedofs[..., -1]
    )

    # Finally, assemble into a 27-dim vector, ordered by joint, then XYZ.
    hand_model_params = torch.zeros(*hand_cont.shape[:-1], 27).to(hand_cont)
    hand_model_params[..., mask_model_params_threedofs] = hand_model_params_threedofs
    hand_model_params[..., mask_model_params_onedofs] = hand_model_params_onedofs

    return hand_model_params


def compact_model_params_to_cont_hand(hand_model_params):
    # These are ordered by joint, not model params ^^
    assert hand_model_params.shape[-1] == 27
    hand_dofs_in_order = torch.tensor([3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 2, 3, 1, 1])
    assert sum(hand_dofs_in_order) == 27
    # Mask of 3DoFs into hand_cont
    mask_cont_threedofs = torch.cat(
        [torch.ones(2 * k).bool() * (k in [3]) for k in hand_dofs_in_order]
    )
    # Mask of 1DoFs (including 2DoF) into hand_cont
    mask_cont_onedofs = torch.cat(
        [torch.ones(2 * k).bool() * (k in [1, 2]) for k in hand_dofs_in_order]
    )
    # Mask of 3DoFs into hand_model_params
    mask_model_params_threedofs = torch.cat(
        [torch.ones(k).bool() * (k in [3]) for k in hand_dofs_in_order]
    )
    # Mask of 1DoFs (including 2DoF) into hand_model_params
    mask_model_params_onedofs = torch.cat(
        [torch.ones(k).bool() * (k in [1, 2]) for k in hand_dofs_in_order]
    )

    # Convert eulers to hand_cont hand_cont
    ## First for 3DoFs
    hand_model_params_threedofs = hand_model_params[
        ..., mask_model_params_threedofs
    ].unflatten(-1, (-1, 3))
    hand_cont_threedofs = batch6DFromXYZ(hand_model_params_threedofs).flatten(-2, -1)
    ## Next for 1DoFs
    hand_model_params_onedofs = hand_model_params[..., mask_model_params_onedofs]
    hand_cont_onedofs = torch.stack(
        [hand_model_params_onedofs.sin(), hand_model_params_onedofs.cos()], dim=-1
    ).flatten(-2, -1)

    # Finally, assemble into a 27-dim vector, ordered by joint, then XYZ.
    hand_cont = torch.zeros(*hand_model_params.shape[:-1], 54).to(hand_model_params)
    hand_cont[..., mask_cont_threedofs] = hand_cont_threedofs
    hand_cont[..., mask_cont_onedofs] = hand_cont_onedofs

    return hand_cont


def batch9Dfrom6D(poses):
    # Args: poses: ... x 6, where "6" is the combined first and second columns
    # First, get the rotaiton matrix
    x_raw = poses[..., :3]
    y_raw = poses[..., 3:]

    x = F.normalize(x_raw, dim=-1)
    z = torch.cross(x, y_raw, dim=-1)
    z = F.normalize(z, dim=-1)
    y = torch.cross(z, x, dim=-1)

    matrix = torch.stack([x, y, z], dim=-1).flatten(-2, -1)  # ... x 3 x 3 -> x9

    return matrix


def batch4Dfrom2D(poses):
    # Args: poses: ... x 2, where "2" is sincos
    poses_norm = F.normalize(poses, dim=-1)

    poses_4d = torch.stack(
        [
            poses_norm[..., 1],
            poses_norm[..., 0],
            -poses_norm[..., 0],
            poses_norm[..., 1],
        ],
        dim=-1,
    )  # Flattened SO2.

    return poses_4d  # .... x 4


def compact_cont_to_rotmat_body(body_pose_cont, inflate_trans=False):
    # fmt: off
    all_param_3dof_rot_idxs = torch.LongTensor([(0, 2, 4), (6, 8, 10), (12, 13, 14), (15, 16, 17), (18, 19, 20), (21, 22, 23), (24, 25, 26), (27, 28, 29), (34, 35, 36), (37, 38, 39), (44, 45, 46), (53, 54, 55), (64, 65, 66), (85, 69, 73), (86, 70, 79), (87, 71, 82), (88, 72, 76), (91, 92, 93), (112, 96, 100), (113, 97, 106), (114, 98, 109), (115, 99, 103), (130, 131, 132)])
    all_param_1dof_rot_idxs = torch.LongTensor([1, 3, 5, 7, 9, 11, 30, 31, 32, 33, 40, 41, 42, 43, 47, 48, 49, 50, 51, 52, 56, 57, 58, 59, 60, 61, 62, 63, 67, 68, 74, 75, 77, 78, 80, 81, 83, 84, 89, 90, 94, 95, 101, 102, 104, 105, 107, 108, 110, 111, 116, 117, 118, 119, 120, 121, 122, 123])
    all_param_1dof_trans_idxs = torch.LongTensor([124, 125, 126, 127, 128, 129])
    # fmt: on
    num_3dof_angles = len(all_param_3dof_rot_idxs) * 3
    num_1dof_angles = len(all_param_1dof_rot_idxs)
    num_1dof_trans = len(all_param_1dof_trans_idxs)
    assert body_pose_cont.shape[-1] == (
        2 * num_3dof_angles + 2 * num_1dof_angles + num_1dof_trans
    )
    # Get subsets
    body_cont_3dofs = body_pose_cont[..., : 2 * num_3dof_angles]
    body_cont_1dofs = body_pose_cont[
        ..., 2 * num_3dof_angles : 2 * num_3dof_angles + 2 * num_1dof_angles
    ]
    body_cont_trans = body_pose_cont[..., 2 * num_3dof_angles + 2 * num_1dof_angles :]
    # Convert conts to model params
    ## First for 3dofs
    body_cont_3dofs = body_cont_3dofs.unflatten(-1, (-1, 6))
    body_rotmat_3dofs = batch9Dfrom6D(body_cont_3dofs).flatten(-2, -1)
    ## Next for 1dofs
    body_cont_1dofs = body_cont_1dofs.unflatten(-1, (-1, 2))  # (sincos)
    body_rotmat_1dofs = batch4Dfrom2D(body_cont_1dofs).flatten(-2, -1)
    if inflate_trans:
        assert (
            False
        ), "This is left as a possibility to increase the space/contribution/supervision trans params gets compared to rots"
    else:
        ## Nothing to do for trans
        body_rotmat_trans = body_cont_trans
    # Put them together
    body_rotmat_params = torch.cat(
        [body_rotmat_3dofs, body_rotmat_1dofs, body_rotmat_trans], dim=-1
    )
    return body_rotmat_params


def compact_cont_to_model_params_body(body_pose_cont):
    # fmt: off
    all_param_3dof_rot_idxs = torch.LongTensor([(0, 2, 4), (6, 8, 10), (12, 13, 14), (15, 16, 17), (18, 19, 20), (21, 22, 23), (24, 25, 26), (27, 28, 29), (34, 35, 36), (37, 38, 39), (44, 45, 46), (53, 54, 55), (64, 65, 66), (85, 69, 73), (86, 70, 79), (87, 71, 82), (88, 72, 76), (91, 92, 93), (112, 96, 100), (113, 97, 106), (114, 98, 109), (115, 99, 103), (130, 131, 132)])
    all_param_1dof_rot_idxs = torch.LongTensor([1, 3, 5, 7, 9, 11, 30, 31, 32, 33, 40, 41, 42, 43, 47, 48, 49, 50, 51, 52, 56, 57, 58, 59, 60, 61, 62, 63, 67, 68, 74, 75, 77, 78, 80, 81, 83, 84, 89, 90, 94, 95, 101, 102, 104, 105, 107, 108, 110, 111, 116, 117, 118, 119, 120, 121, 122, 123])
    all_param_1dof_trans_idxs = torch.LongTensor([124, 125, 126, 127, 128, 129])
    # fmt: on
    num_3dof_angles = len(all_param_3dof_rot_idxs) * 3
    num_1dof_angles = len(all_param_1dof_rot_idxs)
    num_1dof_trans = len(all_param_1dof_trans_idxs)
    assert body_pose_cont.shape[-1] == (
        2 * num_3dof_angles + 2 * num_1dof_angles + num_1dof_trans
    )
    # Get subsets
    body_cont_3dofs = body_pose_cont[..., : 2 * num_3dof_angles]
    body_cont_1dofs = body_pose_cont[
        ..., 2 * num_3dof_angles : 2 * num_3dof_angles + 2 * num_1dof_angles
    ]
    body_cont_trans = body_pose_cont[..., 2 * num_3dof_angles + 2 * num_1dof_angles :]
    # Convert conts to model params
    ## First for 3dofs
    body_cont_3dofs = body_cont_3dofs.unflatten(-1, (-1, 6))
    body_params_3dofs = batchXYZfrom6D(body_cont_3dofs).flatten(-2, -1)
    ## Next for 1dofs
    body_cont_1dofs = body_cont_1dofs.unflatten(-1, (-1, 2))  # (sincos)
    body_params_1dofs = torch.atan2(body_cont_1dofs[..., -2], body_cont_1dofs[..., -1])
    ## Nothing to do for trans
    body_params_trans = body_cont_trans
    # Put them together
    body_pose_params = torch.zeros(*body_pose_cont.shape[:-1], 133).to(body_pose_cont)
    body_pose_params[..., all_param_3dof_rot_idxs.flatten()] = body_params_3dofs
    body_pose_params[..., all_param_1dof_rot_idxs] = body_params_1dofs
    body_pose_params[..., all_param_1dof_trans_idxs] = body_params_trans
    return body_pose_params


def compact_model_params_to_cont_body(body_pose_params):
    # fmt: off
    all_param_3dof_rot_idxs = torch.LongTensor([(0, 2, 4), (6, 8, 10), (12, 13, 14), (15, 16, 17), (18, 19, 20), (21, 22, 23), (24, 25, 26), (27, 28, 29), (34, 35, 36), (37, 38, 39), (44, 45, 46), (53, 54, 55), (64, 65, 66), (85, 69, 73), (86, 70, 79), (87, 71, 82), (88, 72, 76), (91, 92, 93), (112, 96, 100), (113, 97, 106), (114, 98, 109), (115, 99, 103), (130, 131, 132)])
    all_param_1dof_rot_idxs = torch.LongTensor([1, 3, 5, 7, 9, 11, 30, 31, 32, 33, 40, 41, 42, 43, 47, 48, 49, 50, 51, 52, 56, 57, 58, 59, 60, 61, 62, 63, 67, 68, 74, 75, 77, 78, 80, 81, 83, 84, 89, 90, 94, 95, 101, 102, 104, 105, 107, 108, 110, 111, 116, 117, 118, 119, 120, 121, 122, 123])
    all_param_1dof_trans_idxs = torch.LongTensor([124, 125, 126, 127, 128, 129])
    # fmt: on
    num_3dof_angles = len(all_param_3dof_rot_idxs) * 3
    num_1dof_angles = len(all_param_1dof_rot_idxs)
    num_1dof_trans = len(all_param_1dof_trans_idxs)
    assert body_pose_params.shape[-1] == (
        num_3dof_angles + num_1dof_angles + num_1dof_trans
    )
    # Take out params
    body_params_3dofs = body_pose_params[..., all_param_3dof_rot_idxs.flatten()]
    body_params_1dofs = body_pose_params[..., all_param_1dof_rot_idxs]
    body_params_trans = body_pose_params[..., all_param_1dof_trans_idxs]
    # params to cont
    body_cont_3dofs = batch6DFromXYZ(body_params_3dofs.unflatten(-1, (-1, 3))).flatten(
        -2, -1
    )
    body_cont_1dofs = torch.stack(
        [body_params_1dofs.sin(), body_params_1dofs.cos()], dim=-1
    ).flatten(-2, -1)
    body_cont_trans = body_params_trans
    # Put them together
    body_pose_cont = torch.cat(
        [body_cont_3dofs, body_cont_1dofs, body_cont_trans], dim=-1
    )
    return body_pose_cont


def convert_flow_samples_to_mhr_params(
    aa_3dof_samples: torch.Tensor,
    params_1dofs_samples: torch.Tensor,
    pose_mean: torch.Tensor,
) -> torch.Tensor:
    """Convert flow-space axis-angle / 1-DOF samples back to 133D MHR Euler params.

    Args:
        aa_3dof_samples: (B, N, 39) axis-angle for 13 3-DOF joints (excl. hands).
        params_1dofs_samples: (B, N, 34) scalar angles for 1-DOF joints (excl. hands).
        pose_mean: (B, 133) MHR Euler-angle base pose (expanded to (B, N, 133) internally).

    Returns:
        (B, N, 133) MHR Euler-angle pose parameters.
    """
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

    euler_3dof_samples = euler_3dof_samples.flatten(-2, -1)

    pose_mean[..., all_param_3dof_rot_idxs_except_hands.flatten()] = (
        euler_3dof_samples
    )
    pose_mean[..., all_param_1dof_rot_idxs_except_hands] = params_1dofs_samples
    pose_mean[..., mhr_param_hand_mask] = 0
    pose_mean[..., -3:] = 0
    return pose_mean


def convert_pose_cont_to_flow_context(pose_cont: torch.Tensor) -> dict:
    """Convert continuous pose representation to axis-angle / 1-DOF params for flow context.

    Args:
        pose_cont: (B, 204) continuous pose (6D rotations + sin/cos + translations),
            excluding the first 6 global-rotation dims.

    Returns:
        dict with keys ``aa_3dofs`` (B, 39) and ``params_1dofs`` (B, 34).
    """
    assert pose_cont.shape[-1] == (
        2 * num_3dof_angles + 2 * num_1dof_angles + num_1dof_trans
    )
    cont_3dofs = pose_cont[..., : 2 * num_3dof_angles]
    cont_1dofs = pose_cont[
        ..., 2 * num_3dof_angles : 2 * num_3dof_angles + 2 * num_1dof_angles
    ]

    cont_3dofs = cont_3dofs.unflatten(-1, (-1, 6))
    rotmat_3dofs = batch9Dfrom6D(cont_3dofs).unflatten(-1, (3, 3))
    rotmat_3dofs_selected = rotmat_3dofs[:, indices_3dof, ...]  # (B, 13, 3, 3)

    aa_3dofs = matrix_to_axis_angle(rotmat_3dofs_selected).flatten(-2, -1)

    cont_1dofs = cont_1dofs.unflatten(-1, (-1, 2))  # (sincos)
    params_1dofs = torch.atan2(cont_1dofs[..., -2], cont_1dofs[..., -1])
    params_1dofs = params_1dofs[:, indices_1dof]

    return {
        "aa_3dofs": aa_3dofs,
        "params_1dofs": params_1dofs,
        "rotmat_3dofs": rotmat_3dofs_selected,  # (B, 13, 3, 3)
    }


"""
Constants describing the MHR body parameterization.
These mirror the index lists used above for 3-DoF and 1-DoF body joints.
"""

# Number of 3-DoF body joints used in the compact body pose representation.
NUM_BODY_3DOF_JOINTS = 23
NUM_BODY_1DOF_ANGLES = 58

# fmt: off
mhr_param_hand_idxs = [62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115]
mhr_cont_hand_idxs = [72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,190,191,192,193,194,195,196,197,198,199,200,201,202,203,204,205,206,207,208,209,210,211,212,213,214,215,216,217,218,219,220,221,222,223,224,225,226,227,228,229,230,231,232,233,234,235,236,237]
mhr_param_hand_mask = torch.zeros(133).bool(); mhr_param_hand_mask[mhr_param_hand_idxs] = True
mhr_cont_hand_mask = torch.zeros(260).bool(); mhr_cont_hand_mask[mhr_cont_hand_idxs] = True
# fmt: on
