import os
import torch
import pytorch_lightning as pl
import numpy as np


def compute_similarity_transform(S1, S2):
    """
    Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    """

    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.T
        S2 = S2.T
        transposed = True
    assert S2.shape[1] == S1.shape[1]

    # 1. Remove mean.
    mu1 = S1.mean(axis=1, keepdims=True)
    mu2 = S2.mean(axis=1, keepdims=True)
    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = np.sum(X1**2)

    # 3. The outer product of X1 and X2.
    K = X1.dot(X2.T)

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, s, Vh = np.linalg.svd(K)
    V = Vh.T
    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = np.eye(U.shape[0])
    Z[-1, -1] *= np.sign(np.linalg.det(U.dot(V.T)))
    # Construct R.
    R = V.dot(Z.dot(U.T))

    # 5. Recover scale.
    scale = np.trace(R.dot(K)) / var1

    # 6. Recover translation.
    t = mu2 - scale * (R.dot(mu1))

    # 7. Error:
    S1_hat = scale * R.dot(S1) + t

    if transposed:
        S1_hat = S1_hat.T

    return S1_hat


def compute_similarity_transform_batch(S1, S2):
    """Batched version of compute_similarity_transform."""
    S1_hat = np.zeros_like(S1)
    for i in range(S1.shape[0]):
        S1_hat[i] = compute_similarity_transform(S1[i], S2[i])
    return S1_hat


def compute_similarity_transform_batch_torch(S1, S2):
    """
    Batched orthogonal procrustes (with scale + translation) in torch.
    S1, S2: (B, N, 3). Returns aligned S1 of same shape.
    ~100x faster than the numpy version when run on GPU.
    """
    mu1 = S1.mean(dim=-2, keepdim=True)          # (B, 1, 3)
    mu2 = S2.mean(dim=-2, keepdim=True)
    X1 = S1 - mu1
    X2 = S2 - mu2
    var1 = (X1 ** 2).sum(dim=(-2, -1))           # (B,)

    K = X1.transpose(-2, -1) @ X2                # (B, 3, 3)
    U, _, Vh = torch.linalg.svd(K)
    V = Vh.transpose(-2, -1)

    sign = torch.sign(torch.det(U @ Vh))         # (B,)
    Z = torch.eye(3, device=S1.device, dtype=S1.dtype).expand(S1.shape[0], 3, 3).clone()
    Z[:, -1, -1] = Z[:, -1, -1] * sign

    R = V @ Z @ U.transpose(-2, -1)
    trRK = torch.einsum("bij,bji->b", R, K)
    scale = (trRK / var1).view(-1, 1, 1)         # (B, 1, 1)

    S1_rot = S1 @ R.transpose(-2, -1)
    t = mu2 - scale * (mu1 @ R.transpose(-2, -1))
    return scale * S1_rot + t


def reconstruction_error(S1, S2, reduction="mean"):
    """Do Procrustes alignment and compute reconstruction error."""
    S1_hat = compute_similarity_transform_batch(S1, S2)

    re_per_joint = np.sqrt(((S1_hat - S2) ** 2).sum(axis=-1))
    re = re_per_joint
    if reduction == "mean":
        re = re.mean()
    elif reduction == "sum":
        re = re.sum()
    else:
        re = re
    return re, re_per_joint


def scale_and_translation_transform_batch(P, T):
    """
    First Normalises batch of input 3D meshes P such that each mesh has mean (0, 0, 0) and
    RMS distance from mean = 1.
    Then transforms P such that it has the same mean and RMSD as T.
    :param P: (batch_size, N, 3) batch of 3D meshes to transform.
    :param T: (batch_size, N, 3) batch of reference 3D meshes.
    :return: P transformed
    """
    P_mean = np.mean(P, axis=-2, keepdims=True)
    P_trans = P - P_mean
    P_scale = np.sqrt(np.sum(P_trans**2, axis=(-2, -1), keepdims=True) / P.shape[-2])
    P_normalised = P_trans / P_scale

    T_mean = np.mean(T, axis=-2, keepdims=True)
    T_scale = np.sqrt(
        np.sum((T - T_mean) ** 2, axis=(-2, -1), keepdims=True) / T.shape[-2]
    )
    P_transformed = P_normalised * T_scale + T_mean

    return P_transformed


def scale_and_translation_transform_batch_torch(P, T):
    """
    First Normalises batch of input 3D meshes P such that each mesh has mean (0, 0, 0) and
    RMS distance from mean = 1.
    Then transforms P such that it has the same mean and RMSD as T.
    :param P: (batch_size, N, 3) batch of 3D meshes to transform.
    :param T: (batch_size, N, 3) batch of reference 3D meshes.
    :return: P transformed
    """
    P_mean = torch.mean(P, dim=1, keepdim=True)
    P_trans = P - P_mean
    P_scale = torch.sqrt(torch.sum(P_trans**2, dim=(1, 2), keepdim=True) / P.shape[1])
    P_normalised = P_trans / P_scale

    T_mean = torch.mean(T, dim=1, keepdim=True)
    T_scale = torch.sqrt(
        torch.sum((T - T_mean) ** 2, dim=(1, 2), keepdim=True) / T.shape[1]
    )

    P_transformed = P_normalised * T_scale + T_mean

    return P_transformed


def scale_and_translation_transform_batch_masked(P, T, mask):
    """Like ``scale_and_translation_transform_batch`` but the centroid + RMS scale
    are computed only over the masked subset of vertices. The resulting (translation,
    scale) transform is then applied to **all** vertices so the full mesh remains
    available for rendering / downstream use.

    :param P: (..., N, 3) numpy array.
    :param T: (..., N, 3) numpy array (broadcasted to P).
    :param mask: (N,) boolean numpy array selecting vertices used for alignment.
    """
    Pb = P[..., mask, :]
    Tb = T[..., mask, :]
    P_mean = Pb.mean(axis=-2, keepdims=True)
    P_scale = np.sqrt(((Pb - P_mean) ** 2).sum(axis=(-2, -1), keepdims=True) / Pb.shape[-2])
    T_mean = Tb.mean(axis=-2, keepdims=True)
    T_scale = np.sqrt(((Tb - T_mean) ** 2).sum(axis=(-2, -1), keepdims=True) / Tb.shape[-2])
    return (P - P_mean) / P_scale * T_scale + T_mean


def scale_and_translation_transform_batch_torch_masked(P, T, mask):
    """Torch version of ``scale_and_translation_transform_batch_masked``.

    :param P: (B, N, 3) torch tensor.
    :param T: (B, N, 3) torch tensor.
    :param mask: (N,) boolean torch tensor.
    """
    Pb = P[:, mask, :]
    Tb = T[:, mask, :]
    P_mean = Pb.mean(dim=1, keepdim=True)
    P_scale = torch.sqrt(((Pb - P_mean) ** 2).sum(dim=(1, 2), keepdim=True) / Pb.shape[1])
    T_mean = Tb.mean(dim=1, keepdim=True)
    T_scale = torch.sqrt(((Tb - T_mean) ** 2).sum(dim=(1, 2), keepdim=True) / Tb.shape[1])
    return (P - P_mean) / P_scale * T_scale + T_mean


_BODY_MASK_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "assets",
    "head_hand_mask.npz",
)


def load_body_vertex_mask_np(path: str = _BODY_MASK_PATH) -> np.ndarray:
    """Load the body-only vertex mask (head + hands excluded) as a bool array of
    shape (18439,)."""
    data = np.load(path)
    return data["body_mask"] > 0.5


# MHR rig has 127 joints. Body joints exclude hands (42-64 right, 78-100 left),
# face (114-126: jaw, teeth, tongue, eyes), and procedural twist joints (the
# 38 ``*_twist*_proc`` joints driven deterministically by parent rotations
# for skinning correction — not anatomical landmarks).
# Anatomical body joints retained:
#   0-8   : body_world, root, l_upleg, l_lowleg, l_foot, l_talocrural, l_subtalar, l_transversetarsal, l_ball
#   18-24 : r_upleg, r_lowleg, r_foot, r_talocrural, r_subtalar, r_transversetarsal, r_ball
#   34-41 : c_spine0-3, r_clavicle, r_uparm, r_lowarm, r_wrist_twist
#   74-77 : l_clavicle, l_uparm, l_lowarm, l_wrist_twist
#   110   : c_neck
#   113   : c_head
BODY_JOINT_INDICES_127 = (
    list(range(0, 9))
    + list(range(18, 25))
    + list(range(34, 42))
    + list(range(74, 78))
    + [110, 113]
)


def make_body_joint_mask_127() -> np.ndarray:
    mask = np.zeros(127, dtype=bool)
    mask[BODY_JOINT_INDICES_127] = True
    return mask


def compute_similarity_transform_batch_torch_masked(S1, S2, mask):
    """Procrustes (rotation + scale + translation) where the transform is fit
    on ``S1[:, mask, :]`` and ``S2[:, mask, :]`` only, then applied to all of
    S1. Returns aligned S1 of shape (B, N, 3)."""
    S1m = S1[:, mask, :]
    S2m = S2[:, mask, :]
    mu1 = S1m.mean(dim=-2, keepdim=True)
    mu2 = S2m.mean(dim=-2, keepdim=True)
    X1 = S1m - mu1
    X2 = S2m - mu2
    var1 = (X1 ** 2).sum(dim=(-2, -1))

    K = X1.transpose(-2, -1) @ X2
    U, _, Vh = torch.linalg.svd(K)
    V = Vh.transpose(-2, -1)
    sign = torch.sign(torch.det(U @ Vh))
    Z = torch.eye(3, device=S1.device, dtype=S1.dtype).expand(S1.shape[0], 3, 3).clone()
    Z[:, -1, -1] = Z[:, -1, -1] * sign
    R = V @ Z @ U.transpose(-2, -1)
    trRK = torch.einsum("bij,bji->b", R, K)
    scale = (trRK / var1).view(-1, 1, 1)
    S1_rot = S1 @ R.transpose(-2, -1)
    t = mu2 - scale * (mu1 @ R.transpose(-2, -1))
    return scale * S1_rot + t




def mpjpe(pred, gt, reduction="mean"):
    if reduction == "mean":
        return torch.sqrt(((pred - gt) ** 2).sum(dim=-1)).mean()
    elif reduction == "sum":
        return torch.sqrt(((pred - gt) ** 2).sum(dim=-1)).sum()
    else:
        return torch.sqrt(((pred - gt) ** 2).sum(dim=-1))

def pampjpe(pred, gt, reduction="mean"):
    r_error, _ = reconstruction_error(pred, gt, reduction=None)
    if reduction == "mean":
        return r_error.mean()
    elif reduction == "sum":
        return r_error.sum()
    else:
        return r_error

def pve(pred, gt, reduction="mean"):
    if reduction == "mean":
        return torch.sqrt(((pred - gt) ** 2).sum(dim=-1)).mean()
    elif reduction == "sum":
        return torch.sqrt(((pred - gt) ** 2).sum(dim=-1)).sum()
    else:
        return torch.sqrt(((pred - gt) ** 2).sum(dim=-1))

def avg_kp2d_pixel(pred, gt, metrics="l1", reduction="mean"):
    if metrics == "l1":
        if reduction == "mean":
            return torch.abs(pred - gt).mean()
        elif reduction == "sum":
            return torch.abs(pred - gt).sum()
        else:
            return torch.abs(pred - gt)
    elif metrics == "l2":
        if reduction == "mean":
            return torch.sqrt(((pred - gt) ** 2).mean(dim=-1)).mean()
        elif reduction == "sum":
            return torch.sqrt(((pred - gt) ** 2).sum(dim=-1)).sum()
        else:
            return torch.sqrt(((pred - gt) ** 2).sum(dim=-1))

def pvetsc(pred, gt):
    pred_tpose_vertices_sc = scale_and_translation_transform_batch(pred, gt)
    pvet_sc_batch = np.linalg.norm(
        pred_tpose_vertices_sc - gt, axis=-1
    )  # (bs, 6890)
    
    return pvet_sc_batch.mean()


class Metrics(pl.LightningModule):
    def __init__(self, mhr_head=None):
        super().__init__()
        object.__setattr__(self, "mhr_head", mhr_head)
        body_mask_np = load_body_vertex_mask_np()
        self.register_buffer(
            "body_vertex_mask", torch.from_numpy(body_mask_np), persistent=False
        )

    def _neutral_forward(self, shape_params, scale_offsets, templates):
        """Run mhr_forward with zero global_rot/pose/hand/face; returns Y/Z-flipped verts."""
        n = shape_params.shape[0]

        def _zero_like(t):
            return torch.zeros_like(t[:1]).expand(n, *t.shape[1:]).contiguous()

        verts, _, _, _, _ = self.mhr_head.mhr_forward(
            shape_params=shape_params,
            scale_offsets=scale_offsets,
            global_trans=_zero_like(templates["global_rot"]),
            global_rot=_zero_like(templates["global_rot"]),
            body_pose_params=_zero_like(templates["body_pose"]),
            hand_pose_params=_zero_like(templates["hand"]),
            expr_params=_zero_like(templates["face"]),
            return_keypoints=True,
            return_joint_coords=True,
            return_model_params=True,
            return_joint_rotations=True,
            do_pcblend=True,
        )
        verts = verts.clone()
        verts[..., [1, 2]] *= -1
        return verts

    @torch.no_grad()
    @torch.amp.autocast(device_type="cuda", enabled=False)
    def forward(self, predictions, batch):
        metrics = {}
        vis_verts = {}

        if "pred_keypoints_3d" in predictions["mhr"]:
            pred_kp3d = predictions["mhr"]["pred_keypoints_3d"]
            gt_kp3d = batch["keypoints_3d"]

            pred_vertices = predictions["mhr"]["pred_vertices"]
            gt_vertices = batch["vertices"]

            mpjpe_per_joint = mpjpe(
                pred_kp3d[:, :70, :],
                gt_kp3d[:, :70, :],
                reduction="none",
            )
            metrics["mpjpe"] = mpjpe_per_joint.mean(dim=-1) * 1000.0

            pred_kp3d_70 = pred_kp3d[:, :70, :]
            gt_kp3d_70 = gt_kp3d[:, :70, :]
            aligned_kp3d = compute_similarity_transform_batch_torch(pred_kp3d_70, gt_kp3d_70)
            pampjpe_per_joint = torch.sqrt(((aligned_kp3d - gt_kp3d_70) ** 2).sum(dim=-1))
            metrics["pampjpe"] = pampjpe_per_joint.mean(dim=-1) * 1000.0

            # pve_mean = pve(pred_vertices, gt_vertices)
            # metrics["pve"] = pve_mean * 1000.0

        if "kp3d_samples" in predictions:
            # fmt: off
            pred_kp3d_samples = predictions["kp3d_samples"]
            B, N = pred_kp3d_samples.shape[:2]
            gt_kp3d_expanded = batch["keypoints_3d"][:, None].expand(-1, N, -1, -1)

            pred_vertices_samples = predictions["verts_samples"]
            gt_vertices_expanded = batch["vertices"][:, None].expand(-1, N, -1, -1)

            mpjpe_samples = mpjpe(
                pred_kp3d_samples[:, :, :70, :],
                gt_kp3d_expanded[:, :, :70, :],
                reduction="none",
            ).mean(dim=-1)
            metrics["mpjpe_samples"] = mpjpe_samples * 1000.0
            metrics["mpjpe_samples_min"] = mpjpe_samples.min(dim=-1).values * 1000.0

            pred_kp3d_samples_70 = pred_kp3d_samples[:, :, :70, :].flatten(0, 1)
            gt_kp3d_samples_70 = gt_kp3d_expanded[:, :, :70, :].flatten(0, 1)
            aligned_kp3d_samples = compute_similarity_transform_batch_torch(
                pred_kp3d_samples_70, gt_kp3d_samples_70
            )
            pampjpe_samples = (
                torch.sqrt(((aligned_kp3d_samples - gt_kp3d_samples_70) ** 2).sum(dim=-1))
                .mean(dim=-1)
                .reshape(B, N)
            )  # meters
            pampjpe_samples_mm = pampjpe_samples * 1000.0
            metrics["pampjpe_samples"] = pampjpe_samples_mm
            metrics["pampjpe_samples_min"] = pampjpe_samples_mm.min(dim=-1).values

            # pve_samples = pve(
            #     pred_vertices_samples, gt_vertices_expanded, reduction="none"
            # ).mean(dim=-1)  # meters
            # metrics["pve_samples"] = pve_samples.mean() * 1000.0
            # metrics["pve_samples_min"] = (
            #     pve_samples.min(dim=-1).values.mean() * 1000.0
            # )

            if "visibility" in batch:
                # pred_kp3d_samples: (B, N, J, 3)
                centered = pred_kp3d_samples - pred_kp3d_samples.mean(
                    dim=1, keepdim=True
                )
                # Distances of each sample to its joint-wise mean: (B, N, J)
                dists_per_sample = torch.sqrt((centered**2).sum(dim=-1))

                per_joint_spread = dists_per_sample.mean(dim=1)

                kp_visibility = batch["visibility"].bool()  # (B, J)

                visible_mask = kp_visibility.float()
                invisible_mask = (~kp_visibility).float()
                # metrics["spread_visible_kp3d"] = (
                #     (per_joint_spread * visible_mask).sum(dim=1) / visible_mask.sum(dim=1)
                # )
                # metrics["spread_invisible_kp3d"] = (
                #     (per_joint_spread * invisible_mask).sum(dim=1)
                #     / invisible_mask.sum(dim=1)
                # )

                # Sample-wise spread over invisible joints: (B, N)
                invisible_mask_samples = invisible_mask.unsqueeze(1).expand(-1, N, -1)
                metrics["spread_invisible_kp3d_samples"] = (
                    (dists_per_sample * invisible_mask_samples).sum(dim=-1)
                    / invisible_mask_samples.sum(dim=-1).clamp(min=1)
                )
            # fmt: on

        if "kp2d_samples_cropped" in predictions:
            pred_kp2d_norm = predictions["mhr"][
                "pred_keypoints_2d_cropped"
            ]  # (B, J, 2)
            gt_kp2d_norm = batch["keypoints_2d"]  # (B, J, 2)

            pred_kp2d_samples_norm = predictions["kp2d_samples_cropped"]  # (B, N, J, 2)
            B, N = pred_kp2d_samples_norm.shape[:2]
            gt_kp2d_samples_norm = batch["keypoints_2d"][:, None].expand(-1, N, -1, -1)

            # batch["img_size"] is (B_orig, V, 2); flatten to match flattened-person B.
            img_size_flat = batch["img_size"].reshape(B, 2)
            pred_kp2d = (pred_kp2d_norm + 0.5) * img_size_flat.unsqueeze(1)
            gt_kp2d = (gt_kp2d_norm + 0.5) * img_size_flat.unsqueeze(1)

            img_size_b = img_size_flat.view(B, 1, 1, 2)
            pred_kp2d_samples = (pred_kp2d_samples_norm + 0.5) * img_size_b
            gt_kp2d_samples = (gt_kp2d_samples_norm + 0.5) * img_size_b

            kp2d_err = avg_kp2d_pixel(
                pred_kp2d, gt_kp2d, metrics="l2", reduction="none"
            )
            metrics["kp2d_pixel_error"] = kp2d_err.mean(dim=-1)
            kp2d_err_samples = avg_kp2d_pixel(
                pred_kp2d_samples, gt_kp2d_samples, metrics="l2", reduction="none"
            )
            metrics["kp2d_samples_pixel_error"] = kp2d_err_samples.mean(dim=-1)

            if "visibility" in batch:
                visibility_mask = batch["visibility"].bool().unsqueeze(1).expand(-1, N, -1)
                metrics["kp2d_samples_pixel_error_visible"] = (
                    (kp2d_err_samples * visibility_mask.float()).sum(dim=-1)
                    / visibility_mask.float().sum(dim=-1)
                )
                vis_mean = batch["visibility"].bool().float()
                metrics["kp2d_pixel_error_visible"] = (
                    (kp2d_err * vis_mean).sum(dim=-1)
                    / vis_mean.sum(dim=-1).clamp(min=1)
                )

        # --- visualisation-aligned vertices (all kept on GPU as tensors) ---
        if "pred_vertices" in predictions["mhr"]:
            gt_v = batch["vertices"]
            pred_v = predictions["mhr"]["pred_vertices"]
            vis_verts["pa_mean_verts"] = compute_similarity_transform_batch_torch(pred_v, gt_v)

        if "verts_samples" in predictions:
            vs = predictions["verts_samples"]  # (B, N, V, 3)
            B, N = vs.shape[:2]
            gt_v = batch["vertices"]
            gt_v_exp = gt_v[:, None].expand(-1, N, -1, -1)
            pa_flat = compute_similarity_transform_batch_torch(
                vs.flatten(0, 1), gt_v_exp.flatten(0, 1)
            )
            vis_verts["pa_sample_verts"] = pa_flat.reshape(B, N, *pa_flat.shape[1:])

        if self.mhr_head is not None and "shape" in predictions["mhr"]:
            mhr_out = predictions["mhr"]
            templates = {
                k: mhr_out[k] for k in ("global_rot", "body_pose", "hand", "face")
            }
            pred_shape = mhr_out["shape"]
            pred_scale = mhr_out["scale_68D"]
            gt_shape = batch["shape_params"]
            gt_scale = batch["model_params"][:, -68:]

            pred_neutral = self._neutral_forward(pred_shape, pred_scale, templates)
            gt_neutral = self._neutral_forward(gt_shape, gt_scale, templates)
            vis_verts["pred_neutral_verts"] = pred_neutral
            vis_verts["gt_neutral_verts"] = gt_neutral

            metrics["pve"] = torch.sqrt(((pred_neutral - gt_neutral) ** 2).sum(dim=-1)).mean(dim=-1) * 1000.0

            pred_neutral_sc = scale_and_translation_transform_batch_torch(pred_neutral, gt_neutral)
            vis_verts["pred_neutral_verts_sc"] = pred_neutral_sc
            metrics["pvetsc"] = torch.sqrt(
                ((pred_neutral_sc - gt_neutral) ** 2).sum(dim=-1)
            ).mean(dim=-1) * 1000.0

            body_mask = self.body_vertex_mask.to(pred_neutral.device)
            pred_neutral_sc_body = scale_and_translation_transform_batch_torch_masked(
                pred_neutral, gt_neutral, body_mask
            )
            vis_verts["pred_neutral_verts_sc_body"] = pred_neutral_sc_body
            metrics["pvetsc_body"] = torch.sqrt(
                ((pred_neutral_sc_body - gt_neutral) ** 2).sum(dim=-1)[:, body_mask]
            ).mean(dim=-1) * 1000.0

            if "shape_samples" in predictions.get("uncertainty_output", {}):
                unc = predictions["uncertainty_output"]
                shape_s = unc["shape_samples"]
                scale_s = unc["scale_samples"]
                B, S = shape_s.shape[:2]
                sample_neutral = self._neutral_forward(
                    shape_s.reshape(B * S, -1), scale_s.reshape(B * S, -1), templates
                ).reshape(B, S, *pred_neutral.shape[1:])
                vis_verts["sample_neutral_verts"] = sample_neutral

                gt_neutral_exp = gt_neutral[:, None].expand(-1, S, -1, -1)
                sample_neutral_sc = scale_and_translation_transform_batch_torch(
                    sample_neutral.reshape(B * S, *sample_neutral.shape[2:]),
                    gt_neutral_exp.reshape(B * S, *sample_neutral.shape[2:]),
                ).reshape(B, S, *sample_neutral.shape[2:])
                vis_verts["sample_neutral_verts_sc"] = sample_neutral_sc

                sample_neutral_sc_body = scale_and_translation_transform_batch_torch_masked(
                    sample_neutral.reshape(B * S, *sample_neutral.shape[2:]),
                    gt_neutral_exp.reshape(B * S, *sample_neutral.shape[2:]),
                    body_mask,
                ).reshape(B, S, *sample_neutral.shape[2:])
                vis_verts["sample_neutral_verts_sc_body"] = sample_neutral_sc_body

                metrics["pve_samples"] = torch.sqrt(
                    ((sample_neutral - gt_neutral[:, None]) ** 2).sum(dim=-1)
                ).mean(dim=-1) * 1000.0
                metrics["pvetsc_samples"] = torch.sqrt(
                    ((sample_neutral_sc - gt_neutral[:, None]) ** 2).sum(dim=-1)
                ).mean(dim=-1) * 1000.0
                metrics["pvetsc_body_samples"] = torch.sqrt(
                    ((sample_neutral_sc_body - gt_neutral[:, None]) ** 2).sum(dim=-1)[..., body_mask]
                ).mean(dim=-1) * 1000.0

        return metrics, vis_verts



def _pampjpe_torch(pred, gt):
    """Procrustes-aligned MPJPE per item (mean over joints). Returns (B,)."""
    aligned = compute_similarity_transform_batch_torch(pred, gt)
    return torch.sqrt(((aligned - gt) ** 2).sum(dim=-1)).mean(dim=-1)


def _pampjpe_body_torch(pred, gt, body_joint_mask):
    """Body-only PA-MPJPE. Alignment uses only body joints; error reduces over them."""
    aligned = compute_similarity_transform_batch_torch_masked(pred, gt, body_joint_mask)
    return torch.sqrt(((aligned - gt) ** 2).sum(dim=-1)[:, body_joint_mask]).mean(dim=-1)


def _pvetsc_torch(pred, gt):
    """Scale+translation-aligned PVE per item (mean over vertices). Returns (B,)."""
    pred_sc = scale_and_translation_transform_batch_torch(pred, gt)
    return torch.sqrt(((pred_sc - gt) ** 2).sum(dim=-1)).mean(dim=-1)


def _pvetsc_body_torch(pred, gt, body_mask):
    """Body-only PVE-T-SC. Alignment uses only body verts; error is meaned over them."""
    pred_sc_body = scale_and_translation_transform_batch_torch_masked(pred, gt, body_mask)
    return torch.sqrt(((pred_sc_body - gt) ** 2).sum(dim=-1)[:, body_mask]).mean(dim=-1)


def multiframe_metrics(
    all_metrics,
    mhr_dict,
    batch_idx=None,
    save_dir=None
):
    gt_neutral_jcoords = mhr_dict["gt_neutral_jcoords"]
    per_view_neutral_jcoords = mhr_dict["per_view_neutral_jcoords"]
    avg_neutral_jcoords = mhr_dict["avg_neutral_jcoords"]
    merged_neutral_jcoords = mhr_dict["merged_neutral_jcoords"]

    gt_neutral_verts = mhr_dict["gt_neutral_verts"]
    per_view_neutral_verts = mhr_dict["per_view_neutral_verts"]
    avg_neutral_verts = mhr_dict["avg_neutral_verts"]
    merged_neutral_verts = mhr_dict["merged_neutral_verts"]

    # ---------------- mpjpe ----------------
    per_view_mpjpe = torch.sqrt(((per_view_neutral_jcoords - gt_neutral_jcoords) ** 2).sum(dim=-1)).mean(dim=1)
    avg_mpjpe = torch.sqrt(((avg_neutral_jcoords - gt_neutral_jcoords) ** 2).sum(dim=-1)).mean(dim=1)
    merged_mpjpe = torch.sqrt(((merged_neutral_jcoords - gt_neutral_jcoords) ** 2).sum(dim=-1)).mean(dim=1)

    # ---------------- pve ----------------
    per_view_pve = torch.sqrt(((per_view_neutral_verts - gt_neutral_verts) ** 2).sum(dim=-1)).mean(dim=1)
    avg_pve = torch.sqrt(((avg_neutral_verts - gt_neutral_verts) ** 2).sum(dim=-1)).mean(dim=1)
    merged_pve = torch.sqrt(((merged_neutral_verts - gt_neutral_verts) ** 2).sum(dim=-1)).mean(dim=1)

    # ---------------- pampjpe (batched torch Procrustes) ----------------
    per_view_pampjpe = _pampjpe_torch(per_view_neutral_jcoords, gt_neutral_jcoords)
    avg_pampjpe = _pampjpe_torch(avg_neutral_jcoords, gt_neutral_jcoords)
    merged_pampjpe = _pampjpe_torch(merged_neutral_jcoords, gt_neutral_jcoords)

    # ---------------- pampjpe_body (alignment + error over body joints only) -------
    body_joint_mask = torch.from_numpy(make_body_joint_mask_127()).to(per_view_neutral_jcoords.device)
    per_view_pampjpe_body = _pampjpe_body_torch(per_view_neutral_jcoords, gt_neutral_jcoords, body_joint_mask)
    avg_pampjpe_body = _pampjpe_body_torch(avg_neutral_jcoords, gt_neutral_jcoords, body_joint_mask)
    merged_pampjpe_body = _pampjpe_body_torch(merged_neutral_jcoords, gt_neutral_jcoords, body_joint_mask)

    # ---------------- pvetsc (batched torch scale+trans) ----------------
    per_view_pvetsc = _pvetsc_torch(per_view_neutral_verts, gt_neutral_verts)
    merged_pvetsc = _pvetsc_torch(merged_neutral_verts, gt_neutral_verts)
    avg_pvetsc = _pvetsc_torch(avg_neutral_verts, gt_neutral_verts)

    # ---------------- pvetsc_body (alignment + error over body verts only) ----------
    # Compute body-aligned verts once and stash on ``mhr_dict`` so that
    # downstream visualisation (``vis_merging_neutral``) can reuse them rather
    # than redoing the masked Procrustes.
    body_mask = torch.from_numpy(load_body_vertex_mask_np()).to(per_view_neutral_verts.device)

    def _align_body(verts):
        return scale_and_translation_transform_batch_torch_masked(
            verts, gt_neutral_verts, body_mask
        )

    def _pvetsc_from_aligned(aligned):
        return torch.sqrt(((aligned - gt_neutral_verts) ** 2).sum(-1)[:, body_mask]).mean(-1)

    per_view_neutral_verts_sc_body = _align_body(per_view_neutral_verts)
    merged_neutral_verts_sc_body = _align_body(merged_neutral_verts)
    avg_neutral_verts_sc_body = _align_body(avg_neutral_verts)
    mhr_dict["per_view_neutral_verts_sc_body"] = per_view_neutral_verts_sc_body
    mhr_dict["merged_neutral_verts_sc_body"] = merged_neutral_verts_sc_body
    mhr_dict["avg_neutral_verts_sc_body"] = avg_neutral_verts_sc_body

    per_view_pvetsc_body = _pvetsc_from_aligned(per_view_neutral_verts_sc_body)
    merged_pvetsc_body = _pvetsc_from_aligned(merged_neutral_verts_sc_body)
    avg_pvetsc_body = _pvetsc_from_aligned(avg_neutral_verts_sc_body)

    # ---------------- best NF sample by log-prob (per-view argmax of stage-1 log-prob) ----------------
    best_logprob_sample_mpjpe = best_logprob_sample_pve = best_logprob_sample_pampjpe = best_logprob_sample_pvetsc = best_logprob_sample_pvetsc_body = best_logprob_sample_pampjpe_body = None
    if "best_logprob_sample_neutral_verts" in mhr_dict:
        best_logprob_neutral_verts = mhr_dict["best_logprob_sample_neutral_verts"]
        best_logprob_neutral_jcoords = mhr_dict["best_logprob_sample_neutral_jcoords"]

        best_logprob_sample_mpjpe = torch.sqrt(((best_logprob_neutral_jcoords - gt_neutral_jcoords) ** 2).sum(dim=-1)).mean(dim=1)
        best_logprob_sample_pve = torch.sqrt(((best_logprob_neutral_verts - gt_neutral_verts) ** 2).sum(dim=-1)).mean(dim=1)

        best_logprob_sample_pampjpe = _pampjpe_torch(best_logprob_neutral_jcoords, gt_neutral_jcoords)
        best_logprob_sample_pampjpe_body = _pampjpe_body_torch(best_logprob_neutral_jcoords, gt_neutral_jcoords, body_joint_mask)
        best_logprob_sample_pvetsc = _pvetsc_torch(best_logprob_neutral_verts, gt_neutral_verts)
        best_logprob_neutral_verts_sc_body = _align_body(best_logprob_neutral_verts)
        mhr_dict["best_logprob_sample_neutral_verts_sc_body"] = best_logprob_neutral_verts_sc_body
        best_logprob_sample_pvetsc_body = _pvetsc_from_aligned(best_logprob_neutral_verts_sc_body)

    # ---------------- avg / oracle-best over all NF samples ----------------
    avg_sample_mpjpe = avg_sample_pve = avg_sample_pampjpe = avg_sample_pvetsc = avg_sample_pvetsc_body = avg_sample_pampjpe_body = None
    best_metric_sample_mpjpe = best_metric_sample_pve = best_metric_sample_pampjpe = best_metric_sample_pvetsc = best_metric_sample_pvetsc_body = best_metric_sample_pampjpe_body = None
    if "sample_neutral_verts" in mhr_dict:
        sample_verts = mhr_dict["sample_neutral_verts"]    # [B*V, S, n_verts, 3]
        sample_jcoords = mhr_dict["sample_neutral_jcoords"] # [B*V, S, n_joints, 3]
        BV, S = sample_verts.shape[:2]

        gt_verts_exp = gt_neutral_verts.unsqueeze(1)        # [B*V, 1, n_verts, 3]
        gt_jcoords_exp = gt_neutral_jcoords.unsqueeze(1)    # [B*V, 1, n_joints, 3]

        # per-sample pve/mpjpe: [B*V, S]
        sample_pve_per_s = torch.sqrt(((sample_verts - gt_verts_exp) ** 2).sum(dim=-1)).mean(dim=2)
        sample_mpjpe_per_s = torch.sqrt(((sample_jcoords - gt_jcoords_exp) ** 2).sum(dim=-1)).mean(dim=2)
        avg_sample_pve = sample_pve_per_s.mean(dim=1)
        avg_sample_mpjpe = sample_mpjpe_per_s.mean(dim=1)
        best_metric_sample_pve = sample_pve_per_s.min(dim=1).values
        best_metric_sample_mpjpe = sample_mpjpe_per_s.min(dim=1).values

        sample_verts_flat = sample_verts.reshape(BV * S, *sample_verts.shape[2:])
        sample_jcoords_flat = sample_jcoords.reshape(BV * S, *sample_jcoords.shape[2:])
        gt_verts_tiled = gt_neutral_verts.unsqueeze(1).expand(-1, S, -1, -1).reshape(BV * S, *gt_neutral_verts.shape[1:])
        gt_jcoords_tiled = gt_neutral_jcoords.unsqueeze(1).expand(-1, S, -1, -1).reshape(BV * S, *gt_neutral_jcoords.shape[1:])

        sample_pampjpe_per_s = _pampjpe_torch(sample_jcoords_flat, gt_jcoords_tiled).reshape(BV, S)
        avg_sample_pampjpe = sample_pampjpe_per_s.mean(dim=1)
        best_metric_sample_pampjpe = sample_pampjpe_per_s.min(dim=1).values

        sample_pampjpe_body_per_s = _pampjpe_body_torch(sample_jcoords_flat, gt_jcoords_tiled, body_joint_mask).reshape(BV, S)
        avg_sample_pampjpe_body = sample_pampjpe_body_per_s.mean(dim=1)
        best_metric_sample_pampjpe_body = sample_pampjpe_body_per_s.min(dim=1).values

        sample_pvetsc_per_s = _pvetsc_torch(sample_verts_flat, gt_verts_tiled).reshape(BV, S)
        avg_sample_pvetsc = sample_pvetsc_per_s.mean(dim=1)
        best_metric_sample_pvetsc = sample_pvetsc_per_s.min(dim=1).values

        sample_pvetsc_body_per_s = _pvetsc_body_torch(sample_verts_flat, gt_verts_tiled, body_mask).reshape(BV, S)
        avg_sample_pvetsc_body = sample_pvetsc_body_per_s.mean(dim=1)
        best_metric_sample_pvetsc_body = sample_pvetsc_body_per_s.min(dim=1).values

    # ---------------- sample-param-average (mean of residual samples → MHR once) ----------------
    sample_param_avg_pampjpe = sample_param_avg_pvetsc = sample_param_avg_pvetsc_body = sample_param_avg_pampjpe_body = None
    if "sample_param_avg_neutral_verts" in mhr_dict:
        sp_avg_verts = mhr_dict["sample_param_avg_neutral_verts"]
        sp_avg_jcoords = mhr_dict["sample_param_avg_neutral_jcoords"]

        sample_param_avg_pampjpe = _pampjpe_torch(sp_avg_jcoords, gt_neutral_jcoords)
        sample_param_avg_pampjpe_body = _pampjpe_body_torch(sp_avg_jcoords, gt_neutral_jcoords, body_joint_mask)
        sample_param_avg_pvetsc = _pvetsc_torch(sp_avg_verts, gt_neutral_verts)
        sample_param_avg_neutral_verts_sc_body = _align_body(sp_avg_verts)
        mhr_dict["sample_param_avg_neutral_verts_sc_body"] = sample_param_avg_neutral_verts_sc_body
        sample_param_avg_pvetsc_body = _pvetsc_from_aligned(sample_param_avg_neutral_verts_sc_body)

    # print(f"mpjpe: view avg: {per_view_mpjpe.mean():.4f}, view min: {per_view_mpjpe.min():.4f}, mean: {avg_mpjpe.mean():.4f} merged: {merged_mpjpe.mean():.4f}")
    # print(f"pve: view avg: {per_view_pve.mean():.4f}, view min: {per_view_pve.min():.4f}, mean: {avg_pve.mean():.4f}, merged: {merged_pve.mean():.4f}")
    # print(f"pampjpe: view avg: {per_view_pampjpe.mean():.4f}, view min: {per_view_pampjpe.min():.4f}, mean: {avg_pampjpe.mean():.4f}, merged: {merged_pampjpe.mean():.4f}")
    # print(f"pvetsc: view avg: {per_view_pvetsc.mean():.4f}, view min: {per_view_pvetsc.min():.4f}, mean: {avg_pvetsc.mean():.4f}, merged: {merged_pvetsc.mean():.4f}")
    print(f"pvetsc_body: view avg: {per_view_pvetsc_body.mean():.4f}, view min: {per_view_pvetsc_body.min():.4f}, mean: {avg_pvetsc_body.mean():.4f}, merged: {merged_pvetsc_body.mean():.4f}")

    all_metrics["per_view_mpjpe"].append(per_view_mpjpe)
    all_metrics["best_per_view_mpjpe"].append(per_view_mpjpe.min().item())
    all_metrics["avg_mpjpe"].append(avg_mpjpe)
    all_metrics["merged_mpjpe"].append(merged_mpjpe)

    all_metrics["per_view_pve"].append(per_view_pve)
    all_metrics["best_per_view_pve"].append(per_view_pve.min().item())
    all_metrics["avg_pve"].append(avg_pve)
    all_metrics["merged_pve"].append(merged_pve)

    all_metrics["per_view_pampjpe"].append(per_view_pampjpe)
    all_metrics["best_per_view_pampjpe"].append(per_view_pampjpe.min().item())
    all_metrics["avg_pampjpe"].append(avg_pampjpe)
    all_metrics["merged_pampjpe"].append(merged_pampjpe)

    all_metrics["per_view_pampjpe_body"].append(per_view_pampjpe_body)
    all_metrics["best_per_view_pampjpe_body"].append(per_view_pampjpe_body.min().item())
    all_metrics["avg_pampjpe_body"].append(avg_pampjpe_body)
    all_metrics["merged_pampjpe_body"].append(merged_pampjpe_body)

    all_metrics["per_view_pvetsc"].append(per_view_pvetsc)
    all_metrics["best_per_view_pvetsc"].append(per_view_pvetsc.min().item())
    all_metrics["avg_pvetsc"].append(avg_pvetsc)
    all_metrics["merged_pvetsc"].append(merged_pvetsc)

    all_metrics["per_view_pvetsc_body"].append(per_view_pvetsc_body)
    all_metrics["best_per_view_pvetsc_body"].append(per_view_pvetsc_body.min().item())
    all_metrics["avg_pvetsc_body"].append(avg_pvetsc_body)
    all_metrics["merged_pvetsc_body"].append(merged_pvetsc_body)

    # if best_logprob_sample_mpjpe is not None:
    #     all_metrics["best_logprob_sample_mpjpe"].append(best_logprob_sample_mpjpe)
    #     all_metrics["best_logprob_sample_pve"].append(best_logprob_sample_pve)
    #     all_metrics["best_logprob_sample_pampjpe"].append(best_logprob_sample_pampjpe)
    #     all_metrics["best_logprob_sample_pvetsc"].append(best_logprob_sample_pvetsc)
    #     all_metrics["best_logprob_sample_pvetsc_body"].append(best_logprob_sample_pvetsc_body)

    # if avg_sample_mpjpe is not None:
    #     all_metrics["avg_sample_mpjpe"].append(avg_sample_mpjpe)
    #     all_metrics["avg_sample_pve"].append(avg_sample_pve)
    #     all_metrics["avg_sample_pampjpe"].append(avg_sample_pampjpe)
    #     all_metrics["avg_sample_pvetsc"].append(avg_sample_pvetsc)
    #     all_metrics["avg_sample_pvetsc_body"].append(avg_sample_pvetsc_body)
    #     all_metrics["best_pve_sample_pve"].append(best_metric_sample_pve)
    #     all_metrics["best_mpjpe_sample_mpjpe"].append(best_metric_sample_mpjpe)
    #     all_metrics["best_pampjpe_sample_pampjpe"].append(best_metric_sample_pampjpe)
    #     all_metrics["best_pvetsc_sample_pvetsc"].append(best_metric_sample_pvetsc)
    #     all_metrics["best_pvetsc_body_sample_pvetsc_body"].append(best_metric_sample_pvetsc_body)

    if sample_param_avg_pampjpe is not None:
        all_metrics["sample_param_avg_pampjpe"].append(sample_param_avg_pampjpe)
        all_metrics["sample_param_avg_pampjpe_body"].append(sample_param_avg_pampjpe_body)
        all_metrics["sample_param_avg_pvetsc"].append(sample_param_avg_pvetsc)
        all_metrics["sample_param_avg_pvetsc_body"].append(sample_param_avg_pvetsc_body)
        all_metrics["best_sample_param_avg_pampjpe"].append(sample_param_avg_pampjpe.min().item())
        all_metrics["best_sample_param_avg_pampjpe_body"].append(sample_param_avg_pampjpe_body.min().item())
        all_metrics["best_sample_param_avg_pvetsc"].append(sample_param_avg_pvetsc.min().item())
        all_metrics["best_sample_param_avg_pvetsc_body"].append(sample_param_avg_pvetsc_body.min().item())
        # print(f"sample_param_avg_pampjpe: {sample_param_avg_pampjpe}, sample_param_avg_pvetsc: {sample_param_avg_pvetsc}")
        # print(f"sample_param_avg_pvetsc_body: {sample_param_avg_pvetsc_body}")
        # print(f"best_sample_param_avg_pampjpe: {sample_param_avg_pampjpe.min().item():.4f}, best_sample_param_avg_pvetsc: {sample_param_avg_pvetsc.min().item():.4f}")
        # print(f"best_sample_param_avg_pvetsc_body: {sample_param_avg_pvetsc_body.min().item():.4f}")
        # print(f"sample_param_avg_pvetsc: {sample_param_avg_pvetsc}")
        print(f"sample_param_avg_pvetsc_body: {sample_param_avg_pvetsc_body}")
        # print(f"best_sample_param_avg_pvetsc: {sample_param_avg_pvetsc.min().item():.4f}")
        print(f"best_sample_param_avg_pvetsc_body: {sample_param_avg_pvetsc_body.min().item():.4f}")
        print('')


    return all_metrics


def print_multiview_metrics(
    all_metrics,
    save_dir,
):
    avg_metrics = {}
    for k, v in all_metrics.items():
        try:
            avg_metrics[k] = torch.stack(v).mean().item()
        except:
            avg_metrics[k] = np.mean(np.array(v))

    summary_lines = [
        "=" * 60,
        "Average Metrics:",
        "=" * 60,
    ]

    # print(all_metrics)
    # for k, v in all_metrics.items():
    #     print(f"{k}: {type(v)}")
    # import ipdb; ipdb.set_trace()

    for k, v in avg_metrics.items():
        summary_lines.append(f"{k}: {v:.4f}")
    summary_lines.append("=" * 60)

    for line in summary_lines:
        print(line)

    metrics_path = os.path.join(save_dir, "metrics.txt")
    with open(metrics_path, "w") as f:
        f.write("\n".join(summary_lines) + "\n")
    print(f"Saved metrics summary to {metrics_path}")

    return avg_metrics