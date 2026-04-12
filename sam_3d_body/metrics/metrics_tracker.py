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
    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def forward(self, predictions, batch):
        metrics = {}

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

            pampjpe_per_joint = pampjpe(
                pred_kp3d[:, :70, :].cpu().detach().numpy(),
                gt_kp3d[:, :70, :].cpu().detach().numpy(),
                reduction="none",
            )
            metrics["pampjpe"] = pampjpe_per_joint.mean(axis=-1) * 1000.0

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

            pampjpe_samples = pampjpe(
                pred_kp3d_samples[:, :, :70, :].flatten(0, 1).cpu().detach().numpy(),
                gt_kp3d_expanded[:, :, :70, :].flatten(0, 1).cpu().detach().numpy(),
                reduction="none",
            ).reshape(B, N, -1).mean(axis=-1)  # meters
            pampjpe_samples_mm = pampjpe_samples * 1000.0
            metrics["pampjpe_samples"] = pampjpe_samples_mm
            metrics["pampjpe_samples_min"] = pampjpe_samples_mm.min(axis=-1)

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
            img_size = batch["img_size"]

            pred_kp2d_norm = predictions["mhr"][
                "pred_keypoints_2d_cropped"
            ]  # (B, J, 2)
            gt_kp2d_norm = batch["keypoints_2d"]  # (B, J, 2)

            pred_kp2d = (pred_kp2d_norm + 0.5) * img_size.unsqueeze(1)
            gt_kp2d = (gt_kp2d_norm + 0.5) * img_size.unsqueeze(1)

            pred_kp2d_samples_norm = predictions["kp2d_samples_cropped"]  # (B, N, J, 2)
            B, N = pred_kp2d_samples_norm.shape[:2]
            gt_kp2d_samples_norm = batch["keypoints_2d"][:, None].expand(-1, N, -1, -1)

            img_size_b = img_size.view(B, 1, 1, 2)
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

        # for k, v in metrics.items():
        #     print(f"{k}: {v:.4f}")
        # import ipdb; ipdb.set_trace()

        return metrics



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

    # ---------------- pampjpe ----------------
    per_view_pampjpe, _ = reconstruction_error(
        per_view_neutral_jcoords.cpu().detach().numpy(),
        gt_neutral_jcoords.cpu().detach().numpy(),
        reduction="none",
    )
    per_view_pampjpe = per_view_pampjpe.mean(axis=-1)
    avg_pampjpe, _ = reconstruction_error(
        avg_neutral_jcoords.cpu().detach().numpy(),
        gt_neutral_jcoords.cpu().detach().numpy(),
        reduction="none",
    )
    avg_pampjpe = avg_pampjpe.mean(axis=-1)
    merged_pampjpe, _ = reconstruction_error(
        merged_neutral_jcoords.cpu().detach().numpy(),
        gt_neutral_jcoords.cpu().detach().numpy(),
        reduction="none",
    )
    merged_pampjpe = merged_pampjpe.mean(axis=-1)

    # ---------------- pvetsc ----------------
    pred_sc = scale_and_translation_transform_batch(
        per_view_neutral_verts.cpu().detach().numpy(),
        gt_neutral_verts.cpu().detach().numpy(),
    )
    merged_sc = scale_and_translation_transform_batch(
        merged_neutral_verts.cpu().detach().numpy(),
        gt_neutral_verts.cpu().detach().numpy(),
    )
    avg_sc = scale_and_translation_transform_batch(
        avg_neutral_verts.cpu().detach().numpy(),
        gt_neutral_verts.cpu().detach().numpy(),
    )
    per_view_pvetsc = np.linalg.norm(
        pred_sc - gt_neutral_verts.cpu().detach().numpy(), axis=-1
    )
    per_view_pvetsc = per_view_pvetsc.mean(axis=1)
    merged_pvetsc = np.linalg.norm(
        merged_sc - gt_neutral_verts.cpu().detach().numpy(), axis=-1
    )
    merged_pvetsc = merged_pvetsc.mean(axis=1)
    avg_pvetsc = np.linalg.norm(
        avg_sc - gt_neutral_verts.cpu().detach().numpy(), axis=-1
    )
    avg_pvetsc = avg_pvetsc.mean(axis=1)

    # print(per_view_pvetsc)
    # print(merged_pvetsc)
    # import ipdb; ipdb.set_trace()

    # ---------------- best NF sample by log-prob (per-view argmax of stage-1 log-prob) ----------------
    best_logprob_sample_mpjpe = best_logprob_sample_pve = best_logprob_sample_pampjpe = best_logprob_sample_pvetsc = None
    if "best_logprob_sample_neutral_verts" in mhr_dict:
        best_logprob_neutral_verts = mhr_dict["best_logprob_sample_neutral_verts"]
        best_logprob_neutral_jcoords = mhr_dict["best_logprob_sample_neutral_jcoords"]

        best_logprob_sample_mpjpe = torch.sqrt(((best_logprob_neutral_jcoords - gt_neutral_jcoords) ** 2).sum(dim=-1)).mean(dim=1)
        best_logprob_sample_pve = torch.sqrt(((best_logprob_neutral_verts - gt_neutral_verts) ** 2).sum(dim=-1)).mean(dim=1)

        best_logprob_sample_pampjpe, _ = reconstruction_error(
            best_logprob_neutral_jcoords.cpu().detach().numpy(),
            gt_neutral_jcoords.cpu().detach().numpy(),
            reduction="none",
        )
        best_logprob_sample_pampjpe = best_logprob_sample_pampjpe.mean(axis=-1)

        best_logprob_sc = scale_and_translation_transform_batch(
            best_logprob_neutral_verts.cpu().detach().numpy(),
            gt_neutral_verts.cpu().detach().numpy(),
        )
        best_logprob_sample_pvetsc = np.linalg.norm(
            best_logprob_sc - gt_neutral_verts.cpu().detach().numpy(), axis=-1
        ).mean(axis=1)

    # ---------------- avg / oracle-best over all NF samples ----------------
    avg_sample_mpjpe = avg_sample_pve = avg_sample_pampjpe = avg_sample_pvetsc = None
    best_metric_sample_mpjpe = best_metric_sample_pve = best_metric_sample_pampjpe = best_metric_sample_pvetsc = None
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

        sample_pampjpe_flat, _ = reconstruction_error(
            sample_jcoords_flat.cpu().detach().numpy(),
            gt_jcoords_tiled.cpu().detach().numpy(),
            reduction="none",
        )
        sample_pampjpe_per_s = sample_pampjpe_flat.mean(axis=-1).reshape(BV, S)
        avg_sample_pampjpe = sample_pampjpe_per_s.mean(axis=1)
        best_metric_sample_pampjpe = sample_pampjpe_per_s.min(axis=1)

        sample_sc_flat = scale_and_translation_transform_batch(
            sample_verts_flat.cpu().detach().numpy(),
            gt_verts_tiled.cpu().detach().numpy(),
        )
        sample_pvetsc_per_s = np.linalg.norm(
            sample_sc_flat - gt_verts_tiled.cpu().detach().numpy(), axis=-1
        ).mean(axis=1).reshape(BV, S)
        avg_sample_pvetsc = sample_pvetsc_per_s.mean(axis=1)
        best_metric_sample_pvetsc = sample_pvetsc_per_s.min(axis=1)

    print(f"mpjpe: view avg: {per_view_mpjpe.mean():.4f}, view min: {per_view_mpjpe.min():.4f}, mean: {avg_mpjpe.mean():.4f} merged: {merged_mpjpe.mean():.4f}")
    print(f"pve: view avg: {per_view_pve.mean():.4f}, view min: {per_view_pve.min():.4f}, mean: {avg_pve.mean():.4f}, merged: {merged_pve.mean():.4f}")
    print(f"pampjpe: view avg: {per_view_pampjpe.mean():.4f}, view min: {per_view_pampjpe.min():.4f}, mean: {avg_pampjpe.mean():.4f}, merged: {merged_pampjpe.mean():.4f}")
    print(f"pvetsc: view avg: {per_view_pvetsc.mean():.4f}, view min: {per_view_pvetsc.min():.4f}, mean: {avg_pvetsc.mean():.4f}, merged: {merged_pvetsc.mean():.4f}")

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

    all_metrics["per_view_pvetsc"].append(per_view_pvetsc)
    all_metrics["best_per_view_pvetsc"].append(per_view_pvetsc.min().item())
    all_metrics["avg_pvetsc"].append(avg_pvetsc)
    all_metrics["merged_pvetsc"].append(merged_pvetsc)

    if best_logprob_sample_mpjpe is not None:
        all_metrics["best_logprob_sample_mpjpe"].append(best_logprob_sample_mpjpe)
        all_metrics["best_logprob_sample_pve"].append(best_logprob_sample_pve)
        all_metrics["best_logprob_sample_pampjpe"].append(best_logprob_sample_pampjpe)
        all_metrics["best_logprob_sample_pvetsc"].append(best_logprob_sample_pvetsc)

    if avg_sample_mpjpe is not None:
        all_metrics["avg_sample_mpjpe"].append(avg_sample_mpjpe)
        all_metrics["avg_sample_pve"].append(avg_sample_pve)
        all_metrics["avg_sample_pampjpe"].append(avg_sample_pampjpe)
        all_metrics["avg_sample_pvetsc"].append(avg_sample_pvetsc)
        all_metrics["best_pve_sample_pve"].append(best_metric_sample_pve)
        all_metrics["best_mpjpe_sample_mpjpe"].append(best_metric_sample_mpjpe)
        all_metrics["best_pampjpe_sample_pampjpe"].append(best_metric_sample_pampjpe)
        all_metrics["best_pvetsc_sample_pvetsc"].append(best_metric_sample_pvetsc)

    # if batch_idx == 3:
    #     for k, v in all_metrics.items():
    #         print(k, v)

    # x = np.linalg.norm(merged_sc - gt_neutral_verts.cpu().detach().numpy(), axis=-1) # (B, 18439)
    # y = np.linalg.norm(pred_sc - gt_neutral_verts.cpu().detach().numpy(), axis=-1) # (B, 18439)

    # import matplotlib.pyplot as plt
    # all_dists_for_color = np.concatenate([x.reshape(-1), y.reshape(-1)])
    # max_dist = float(all_dists_for_color.max()) if all_dists_for_color.size > 0 else 0.1
    # if max_dist <= 0:
    #     max_dist = 0.1

    # bins = np.linspace(0.0, max_dist, 51)
    # fig, axs = plt.subplots(5, 1, figsize=(6, 15), sharex=True)

    # def plot_hist_inferno(ax, data, *, title: str, alpha: float = 0.7):
    #     counts, edges = np.histogram(data, bins=bins)
    #     bin_centers = 0.5 * (edges[:-1] + edges[1:])

    #     # Anchor at 0, exactly matching mesh color normalization.
    #     denom = max_dist - 0.0
    #     if denom <= 0:
    #         denom = 1.0
    #     normalized = np.clip((bin_centers - 0.0) / denom, 0.0, 1.0)
    #     cmap = plt.get_cmap("inferno")
    #     rgba = cmap(normalized)  # (N,4)

    #     ax.bar(
    #         edges[:-1],
    #         counts,
    #         width=np.diff(edges),
    #         align="edge",
    #         color=rgba,
    #         alpha=alpha,
    #         linewidth=0,
    #     )
    #     ax.set_title(title)
    #     ax.set_ylabel("Frequency")
    #     ax.grid(True)

    # plot_hist_inferno(axs[0], x[0], title="merged_sc")
    # for i in range(4):
    #     plot_hist_inferno(axs[i + 1], y[i], title=f"pred_sc{i}")

    # axs[-1].set_xlabel("Distance")
    # plt.tight_layout()
    # hist_path = os.path.join(save_dir, f"b{batch_idx:03d}_error_hist.png")
    # plt.savefig(hist_path)
    # print(f"Saved histogram column to {hist_path}")
    # plt.close()

    # # print(x.shape, y.shape)
    # # import ipdb; ipdb.set_trace()


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