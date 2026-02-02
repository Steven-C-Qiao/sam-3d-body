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


class Metrics(pl.LightningModule):
    def __init__(self):
        super().__init__()

    def forward(self, predictions, batch):
        metrics = {}

        if "pred_keypoints_3d" in predictions["mhr"]:
            gt_kp3d = batch["keypoints_3d"]
            pred_kp3d = predictions["mhr"]["pred_keypoints_3d"]

            mpjpe_mean = self.mpjpe(
                pred_kp3d[:, :70, :],
                gt_kp3d[:, :70, :],
            )
            metrics["mpjpe"] = mpjpe_mean

            pampjpe_mean = self.pampjpe(
                pred_kp3d[:, :70, :].cpu().detach().numpy(),
                gt_kp3d[:, :70, :].cpu().detach().numpy(),
            )
            metrics["pampjpe"] = pampjpe_mean

            if pred_kp3d.shape[1] > 70:
                pve_mean = self.pve(pred_kp3d[:, 70:, :], gt_kp3d[:, 70:, :])
                metrics["pve"] = pve_mean

        if "mhr_samples_keypoints_3d" in predictions:
            pred_kp3d_samples = predictions["mhr_samples_keypoints_3d"]
            num_samples = pred_kp3d_samples.shape[1]
            gt_kp3d_samples = batch["keypoints_3d"][:, None].expand(
                -1, num_samples, -1, -1
            )

            mpjpe_samples = self.mpjpe(
                pred_kp3d_samples[:, :, :70, :], gt_kp3d_samples[:, :, :70, :]
            )
            metrics["mpjpe_samples"] = mpjpe_samples

            pampjpe_samples = self.pampjpe(
                pred_kp3d_samples[:, :, :70, :].flatten(0, 1).cpu().detach().numpy(),
                gt_kp3d_samples[:, :, :70, :].flatten(0, 1).cpu().detach().numpy(),
            )
            metrics["pampjpe_samples"] = pampjpe_samples

            if pred_kp3d_samples.shape[2] > 70:
                pve_samples = self.pve(
                    pred_kp3d_samples[:, :, 70:, :],
                    gt_kp3d_samples[:, :, 70:, :],
                )
                metrics["pve_samples"] = pve_samples

        # # Compute 2D keypoint L1 distance metrics for mean prediction
        # if "mhr" in predictions and "pred_keypoints_2d_cropped" in predictions["mhr"]:
        #     gt_kp2d_mean = batch[
        #         "keypoints_2d"
        #     ]  # [B, N, 2] (in normalized cropped coords [-0.5, 0.5])
        #     pred_kp2d_mean = predictions["mhr"][
        #         "pred_keypoints_2d_cropped"
        #     ]  # [B, N, 2] (in normalized cropped coords [-0.5, 0.5])

        #     # Ensure shapes match (handle batch dimension if needed)
        #     if gt_kp2d_mean.shape[0] != pred_kp2d_mean.shape[0]:
        #         gt_kp2d_mean = gt_kp2d_mean[: pred_kp2d_mean.shape[0]]

        #     # Compute L1 distance for mean prediction
        #     kp2d_l1_mean = self.avg_kp2d_l1_dist(pred_kp2d_mean, gt_kp2d_mean)
        #     metrics["kp2d_l1"] = kp2d_l1_mean

        # # Compute 2D keypoint L1 distance metrics for samples
        # if "mhr_samples_keypoints_2d_cropped" in predictions:
        #     num_samples = predictions["mhr_samples_keypoints_2d_cropped"].shape[1]
        #     gt_kp2d_samples = batch["keypoints_2d"][:, None].expand(
        #         -1, num_samples, -1, -1
        #     )  # [B, num_samples, N, 2]
        #     pred_kp2d_samples = predictions[
        #         "mhr_samples_keypoints_2d_cropped"
        #     ]  # [B, num_samples, N, 2]

        #     # Compute L1 distance for samples
        #     kp2d_l1_samples = self.avg_kp2d_l1_dist(pred_kp2d_samples, gt_kp2d_samples)
        #     metrics["kp2d_l1_samples"] = kp2d_l1_samples

        # print(metrics)
        # import ipdb; ipdb.set_trace()

        return metrics

    def mpjpe(self, pred, gt):
        return torch.sqrt(((pred - gt) ** 2).sum(dim=-1)).mean()

    def pampjpe(self, pred, gt):
        r_error, _ = reconstruction_error(pred, gt, reduction=None)
        return r_error.mean()

    def pve(self, pred, gt):
        return torch.sqrt(((pred - gt) ** 2).sum(dim=-1)).mean()

    def avg_kp2d_l1_dist(self, pred, gt):
        return torch.abs(pred - gt).mean()

    def pvetsc(self, pred, gt):
        pred_tpose_vertices_sc = scale_and_translation_transform_batch(pred, gt)
        pvet_sc_batch = np.linalg.norm(
            pred_tpose_vertices_sc - gt, axis=-1
        )  # (bs, 6890)
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        # Only plot the first sample for visualization
        if pred.ndim == 3 and gt.ndim == 3:
            pred_plot = pred[0]
            gt_plot = gt[0]
        else:
            pred_plot = pred
            gt_plot = gt

        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(pred_plot[:, 0], pred_plot[:, 1], pred_plot[:, 2], c='r', label='pred', alpha=0.6)
        # ax.scatter(gt_plot[:, 0], gt_plot[:, 1], gt_plot[:, 2], c='b', label='gt', alpha=0.6)
        # ax.set_title("3D scatter of pred (red) and gt (blue)")
        # ax.legend()
        # plt.savefig('pvetsc.png')
        # plt.close()
        # import ipdb; ipdb.set_trace()
        return pvet_sc_batch.mean()
