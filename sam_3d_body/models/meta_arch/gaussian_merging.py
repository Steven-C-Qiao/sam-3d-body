import torch
def merge_predictions(self, mu, sigma):
    """
    Merge predictions from N images
    Args:
        mu: (N, D)
        sigma: (N, D, D)
    Returns:
        mu_star: (D,)
        sigma_star: (D, D)
    """
    inv_sigma = torch.linalg.inv(sigma)

    precision_mat = torch.sum(inv_sigma, dim=0)

    sigma_star = torch.linalg.inv(precision_mat)

    mu_star = sigma_star @ torch.einsum("nij,nj->ni", inv_sigma, mu).sum(dim=0)

    return mu_star, sigma_star

def merge_predictions_batch(self, mu, sigma):
    """
    Batch version: Merge predictions from N images for B batches
    Args:
        mu: (B, N, D)
        sigma: (B, N, D, D)
    Returns:
        mu_star: (B, D)
        sigma_star: (B, D, D)
    """
    inv_sigma = torch.linalg.inv(sigma)  # (B, N, D, D)

    precision_mat = torch.sum(inv_sigma, dim=1)  # (B, D, D)

    sigma_star = torch.linalg.inv(precision_mat)  # (B, D, D)

    # einsum: bnij,bnj->bni, then sum over N to get (B, D)
    weighted_mu = torch.einsum("bnij,bnj->bni", inv_sigma, mu).sum(dim=1)  # (B, D)

    # Batch matrix multiplication: (B, D, D) @ (B, D) -> (B, D)
    mu_star = torch.bmm(sigma_star, weighted_mu.unsqueeze(-1)).squeeze(-1)  # (B, D)

    return mu_star, sigma_star


def merge_parameters_gaussian(self, num_views, bs, outputs):
    pred_shape = outputs["mhr"]["shape"]
    pred_scale = outputs["mhr"]["scale"]

    # Uncertainties come from uncertainty head (separate from mhr output)
    indices = [3, 4, 5, 6, 7, 10, 11, 12, 13, 14]
    shape_var = outputs["uncertainty_output"]["shape_uncertainty"]
    scale_var = outputs["uncertainty_output"]["scale_uncertainty"]

    # shape_var: [batch, D], want [batch, D, D] with diag elements
    shape_var_diag = torch.diag_embed(shape_var)
    scale_var_diag = torch.diag_embed(scale_var)

    pred_shape = pred_shape.unflatten(0, (bs, num_views))
    pred_scale = pred_scale.unflatten(0, (bs, num_views))
    shape_var_diag = shape_var_diag.unflatten(0, (bs, num_views))
    scale_var_diag = scale_var_diag.unflatten(0, (bs, num_views))

    shape_mu_star, shape_sigma_star = self.merge_predictions_batch(
        pred_shape, shape_var_diag
    )

    # Variance for per-view and merged shape parameters
    shape_var_unflattened = shape_var.unflatten(0, (bs, num_views))
    merged_shape_var = torch.diagonal(shape_sigma_star, dim1=-2, dim2=-1)

    scale_mu_star, scale_sigma_star = self.merge_predictions_batch(
        pred_scale[..., indices], scale_var_diag
    )
    scale_mu_star_full = pred_scale.mean(dim=1)
    scale_mu_star_full[..., indices] = scale_mu_star

    shape_mean = pred_shape.mean(dim=1).repeat_interleave(
        num_views, dim=0
    )  # naive average of parameters
    scale_mean = pred_scale.mean(dim=1).repeat_interleave(num_views, dim=0)

    param_dict = {
        "pred_shape": pred_shape,
        "pred_scale": pred_scale,
        "shape_mu_star": shape_mu_star,
        "shape_var_unflattened": shape_var_unflattened,
        "merged_shape_var": merged_shape_var,
        "scale_mu_star_full": scale_mu_star_full,
        "shape_mean": shape_mean,
        "scale_mean": scale_mean,
    }
    return param_dict
