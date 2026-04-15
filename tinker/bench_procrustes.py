"""Benchmark numpy vs torch procrustes / scale-translation for B=64, S=25, V=18000."""
import time

import numpy as np
import torch

from sam_3d_body.metrics.metrics_tracker import (
    compute_similarity_transform_batch,
    scale_and_translation_transform_batch,
    scale_and_translation_transform_batch_torch,
)


def compute_similarity_transform_batch_torch(S1, S2):
    """
    Batched procrustes in torch. S1, S2: (B, N, 3). Returns aligned S1 of same shape.
    """
    mu1 = S1.mean(dim=-2, keepdim=True)   # (B, 1, 3)
    mu2 = S2.mean(dim=-2, keepdim=True)
    X1 = S1 - mu1                          # (B, N, 3)
    X2 = S2 - mu2
    var1 = (X1 ** 2).sum(dim=(-2, -1))     # (B,)

    K = X1.transpose(-2, -1) @ X2          # (B, 3, 3)
    U, _, Vh = torch.linalg.svd(K)         # (B,3,3), (B,3), (B,3,3)
    V = Vh.transpose(-2, -1)

    # Z diag with last entry = sign(det(U V^T)) = sign(det(U @ Vh))
    sign = torch.sign(torch.det(U @ Vh))   # (B,)
    Z = torch.eye(3, device=S1.device, dtype=S1.dtype).expand(S1.shape[0], 3, 3).clone()
    Z[:, -1, -1] = Z[:, -1, -1] * sign

    R = V @ Z @ U.transpose(-2, -1)        # (B, 3, 3)
    # trace(R @ K) = sum_ij R_ij * K_ij (since trace(AB) = sum_ij A_ij B_ji, and we want trace(R @ K))
    # trace(R @ K) = sum_ij R_ik K_ki = einsum('bij,bji->b', R, K)
    trRK = torch.einsum('bij,bji->b', R, K)
    scale = trRK / var1                    # (B,)

    # S1_hat = scale * R @ S1 + t, where t = mu2 - scale * R @ mu1
    # R applies on (3,) points; for (B, N, 3) points: (R @ S1.T).T = S1 @ R.T
    S1_rot = S1 @ R.transpose(-2, -1)      # (B, N, 3)
    t = mu2 - scale.view(-1, 1, 1) * (mu1 @ R.transpose(-2, -1))
    S1_hat = scale.view(-1, 1, 1) * S1_rot + t
    return S1_hat


def bench(fn, *args, n=3, sync=False):
    if sync:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        out = fn(*args)
        if sync:
            torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / n
    return dt, out


def main():
    B, S, V = 64, 25, 18000
    BS = B * S
    print(f"Shapes: ({BS}, {V}, 3)")

    rng = np.random.default_rng(0)
    P_np = rng.standard_normal((BS, V, 3)).astype(np.float32)
    T_np = rng.standard_normal((BS, V, 3)).astype(np.float32)

    # Warmup numpy
    _ = compute_similarity_transform_batch(P_np[:4], T_np[:4])

    # NUMPY procrustes
    dt, aligned_np = bench(compute_similarity_transform_batch, P_np, T_np, n=2)
    print(f"[numpy]  procrustes_batch       : {dt*1000:8.1f} ms")

    # NUMPY scale+trans
    dt, _ = bench(scale_and_translation_transform_batch, P_np, T_np, n=2)
    print(f"[numpy]  scale_trans_batch      : {dt*1000:8.1f} ms")

    # torch CPU
    P_cpu = torch.from_numpy(P_np)
    T_cpu = torch.from_numpy(T_np)
    dt, aligned_cpu = bench(compute_similarity_transform_batch_torch, P_cpu, T_cpu, n=2)
    print(f"[torch-cpu] procrustes_batch    : {dt*1000:8.1f} ms")
    dt, _ = bench(scale_and_translation_transform_batch_torch, P_cpu, T_cpu, n=2)
    print(f"[torch-cpu] scale_trans_batch   : {dt*1000:8.1f} ms")

    # torch CUDA
    if torch.cuda.is_available():
        P_g = P_cpu.cuda()
        T_g = T_cpu.cuda()
        # warmup
        _ = compute_similarity_transform_batch_torch(P_g, T_g)
        torch.cuda.synchronize()
        dt, aligned_g = bench(compute_similarity_transform_batch_torch, P_g, T_g, n=3, sync=True)
        print(f"[torch-gpu] procrustes_batch    : {dt*1000:8.1f} ms")
        dt, _ = bench(scale_and_translation_transform_batch_torch, P_g, T_g, n=3, sync=True)
        print(f"[torch-gpu] scale_trans_batch   : {dt*1000:8.1f} ms")

        # Correctness check vs numpy
        max_err = (aligned_g.cpu().numpy() - aligned_np).__abs__().max()
        print(f"[correct] procrustes torch-gpu vs numpy max abs err: {max_err:.2e}")

    # Also smaller batch to see per-B cost
    for B2 in [16, 64, 256, 1600]:
        P_s = P_np[:B2]
        T_s = T_np[:B2]
        dt, _ = bench(compute_similarity_transform_batch, P_s, T_s, n=2)
        print(f"[numpy] procrustes B={B2:4d}: {dt*1000:8.1f} ms")


if __name__ == "__main__":
    main()
