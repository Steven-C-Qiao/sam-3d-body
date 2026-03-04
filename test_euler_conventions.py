"""
Minimal test comparing two Euler-conversion paths on random 6D rotations.

Path 1:
    6D -> rotation matrix -> Euler (batchXYZfrom6D)

Path 2:
    6D -> 9D rotation rep -> 3x3 rotation matrix -> axis-angle -> matrix -> Euler
"""

import torch

from sam_3d_body.models.modules.mhr_utils import batchXYZfrom6D, batch9Dfrom6D
from pytorch3d.transforms import (
    matrix_to_axis_angle,
    axis_angle_to_matrix,
    matrix_to_euler_angles,
)


def main(num_samples: int = 1024, num_joints: int = 23, device: str = "cpu") -> None:
    torch.manual_seed(0)

    # Random 6D rotations for [B, J] joints
    poses_6d = torch.randn(num_samples, num_joints, 6, device=device)

    # Path 1: 6D -> rotmat -> Euler via batchXYZfrom6D
    euler1 = batchXYZfrom6D(poses_6d)  # [B, J, 3]

    # Path 2: 6D -> 9D -> 3x3 -> axis-angle -> 3x3 -> Euler
    rotmat9 = batch9Dfrom6D(poses_6d)          # [B, J, 9]
    rotmat = rotmat9.unflatten(-1, (3, 3))     # [B, J, 3, 3]
    aa = matrix_to_axis_angle(rotmat)         # [B, J, 3]
    rotmat2 = axis_angle_to_matrix(aa)        # [B, J, 3, 3]
    euler2 = matrix_to_euler_angles(rotmat2, "XYZ")  # [B, J, 3]

    diff = (euler1 - euler2).abs()

    print("=== Euler difference statistics (path1 - path2) ===")
    print(f"num_samples     : {num_samples}")
    print(f"num_joints      : {num_joints}")
    print(f"mean(|diff|)    : {diff.mean().item():.6f}")
    print(f"max(|diff|)     : {diff.max().item():.6f}")


    # Per-joint stats (max over 3 Euler angles, then mean/max over batch)
    diff_per_joint = diff.max(dim=-1)[0]      # [B, J]
    mean_per_joint = diff_per_joint.mean(dim=0)
    max_per_joint = diff_per_joint.max(dim=0)[0]

    print("\nPer-joint max |diff| over Euler components:")
    for j, (m, M) in enumerate(zip(mean_per_joint, max_per_joint)):
        print(f"  joint {j:2d}: mean_max={m.item():.6f}, max_max={M.item():.6f}")


if __name__ == "__main__":
    main()

