"""For each top dim (uncertain + disagreement), report per-view σ/σ_prior
to check whether the flow is appropriately uncertain on geometrically-null dims.
"""
import os
import sys
from pathlib import Path

import torch
from loguru import logger

sys.path.append(".")

from sam_3d_body.configs.config import get_config_defaults

UNC_DIMS = [43, 24, 7, 31, 28]
DIS_DIMS = [10, 34, 25, 18, 11]


def load_trainer(exp_dir, load_path, device):
    from sam_3d_body.trainer import Trainer
    cfg = get_config_defaults()
    cfg.merge_from_file(str(Path(exp_dir) / "config.yaml"))
    cfg.MODEL.MHR_HEAD.MHR_MODEL_PATH = (
        "checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt"
    )
    trainer = Trainer(cfg=cfg, vis_save_dir=str(Path(exp_dir) / "diag_tmp")).to(device)
    ckpt = torch.load(load_path, weights_only=False, map_location="cpu")
    sd = {k[6:] if k.startswith("model.") else k: v for k, v in ckpt["state_dict"].items()}
    trainer.model.load_state_dict(sd, strict=False)
    trainer.model.eval()
    return trainer


@torch.no_grad()
def main():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["EGL_DEVICE_ID"] = "0"
    device = torch.device("cuda")

    trainer = load_trainer("exp/exp_071_crop_shape",
                           "exp/exp_071_crop_shape/saved_models/last.ckpt", device)
    nf = trainer.model.nf_head

    # Load shape_std (population)
    stds = torch.load("checkpoints/sam-3d-body-dinov3/shape_scale_std.pt",
                       weights_only=False, map_location=device)
    shape_std = stds["shape_std"].to(device).float()  # [45]

    loader = trainer.multiview_eval_dataloader(num_view=4, batch_size=1, dataset_name="4d-dress")

    sigmas = []
    for bidx, batch in enumerate(loader):
        if bidx >= 3:
            break
        for k, v in list(batch.items()):
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)
        bs, num_views = batch["img"].shape[:2]
        for k, v in list(batch.items()):
            if isinstance(v, torch.Tensor):
                if v.dim() >= 2 and v.shape[0] == bs and v.shape[1] == num_views:
                    batch[k] = v.flatten(0, 1)
        batch = trainer.preprocess(batch)
        outputs = trainer.model(batch, num_samples=100)
        u = outputs["uncertainty_output"]
        shape_s = u["shape_samples"].unflatten(0, (bs, num_views))  # [B, V, S, 45]
        sigma = shape_s.std(dim=2)                                  # [B, V, 45]
        sigmas.append(sigma.flatten(0, 1).cpu())
    sigma_all = torch.cat(sigmas, dim=0).float()  # [B*V, 45]
    mean_sigma = sigma_all.mean(dim=0)            # [45]
    ratio = mean_sigma / shape_std.cpu()          # [45]

    print("\n=== per-view σ / σ_prior, with body-effect ===")
    print(f"{'dim':>5} {'set':>5} {'σ_prior':>9} {'mean σ':>9} {'σ/σ_prior':>11}")
    print("-" * 45)
    for dim in UNC_DIMS:
        print(f"d{dim:>4} {'unc':>5} {shape_std[dim].item():>9.3f} "
              f"{mean_sigma[dim].item():>9.3f} {ratio[dim].item():>11.3f}")
    for dim in DIS_DIMS:
        print(f"d{dim:>4} {'dis':>5} {shape_std[dim].item():>9.3f} "
              f"{mean_sigma[dim].item():>9.3f} {ratio[dim].item():>11.3f}")
    print()
    print(f"Median across all 45 shape dims: σ/σ_prior = {ratio.median().item():.3f}")
    print(f"Min:    σ/σ_prior = {ratio.min().item():.3f}  at dim {ratio.argmin().item()}")
    print(f"Max:    σ/σ_prior = {ratio.max().item():.3f}  at dim {ratio.argmax().item()}")


if __name__ == "__main__":
    main()
