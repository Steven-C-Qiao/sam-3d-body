"""Multi-view NF sample + merge pipeline for real images.

Takes N photos of a single subject (no masks / intrinsics / GT required),
runs them through SAM3DBody with the NF-AR head, visualises per-view
samples, performs multi-view shape/scale merging, and visualises the
merged result.

Example:
    python scripts/demo_nf_multiview.py \
        --image_folder assets/demo/roberto \
        --checkpoint exp/exp_058_so3_c4/saved_models/val_loss_epoch=006.ckpt \
        --config    exp/exp_058_so3_c4/config.yaml \
        --output_dir output/roberto_nf \
        --num_samples 100 \
        --method psis
"""

import argparse
import glob
import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
from loguru import logger

from sam_3d_body.configs.config import get_config_defaults
from sam_3d_body.models.meta_arch.nf_merging import (
    get_mhr_outputs,
    merge_params_nf,
    resample_cam_for_merged_shape,
)
from sam_3d_body.models.meta_arch.sam3d_body import SAM3DBody
from sam_3d_body.sam_3d_body_estimator import SAM3DBodyEstimator
from sam_3d_body.visualization.real_image_vis import vis_merged, vis_per_view_samples


IMG_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
DEFAULT_MHR_PATH = "checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt"


def _load_cfg(config_path: str) -> object:
    cfg = get_config_defaults()
    if config_path and os.path.exists(config_path):
        logger.info(f"Merging config overrides from {config_path}")
        cfg.merge_from_file(config_path)
    cfg.MODEL.MHR_HEAD.MHR_MODEL_PATH = DEFAULT_MHR_PATH
    return cfg


def _build_model(cfg, checkpoint_path: str, device: torch.device) -> SAM3DBody:
    model = SAM3DBody(cfg).to(device).eval()
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    cleaned = {
        (k[len("model."):] if k.startswith("model.") else k): v
        for k, v in state_dict.items()
    }
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    logger.info(f"Loaded {len(cleaned)} params; {len(missing)} missing, "
                f"{len(unexpected)} unexpected")
    if missing:
        logger.warning(f"Missing keys (showing up to 10): {list(missing)[:10]}")
    return model


def _build_estimator(model, cfg, device: torch.device) -> SAM3DBodyEstimator:
    from tools.build_detector import HumanDetector
    from tools.build_fov_estimator import FOVEstimator

    detector_path = os.environ.get("SAM3D_DETECTOR_PATH", "")
    fov_path = os.environ.get("SAM3D_FOV_PATH", "")

    logger.info("Building ViTDet human detector...")
    detector = HumanDetector(name="vitdet", device=device, path=detector_path)

    logger.info("Building MoGe2 FOV estimator...")
    fov = FOVEstimator(name="moge2", device=device, path=fov_path)

    return SAM3DBodyEstimator(
        sam_3d_body_model=model,
        model_cfg=cfg,
        human_detector=detector,
        human_segmentor=None,
        fov_estimator=fov,
    )


def _stack_multiview(per_view: list) -> dict:
    """Concatenate per-image batches into a flat (V, 1, ...) multi-view batch.

    Each per-image batch from ``prepare_batch`` has tensor keys of shape
    ``(1, 1, ...)`` (``(batch_size=1, num_persons=1, ...)``). The model's
    multi-view forward expects the already-flattened layout
    ``(bs*num_views, num_persons=1, ...)`` — i.e., the view dim merged into
    the batch dim. We reach that by simply concatenating along dim 0.
    ``cam_int`` and ``img_ori`` follow the per-view convention as in the
    regular multi-view dataloader.
    """
    keys_1N = ["img", "img_size", "ori_img_size", "bbox_center",
               "bbox_scale", "bbox", "affine_trans", "mask",
               "mask_score", "person_valid"]

    stacked = {}
    for k in keys_1N:
        if k not in per_view[0]:
            continue
        stacked[k] = torch.cat([b[k] for b in per_view], dim=0)

    stacked["cam_int"] = torch.cat([b["cam_int"] for b in per_view], dim=0)
    stacked["img_ori"] = [b["img_ori"][0] for b in per_view]

    return stacked


def _inject_dummy_gt(batch: dict, mhr_out: dict) -> None:
    """get_mhr_outputs reads batch['shape_params' / 'model_params' /
    'face_expr_coeffs'] to compute gt_neutral_verts. We don't have GT;
    fill with zeros so the call doesn't KeyError. Those GT-derived
    outputs won't be rendered for real images."""
    V = mhr_out["shape"].shape[0]
    device = mhr_out["shape"].device
    dtype = mhr_out["shape"].dtype
    batch["shape_params"] = torch.zeros_like(mhr_out["shape"])
    batch["face_expr_coeffs"] = torch.zeros_like(mhr_out["face"])
    batch["model_params"] = torch.zeros(V, 68, device=device, dtype=dtype)


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    image_paths = sorted(
        p for ext in IMG_EXTS
        for p in glob.glob(os.path.join(args.image_folder, ext))
    )
    if not image_paths:
        raise FileNotFoundError(f"No images found under {args.image_folder}")
    logger.info(f"Found {len(image_paths)} images in {args.image_folder}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = _load_cfg(args.config)
    model = _build_model(cfg, args.checkpoint, device)
    estimator = _build_estimator(model, cfg, device)
    faces = estimator.faces

    logger.info("Preprocessing each view (detector + FOV + crop)...")
    per_view = []
    for p in image_paths:
        batch_i = estimator.preprocess_image(
            p, bbox_thr=args.bbox_thr, single_person=True
        )
        if batch_i is None:
            logger.warning(f"No person detected in {p}; skipping.")
            continue
        per_view.append(batch_i)
    V = len(per_view)
    if V < 2:
        raise RuntimeError(
            f"Need at least 2 views for multi-view merging; got {V}"
        )
    logger.info(f"Prepared {V} views.")

    logger.info("Stacking multi-view batch...")
    batch = _stack_multiview(per_view)
    bs, num_views = 1, V
    assert batch["img"].shape[0] == V and batch["img"].shape[1] == 1

    logger.info(f"Running model forward with num_samples={args.num_samples}...")
    with torch.no_grad():
        outputs = model(batch, num_samples=args.num_samples)

    logger.info("Visualising per-view samples...")
    vis_per_view_samples(
        outputs, batch, faces=faces, save_dir=args.output_dir,
        max_samples=args.max_vis_samples,
    )

    logger.info(f"Running merge method={args.method}...")
    _inject_dummy_gt(batch, outputs["mhr"])
    param_dict = merge_params_nf(
        model.nf_head,
        outputs["mhr"],
        outputs["uncertainty_output"],
        bs=bs,
        num_views=num_views,
        num_samples=args.num_samples,
        method=args.method,
    )
    outs = get_mhr_outputs(
        mhr_head=model.head_pose,
        mhr_out=outputs["mhr"],
        param_dict=param_dict,
        batch=batch,
        bs=bs,
        num_views=num_views,
        uncertainty_out=outputs["uncertainty_output"],
    )

    # Stage-2 cam resample: sample a camera consistent with the merged shape.
    merged_pred_cam_t = resample_cam_for_merged_shape(
        model=model,
        mhr_out=outputs["mhr"],
        uncertainty_out=outputs["uncertainty_output"],
        param_dict=param_dict,
        batch=batch,
        bs=bs,
        num_views=num_views,
    )

    logger.info("Visualising merged result...")
    vis_merged(
        outs, outputs, batch, faces=faces, save_dir=args.output_dir,
        merged_cam_t=merged_pred_cam_t,
    )

    logger.info(f"Done. Outputs in {args.output_dir}")


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-view NF sample + merge pipeline for real images."
    )
    parser.add_argument("--image_folder", required=True, type=str)
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument(
        "--config", type=str, default="",
        help="Path to experiment config.yaml. Merged into the defaults.",
    )
    parser.add_argument("--output_dir", type=str, default="output/demo_nf_multiview")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--max_vis_samples", type=int, default=8,
                        help="Max samples to render per view in the sample grid.")
    parser.add_argument(
        "--method", type=str, default="psis",
        choices=["psis", "tempered", "gaussian", "is", "langevin"],
    )
    parser.add_argument("--bbox_thr", type=float, default=0.5)
    parser.add_argument("--gpus", type=str, default="0")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    main(args)
