"""
Convert SMPL/SMPL-X parameters to MHR format, conditioning the shape/scale
optimisation on predictions from the pre-trained SAM-3D-Body model.

This removes the scale-shape ambiguity in the vanilla SMPL→MHR conversion:
the optimiser is regularised toward the model's predicted (identity, scale),
anchoring the GT labels to the model's prediction space.

Parallelisation (mirrors convert_bedlam.py):
  - Sequences are distributed round-robin across GPUs.
  - Each GPU runs one ConvertBedlamWithPrior instance in a dedicated thread.
  - Within each GPU, --jobs_per_gpu sequences can run concurrently via a
    ThreadPoolExecutor (default 1 to avoid OOM from multiple inference passes).

Usage:
    # Single sequence, one GPU (for testing):
    python scripts/convert_with_model_prior.py --dataset bedlam \
        --sequences closeup-suburbc-bmi --gpus 0

    # Full BEDLAM, two GPUs:
    python scripts/convert_with_model_prior.py --dataset bedlam --gpus 0,1

    # 4D-Dress:
    python scripts/convert_with_model_prior.py --dataset 4d-dress --gpus 0
"""

import os
import sys
import argparse
import threading
import pickle
import cv2
import numpy as np
import torch
import smplx
import trimesh
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — required for 3D projection
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, "/scratch/cq244/MHR/tools/mhr_smpl_conversion")

from sam_3d_body.configs.config import (
    get_config_defaults,
    DATASET_FILES,
    DATASET_FOLDERS,
)
from sam_3d_body.models.meta_arch.sam3d_body import SAM3DBody
from sam_3d_body.data.bedlam_dataset import DatasetHMR, FakeGetBBoxCenterScale
from sam_3d_body.data.transforms.common import Compose, TopdownAffine, VisionTransformWrapper
from torchvision.transforms import ToTensor
from torch.utils.data._utils.collate import default_collate

from mhr.mhr import MHR
from conversion import Conversion


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BEDLAM_EXTRA_PATH = "/scratch/cq244/BEDLAM/data/training_labels/all_npz_12_training_extra"
BEDLAM_SAVE_PATH  = "/scratch/cq244/BEDLAM/data/training_labels/all_npz_12_training_mhr_conditioned"

D4DRESS_DATA_PATH   = "/scratches/kyuban/share/4DDress"
MHR_ASSETS_PATH     = "/scratch/cq244/MHR/assets"
MHR_MODEL_FILES_PATH = "/scratch/cq244/MHR/model_files"
SAM3D_CKPT_PATH     = "checkpoints/sam-3d-body-dinov3/model.ckpt"
SAM3D_MHR_MODEL_PATH = "checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_sam3d_model(device: torch.device) -> SAM3DBody:
    cfg = get_config_defaults()
    cfg.MODEL.MHR_HEAD.MHR_MODEL_PATH = SAM3D_MHR_MODEL_PATH
    model = SAM3DBody(cfg)
    checkpoint = torch.load(SAM3D_CKPT_PATH, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.to(device)
    return model


def build_prior_params(
    shape_pred: np.ndarray,
    scale_pred: np.ndarray,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """
    shape_pred: (45,) identity_coeffs
    scale_pred: (68,) scale_68D
    Returns dict matching MHR optimizer variable names.
    """
    shape_t = torch.tensor(shape_pred, dtype=torch.float32, device=device).unsqueeze(0)
    scale_t  = torch.tensor(scale_pred,  dtype=torch.float32, device=device).unsqueeze(0)
    return {
        "body_identity_coeffs": shape_t[:, :20],
        "head_identity_coeffs": shape_t[:, 20:40],
        "hand_identity_coeffs": shape_t[:, 40:45],
        "scale_params":         scale_t,
    }


def _save_comparison_vis(
    ds_name: str,
    serno: int,
    smpl_verts: np.ndarray,
    mhr_verts: np.ndarray,
    save_dir: str,
) -> None:
    """
    Save a side-by-side scatter plot comparing SMPL-X and MHR vertices for one frame.

    smpl_verts: (N_smpl, 3) in metres
    mhr_verts:  (N_mhr,  3) in centimetres — converted to metres here
    """
    mhr_verts_m = mhr_verts / 100.0

    fig = plt.figure(figsize=(18, 6))

    for col, (verts, title) in enumerate([
        (smpl_verts, "SMPL-X (input)"),
        (mhr_verts_m, "MHR (converted)"),
    ], start=1):
        ax = fig.add_subplot(1, 3, col, projection="3d")
        ax.scatter(verts[:, 0], verts[:, 2], verts[:, 1],
                   c=verts[:, 1], cmap="viridis", s=1, alpha=0.6)
        ax.set_title(title)
        ax.set_xlabel("X"); ax.set_ylabel("Z"); ax.set_zlabel("Y")

        # Equal aspect ratio
        extent = max(verts.max(0) - verts.min(0)) / 2
        mid    = (verts.max(0) + verts.min(0)) / 2
        ax.set_xlim(mid[0] - extent, mid[0] + extent)
        ax.set_ylim(mid[2] - extent, mid[2] + extent)
        ax.set_zlim(mid[1] - extent, mid[1] + extent)
        ax.view_init(elev=10, azim=-80)

    # Third plot: overlay
    all_verts = np.concatenate([smpl_verts, mhr_verts_m], axis=0)
    extent = max(all_verts.max(0) - all_verts.min(0)) / 2
    mid    = (all_verts.max(0) + all_verts.min(0)) / 2

    ax3 = fig.add_subplot(1, 3, 3, projection="3d")
    ax3.scatter(smpl_verts[:, 0], smpl_verts[:, 2], smpl_verts[:, 1],
                c="steelblue", s=1, alpha=0.5, label="SMPL-X")
    ax3.scatter(mhr_verts_m[:, 0], mhr_verts_m[:, 2], mhr_verts_m[:, 1],
                c="tomato", s=1, alpha=0.5, label="MHR")
    ax3.set_title("Overlay")
    ax3.set_xlabel("X"); ax3.set_ylabel("Z"); ax3.set_zlabel("Y")
    ax3.set_xlim(mid[0] - extent, mid[0] + extent)
    ax3.set_ylim(mid[2] - extent, mid[2] + extent)
    ax3.set_zlim(mid[1] - extent, mid[1] + extent)
    ax3.view_init(elev=10, azim=-80)
    ax3.legend(loc="upper right", markerscale=5)

    fig.suptitle(f"{ds_name}  serno={serno}", fontsize=11)
    fig.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f"{ds_name}.png")
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  [vis] saved → {out_path}")


def _build_npz_to_dataset_map() -> dict[str, str]:
    """Reverse map: npz basename → dataset name (training datasets only)."""
    return {
        os.path.basename(npz_path): ds_name
        for ds_name, npz_path in DATASET_FILES[1].items()
        if npz_path.endswith(".npz")
    }


# ---------------------------------------------------------------------------
# Converter class (one per GPU, mirrors ConvertBedlam)
# ---------------------------------------------------------------------------

class ConvertBedlamWithPrior:
    """
    Holds a SAM-3D-Body model and an MHR converter on a single GPU.
    Runs inference and prior-conditioned MHR fitting for a list of sequences.
    """

    def __init__(self, device_id: int = 0):
        self._device_id = device_id
        self._device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.cuda.set_device(device_id)
        print(f"[GPU {device_id}] Loading models...")

        cfg = get_config_defaults()
        cfg.MODEL.MHR_HEAD.MHR_MODEL_PATH = SAM3D_MHR_MODEL_PATH
        self._cfg = cfg

        self._model = load_sam3d_model(self._device)

        self._smplx = smplx.create(
            model_type="smplx",
            model_path=MHR_MODEL_FILES_PATH,
            num_betas=11,
            gender="neutral",
            use_pca=False,
            flat_hand_mean=True,
        ).to(self._device)

        mhr_model = MHR.from_files(folder=Path(MHR_ASSETS_PATH), lod=1, device=self._device)
        self._converter = Conversion(mhr_model=mhr_model, smpl_model=self._smplx, method="pytorch")

    # ---- inference ----

    @torch.no_grad()
    def _run_inference(self, ds_name: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """
        Run SAM-3D-Body on all frames of ds_name.

        Returns:
            {relative_imgname: (shape_45, scale_68, lbs_model_params_275)}
        """
        if torch.cuda.is_available():
            torch.cuda.set_device(self._device_id)
        dataset = DatasetHMR(self._cfg.DATASET, ds_name, use_augmentation=False, is_train=True)
        # Override __len__ so we iterate ALL frames, not the CROP_PERCENT subset.
        dataset.is_train = False

        loader = DataLoader(
            dataset,
            batch_size=32,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
            drop_last=False,
        )

        preds: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for batch in tqdm(loader, desc=f"[GPU {self._device}][{ds_name}] Inference",
                          unit="batch", leave=False):
            # Collect relative imgnames (strip img_dir prefix)
            raw_names = batch["imgname"]
            if isinstance(raw_names, str):
                raw_names = [raw_names]
            rel_names = []
            for p in raw_names:
                p = str(p)
                rel_names.append(
                    os.path.relpath(p, dataset.img_dir) if p.startswith(dataset.img_dir) else p
                )

            model_batch = {
                k: v.to(self._device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            out = self._model(model_batch, num_samples=0)
            mhr_out = out["mhr"]
            shapes          = mhr_out["shape"].float().cpu().numpy()          # (B, 45)
            scales          = mhr_out["scale_68D"].float().cpu().numpy()      # (B, 68)
            mhr_model_params = mhr_out["mhr_model_params"].float().cpu().numpy()  # (B, 275)

            for name, shape, scale, lbs in zip(rel_names, shapes, scales, mhr_model_params):
                preds[name] = (shape, scale, lbs)

        return preds

    # ---- conversion for one sequence ----

    def process_sequence(
        self,
        ds_name: str,
        npz_extra_path: str,
        prior_reg_weight: float,
        save_path: str,
        process_id: int | None = None,
    ) -> None:
        if torch.cuda.is_available():
            torch.cuda.set_device(self._device_id)
        label = f"[GPU {self._device}][{ds_name}]"
        npz_basename = os.path.basename(npz_extra_path)
        output_path  = os.path.join(save_path, npz_basename)

        if os.path.exists(output_path):
            print(f"{label} output already exists, skipping.")
            return

        # --- inference ---
        imgname_preds = self._run_inference(ds_name)
        print(f"{label} Inference done ({len(imgname_preds)} frames).")

        # --- load SMPL-X extra npz ---
        original_data = np.load(npz_extra_path)
        serno         = original_data["serno"]
        extra_imgnames = original_data["imgname"]
        total_frames  = len(extra_imgnames)

        unique_sernos = np.unique(serno)
        params_with_indices: dict[str, list] = defaultdict(list)
        vis_dir = os.path.join(save_path, "_vis")
        is_first_serno = True

        for unique_serno in tqdm(unique_sernos, desc=f"{label} Converting", unit="person", leave=True):
            serno_indices = np.where(serno == unique_serno)[0]

            # Collect predictions for this person's frames
            frame_preds = [
                imgname_preds.get(str(extra_imgnames[i]))
                for i in serno_indices
            ]
            # Build per-frame pose init from model predictions (None where no prediction).
            # Use body pose only (lbs[6:136], 130D) — global rot/transl are excluded so
            # the optimizer can find the correct alignment independently.
            pose_list = []
            for p in frame_preds:
                if p is not None:
                    lbs = p[2]          # (275,): [transl(3), rots(3), body_pose(130), hands(71), scale(68)]
                    pose_list.append(lbs[6:136])   # body pose, 130D
                else:
                    pose_list.append(None)

            frame_preds_valid = [p for p in frame_preds if p is not None]
            if not frame_preds_valid:
                continue

            mean_shape = np.mean([p[0] for p in frame_preds_valid], axis=0)  # (45,)
            mean_scale = np.mean([p[1] for p in frame_preds_valid], axis=0)  # (68,)
            # prior_params carries shape/scale for use as initialisation (merged into
            # initial_parameter_values in _optimize_mhr); prior_reg_weight defaults to 0
            # so no regularisation loss is applied.
            prior_params = build_prior_params(mean_shape, mean_scale, self._device)

            # Per-frame body pose initialisation; fall back to zeros where no prediction.
            pose_dim  = next(p for p in pose_list if p is not None).shape[0]
            pose_init = np.stack([
                p if p is not None else np.zeros(pose_dim) for p in pose_list
            ], axis=0)  # (N_frames, 130)
            initial_parameter_values = {
                "pose_params": torch.tensor(pose_init, dtype=torch.float32, device=self._device),
            }

            smpl_inputs = {
                "betas":           torch.tensor(original_data["shape"][serno_indices]).float().to(self._device),
                "body_pose":       torch.tensor(original_data["pose_cam"][serno_indices, 3:66]).float().to(self._device),
                "global_orient":   torch.tensor(original_data["pose_cam"][serno_indices, :3]).float().to(self._device),
                "left_hand_pose":  torch.tensor(original_data["pose_cam"][serno_indices, 75:120]).float().to(self._device),
                "right_hand_pose": torch.tensor(original_data["pose_cam"][serno_indices, 120:165]).float().to(self._device),
                "jaw_pose":   torch.zeros(len(serno_indices), 3, device=self._device),
                "leye_pose":  torch.zeros(len(serno_indices), 3, device=self._device),
                "reye_pose":  torch.zeros(len(serno_indices), 3, device=self._device),
                "expression": torch.zeros(len(serno_indices), self._smplx.num_expression_coeffs,
                                          device=self._device),
            }
            smpl_vertices = self._smplx(**smpl_inputs).vertices.detach().cpu().numpy()

            results = self._converter.convert_smpl2mhr(
                smpl_vertices=smpl_vertices,
                smpl_parameters=None,
                single_identity=True,
                return_mhr_parameters=True,
                return_mhr_vertices=is_first_serno,
                return_fitting_errors=True,
                initial_parameter_values=initial_parameter_values,
                prior_params=prior_params,
                prior_reg_weight=prior_reg_weight,
            )

            if is_first_serno and results.result_vertices is not None:
                _save_comparison_vis(
                    ds_name=ds_name,
                    serno=int(unique_serno),
                    smpl_verts=smpl_vertices[0],   # first frame, metres
                    mhr_verts=results.result_vertices[0],  # first frame, cm
                    save_dir=vis_dir,
                )
                is_first_serno = False

            for k, v in results.result_parameters.items():
                params_with_indices[k].append((serno_indices, v.detach().cpu().numpy()))

        # --- reconstruct full-length arrays and save ---
        params_list = {}
        for k, pairs in params_with_indices.items():
            s = pairs[0][1].shape
            arr = np.zeros((total_frames,) + s[1:], dtype=pairs[0][1].dtype)
            for idx, vals in pairs:
                arr[idx] = vals
            params_list[k] = arr

        combined = {key: original_data[key] for key in original_data.keys()}
        combined.update(params_list)

        os.makedirs(save_path, exist_ok=True)
        np.savez(output_path, **combined)
        print(f"{label} Saved → {output_path}")

    # ---- entry point for a chunk of sequences ----

    def convert(self, work_items: list[dict], jobs_per_gpu: int = 1) -> None:
        """
        Run work_items on this GPU with up to jobs_per_gpu concurrent jobs.

        For jobs_per_gpu > 1, each concurrent slot gets its own
        ConvertBedlamWithPrior instance (avoids shared mutable state in the
        MHR optimizer).  Instances are created once upfront and reused across
        all work items via a thread-safe queue.
        """
        import traceback
        from queue import Queue

        # Build a pool of converter instances — one per concurrent slot.
        inst_queue: Queue = Queue()
        for _ in range(jobs_per_gpu):
            inst_queue.put(ConvertBedlamWithPrior(device_id=self._device_id))

        def _run(item):
            inst = inst_queue.get()
            try:
                inst.process_sequence(
                    item["ds_name"],
                    item["npz_extra_path"],
                    item["prior_reg_weight"],
                    item["save_path"],
                )
            finally:
                inst_queue.put(inst)  # return to pool even on exception

        with ThreadPoolExecutor(max_workers=jobs_per_gpu) as executor:
            futures = {executor.submit(_run, item): item["ds_name"] for item in work_items}
            for future in as_completed(futures):
                ds_name = futures[future]
                try:
                    future.result()
                except Exception as e:
                    print(f"[ERROR] {ds_name}: {e}")
                    traceback.print_exc()


# ---------------------------------------------------------------------------
# 4D-Dress conversion (single GPU, unchanged logic)
# ---------------------------------------------------------------------------

def convert_4ddress(
    model: SAM3DBody,
    converter: Conversion,
    device: torch.device,
    prior_reg_weight: float,
) -> None:
    """Convert 4D-Dress SMPL meshes to MHR using model-conditioned fitting."""
    from sam_3d_body.data.d4dress_dataset import MultiD4DressDataset

    cfg = get_config_defaults()
    cfg.MODEL.MHR_HEAD.MHR_MODEL_PATH = SAM3D_MHR_MODEL_PATH
    MultiD4DressDataset(ids=None, cfg=cfg)  # kept for side-effects / validation

    subject_ids = [
        "00122", "00123", "00127", "00129", "00134", "00135", "00136", "00137",
        "00140", "00147", "00148", "00149", "00151", "00152", "00154", "00156",
        "00160", "00163", "00167", "00168", "00169", "00170", "00174", "00175",
        "00176", "00179", "00180", "00185", "00187", "00190",
    ]

    # smplx_model = converter.smpl_model

    for subject_id in subject_ids:
        takes = sorted(os.listdir(os.path.join(D4DRESS_DATA_PATH, subject_id, "Inner")))
        for take in takes:
            take_dir = os.path.join(D4DRESS_DATA_PATH, subject_id, "Inner", take)
            smpl_dir = os.path.join(take_dir, "SMPLX")
            if not os.path.isdir(smpl_dir):
                continue

            files = sorted([f for f in os.listdir(smpl_dir) if f.endswith(".ply")])
            if not files:
                continue

            print(f"[4D-Dress] {subject_id}/{take}: {len(files)} frames")
            vertices = np.stack(
                [trimesh.load(os.path.join(smpl_dir, f)).vertices for f in files], axis=0
            )

            camera_ids = ["0004", "0028", "0052", "0076"]
            shape_preds_list, scale_preds_list = [], []

            for cam_id in camera_ids:
                cam_img_dir = os.path.join(take_dir, "Capture", cam_id, "images")
                if not os.path.isdir(cam_img_dir):
                    continue
                cam_pkl = os.path.join(take_dir, "Capture", "cameras.pkl")
                if not os.path.exists(cam_pkl):
                    continue
                cameras = pickle.load(open(cam_pkl, "rb"))
                cam_data = cameras[cam_id]
                basic_info = pickle.load(open(os.path.join(take_dir, "basic_info.pkl"), "rb"))
                scan_frames = basic_info["scan_frames"]

                transform = Compose([
                    FakeGetBBoxCenterScale(),
                    TopdownAffine(input_size=cfg.DATASET.IMAGE_SIZE, use_udp=False),
                    VisionTransformWrapper(ToTensor()),
                ])
                cam_int = torch.tensor(cam_data["intrinsics"], dtype=torch.float32)

                for frame_idx in scan_frames:
                    img_path = os.path.join(cam_img_dir, f"capture-f{int(frame_idx):05d}.png")
                    if not os.path.exists(img_path):
                        continue
                    img = cv2.imread(img_path)
                    if img is None:
                        continue
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    h, w = img.shape[:2]

                    data_item = transform(dict(
                        img=img,
                        center=np.array([w / 2, h / 2], dtype=np.float32),
                        scale=float(max(h, w)),
                        bbox_format="xyxy",
                        mask=np.zeros((h, w), dtype=np.uint8),
                    ))
                    batch = default_collate([data_item])

                    bbox_center = batch["bbox_center"].to(device).float()
                    bbox_scale = batch["bbox_scale"].to(device).float()
                    ori_img_size = batch["ori_img_size"].to(device).float()
                    img_size = batch["img_size"].to(device).float()
                    affine_trans = batch["affine_trans"].to(device).float()

                    model_batch = {
                        "img":          batch["img"].unsqueeze(1).to(device),
                        "cam_int":      cam_int.unsqueeze(0).to(device),
                        "mask":         batch["mask"].unsqueeze(1).float().to(device) * -1.0,
                        "mask_score":   torch.ones(1, 1, 1, 1, device=device),
                        "person_valid": torch.ones(1, 1, device=device),
                        # Required by SAM3DBody.get_ray_condition() and decoder conditioning.
                        "affine_trans": affine_trans.unsqueeze(1),  # (B, N, 2, 3)
                        "bbox_center":  bbox_center.unsqueeze(1),   # (B, N, 2)
                        "bbox_scale":   bbox_scale.unsqueeze(1),    # (B, N, 2)
                        "ori_img_size": ori_img_size.unsqueeze(1),  # (B, N, 2)
                        "img_size":     img_size.unsqueeze(1),      # (B, N, 2)
                    }
                    with torch.no_grad():
                        out = model(model_batch, num_samples=0)
                    shape_preds_list.append(out["mhr"]["shape"].float().cpu().numpy())
                    scale_preds_list.append(out["mhr"]["scale_68D"].float().cpu().numpy())

            if not shape_preds_list:
                print(f"  No valid camera images found, skipping.")
                continue

            mean_shape = np.mean(shape_preds_list, axis=0).squeeze()
            mean_scale = np.mean(scale_preds_list, axis=0).squeeze()
            prior_params = build_prior_params(mean_shape, mean_scale, device)

            results = converter.convert_smpl2mhr(
                smpl_vertices=vertices,
                smpl_parameters=None,
                single_identity=True,
                return_mhr_parameters=True,
                return_fitting_errors=True,
                prior_params=prior_params,
                prior_reg_weight=prior_reg_weight,
            )
            mhr_parameters = results.result_parameters

            output_path = os.path.join(take_dir, "MHR_params_conditioned.npz")
            np.savez(
                output_path,
                lbs_params_np=mhr_parameters["lbs_model_params"].detach().cpu().numpy(),
                identity_coeffs_np=mhr_parameters["identity_coeffs"].detach().cpu().numpy(),
                face_expr_coeffs_np=mhr_parameters["face_expr_coeffs"].detach().cpu().numpy(),
            )
            print(f"  Saved → {output_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Model-conditioned SMPL→MHR conversion")
    parser.add_argument("--dataset",  type=str, required=True, choices=["bedlam", "4d-dress"])
    parser.add_argument("--gpus",     type=str, default="0",
                        help="Comma-separated GPU IDs, e.g. '0,1'")
    parser.add_argument("--prior_reg_weight", type=float, default=0,
                        help="Regularisation weight toward model prediction")
    parser.add_argument("--sequences", type=str, default=None,
                        help="Comma-separated dataset names to process (BEDLAM only)")
    parser.add_argument("--jobs_per_gpu", type=int, default=1,
                        help="Concurrent sequences per GPU (default 1; increase with care)")
    args = parser.parse_args()

    gpu_ids = [int(g.strip()) for g in args.gpus.split(",")]
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)
    print(f"CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")

    if args.dataset == "bedlam":
        npz_to_ds = _build_npz_to_dataset_map()
        requested = set(args.sequences.split(",")) if args.sequences else set(npz_to_ds.values())

        work_items = []
        for npz_basename, ds_name in sorted(npz_to_ds.items()):
            if ds_name not in requested:
                continue
            npz_extra_path = os.path.join(BEDLAM_EXTRA_PATH, npz_basename)
            if not os.path.exists(npz_extra_path):
                print(f"  [{ds_name}] extra npz not found, skipping")
                continue
            output_path = os.path.join(BEDLAM_SAVE_PATH, npz_basename)
            if os.path.exists(output_path):
                print(f"  [{ds_name}] already converted, skipping")
                continue
            work_items.append({
                "ds_name":          ds_name,
                "npz_extra_path":   npz_extra_path,
                "prior_reg_weight": args.prior_reg_weight,
                "save_path":        BEDLAM_SAVE_PATH,
            })

        if not work_items:
            print("Nothing to convert.")
            return

        n_gpus = len(gpu_ids)
        # Distribute work round-robin across GPUs (visible IDs are 0..n_gpus-1)
        chunks = [work_items[i::n_gpus] for i in range(n_gpus)]

        print(f"Converting {len(work_items)} sequences on {n_gpus} GPU(s), "
              f"{args.jobs_per_gpu} job(s)/GPU")

        def run_on_gpu(visible_gpu_id: int, chunk: list) -> None:
            if not chunk:
                return
            conv = ConvertBedlamWithPrior(device_id=visible_gpu_id)
            conv.convert(chunk, jobs_per_gpu=args.jobs_per_gpu)

        if n_gpus == 1:
            run_on_gpu(0, work_items)
        else:
            threads = [
                threading.Thread(target=run_on_gpu, args=(i, chunk), daemon=True)
                for i, chunk in enumerate(chunks)
                if chunk
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

    elif args.dataset == "4d-dress":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Loading SAM-3D-Body model for 4D-Dress...")
        model = load_sam3d_model(device)
        smplx_model = smplx.create(
            model_type="smplx", model_path=MHR_MODEL_FILES_PATH,
            num_betas=11, gender="neutral", use_pca=False, flat_hand_mean=True,
        ).to(device)
        mhr_model = MHR.from_files(folder=Path(MHR_ASSETS_PATH), lod=1, device=device)
        converter  = Conversion(mhr_model=mhr_model, smpl_model=smplx_model, method="pytorch")
        convert_4ddress(model=model, converter=converter, device=device,
                        prior_reg_weight=args.prior_reg_weight)


if __name__ == "__main__":
    main()
