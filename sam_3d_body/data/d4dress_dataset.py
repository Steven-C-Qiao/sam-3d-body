import os
import torch
import smplx
import pickle
import random
import trimesh
import numpy as np
import torch.nn as nn
from PIL import Image
from typing import Sequence, Dict, Optional
from torchvision.transforms import ToTensor
from collections import defaultdict
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
from sam_3d_body.data.transforms import (
    Compose,
    TopdownAffine,
    VisionTransformWrapper,
)


def load_pickle(pkl_dir):
    with open(pkl_dir, "rb") as f:
        try:
            return pickle.load(f)
        except LookupError as e:
            # Try with encoding for python2 pickles loaded in python3
            f.seek(0)
            return pickle.load(f, encoding="latin1")


# save data to pkl_dir
def save_pickle(pkl_dir, data):
    pickle.dump(data, open(pkl_dir, "wb"))


# load image as numpy array
def load_image(img_dir):
    return np.array(Image.open(img_dir))


# save numpy array image
def save_image(img_dir, img):
    Image.fromarray(img).save(img_dir)


# get xyz rotation matrix
def rotation_matrix(angle, axis="x"):
    # get cos and sin from angle
    c, s = np.cos(np.radians(angle)), np.sin(np.radians(angle))
    # get totation matrix
    R = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "y":
        R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    if axis == "z":
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return R


def convert_intrinsics_to_pytorch3d_convention(intrinsics):
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    px = intrinsics[0, 2]
    py = intrinsics[1, 2]
    K = np.array([[-fx, 0, px, 0], [0, -fy, py, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    return K


def get_R_T_from_extrinsics(extrinsics):
    R = extrinsics[:3, :3]
    T = extrinsics[:3, 3]
    return R, T


def d4dress_cameras_to_pytorch3d_cameras(cameras, ids):
    Rs, Ts, Ks = [], [], []
    for camera_id in ids:
        camera = cameras[camera_id]
        R, T = get_R_T_from_extrinsics(camera["extrinsics"])
        K = convert_intrinsics_to_pytorch3d_convention(camera["intrinsics"])
        Rs.append(R.T)
        Ts.append(T)
        Ks.append(K)

    Rs = np.stack(Rs)
    Ts = np.stack(Ts)
    Ks = np.stack(Ks)

    return Rs, Ts, Ks


class FakeGetBBoxCenterScale(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, results: Dict) -> Optional[dict]:
        results["bbox_center"] = results["center"].astype(np.float32)
        results["bbox_scale"] = (
            np.array([results["scale"], results["scale"]], dtype=np.float32) * 200.0
        )
        return results


PATH_TO_DATASET = "/scratches/kyuban/share/4DDress"

# Use timm's names
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


class MultiD4DressDataset(Dataset):
    def __init__(self, ids, cfg):
        self.cfg = cfg
        self.num_frames_pp = 4
        self.lengthen_by = 1

        self.body_model = "smplx"
        self.num_joints = 55

        self.subject_ids = ids
        self.camera_ids = ["0004", "0028", "0052", "0076"]

        self.takes = defaultdict(list)
        self.num_takes = defaultdict(int)

        for subject_id in self.subject_ids:
            inner_takes = sorted(
                os.listdir(os.path.join(PATH_TO_DATASET, subject_id, "Inner"))
            )
            inner_takes = [
                (take, "Inner") for take in inner_takes if take.startswith("Take")
            ]

            self.takes[subject_id] = inner_takes
            self.num_takes[subject_id] = len(inner_takes)

        self.transform = Compose(
            [
                FakeGetBBoxCenterScale(),
                TopdownAffine(input_size=self.cfg.DATASET.IMAGE_SIZE, use_udp=False),
                VisionTransformWrapper(ToTensor()),
            ]
        )

    def __len__(self):
        return int(len(self.subject_ids) * self.lengthen_by)

    def __getitem__(self, index):
        item = defaultdict(list)

        subject_id = self.subject_ids[index // self.lengthen_by]

        num_takes = self.num_takes[subject_id]
        sampled_take = self.takes[subject_id][torch.randint(0, num_takes, (1,)).item()]
        take_dir = os.path.join(
            PATH_TO_DATASET, subject_id, sampled_take[1], sampled_take[0]
        )

        item["take_dir"] = take_dir
        item["scan_ids"] = subject_id

        mhr_params = np.load(os.path.join(take_dir, "MHR_params.npz"))
        mhr_shape = mhr_params["identity_coeffs_np"]
        mhr_pose = mhr_params["lbs_params_np"]
        mhr_expr = mhr_params["face_expr_coeffs_np"]


        # Load basic_info from main take
        basic_info = load_pickle(os.path.join(take_dir, "basic_info.pkl"))
        scan_frames, scan_rotation = basic_info["scan_frames"], basic_info["rotation"]
        
        # Convert scan_frames to numpy array if it's a list
        scan_frames = np.array(scan_frames) if not isinstance(scan_frames, np.ndarray) else scan_frames

        # Sample frame indices (not frame numbers) to correctly index into MHR params
        num_scan_frames = len(scan_frames)
        sampled_indices = np.random.choice(
            num_scan_frames, size=self.num_frames_pp, replace=False
        )
        sampled_frames = scan_frames[sampled_indices]

        # Sample cameras
        sampled_cameras = np.random.choice(
            self.camera_ids, size=self.num_frames_pp, replace=False
        )

        # Load camera params
        camera_params = load_pickle(os.path.join(take_dir, "Capture", "cameras.pkl"))

        cam_int = [camera_params[camera_id]["intrinsics"] for camera_id in sampled_cameras]
        cam_ext = [camera_params[camera_id]["extrinsics"] for camera_id in sampled_cameras]
        # trans_cam = [cam_ext[i][:3, 3] for i in range(len(cam_ext))]
        cam_int = np.stack(cam_int).astype(np.float32)
        cam_ext = np.stack(cam_ext).astype(np.float32)

        # trans_cam = np.stack(trans_cam).astype(np.float32)

        item["cam_int"] = cam_int
        item["cam_ext"] = cam_ext

        views = []
        for i, (sampled_idx, sampled_frame) in enumerate(zip(sampled_indices, sampled_frames)):
            view = {}
            img_fname = os.path.join(
                take_dir,
                "Capture",
                sampled_cameras[i],
                "images",
                "capture-f{}.png".format(sampled_frame),
            )
            mask_fname = os.path.join(
                take_dir,
                "Capture",
                sampled_cameras[i],
                "masks",
                "mask-f{}.png".format(sampled_frame),
            )

            # Load images and apply transforms
            img = load_image(img_fname)
            mask = load_image(mask_fname)

            view['img_ori'] = img

            img_h, img_w = img.shape[:2]
            assert (
                img_h >= img_w
            ), f"D4Dress images expected portrait mode (H>=W), got H={img_h}, W={img_w}"
            # bbox_center = np.array([img_w / 4.0, img_h / 4.0], dtype=np.float32)
            bbox_center = np.array([cam_int[i][0, 2], cam_int[i][1, 2]], dtype=np.float32)
            bbox_scale = img_w / 200.0

            # view['center'] = bbox_center
            # view['scale'] = bbox_scale

            data_info = dict(
                img=img,
                center=bbox_center,
                scale=bbox_scale,
                bbox_format="xyxy",
                mask=mask,
            )
            data_list = [self.transform(data_info)]
            data = default_collate(data_list)

            for key in data:
                view[key] = data[key]

            views.append(view)

        for field in set(views[0].keys()):
            try:
                stacked = torch.stack([v[field] for v in views], dim=0)
                item[field] = stacked
            except:
                item[field] = [v[field] for v in views]

        item['num_views'] = len(views)
        item['mask'] = item['mask'].float().unsqueeze(-3)
        item['mask_score'] = torch.ones((len(views), 1, 1, 1))
        item["person_valid"] = torch.ones((len(views), 1, 1))
        item["shape_params"] = mhr_shape[sampled_indices]
        item["model_params"] = mhr_pose[sampled_indices]
        item["face_expr_coeffs"] = mhr_expr[sampled_indices]
        item['dataset_name'] = '4d-dress'

        return item