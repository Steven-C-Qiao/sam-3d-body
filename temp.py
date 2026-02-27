import numpy as np
import os

DATA_BASE_PATH = "/scratches/columbo2/cq244/BEDLAM/data/"
NPZ_PATH = os.path.join(
    DATA_BASE_PATH,
    "training_labels/all_npz_12_training_extra_mhr/20221010_3_1000_batch01hand_6fps.npz",
)
IMAGE_DIR = os.path.join(
    DATA_BASE_PATH, "training_images/20221010_3_1000_batch01hand_6fps"
)
MHR_MODEL_PATH = (
    "/scratches/columbo2/cq244/sam-3d-body/checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt"
)


all_npzs = sorted([
        os.path.join(DATA_BASE_PATH, "training_labels/all_npz_12_training_extra_mhr", f)
        for f in os.listdir(
            os.path.join(
                DATA_BASE_PATH, "training_labels/all_npz_12_training_extra_mhr"
            )
        )
        if f.endswith(".npz")
    ])
