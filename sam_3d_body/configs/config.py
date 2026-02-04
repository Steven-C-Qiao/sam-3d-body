from yacs.config import CfgNode 

INDICES_PATH = "/scratches/juban/cq244/sam-3d-body/tinker/mhr_kp_sample_128.npy"

_C = CfgNode()

_C.TRAIN = CfgNode()
_C.TRAIN.MODEL_TYPE = "full"  # Options: "full" (SAM3DBody) or "toy" (ToyModel)
_C.TRAIN.USE_FP16 = True
_C.TRAIN.FP16_TYPE = "high"
_C.TRAIN.LR = 2e-5
_C.TRAIN.NUM_EPOCHS = 50
_C.TRAIN.CKPT_PATH = "checkpoints/sam-3d-body-dinov3/model.ckpt"
_C.TRAIN.FREEZE_BACKBONE = False


_C.LOSS = CfgNode()
_C.LOSS.SHAPE_PARAM_WEIGHT = 1.0
_C.LOSS.SCALE_PARAM_WEIGHT = 1.0
_C.LOSS.POSE_PARAM_WEIGHT = 1.0
_C.LOSS.KP2D_WEIGHT = 100.0
_C.LOSS.KP3D_WEIGHT = 100.0


# Dataset hparams
_C.DATASET = CfgNode()
_C.DATASET.BATCH_SIZE = 32
_C.DATASET.NUM_WORKERS = 64
_C.DATASET.NOISE_FACTOR = 0.4
_C.DATASET.SCALE_FACTOR = 0.0
_C.DATASET.CROP_PROB = 0.0
_C.DATASET.CROP_FACTOR = 0.0
_C.DATASET.PIN_MEMORY = True
_C.DATASET.SHUFFLE_TRAIN = True
_C.DATASET.TRAIN_DS = 'all'
_C.DATASET.VAL_DS = 'orbit-archviz-15'
_C.DATASET.IMG_RES = 256
_C.DATASET.MESH_COLOR = 'pinkish'
_C.DATASET.DATASETS_AND_RATIOS = 'static-hdri_zoom-suburbd_zoom-gym_static-office_orbit-office_pitchup-stadium_pitchdown-stadium_static-hdri-bmi_closeup-suburbb-bmi_closeup-suburbc-bmi_zoom-gym-bmi_static-office-hair_zoom-suburbd-hair_static-gym-hair_orbit-archviz-19_orbit-archviz-12_orbit-archviz-10'
_C.DATASET.CROP_PERCENT = 0.8
_C.DATASET.ALB = True
_C.DATASET.ALB_PROB = 0.3
_C.DATASET.proj_verts = False
_C.DATASET.FOCAL_LENGTH = 5000


_C.MODEL = CfgNode()
_C.MODEL.IMAGE_SIZE = [512, 512]
_C.MODEL.IMAGE_MEAN = [0.485, 0.456, 0.406]
_C.MODEL.IMAGE_STD = [0.229, 0.224, 0.225]



_C.MODEL.ENABLE_BODY = True
_C.MODEL.ENABLE_HAND = True
_C.MODEL.DENSE_KEYPOINTS = True
_C.MODEL.SAMPLE_SHAPE = True
_C.MODEL.SAMPLE_SCALE = True
_C.MODEL.SAMPLE_POSE = True



_C.MODEL.BACKBONE = CfgNode()
_C.MODEL.BACKBONE.TYPE = "dinov3_vith16plus"
_C.MODEL.BACKBONE.PRETRAINED_WEIGHTS = ""
_C.MODEL.BACKBONE.FROZEN_STAGES = -1
_C.MODEL.BACKBONE.DROP_PATH_RATE = 0.1

_C.MODEL.DECODER = CfgNode()
_C.MODEL.DECODER.TYPE = "sam"
_C.MODEL.DECODER.DIM = 1024
_C.MODEL.DECODER.DEPTH = 6
_C.MODEL.DECODER.HEADS = 8
_C.MODEL.DECODER.MLP_DIM = 1024
_C.MODEL.DECODER.DIM_HEAD = 64
_C.MODEL.DECODER.LAYER_SCALE_INIT = 0.0
_C.MODEL.DECODER.DROP_RATE = 0.0
_C.MODEL.DECODER.ATTN_DROP_RATE = 0.0
_C.MODEL.DECODER.DROP_PATH_RATE = 0.0
_C.MODEL.DECODER.FFN_TYPE = "origin"
_C.MODEL.DECODER.ENABLE_TWOWAY = False
_C.MODEL.DECODER.REPEAT_PE = True
_C.MODEL.DECODER.FROZEN = False
_C.MODEL.DECODER.CONDITION_TYPE = "cliff"
_C.MODEL.DECODER.USE_INTRIN_CENTER = True
_C.MODEL.DECODER.DO_INTERM_PREDS = True
_C.MODEL.DECODER.DO_INTERM_SUP = True
_C.MODEL.DECODER.DO_KEYPOINT_TOKENS = True
_C.MODEL.DECODER.DO_HAND_DETECT_TOKENS = True
_C.MODEL.DECODER.KEYPOINT_TOKEN_UPDATE = "v2"
_C.MODEL.DECODER.KEYPOINT_TOKEN_UPDATE_COORD_EMB_USE_MLP = True
_C.MODEL.DECODER.DO_KEYPOINT3D_TOKENS = True

_C.MODEL.PROMPT_ENCODER = CfgNode()
_C.MODEL.PROMPT_ENCODER.ENABLE = True
_C.MODEL.PROMPT_ENCODER.MAX_NUM_CLICKS = 2
_C.MODEL.PROMPT_ENCODER.PROMPT_KEYPOINTS = "mhr70"
_C.MODEL.PROMPT_ENCODER.FROZEN = False
_C.MODEL.PROMPT_ENCODER.KEYPOINT_SAMPLER = CfgNode()
_C.MODEL.PROMPT_ENCODER.KEYPOINT_SAMPLER.TYPE = "v1"
_C.MODEL.PROMPT_ENCODER.KEYPOINT_SAMPLER.WORST_RATIO = 0.8
_C.MODEL.PROMPT_ENCODER.KEYPOINT_SAMPLER.KEYBODY_RATIO = 0.8
_C.MODEL.PROMPT_ENCODER.KEYPOINT_SAMPLER.NEGATIVE_RATIO = 0.1
_C.MODEL.PROMPT_ENCODER.KEYPOINT_SAMPLER.DUMMY_RATIO = 0.1
_C.MODEL.PROMPT_ENCODER.KEYPOINT_SAMPLER.DISTANCE_THRESH = 0.0001
_C.MODEL.PROMPT_ENCODER.MASK_EMBED_TYPE = "v2"
_C.MODEL.PROMPT_ENCODER.MASK_PROMPT = "v1"

_C.MODEL.PERSON_HEAD = CfgNode()
_C.MODEL.PERSON_HEAD.POSE_TYPE = "uncertainty"
_C.MODEL.PERSON_HEAD.CAMERA_ENABLE = True
_C.MODEL.PERSON_HEAD.CAMERA_TYPE = "perspective"
_C.MODEL.PERSON_HEAD.ZERO_POSE_INIT = True
_C.MODEL.PERSON_HEAD.ZERO_POSE_INIT_BODY_FACTOR = 1

_C.MODEL.MHR_HEAD = CfgNode()
_C.MODEL.MHR_HEAD.MLP_DEPTH = 2
_C.MODEL.MHR_HEAD.MLP_CHANNEL_DIV_FACTOR = 1
_C.MODEL.MHR_HEAD.DEFAULT_SCALE_FACTOR_HAND = 10
_C.MODEL.MHR_HEAD.ENABLE_BODY = True
_C.MODEL.MHR_HEAD.ENABLE_HAND = True
_C.MODEL.MHR_HEAD.MHR_MODEL_PATH = "checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt"

_C.MODEL.CAMERA_HEAD = CfgNode()
_C.MODEL.CAMERA_HEAD.MLP_DEPTH = 2
_C.MODEL.CAMERA_HEAD.MLP_CHANNEL_DIV_FACTOR = 1
_C.MODEL.CAMERA_HEAD.DEFAULT_SCALE_FACTOR_HAND = 10



def get_config_defaults():
    return _C.clone()


import os 
PATH = "/scratches/juban/cq244/BEDLAM/"


SMPL_MODEL_DIR = os.path.join(PATH, 'data/body_models/SMPL_python_v.1.1.0/smpl/models')
SMPLX_MODEL_DIR = os.path.join(PATH, 'data/body_models/smplx/models/smplx')
MANO_MODEL_DIR = os.path.join(PATH, 'data/body_models/mano/mano_v1_2/models/')

JOINT_REGRESSOR_TRAIN_EXTRA = os.path.join(PATH, 'data/utils/J_regressor_extra.npy')
JOINT_REGRESSOR_H36M = os.path.join(PATH, 'data/utils/J_regressor_h36m.npy')
SMPL_MEAN_PARAMS = os.path.join(PATH, 'data/utils/smpl_mean_params.npz')
JOINT_REGRESSOR_14 = os.path.join(PATH, 'data/utils/SMPLX_to_J14.pkl')
SMPLX2SMPL = os.path.join(PATH, 'data/utils/smplx2smpl.pkl')
MEAN_PARAMS = os.path.join(PATH, 'data/utils/all_means.pkl')
DOWNSAMPLE_MAT_SMPLX_PATH = os.path.join(PATH, 'data/utils/downsample_mat_smplx.pkl')

DATASET_FOLDERS = {
    '3dpw-test-cam': os.path.join(PATH, 'data/test_images/3DPW'),
    '3dpw-val-cam': os.path.join(PATH, 'data/test_images/3DPW'),
    'rich': os.path.join(PATH, 'data/test_images/RICH'),
    'h36m-p1': os.path.join(PATH, 'data/test_images/h36m/'),

    'agora-bfh': os.path.join(PATH, 'data/training_images/images/'),
    'agora-body': os.path.join(PATH, 'data/training_images/images/'),
    'zoom-suburbd': os.path.join(PATH, 'data/training_images/20221010_3-10_500_batch01hand_zoom_suburb_d_6fps/png'),
    'closeup-suburba': os.path.join(PATH, 'data/training_images/20221011_1_250_batch01hand_closeup_suburb_a_6fps/png'),
    'closeup-suburbb': os.path.join(PATH, 'data/training_images/20221011_1_250_batch01hand_closeup_suburb_b_6fps/png'),
    'closeup-suburbc': os.path.join(PATH, 'data/training_images/20221011_1_250_batch01hand_closeup_suburb_c_6fps/png'),
    'closeup-suburbd': os.path.join(PATH, 'data/training_images/20221011_1_250_batch01hand_closeup_suburb_d_6fps/png'),
    'closeup-gym': os.path.join(PATH, 'data/training_images/20221012_1_500_batch01hand_closeup_highSchoolGym_6fps/png'),
    'zoom-gym': os.path.join(PATH, 'data/training_images/20221012_3-10_500_batch01hand_zoom_highSchoolGym_6fps/png'),
    'static-gym': os.path.join(PATH, 'data/training_images/20221013_3-10_500_batch01hand_static_highSchoolGym_6fps/png'),
    'static-office': os.path.join(PATH, 'data/training_images/20221013_3_250_batch01hand_static_bigOffice_6fps/png'),
    'orbit-office': os.path.join(PATH, 'data/training_images/20221013_3_250_batch01hand_orbit_bigOffice_6fps/png'),
    'orbit-archviz-15': os.path.join(PATH, 'data/training_images/20221014_3_250_batch01hand_orbit_archVizUI3_time15_6fps/png'),
    'orbit-archviz-19': os.path.join(PATH, 'data/training_images/20221015_3_250_batch01hand_orbit_archVizUI3_time19_6fps/png'),
    'orbit-archviz-12': os.path.join(PATH, 'data/training_images/20221015_3_250_batch01hand_orbit_archVizUI3_time12_6fps/png'),
    'orbit-archviz-10': os.path.join(PATH, 'data/training_images/20221015_3_250_batch01hand_orbit_archVizUI3_time10_6fps/png'),
    'static-hdri': os.path.join(PATH, 'data/training_images/20221010_3_1000_batch01hand_6fps/png'),
    'static-hdri-zoomed': os.path.join(PATH, 'data/training_images/20221017_3_1000_batch01hand_6fps/png'),
    'staticzoomed-suburba-frameocc': os.path.join(PATH, 'data/training_images/20221017_1_250_batch01hand_closeup_suburb_a_6fps/png'),
    'zoom-suburbb-frameocc': os.path.join(PATH, 'data/training_images/20221018_1_250_batch01hand_zoom_suburb_b_6fps/png'),
    'static-hdri-frameocc': os.path.join(PATH, 'data/training_images/20221018_3-8_250_batch01hand_6fps/png'),
    'orbit-archviz-objocc': os.path.join(PATH, 'data/training_images/20221018_3_250_batch01hand_orbit_archVizUI3_time15_6fps/png'),
    'pitchup-stadium': os.path.join(PATH, 'data/training_images/20221018_3-8_250_batch01hand_pitchUp52_stadium_6fps/png'),
    'pitchdown-stadium': os.path.join(PATH, 'data/training_images/20221018_3-8_250_batch01hand_pitchDown52_stadium_6fps/png'),
    'static-hdri-bmi': os.path.join(PATH, 'data/training_images/20221019_3_250_highbmihand_6fps/png'),
    'closeup-suburbb-bmi': os.path.join(PATH, 'data/training_images/20221019_1_250_highbmihand_closeup_suburb_b_6fps/png'),
    'closeup-suburbc-bmi': os.path.join(PATH, 'data/training_images/20221019_1_250_highbmihand_closeup_suburb_c_6fps/png'),
    'static-stadium-bmi': os.path.join(PATH, 'data/training_images/20221019_3-8_250_highbmihand_static_stadium_6fps/png'),
    'orbit-stadium-bmi': os.path.join(PATH, 'data/training_images/20221019_3-8_250_highbmihand_orbit_stadium_6fps/png'),
    'static-suburbd-bmi': os.path.join(PATH, 'data/training_images/20221019_3-8_1000_highbmihand_static_suburb_d_6fps/png'),
    'zoom-gym-bmi': os.path.join(PATH, 'data/training_images/20221020-3-8_250_highbmihand_zoom_highSchoolGym_a_6fps/png'),
    'static-office-hair': os.path.join(PATH, 'data/training_images/20221022_3_250_batch01handhair_static_bigOffice_30fps/png'),
    'zoom-suburbd-hair': os.path.join(PATH, 'data/training_images/20221024_10_100_batch01handhair_zoom_suburb_d_30fps/png'),
    'static-gym-hair': os.path.join(PATH, 'data/training_images/20221024_3-10_100_batch01handhair_static_highSchoolGym_30fps/png'),

    'coco': os.path.join(PATH, 'data/real_training_images/coco'),
    'mpii': os.path.join(PATH, 'data/real_training_images/mpii'),
    'h36m': os.path.join(PATH, 'data/real_training_images/h36m'),
    'mpi-inf-3dhp': os.path.join(PATH, 'data/real_training_images/mpi_inf_3dhp'),
    '3dpw-train-smpl': os.path.join(PATH, 'data/real_training_images/3DPW'),
    
    '3dpw-train-smplx': os.path.join(PATH, 'data/real_training_images/3DPW'),
}

DATASET_FILES = [
    {
        '3dpw-test-cam': os.path.join(PATH, 'data/eval_data_parsed/3dpw_test.npz'),
        '3dpw-val-cam': os.path.join(PATH, 'data/eval_data_parsed/3dpw_validation.npz'),
        'rich': os.path.join(PATH, 'data/eval_data_parsed/rich_test.npz'),
        'h36m-p1': os.path.join(PATH, 'data/eval_data_parsed/h36m_valid_protocol1.npz'),
        'orbit-stadium-bmi': os.path.join(PATH, 'data/training_labels/all_npz_12_training_extra_mhr/20221019_3-8_250_highbmihand_orbit_stadium_6fps.npz'),
        'orbit-archviz-objocc': os.path.join(PATH, 'data/training_labels/all_npz_12_training_extra_mhr/20221018_3_250_batch01hand_orbit_archVizUI3_time15_6fps.npz'),
        'zoom-suburbb-frameocc': os.path.join(PATH, 'data/training_labels/all_npz_12_training_extra_mhr/20221018_1_250_batch01hand_zoom_suburb_b_6fps.npz'),
        'static-hdri-frameocc': os.path.join(PATH, 'data/training_labels/all_npz_12_training_extra_mhr/20221018_3-8_250_batch01hand_6fps.npz'),
        'zoom-gym': os.path.join(PATH, 'data/training_labels/all_npz_12_training_extra_mhr/20221012_3-10_500_batch01hand_zoom_highSchoolGym_6fps.npz'),
        'static-gym': os.path.join(PATH, 'data/training_labels/all_npz_12_training_extra_mhr/20221013_3-10_500_batch01hand_static_highSchoolGym_6fps.npz'),
        'orbit-archviz-15': os.path.join(PATH, 'data/training_labels/all_npz_12_training_extra_mhr/20221014_3_250_batch01hand_orbit_archVizUI3_time15_6fps.npz'),
        # NOTE: Temporarily added zoom-suburbd for code testing
    },
    {
        'agora-bfh': os.path.join(PATH, 'data/training_labels/all_npz_12_training_extra_mhr/agora-bfh.npz'),
        'agora-body': os.path.join(PATH, 'data/training_labels/all_npz_12_training_extra_mhr/agora-body.npz'),
        '3dpw-train-smplx': os.path.join(PATH, 'data/training_labels/3dpw_train_smplx.npz'),

        'zoom-suburbd': os.path.join(PATH, 'data/training_labels/all_npz_12_training_extra_mhr/20221010_3-10_500_batch01hand_zoom_suburb_d_6fps.npz'),
        'closeup-suburba': os.path.join(PATH, 'data/training_labels/all_npz_12_training_extra_mhr/20221011_1_250_batch01hand_closeup_suburb_a_6fps.npz'),
        'closeup-suburbb': os.path.join(PATH, 'data/training_labels/all_npz_12_training_extra_mhr/20221011_1_250_batch01hand_closeup_suburb_b_6fps.npz'),
        'closeup-suburbc': os.path.join(PATH, 'data/training_labels/all_npz_12_training_extra_mhr/20221011_1_250_batch01hand_closeup_suburb_c_6fps.npz'),
        'closeup-suburbd': os.path.join(PATH, 'data/training_labels/all_npz_12_training_extra_mhr/20221011_1_250_batch01hand_closeup_suburb_d_6fps.npz'),
        'closeup-gym': os.path.join(PATH, 'data/training_labels/all_npz_12_training_extra_mhr/20221012_1_500_batch01hand_closeup_highSchoolGym_6fps.npz'),
        'zoom-gym': os.path.join(PATH, 'data/training_labels/all_npz_12_training_extra_mhr/20221012_3-10_500_batch01hand_zoom_highSchoolGym_6fps.npz'),
        'static-gym': os.path.join(PATH, 'data/training_labels/all_npz_12_training_extra_mhr/20221013_3-10_500_batch01hand_static_highSchoolGym_6fps.npz'),
        'static-office': os.path.join(PATH, 'data/training_labels/all_npz_12_training_extra_mhr/20221013_3_250_batch01hand_static_bigOffice_6fps.npz'),
        'orbit-office': os.path.join(PATH, 'data/training_labels/all_npz_12_training_extra_mhr/20221013_3_250_batch01hand_orbit_bigOffice_6fps.npz'),
        'orbit-archviz-15': os.path.join(PATH, 'data/training_labels/all_npz_12_training_extra_mhr/20221014_3_250_batch01hand_orbit_archVizUI3_time15_6fps.npz'),
        'orbit-archviz-19': os.path.join(PATH, 'data/training_labels/all_npz_12_training_extra_mhr/20221015_3_250_batch01hand_orbit_archVizUI3_time19_6fps.npz'),
        'orbit-archviz-12': os.path.join(PATH, 'data/training_labels/all_npz_12_training_extra_mhr/20221015_3_250_batch01hand_orbit_archVizUI3_time12_6fps.npz'),
        'orbit-archviz-10': os.path.join(PATH, 'data/training_labels/all_npz_12_training_extra_mhr/20221015_3_250_batch01hand_orbit_archVizUI3_time10_6fps.npz'),
        'static-hdri': os.path.join(PATH, 'data/training_labels/all_npz_12_training_extra_mhr/20221010_3_1000_batch01hand_6fps.npz'),
        'static-hdri-zoomed': os.path.join(PATH, 'data/training_labels/all_npz_12_training_extra_mhr/20221017_3_1000_batch01hand_6fps.npz'),
        'staticzoomed-suburba-frameocc': os.path.join(PATH, 'data/training_labels/all_npz_12_training_extra_mhr/20221017_1_250_batch01hand_closeup_suburb_a_6fps.npz'),
        'pitchup-stadium': os.path.join(PATH, 'data/training_labels/all_npz_12_training_extra_mhr/20221018_3-8_250_batch01hand_pitchUp52_stadium_6fps.npz'),
        'static-hdri-bmi': os.path.join(PATH, 'data/training_labels/all_npz_12_training_extra_mhr/20221019_3_250_highbmihand_6fps.npz'),
        'closeup-suburbb-bmi': os.path.join(PATH, 'data/training_labels/all_npz_12_training_extra_mhr/20221019_1_250_highbmihand_closeup_suburb_b_6fps.npz'),
        'closeup-suburbc-bmi': os.path.join(PATH, 'data/training_labels/all_npz_12_training_extra_mhr/20221019_1_250_highbmihand_closeup_suburb_c_6fps.npz'),
        'static-suburbd-bmi': os.path.join(PATH, 'data/training_labels/all_npz_12_training_extra_mhr/20221019_3-8_1000_highbmihand_static_suburb_d_6fps.npz'),
        'zoom-gym-bmi': os.path.join(PATH, 'data/training_labels/all_npz_12_training_extra_mhr/20221020-3-8_250_highbmihand_zoom_highSchoolGym_a_6fps.npz'),
        'pitchdown-stadium': os.path.join(PATH, 'data/training_labels/all_npz_12_training_extra_mhr/20221018_3-8_250_batch01hand_pitchDown52_stadium_6fps.npz'),
        'static-office-hair': os.path.join(PATH, 'data/training_labels/all_npz_12_training_extra_mhr/20221022_3_250_batch01handhair_static_bigOffice_30fps.npz'),
        'zoom-suburbd-hair': os.path.join(PATH, 'data/training_labels/all_npz_12_training_extra_mhr/20221024_10_100_batch01handhair_zoom_suburb_d_30fps.npz'),
        'static-gym-hair': os.path.join(PATH, 'data/training_labels/all_npz_12_training_extra_mhr/20221024_3-10_100_batch01handhair_static_highSchoolGym_30fps.npz'),
        'orbit-stadium-bmi': os.path.join(PATH, 'data/training_labels/all_npz_12_training_extra_mhr/20221019_3-8_250_highbmihand_orbit_stadium_6fps.npz'),

        'coco': os.path.join(PATH, 'data/real_training_labels/coco.npz'),
        'mpii': os.path.join(PATH, 'data/real_training_labels//mpii.npz'),
        'h36m': os.path.join(PATH, 'data/real_training_labels//h36m_train.npz'),
        'mpi-inf-3dhp': os.path.join(PATH, 'data/real_training_labels//mpi_inf_3dhp_train.npz'),
        '3dpw-train-smpl': os.path.join(PATH, 'data/real_training_labels/3dpw_train.npz'),
    }
]

# Download the models from https://github.com/leoxiaobin/deep-high-resolution-net.pytorch and update the path
PRETRAINED_CKPT_FOLDER = {
    'hrnet_w32-coco': 'data/ckpt/pretrained/pose_hrnet_w32_256x192.pth',
    'hrnet_w32-imagenet': 'data/ckpt/pretrained/hrnetv2_w32_imagenet_pretrained.pth',
    'hrnet_w32-scratch': '',
    'hrnet_w48-coco': 'data/ckpt/pretrained/pose_hrnet_w48_256x192.pth',
    'hrnet_w48-imagenet': 'data/ckpt/pretrained/hrnetv2_w48_imagenet_pretrained.pth',
    'hrnet_w48-scratch': '',

}
