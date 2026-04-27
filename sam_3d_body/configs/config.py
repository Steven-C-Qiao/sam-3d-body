import os
from yacs.config import CfgNode
from sam_3d_body.configs.paths import BEDLAM1_PATH, BEDLAM2_PATH

PATH = BEDLAM1_PATH  # legacy alias; body-model assets live under BEDLAM1 checkout
BEDLAM2_LABELS_DIR = os.path.join(BEDLAM2_PATH, "bedlam2_labels_mhr_conditioned")

INDICES_PATH = "checkpoints/sam-3d-body-dinov3/assets/mhr_kp_sample_128.npy"

_C = CfgNode()

_C.TRAIN = CfgNode()
_C.TRAIN.MODEL_TYPE = "full"  # Options: "full" (SAM3DBody) or "toy" (ToyModel)
_C.TRAIN.USE_FP16 = True
_C.TRAIN.FP16_TYPE = "high"
_C.TRAIN.LR = 2e-5
_C.TRAIN.NUM_EPOCHS = 50
_C.TRAIN.MAX_STEPS = -1
_C.TRAIN.CKPT_PATH = "checkpoints/sam-3d-body-dinov3/model.ckpt"
_C.TRAIN.FREEZE_BACKBONE = True
_C.TRAIN.GRAD_NORM_PROBE = 0  # >0: every N steps, print per-loss grad norm w.r.t. trainable params


_C.MODEL = CfgNode()
_C.MODEL.DECODER = CfgNode()
_C.MODEL.ENABLE_BODY = True
_C.MODEL.ENABLE_HAND = True
_C.MODEL.DENSE_KEYPOINTS = True
_C.MODEL.SAMPLE_SHAPE = True
_C.MODEL.SAMPLE_SCALE = True
_C.MODEL.SAMPLE_POSE = True
_C.MODEL.FULL_COV = True
_C.MODEL.MODEL_GLOB_ROT = True
_C.MODEL.MODEL_SHAPE = True
_C.MODEL.MODEL_SCALE = True
_C.MODEL.MODEL_CAM = True

##########################################
_C.MODEL.DECODER.USE_LORA = True
_C.MODEL.NUM_SAMPLES = 25
_C.MODEL.HEAD_TYPE = "nf_ar"
_C.MODEL.FLOW_COUPLING = "clamped_affine"
_C.MODEL.FLOW_NUM_LAYERS = 8
_C.MODEL.FLOW_HIDDEN_FEATURES = 1024
_C.MODEL.FLOW_DROPOUT = 0.2
_C.MODEL.FLOW_BATCH_NORM = False
_C.MODEL.FLOW_BASE_STD = 1.0
_C.MODEL.FLOW_SPLINE_NUM_BINS = 10
_C.MODEL.FLOW_SPLINE_TAIL_BOUND = 3.0
_C.MODEL.FLOW_SPLINE_TAILS = "linear"
_C.MODEL.SHAPE_PERTURB_SCALE = 0.0   # multiplier on per-dim GT std noise for shape (45D); 0 = disabled
_C.MODEL.SCALE_PERTURB_SCALE = 0.0   # multiplier on per-dim GT std noise for scale (10D selected); 0 = disabled
_C.MODEL.BETA_PERTURB_DETACH = True  # detach perturbed betas from stage-1 graph
_C.MODEL.BETA_PERTURB_STATS_PATH = "checkpoints/sam-3d-body-dinov3/shape_scale_std.pt"  # per-dim GT stds


_C.LOSS = CfgNode()
_C.LOSS.SHAPE_PARAM_WEIGHT = 0.0
_C.LOSS.SCALE_PARAM_WEIGHT = 0.0
_C.LOSS.POSE_PARAM_WEIGHT = 0.0
_C.LOSS.JOINTS_3D_WEIGHT = 0.0
_C.LOSS.JOINTS_2D_WEIGHT = 0.0
_C.LOSS.KP2D_WEIGHT = 500.0
_C.LOSS.KP3D_WEIGHT = 0.0
_C.LOSS.PARAM_NLL_WEIGHT = 0.5
_C.LOSS.PARAM_L2_WEIGHT = 0.0
# Diversity options
_C.LOSS.BYPASS_VISIBILITY = False
_C.LOSS.KP3D_ON_SAMPLES = True    # Default: True,  False: skip KP3D loss on NF samples entirely
_C.LOSS.KP2D_BEST_OF_N = False    # Default: False, True: penalise only the closest sample to GT (min-over-N)
_C.LOSS.ENTROPY_WEIGHT = 0.0      # Default: 0.0,   >0: add sample-variance entropy bonus (DEPRECATED: use KP3D_INVISIBLE_SPREAD_WEIGHT)
_C.LOSS.KP3D_INVISIBLE_SPREAD_WEIGHT = 0.0  # >0: maximise 3D keypoint spread over invisible joints only
_C.LOSS.KP3D_ALONG_RAY_WEIGHT = 0.0   # >0: reward sample spread along camera ray (depth diversity for visible joints)
_C.LOSS.KP3D_PERP_RAY_WEIGHT  = 0.0   # >0: penalise sample spread perpendicular to ray (explicit 2D consistency)
##########################################


# Dataset hparams
_C.DATASET = CfgNode()
_C.DATASET.BATCH_SIZE = 64
_C.DATASET.NUM_WORKERS = 32
_C.DATASET.NOISE_FACTOR = 0.4
_C.DATASET.SCALE_FACTOR = 0.25
_C.DATASET.CROP_PROB = 0.5
_C.DATASET.CROP_FACTOR = [0.2, 0.4]
_C.DATASET.EXTREME_CROP_PROB = 0.0
_C.DATASET.EXTREME_CROP_LEVEL = 1
_C.DATASET.PIN_MEMORY = True
_C.DATASET.SHUFFLE_TRAIN = True
_C.DATASET.TRAIN_DS = 'all'
_C.DATASET.VAL_DS = 'orbit-archviz-15-bbox44-smplx_static-hdri-bmi-bbox44-smplx'
_C.DATASET.MESH_COLOR = 'pinkish'
_C.DATASET.DATASETS_AND_RATIOS_FULL = 'static-hdri-bbox44-smplx_agora-body-bbox44-smplx_zoom-suburbd-bbox44-smplx_closeup-suburba-bbox44-smplx_closeup-suburbb-bbox44-smplx_closeup-suburbc-bbox44-smplx_closeup-suburbd-bbox44-smplx_closeup-gym-bbox44-smplx_zoom-gym-bbox44-smplx_static-gym-bbox44-smplx_static-office-bbox44-smplx_orbit-office-bbox44-smplx_static-hdri-zoomed-bbox44-smplx_pitchup-stadium-bbox44-smplx_pitchdown-stadium-bbox44-smplx_static-hdri-bmi-bbox44-smplx_closeup-suburbb-bmi-bbox44-smplx_closeup-suburbc-bmi-bbox44-smplx_static-suburbd-bmi-bbox44-smplx_zoom-gym-bmi-bbox44-smplx_static-office-hair-bbox44-smplx_zoom-suburbd-hair-bbox44-smplx_static-gym-hair-bbox44-smplx_orbit-archviz-15-bbox44-smplx_orbit-archviz-19-bbox44-smplx_orbit-archviz-12-bbox44-smplx_orbit-archviz-10-bbox44-smplx'
_C.DATASET.DATASETS_AND_RATIOS = 'static-hdri-bbox44-smplx_agora-body-bbox44-smplx_zoom-suburbd-bbox44-smplx_closeup-suburba-bbox44-smplx_closeup-suburbb-bbox44-smplx_closeup-suburbc-bbox44-smplx_closeup-suburbd-bbox44-smplx_closeup-gym-bbox44-smplx_zoom-gym-bbox44-smplx_static-gym-bbox44-smplx_static-office-bbox44-smplx_orbit-office-bbox44-smplx_static-hdri-zoomed-bbox44-smplx_pitchup-stadium-bbox44-smplx_pitchdown-stadium-bbox44-smplx_closeup-suburbb-bmi-bbox44-smplx_closeup-suburbc-bmi-bbox44-smplx_static-suburbd-bmi-bbox44-smplx_zoom-gym-bmi-bbox44-smplx_static-office-hair-bbox44-smplx_zoom-suburbd-hair-bbox44-smplx_static-gym-hair-bbox44-smplx_orbit-archviz-19-bbox44-smplx_orbit-archviz-12-bbox44-smplx_orbit-archviz-10-bbox44-smplx'

_C.DATASET.CROP_PERCENT = 0.8
_C.DATASET.MAX_SAMPLES_PER_DS = -1  # >0: cap each sub-dataset length for fast iteration
_C.DATASET.ALB = True
_C.DATASET.ALB_PROB = 0.5
_C.DATASET.proj_verts = False
_C.DATASET.FOCAL_LENGTH = 5000


_C.MODEL.IMAGE_MEAN = [0.485, 0.456, 0.406]
_C.MODEL.IMAGE_STD = [0.229, 0.224, 0.225]

_C.DATASET.IMAGE_SIZE = (512, 512)
_C.MODEL.IMAGE_SIZE = (512, 512)


_C.MODEL.BACKBONE = CfgNode()
_C.MODEL.BACKBONE.TYPE = "dinov3_vith16plus"
_C.MODEL.BACKBONE.PRETRAINED_WEIGHTS = ""
_C.MODEL.BACKBONE.FROZEN_STAGES = -1
_C.MODEL.BACKBONE.DROP_PATH_RATE = 0.1

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

_C.MODEL.DECODER.LORA_R = 64
_C.MODEL.DECODER.LORA_ALPHA = 128
_C.MODEL.DECODER.LORA_DROPOUT = 0.0
_C.MODEL.DECODER.LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "proj", "layers.1", "ffn.layers.0.0"]

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
_C.MODEL.PERSON_HEAD.POSE_TYPE = "mhr"
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

    # BEDLAM 1 (SMPLX labels in all_npz_12_training_mhr_conditioned)
    'agora-body-bbox44-smplx': os.path.join(PATH, 'data/training_images/images/'),
    'zoom-suburbd-bbox44-smplx': os.path.join(PATH, 'data/training_images/20221010_3-10_500_batch01hand_zoom_suburb_d_6fps/png'),
    'closeup-suburba-bbox44-smplx': os.path.join(PATH, 'data/training_images/20221011_1_250_batch01hand_closeup_suburb_a_6fps/png'),
    'closeup-suburbb-bbox44-smplx': os.path.join(PATH, 'data/training_images/20221011_1_250_batch01hand_closeup_suburb_b_6fps/png'),
    'closeup-suburbc-bbox44-smplx': os.path.join(PATH, 'data/training_images/20221011_1_250_batch01hand_closeup_suburb_c_6fps/png'),
    'closeup-suburbd-bbox44-smplx': os.path.join(PATH, 'data/training_images/20221011_1_250_batch01hand_closeup_suburb_d_6fps/png'),
    'closeup-gym-bbox44-smplx': os.path.join(PATH, 'data/training_images/20221012_1_500_batch01hand_closeup_highSchoolGym_6fps/png'),
    'zoom-gym-bbox44-smplx': os.path.join(PATH, 'data/training_images/20221012_3-10_500_batch01hand_zoom_highSchoolGym_6fps/png'),
    'static-gym-bbox44-smplx': os.path.join(PATH, 'data/training_images/20221013_3-10_500_batch01hand_static_highSchoolGym_6fps/png'),
    'static-office-bbox44-smplx': os.path.join(PATH, 'data/training_images/20221013_3_250_batch01hand_static_bigOffice_6fps/png'),
    'orbit-office-bbox44-smplx': os.path.join(PATH, 'data/training_images/20221013_3_250_batch01hand_orbit_bigOffice_6fps/png'),
    'orbit-archviz-15-bbox44-smplx': os.path.join(PATH, 'data/training_images/20221014_3_250_batch01hand_orbit_archVizUI3_time15_6fps/png'),
    'orbit-archviz-19-bbox44-smplx': os.path.join(PATH, 'data/training_images/20221015_3_250_batch01hand_orbit_archVizUI3_time19_6fps/png'),
    'orbit-archviz-12-bbox44-smplx': os.path.join(PATH, 'data/training_images/20221015_3_250_batch01hand_orbit_archVizUI3_time12_6fps/png'),
    'orbit-archviz-10-bbox44-smplx': os.path.join(PATH, 'data/training_images/20221015_3_250_batch01hand_orbit_archVizUI3_time10_6fps/png'),
    'static-hdri-bbox44-smplx': os.path.join(PATH, 'data/training_images/20221010_3_1000_batch01hand_6fps/png'),
    'static-hdri-zoomed-bbox44-smplx': os.path.join(PATH, 'data/training_images/20221017_3_1000_batch01hand_6fps/png'),
    'staticzoomed-suburba-frameocc-bbox44-smplx': os.path.join(PATH, 'data/training_images/20221017_1_250_batch01hand_closeup_suburb_a_6fps/png'),
    'zoom-suburbb-frameocc-bbox44-smplx': os.path.join(PATH, 'data/training_images/20221018_1_250_batch01hand_zoom_suburb_b_6fps/png'),
    'static-hdri-frameocc-bbox44-smplx': os.path.join(PATH, 'data/training_images/20221018_3-8_250_batch01hand_6fps/png'),
    'orbit-archviz-objocc-bbox44-smplx': os.path.join(PATH, 'data/training_images/20221018_3_250_batch01hand_orbit_archVizUI3_time15_6fps/png'),
    'pitchup-stadium-bbox44-smplx': os.path.join(PATH, 'data/training_images/20221018_3-8_250_batch01hand_pitchUp52_stadium_6fps/png'),
    'pitchdown-stadium-bbox44-smplx': os.path.join(PATH, 'data/training_images/20221018_3-8_250_batch01hand_pitchDown52_stadium_6fps/png'),
    'static-hdri-bmi-bbox44-smplx': os.path.join(PATH, 'data/training_images/20221019_3_250_highbmihand_6fps/png'),
    'closeup-suburbb-bmi-bbox44-smplx': os.path.join(PATH, 'data/training_images/20221019_1_250_highbmihand_closeup_suburb_b_6fps/png'),
    'closeup-suburbc-bmi-bbox44-smplx': os.path.join(PATH, 'data/training_images/20221019_1_250_highbmihand_closeup_suburb_c_6fps/png'),
    'static-stadium-bmi-bbox44-smplx': os.path.join(PATH, 'data/training_images/20221019_3-8_250_highbmihand_static_stadium_6fps/png'),
    'orbit-stadium-bmi-bbox44-smplx': os.path.join(PATH, 'data/training_images/20221019_3-8_250_highbmihand_orbit_stadium_6fps/png'),
    'static-suburbd-bmi-bbox44-smplx': os.path.join(PATH, 'data/training_images/20221019_3-8_1000_highbmihand_static_suburb_d_6fps/png'),
    'zoom-gym-bmi-bbox44-smplx': os.path.join(PATH, 'data/training_images/20221020-3-8_250_highbmihand_zoom_highSchoolGym_a_6fps/png'),
    'static-office-hair-bbox44-smplx': os.path.join(PATH, 'data/training_images/20221022_3_250_batch01handhair_static_bigOffice_30fps/png'),
    'zoom-suburbd-hair-bbox44-smplx': os.path.join(PATH, 'data/training_images/20221024_10_100_batch01handhair_zoom_suburb_d_30fps/png'),
    'static-gym-hair-bbox44-smplx': os.path.join(PATH, 'data/training_images/20221024_3-10_100_batch01handhair_static_highSchoolGym_30fps/png'),

    # BEDLAM 2
    'city-dolly-moyo1-smplx-notest': os.path.join(BEDLAM2_PATH, '20240425_1_171_citysample_dolly/png'),
    'yoga-orbit-moyo-smplx-notest': os.path.join(BEDLAM2_PATH, '20240416_1_171_yogastudio_orbit_timeofday/png'),
    'yoga-static-moyo-smplx-notest': os.path.join(BEDLAM2_PATH, '20240423_1_171_yogastudio_staticloc_timeofday/png'),
    'city-orbit-moyo1-smplx-notest': os.path.join(BEDLAM2_PATH, '20240424_1_171_citysample_orbit/png'),
    'hdri-moyo-smplx-notest': os.path.join(BEDLAM2_PATH, '20240425_1_171_hdri/png'),
    'city-orbit-moyo2-smplx-notest': os.path.join(BEDLAM2_PATH, '20240426_5_100_citysample_orbit/png'),
    'stadium-moyo-smplx-notest': os.path.join(BEDLAM2_PATH, '20240429_1_171_stadium/png'),
    'city-dolly-moyo2-smplx-notest': os.path.join(BEDLAM2_PATH, '20240502_5_200_citysample_dolly/png'),
    'hdri-moyo2-smplx-notest': os.path.join(BEDLAM2_PATH, '20240506_10_200_hdri/png'),
    'city-orbit-moyo3-smplx-notest': os.path.join(BEDLAM2_PATH, '20240506_5_200_citysample_orbit/png'),
    'city-dollyz-moyo-smplx-notest': os.path.join(BEDLAM2_PATH, '20240507_5_200_citysample_dollyz/png'),
    'city-tracking-b2v01-smplx-notest': os.path.join(BEDLAM2_PATH, '20240514_1_1001_citysample_tracking/png'),
    'city-tracking-b2v02-smplx-notest': os.path.join(BEDLAM2_PATH, '20240604_5_500_citysample_tracking/png'),
    'bus-tracking-b2v01-smplx-notest': os.path.join(BEDLAM2_PATH, '20240605_3_500_busstation_tracking/png'),
    'bus-orbit-b2v01-smplx-notest': os.path.join(BEDLAM2_PATH, '20240606_4_250_busstation_orbit/png'),
    'stadium-b2v01-smplx-notest': os.path.join(BEDLAM2_PATH, '20240606_1_500_stadium_closeup/png'),
    'archmodel-dolly-b2v01-smplx-notest': os.path.join(BEDLAM2_PATH, '20240611_5_250_archmodelsvol8_dolly/png'),
    'hdri-b2v01-smplx-notest': os.path.join(BEDLAM2_PATH, '20240613_1_200_hdri/png'),
    'citynight-tracking-b2v01-smplx-notest': os.path.join(BEDLAM2_PATH, '20240614_5_200_citysamplenight_tracking/png'),
    'hdri-b2v02-smplx-notest': os.path.join(BEDLAM2_PATH, '20240614_1_300_hdri/png'),
    'hdri-b2v03-smplx-notest': os.path.join(BEDLAM2_PATH, '20240617_10_500_hdri/png'),
    'ai0805-orbit-b2v01-smplx-notest': os.path.join(BEDLAM2_PATH, '20240618_1_500_ai0805_orbit/png'),
    'ai1004-orbit-b2v01-smplx-notest': os.path.join(BEDLAM2_PATH, '20240619_2_250_ai1004_orbit/png'),
    'ai1004-tracking-b2v01-smplx-notest': os.path.join(BEDLAM2_PATH, '20240619_1_250_ai1004_tracking/png'),
    'archmodel-dollyz-b2v01-smplx-notest': os.path.join(BEDLAM2_PATH, '20240620_5_250_archmodelsvol8_dollyz/png'),
    'hdri-b2v11-smplx-notest': os.path.join(BEDLAM2_PATH, '20240625_1_2337_hdri/png'),
    'ai1004-tracking-b2v11-smplx-notest': os.path.join(BEDLAM2_PATH, '20240628_1_250_ai1004_tracking/png'),
    'bus-tracking-b2v11-smplx-notest': os.path.join(BEDLAM2_PATH, '20240628_4_250_busstation_orbit/png'),
    'ai0901-lookat-b2v11-smplx-notest': os.path.join(BEDLAM2_PATH, '20240701_1_250_ai0901_lookat/png'),
    'ai0901-orbit-portrait-b2v11-smplx-notest': os.path.join(BEDLAM2_PATH, '20240703_1_250_ai0901_orbit_portrait/png'),
    'ai0901-static-portrait-b2v11-smplx-notest': os.path.join(BEDLAM2_PATH, '20240708_1_250_ai0901_static_portrait/png'),
    'archmodel-zoom-b2v11-smplx-notest': os.path.join(BEDLAM2_PATH, '20240709_5_250_archmodelsvol8_zoom/png'),
    'ai0805-orbit-portrait-b2v11-smplx-notest': os.path.join(BEDLAM2_PATH, '20240710_1_250_ai0805_orbit_portrait/png'),
    'bus-orbit-zoom-b2v11-smplx-notest': os.path.join(BEDLAM2_PATH, '20240711_5-10_250_busstation_orbit_zoom/png'),
    'ai0805-vcam-b2v11-smplx-notest': os.path.join(BEDLAM2_PATH, '20240725_1_250_ai0805_vcam/png'),
    'ai0805-vcam-b2v12-smplx-notest': os.path.join(BEDLAM2_PATH, '20240726_1_250_ai0805_vcam/png'),
    'ai1004-vcam-portrait-b2v11-smplx-notest': os.path.join(BEDLAM2_PATH, '20240729_1_250_ai1004_vcam/png'),
    'ai1101-vcam-portrait-b2v11-smplx-notest': os.path.join(BEDLAM2_PATH, '20240730_1_250_ai1101_vcam/png'),
    'hdri-b2v21-smplx-notest': os.path.join(BEDLAM2_PATH, '20240731_1_1827_hdri/png'),
    'bus-orbit-zoom-b2v21-smplx-notest': os.path.join(BEDLAM2_PATH, '20240805_5-10_250_busstation_orbit_zoom/png'),
    'ai1101-vcam-portrait-b2v21-smplx-notest': os.path.join(BEDLAM2_PATH, '20240806_1_250_ai1101_vcam/png'),
    'ai1105-vcam-b2v21-smplx-notest': os.path.join(BEDLAM2_PATH, '20240808_1_250_ai1105_vcam/png'),
    'ai1102-vcam-portrait-b2v21-smplx-notest': os.path.join(BEDLAM2_PATH, '20240809_1_250_ai1102_vcam/png'),
    'ai1004-tracking-b2v21-smplx-notest': os.path.join(BEDLAM2_PATH, '20240813_1_250_ai1004_tracking/png'),
    'bus-orbit-zoom-b2v22-smplx-notest': os.path.join(BEDLAM2_PATH, '20241001_5-10_250_busstation_orbit_zoom/png'),
    'archmodel-tracking-b2v02-smplx-notest': os.path.join(BEDLAM2_PATH, '20241107_1_250_archmodelsvol8_tracking/png'),
    'hdri-b2v30-smplx-notest': os.path.join(BEDLAM2_PATH, '20241114_1_4619_hdri/png'),
    'hdri-b2v40-smplx-notest': os.path.join(BEDLAM2_PATH, '20241204_1_2120_hdri/png'),
    'rome-dollyz-zoom-b2v40-smplx-notest': os.path.join(BEDLAM2_PATH, '20241210_5-10_250_rome_dollyz_zoom/png'),
    'rome-orbit-zoom-b2v40-smplx-notest': os.path.join(BEDLAM2_PATH, '20241211_5-10_250_rome_orbit_zoom/png'),
    'rome-dolly-zoom-b2v40-smplx-notest': os.path.join(BEDLAM2_PATH, '20241212_5-10_250_rome_dolly_zoom/png'),
    'rome-tracking-b2v40-smplx-notest': os.path.join(BEDLAM2_PATH, '20241213_1_250_rome_tracking/png'),
    'rome-vcam-portrait-b2v40-smplx-notest': os.path.join(BEDLAM2_PATH, '20241217_1_250_rome_vcam/png'),
    'chemicalplant-dollyz-zoom-b2v30-smplx-notest': os.path.join(BEDLAM2_PATH, '20241219_5_250_chemicalplant_dollyz_zoom/png'),
    'rome-vcam-portrait-b2v30-smplx-notest': os.path.join(BEDLAM2_PATH, '20250103_1_250_rome_vcam/png'),
    'chemicalplant-vcam-portrait-b2v30-smplx-notest': os.path.join(BEDLAM2_PATH, '20250110_1_250_chemicalplant_vcam/png'),
    'rome-vcam-b2v31-smplx-notest': os.path.join(BEDLAM2_PATH, '20250113_1_250_rome_vcam/png'),
    'chemicalplant-dolly-zoom-b2v30-smplx-notest': os.path.join(BEDLAM2_PATH, '20250114_4-5_250_chemicalplant_dolly_zoom/png'),
    'chemicalplant-vcamego-b2v30-smplx-notest': os.path.join(BEDLAM2_PATH, '20250123_1_250_chemicalplant_vcamego/png'),
    'ai1102-vcamego-b2v30-smplx-notest': os.path.join(BEDLAM2_PATH, '20250131_1_250_ai1102_vcamego/png'),
    'yakohama-vcamego-b2v30-smplx-notest': os.path.join(BEDLAM2_PATH, '20250206_4-7_250_yakohama_vcamego_approach/png'),
    'ai1105-upperbody-b2v30-smplx-notest': os.path.join(BEDLAM2_PATH, '20250211_1_250_ai1105_upperbody/png'),
    'yakohama-upperbody-b2v30-smplx-notest': os.path.join(BEDLAM2_PATH, '20250212_1_250_yakohama_upperbody/png'),
    'chemicalplant-upperbody-b2v30-smplx-notest': os.path.join(BEDLAM2_PATH, '20250214_1_250_chemicalplant_upperbody/png'),
    'middleeasy-upperbody-b2v30-smplx-notest': os.path.join(BEDLAM2_PATH, '20250218_2-3_250_middleeast_upperbody/png'),
    'middleeast-vacam-b2v40-smplx-notest': os.path.join(BEDLAM2_PATH, '20250219_3-4_250_middleeast_vcam_approach/png'),

    # Real-image training sets
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
        'orbit-stadium-bmi-bbox44-smplx': os.path.join(PATH, 'data/training_labels/all_npz_12_training_mhr_conditioned/20221019_3-8_250_highbmihand_orbit_stadium_6fps.npz'),
        'orbit-archviz-objocc-bbox44-smplx': os.path.join(PATH, 'data/training_labels/all_npz_12_training_mhr_conditioned/20221018_3_250_batch01hand_orbit_archVizUI3_time15_6fps.npz'),
        'zoom-suburbb-frameocc-bbox44-smplx': os.path.join(PATH, 'data/training_labels/all_npz_12_training_mhr_conditioned/20221018_1_250_batch01hand_zoom_suburb_b_6fps.npz'),
        'static-hdri-frameocc-bbox44-smplx': os.path.join(PATH, 'data/training_labels/all_npz_12_training_mhr_conditioned/20221018_3-8_250_batch01hand_6fps.npz'),
        'zoom-gym-bbox44-smplx': os.path.join(PATH, 'data/training_labels/all_npz_12_training_mhr_conditioned/20221012_3-10_500_batch01hand_zoom_highSchoolGym_6fps.npz'),
        'static-gym-bbox44-smplx': os.path.join(PATH, 'data/training_labels/all_npz_12_training_mhr_conditioned/20221013_3-10_500_batch01hand_static_highSchoolGym_6fps.npz'),
        'orbit-archviz-15-bbox44-smplx': os.path.join(PATH, 'data/training_labels/all_npz_12_training_mhr_conditioned/20221014_3_250_batch01hand_orbit_archVizUI3_time15_6fps.npz'),
        'static-hdri-bmi-bbox44-smplx': os.path.join(PATH, 'data/training_labels/all_npz_12_training_mhr_conditioned/20221019_3_250_highbmihand_6fps.npz'),
        'city-dolly-moyo1-smplx-notest': os.path.join(BEDLAM2_LABELS_DIR, '20240425_1_171_citysample_dolly.npz'),
    },
    {
        # BEDLAM 1 (SMPLX)
        'agora-body-bbox44-smplx': os.path.join(PATH, 'data/training_labels/all_npz_12_training_mhr_conditioned/agora-body.npz'),
        '3dpw-train-smplx': os.path.join(PATH, 'data/training_labels/3dpw_train_smplx.npz'),

        'zoom-suburbd-bbox44-smplx': os.path.join(PATH, 'data/training_labels/all_npz_12_training_mhr_conditioned/20221010_3-10_500_batch01hand_zoom_suburb_d_6fps.npz'),
        'closeup-suburba-bbox44-smplx': os.path.join(PATH, 'data/training_labels/all_npz_12_training_mhr_conditioned/20221011_1_250_batch01hand_closeup_suburb_a_6fps.npz'),
        'closeup-suburbb-bbox44-smplx': os.path.join(PATH, 'data/training_labels/all_npz_12_training_mhr_conditioned/20221011_1_250_batch01hand_closeup_suburb_b_6fps.npz'),
        'closeup-suburbc-bbox44-smplx': os.path.join(PATH, 'data/training_labels/all_npz_12_training_mhr_conditioned/20221011_1_250_batch01hand_closeup_suburb_c_6fps.npz'),
        'closeup-suburbd-bbox44-smplx': os.path.join(PATH, 'data/training_labels/all_npz_12_training_mhr_conditioned/20221011_1_250_batch01hand_closeup_suburb_d_6fps.npz'),
        'closeup-gym-bbox44-smplx': os.path.join(PATH, 'data/training_labels/all_npz_12_training_mhr_conditioned/20221012_1_500_batch01hand_closeup_highSchoolGym_6fps.npz'),
        'zoom-gym-bbox44-smplx': os.path.join(PATH, 'data/training_labels/all_npz_12_training_mhr_conditioned/20221012_3-10_500_batch01hand_zoom_highSchoolGym_6fps.npz'),
        'static-gym-bbox44-smplx': os.path.join(PATH, 'data/training_labels/all_npz_12_training_mhr_conditioned/20221013_3-10_500_batch01hand_static_highSchoolGym_6fps.npz'),
        'static-office-bbox44-smplx': os.path.join(PATH, 'data/training_labels/all_npz_12_training_mhr_conditioned/20221013_3_250_batch01hand_static_bigOffice_6fps.npz'),
        'orbit-office-bbox44-smplx': os.path.join(PATH, 'data/training_labels/all_npz_12_training_mhr_conditioned/20221013_3_250_batch01hand_orbit_bigOffice_6fps.npz'),
        'orbit-archviz-15-bbox44-smplx': os.path.join(PATH, 'data/training_labels/all_npz_12_training_mhr_conditioned/20221014_3_250_batch01hand_orbit_archVizUI3_time15_6fps.npz'),
        'orbit-archviz-19-bbox44-smplx': os.path.join(PATH, 'data/training_labels/all_npz_12_training_mhr_conditioned/20221015_3_250_batch01hand_orbit_archVizUI3_time19_6fps.npz'),
        'orbit-archviz-12-bbox44-smplx': os.path.join(PATH, 'data/training_labels/all_npz_12_training_mhr_conditioned/20221015_3_250_batch01hand_orbit_archVizUI3_time12_6fps.npz'),
        'orbit-archviz-10-bbox44-smplx': os.path.join(PATH, 'data/training_labels/all_npz_12_training_mhr_conditioned/20221015_3_250_batch01hand_orbit_archVizUI3_time10_6fps.npz'),
        'static-hdri-bbox44-smplx': os.path.join(PATH, 'data/training_labels/all_npz_12_training_mhr_conditioned/20221010_3_1000_batch01hand_6fps.npz'),
        'static-hdri-zoomed-bbox44-smplx': os.path.join(PATH, 'data/training_labels/all_npz_12_training_mhr_conditioned/20221017_3_1000_batch01hand_6fps.npz'),
        'staticzoomed-suburba-frameocc-bbox44-smplx': os.path.join(PATH, 'data/training_labels/all_npz_12_training_mhr_conditioned/20221017_1_250_batch01hand_closeup_suburb_a_6fps.npz'),
        'pitchup-stadium-bbox44-smplx': os.path.join(PATH, 'data/training_labels/all_npz_12_training_mhr_conditioned/20221018_3-8_250_batch01hand_pitchUp52_stadium_6fps.npz'),
        'static-hdri-bmi-bbox44-smplx': os.path.join(PATH, 'data/training_labels/all_npz_12_training_mhr_conditioned/20221019_3_250_highbmihand_6fps.npz'),
        'closeup-suburbb-bmi-bbox44-smplx': os.path.join(PATH, 'data/training_labels/all_npz_12_training_mhr_conditioned/20221019_1_250_highbmihand_closeup_suburb_b_6fps.npz'),
        'closeup-suburbc-bmi-bbox44-smplx': os.path.join(PATH, 'data/training_labels/all_npz_12_training_mhr_conditioned/20221019_1_250_highbmihand_closeup_suburb_c_6fps.npz'),
        'static-suburbd-bmi-bbox44-smplx': os.path.join(PATH, 'data/training_labels/all_npz_12_training_mhr_conditioned/20221019_3-8_1000_highbmihand_static_suburb_d_6fps.npz'),
        'zoom-gym-bmi-bbox44-smplx': os.path.join(PATH, 'data/training_labels/all_npz_12_training_mhr_conditioned/20221020-3-8_250_highbmihand_zoom_highSchoolGym_a_6fps.npz'),
        'pitchdown-stadium-bbox44-smplx': os.path.join(PATH, 'data/training_labels/all_npz_12_training_mhr_conditioned/20221018_3-8_250_batch01hand_pitchDown52_stadium_6fps.npz'),
        'static-office-hair-bbox44-smplx': os.path.join(PATH, 'data/training_labels/all_npz_12_training_mhr_conditioned/20221022_3_250_batch01handhair_static_bigOffice_30fps.npz'),
        'zoom-suburbd-hair-bbox44-smplx': os.path.join(PATH, 'data/training_labels/all_npz_12_training_mhr_conditioned/20221024_10_100_batch01handhair_zoom_suburb_d_30fps.npz'),
        'static-gym-hair-bbox44-smplx': os.path.join(PATH, 'data/training_labels/all_npz_12_training_mhr_conditioned/20221024_3-10_100_batch01handhair_static_highSchoolGym_30fps.npz'),
        'orbit-stadium-bmi-bbox44-smplx': os.path.join(PATH, 'data/training_labels/all_npz_12_training_mhr_conditioned/20221019_3-8_250_highbmihand_orbit_stadium_6fps.npz'),

        # BEDLAM 2
        'city-dolly-moyo1-smplx-notest': os.path.join(BEDLAM2_LABELS_DIR, '20240425_1_171_citysample_dolly.npz'),
        'yoga-orbit-moyo-smplx-notest': os.path.join(BEDLAM2_LABELS_DIR, '20240416_1_171_yogastudio_orbit_timeofday.npz'),
        'yoga-static-moyo-smplx-notest': os.path.join(BEDLAM2_LABELS_DIR, '20240423_1_171_yogastudio_staticloc_timeofday.npz'),
        'city-orbit-moyo1-smplx-notest': os.path.join(BEDLAM2_LABELS_DIR, '20240424_1_171_citysample_orbit.npz'),
        'hdri-moyo-smplx-notest': os.path.join(BEDLAM2_LABELS_DIR, '20240425_1_171_hdri.npz'),
        'city-orbit-moyo2-smplx-notest': os.path.join(BEDLAM2_LABELS_DIR, '20240426_5_100_citysample_orbit.npz'),
        'stadium-moyo-smplx-notest': os.path.join(BEDLAM2_LABELS_DIR, '20240429_1_171_stadium.npz'),
        'city-dolly-moyo2-smplx-notest': os.path.join(BEDLAM2_LABELS_DIR, '20240502_5_200_citysample_dolly.npz'),
        'hdri-moyo2-smplx-notest': os.path.join(BEDLAM2_LABELS_DIR, '20240506_10_200_hdri.npz'),
        'city-orbit-moyo3-smplx-notest': os.path.join(BEDLAM2_LABELS_DIR, '20240506_5_200_citysample_orbit.npz'),
        'city-dollyz-moyo-smplx-notest': os.path.join(BEDLAM2_LABELS_DIR, '20240507_5_200_citysample_dollyz.npz'),
        'city-tracking-b2v01-smplx-notest': os.path.join(BEDLAM2_LABELS_DIR, '20240514_1_1001_citysample_tracking.npz'),
        'city-tracking-b2v02-smplx-notest': os.path.join(BEDLAM2_LABELS_DIR, '20240604_5_500_citysample_tracking.npz'),
        'bus-tracking-b2v01-smplx-notest': os.path.join(BEDLAM2_LABELS_DIR, '20240605_3_500_busstation_tracking.npz'),
        'bus-orbit-b2v01-smplx-notest': os.path.join(BEDLAM2_LABELS_DIR, '20240606_4_250_busstation_orbit.npz'),
        'stadium-b2v01-smplx-notest': os.path.join(BEDLAM2_LABELS_DIR, '20240606_1_500_stadium_closeup.npz'),
        'archmodel-dolly-b2v01-smplx-notest': os.path.join(BEDLAM2_LABELS_DIR, '20240611_5_250_archmodelsvol8_dolly.npz'),
        'hdri-b2v01-smplx-notest': os.path.join(BEDLAM2_LABELS_DIR, '20240613_1_200_hdri.npz'),
        'citynight-tracking-b2v01-smplx-notest': os.path.join(BEDLAM2_LABELS_DIR, '20240614_5_200_citysamplenight_tracking.npz'),
        'hdri-b2v02-smplx-notest': os.path.join(BEDLAM2_LABELS_DIR, '20240614_1_300_hdri.npz'),
        'hdri-b2v03-smplx-notest': os.path.join(BEDLAM2_LABELS_DIR, '20240617_10_500_hdri.npz'),
        'ai0805-orbit-b2v01-smplx-notest': os.path.join(BEDLAM2_LABELS_DIR, '20240618_1_500_ai0805_orbit.npz'),
        'ai1004-orbit-b2v01-smplx-notest': os.path.join(BEDLAM2_LABELS_DIR, '20240619_2_250_ai1004_orbit.npz'),
        'ai1004-tracking-b2v01-smplx-notest': os.path.join(BEDLAM2_LABELS_DIR, '20240619_1_250_ai1004_tracking.npz'),
        'archmodel-dollyz-b2v01-smplx-notest': os.path.join(BEDLAM2_LABELS_DIR, '20240620_5_250_archmodelsvol8_dollyz.npz'),
        'hdri-b2v11-smplx-notest': os.path.join(BEDLAM2_LABELS_DIR, '20240625_1_2337_hdri.npz'),
        'ai1004-tracking-b2v11-smplx-notest': os.path.join(BEDLAM2_LABELS_DIR, '20240628_1_250_ai1004_tracking.npz'),
        'bus-tracking-b2v11-smplx-notest': os.path.join(BEDLAM2_LABELS_DIR, '20240628_4_250_busstation_orbit.npz'),
        'ai0901-lookat-b2v11-smplx-notest': os.path.join(BEDLAM2_LABELS_DIR, '20240701_1_250_ai0901_lookat.npz'),
        'ai0901-orbit-portrait-b2v11-smplx-notest': os.path.join(BEDLAM2_LABELS_DIR, '20240703_1_250_ai0901_orbit_portrait.npz'),
        'ai0901-static-portrait-b2v11-smplx-notest': os.path.join(BEDLAM2_LABELS_DIR, '20240708_1_250_ai0901_static_portrait.npz'),
        'archmodel-zoom-b2v11-smplx-notest': os.path.join(BEDLAM2_LABELS_DIR, '20240709_5_250_archmodelsvol8_zoom.npz'),
        'ai0805-orbit-portrait-b2v11-smplx-notest': os.path.join(BEDLAM2_LABELS_DIR, '20240710_1_250_ai0805_orbit_portrait.npz'),
        'bus-orbit-zoom-b2v11-smplx-notest': os.path.join(BEDLAM2_LABELS_DIR, '20240711_5-10_250_busstation_orbit_zoom.npz'),
        'ai0805-vcam-b2v11-smplx-notest': os.path.join(BEDLAM2_LABELS_DIR, '20240725_1_250_ai0805_vcam.npz'),
        'ai0805-vcam-b2v12-smplx-notest': os.path.join(BEDLAM2_LABELS_DIR, '20240726_1_250_ai0805_vcam.npz'),
        'ai1004-vcam-portrait-b2v11-smplx-notest': os.path.join(BEDLAM2_LABELS_DIR, '20240729_1_250_ai1004_vcam.npz'),
        'ai1101-vcam-portrait-b2v11-smplx-notest': os.path.join(BEDLAM2_LABELS_DIR, '20240730_1_250_ai1101_vcam.npz'),
        'hdri-b2v21-smplx-notest': os.path.join(BEDLAM2_LABELS_DIR, '20240731_1_1827_hdri.npz'),
        'bus-orbit-zoom-b2v21-smplx-notest': os.path.join(BEDLAM2_LABELS_DIR, '20240805_5-10_250_busstation_orbit_zoom.npz'),
        'ai1101-vcam-portrait-b2v21-smplx-notest': os.path.join(BEDLAM2_LABELS_DIR, '20240806_1_250_ai1101_vcam.npz'),
        'ai1105-vcam-b2v21-smplx-notest': os.path.join(BEDLAM2_LABELS_DIR, '20240808_1_250_ai1105_vcam.npz'),
        'ai1102-vcam-portrait-b2v21-smplx-notest': os.path.join(BEDLAM2_LABELS_DIR, '20240809_1_250_ai1102_vcam.npz'),
        'ai1004-tracking-b2v21-smplx-notest': os.path.join(BEDLAM2_LABELS_DIR, '20240813_1_250_ai1004_tracking.npz'),
        'bus-orbit-zoom-b2v22-smplx-notest': os.path.join(BEDLAM2_LABELS_DIR, '20241001_5-10_250_busstation_orbit_zoom.npz'),
        'archmodel-tracking-b2v02-smplx-notest': os.path.join(BEDLAM2_LABELS_DIR, '20241107_1_250_archmodelsvol8_tracking.npz'),
        'hdri-b2v30-smplx-notest': os.path.join(BEDLAM2_LABELS_DIR, '20241114_1_4619_hdri.npz'),
        'hdri-b2v40-smplx-notest': os.path.join(BEDLAM2_LABELS_DIR, '20241204_1_2120_hdri.npz'),
        'rome-dollyz-zoom-b2v40-smplx-notest': os.path.join(BEDLAM2_LABELS_DIR, '20241210_5-10_250_rome_dollyz_zoom.npz'),
        'rome-orbit-zoom-b2v40-smplx-notest': os.path.join(BEDLAM2_LABELS_DIR, '20241211_5-10_250_rome_orbit_zoom.npz'),
        'rome-dolly-zoom-b2v40-smplx-notest': os.path.join(BEDLAM2_LABELS_DIR, '20241212_5-10_250_rome_dolly_zoom.npz'),
        'rome-tracking-b2v40-smplx-notest': os.path.join(BEDLAM2_LABELS_DIR, '20241213_1_250_rome_tracking.npz'),
        'rome-vcam-portrait-b2v40-smplx-notest': os.path.join(BEDLAM2_LABELS_DIR, '20241217_1_250_rome_vcam.npz'),
        'chemicalplant-dollyz-zoom-b2v30-smplx-notest': os.path.join(BEDLAM2_LABELS_DIR, '20241219_5_250_chemicalplant_dollyz_zoom.npz'),
        'rome-vcam-portrait-b2v30-smplx-notest': os.path.join(BEDLAM2_LABELS_DIR, '20250103_1_250_rome_vcam.npz'),
        'chemicalplant-vcam-portrait-b2v30-smplx-notest': os.path.join(BEDLAM2_LABELS_DIR, '20250110_1_250_chemicalplant_vcam.npz'),
        'rome-vcam-b2v31-smplx-notest': os.path.join(BEDLAM2_LABELS_DIR, '20250113_1_250_rome_vcam.npz'),
        'chemicalplant-dolly-zoom-b2v30-smplx-notest': os.path.join(BEDLAM2_LABELS_DIR, '20250114_4-5_250_chemicalplant_dolly_zoom.npz'),
        'chemicalplant-vcamego-b2v30-smplx-notest': os.path.join(BEDLAM2_LABELS_DIR, '20250123_1_250_chemicalplant_vcamego.npz'),
        'ai1102-vcamego-b2v30-smplx-notest': os.path.join(BEDLAM2_LABELS_DIR, '20250131_1_250_ai1102_vcamego.npz'),
        'yakohama-vcamego-b2v30-smplx-notest': os.path.join(BEDLAM2_LABELS_DIR, '20250206_4-7_250_yakohama_vcamego_approach.npz'),
        'ai1105-upperbody-b2v30-smplx-notest': os.path.join(BEDLAM2_LABELS_DIR, '20250211_1_250_ai1105_upperbody.npz'),
        'yakohama-upperbody-b2v30-smplx-notest': os.path.join(BEDLAM2_LABELS_DIR, '20250212_1_250_yakohama_upperbody.npz'),
        'chemicalplant-upperbody-b2v30-smplx-notest': os.path.join(BEDLAM2_LABELS_DIR, '20250214_1_250_chemicalplant_upperbody.npz'),
        'middleeasy-upperbody-b2v30-smplx-notest': os.path.join(BEDLAM2_LABELS_DIR, '20250218_2-3_250_middleeast_upperbody.npz'),
        'middleeast-vacam-b2v40-smplx-notest': os.path.join(BEDLAM2_LABELS_DIR, '20250219_3-4_250_middleeast_vcam_approach.npz'),

        # Real-image training sets
        'coco': os.path.join(PATH, 'data/real_training_labels/coco.npz'),
        'mpii': os.path.join(PATH, 'data/real_training_labels/mpii.npz'),
        'h36m': os.path.join(PATH, 'data/real_training_labels/h36m_train.npz'),
        'mpi-inf-3dhp': os.path.join(PATH, 'data/real_training_labels/mpi_inf_3dhp_train.npz'),
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
