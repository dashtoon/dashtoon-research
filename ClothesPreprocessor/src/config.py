from densepose.config import add_densepose_config
from detectron2.config import CfgNode as CN


def add_deeplab_config(cfg):
    """
    Add config for DeepLab.
    """
    # We retry random cropping until no single category in semantic segmentation GT occupies more
    # than `SINGLE_CATEGORY_MAX_AREA` part of the crop.
    cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA = 1.0
    # Used for `poly` learning rate schedule.
    cfg.SOLVER.POLY_LR_POWER = 0.9
    cfg.SOLVER.POLY_LR_CONSTANT_ENDING = 0.0
    # Loss type, choose from `cross_entropy`, `hard_pixel_mining`.
    cfg.MODEL.SEM_SEG_HEAD.LOSS_TYPE = "hard_pixel_mining"
    # DeepLab settings
    cfg.MODEL.SEM_SEG_HEAD.PROJECT_FEATURES = ["res2"]
    cfg.MODEL.SEM_SEG_HEAD.PROJECT_CHANNELS = [48]
    cfg.MODEL.SEM_SEG_HEAD.ASPP_CHANNELS = 256
    cfg.MODEL.SEM_SEG_HEAD.ASPP_DILATIONS = [6, 12, 18]
    cfg.MODEL.SEM_SEG_HEAD.ASPP_DROPOUT = 0.1
    cfg.MODEL.SEM_SEG_HEAD.USE_DEPTHWISE_SEPARABLE_CONV = False
    cfg.MODEL.SEM_SEG_HEAD.DROPOUT_RATIO = 0.1
    cfg.MODEL.SEM_SEG_HEAD.IN_CHANNELS = [256, 256, 256, 256]
    # Backbone new configs
    cfg.MODEL.RESNETS.RES4_DILATION = 1
    cfg.MODEL.RESNETS.RES5_MULTI_GRID = [1, 2, 4]
    # ResNet stem type from: `basic`, `deeplab`
    cfg.MODEL.RESNETS.STEM_TYPE = "deeplab"


def add_custom_config(cfg: CN):
    _C = cfg

    add_densepose_config(_C)
    add_deeplab_config(_C)

    _C.MODEL.TIMMNETS = CN()
    _C.MODEL.TIMMNETS.NAME = "efficientnetv2_rw_s"
    _C.MODEL.TIMMNETS.OUT_FEATURES = [2, 3, 4]
    _C.MODEL.TIMMNETS.PRETRAINED = True
    _C.MODEL.TIMMNETS.BACKBONE_ARGS = CN()
    _C.MODEL.TIMMNETS.BACKBONE_ARGS.drop_path_rate = 0.2
    _C.MODEL.TIMMNETS.NORM = "FrozenBN"

    _C.MODEL.BIFPN = CN()
    _C.MODEL.BIFPN.NUM_LEVELS = 5
    _C.MODEL.BIFPN.NUM_BIFPN = 6
    _C.MODEL.BIFPN.NORM = "BN"
    _C.MODEL.BIFPN.OUT_CHANNELS = 256
    _C.MODEL.BIFPN.SEPARABLE_CONV = False

    _C.MODEL.ROI_DENSEPOSE_HEAD.DECODER_ISSIMPLE = False

    # for segmentation
    _C.MODEL.SEM_SEG_ON = True

    _C.INPUT.CUSTOM_AUG = ""
    _C.INPUT.TRAIN_SIZE = 640
    _C.INPUT.TEST_SIZE = 640
    _C.INPUT.SCALE_RANGE = (0.1, 2.0)
    # 'default' for fixed short/ long edge, 'square' for max size=INPUT.SIZE
    _C.INPUT.TEST_INPUT_TYPE = "default"
