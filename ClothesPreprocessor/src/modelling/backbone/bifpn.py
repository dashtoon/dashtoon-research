from typing import Callable, List, Union

import timm
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from detectron2.layers import (
    BatchNorm2d,
    FrozenBatchNorm2d,
    NaiveSyncBatchNorm,
    ShapeSpec,
    get_norm,
)
from detectron2.modeling import BACKBONE_REGISTRY, Backbone
from detectron2.modeling.backbone.fpn import LastLevelMaxPool
from timm.models._efficientnet_builder import BN_EPS_TF_DEFAULT, BN_MOMENTUM_TF_DEFAULT

__all__ = ["BiFPN", "build_timm_backbone"]


def get_world_size() -> int:
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


class DepthwiseSeparableConv2d(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
    ):
        dephtwise_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=False,
        )
        pointwise_conv = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=1,
            bias=bias,
        )
        super().__init__(dephtwise_conv, pointwise_conv)


class Conv3x3BnReLU(nn.Sequential):
    def __init__(self, in_channels, stride=1):
        conv = DepthwiseSeparableConv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            bias=False,
            padding=1,
            stride=stride,
        )
        if get_world_size() > 1:
            bn = nn.SyncBatchNorm(in_channels, momentum=0.03)
        else:
            bn = nn.BatchNorm2d(in_channels, momentum=0.03)
        relu = nn.ReLU(inplace=True)
        super().__init__(conv, bn, relu)


class FastNormalizedFusion(nn.Module):
    def __init__(self, in_nodes):
        super().__init__()
        self.in_nodes = in_nodes
        self.weight = nn.Parameter(torch.ones(in_nodes, dtype=torch.float32))
        self.register_buffer("eps", torch.tensor(0.0001))

    def forward(self, x: List[torch.Tensor]):
        if len(x) != self.in_nodes:
            raise RuntimeError("Expected to have {} input nodes, but have {}.".format(self.in_nodes, len(x)))

        # where wi ≥ 0 is ensured by applying a relu after each wi (paper)
        weight = F.relu(self.weight)
        x_sum = 0
        for xi, wi in zip(x, weight):
            x_sum = x_sum + xi * wi
        normalized_weighted_x = x_sum / (weight.sum() + self.eps)
        return normalized_weighted_x


class BiFPN(Backbone):
    """
    This module implements Feature Pyramid Network.
    It creates pyramid features built on top of some input feature maps.
    """

    def __init__(self, bottom_up, out_channels, top_block=None):
        super().__init__()

        self.bottom_up = bottom_up
        self.top_block = top_block

        self.l5 = nn.Conv2d(bottom_up.feature_info[4]["num_chs"], out_channels, kernel_size=1)
        self.l4 = nn.Conv2d(bottom_up.feature_info[3]["num_chs"], out_channels, kernel_size=1)
        self.l3 = nn.Conv2d(bottom_up.feature_info[2]["num_chs"], out_channels, kernel_size=1)
        self.l2 = nn.Conv2d(bottom_up.feature_info[1]["num_chs"], out_channels, kernel_size=1)

        self.p4_tr = Conv3x3BnReLU(out_channels)
        self.p3_tr = Conv3x3BnReLU(out_channels)

        self.up = nn.Upsample(scale_factor=2, mode="nearest")

        self.fuse_p4_tr = FastNormalizedFusion(in_nodes=2)
        self.fuse_p3_tr = FastNormalizedFusion(in_nodes=2)

        self.down_p2 = Conv3x3BnReLU(out_channels, stride=2)
        self.down_p3 = Conv3x3BnReLU(out_channels, stride=2)
        self.down_p4 = Conv3x3BnReLU(out_channels, stride=2)

        self.fuse_p5_out = FastNormalizedFusion(in_nodes=2)
        self.fuse_p4_out = FastNormalizedFusion(in_nodes=3)
        self.fuse_p3_out = FastNormalizedFusion(in_nodes=3)
        self.fuse_p2_out = FastNormalizedFusion(in_nodes=2)

        self.p5_out = Conv3x3BnReLU(out_channels)
        self.p4_out = Conv3x3BnReLU(out_channels)
        self.p3_out = Conv3x3BnReLU(out_channels)
        self.p2_out = Conv3x3BnReLU(out_channels)

        self._out_features = ["p2", "p3", "p4", "p5", "p6"]
        self._out_feature_channels = {k: out_channels for k in self._out_features}
        self._size_divisibility = 32
        self._out_feature_strides = {}
        for k, name in enumerate(self._out_features):
            self._out_feature_strides[name] = 2 ** (k + 2)

    @property
    def size_divisibility(self):
        return self._size_divisibility

    def forward(self, x):
        p2, p3, p4, p5 = self.bottom_up(x)

        if self.training:
            _dummy = sum(x.view(-1)[0] for x in self.bottom_up.parameters()) * 0.0
            p5 = p5 + _dummy

        p5 = self.l5(p5)
        p4 = self.l4(p4)
        p3 = self.l3(p3)
        p2 = self.l2(p2)

        p4_tr = self.p4_tr(self.fuse_p4_tr([p4, self.up(p5)]))
        p3_tr = self.p3_tr(self.fuse_p3_tr([p3, self.up(p4_tr)]))

        p2_out = self.p2_out(self.fuse_p2_out([p2, self.up(p3_tr)]))
        p3_out = self.p3_out(self.fuse_p3_out([p3, p3_tr, self.down_p2(p2_out)]))
        p4_out = self.p4_out(self.fuse_p4_out([p4, p4_tr, self.down_p3(p3_out)]))
        p5_out = self.p5_out(self.fuse_p5_out([p5, self.down_p4(p4_out)]))

        return {"p2": p2_out, "p3": p3_out, "p4": p4_out, "p5": p5_out, "p6": self.top_block(p5_out)[0]}

    def output_shape(self):
        return {
            name: ShapeSpec(channels=self._out_feature_channels[name], stride=self._out_feature_strides[name])
            for name in self._out_features
        }


@BACKBONE_REGISTRY.register()
def build_timm_backbone(cfg, input_shape):
    """
    Create a TimmNet instance from config.

    Returns:
        TimmNet: a :class:`TimmNet` instance.
    """
    norm = cfg.MODEL.TIMMNETS.NORM
    out_features = cfg.MODEL.TIMMNETS.OUT_FEATURES
    model_name = cfg.MODEL.TIMMNETS.NAME
    pretrained = cfg.MODEL.TIMMNETS.PRETRAINED
    scriptable = cfg.MODEL.TIMMNETS.SCRIPTABLE
    exportable = cfg.MODEL.TIMMNETS.EXPORTABLE

    # GET MODEL BY NAME
    model = timm.create_model(
        model_name,
        pretrained,
        features_only=True,
        out_indices=out_features,
        scriptable=scriptable,
        exportable=exportable,
        feature_location="expansion",
    )

    # LOAD MODEL AND CONVERT NORM
    # NOTE: why I use if/else: see the strange function _load_from_state_dict in FrozenBatchNorm2d
    assert norm in ["FrozenBN", "SyncBN", "BN"]
    if norm == "FrozenBN":
        model = FrozenBatchNorm2d.convert_frozen_batchnorm(model)
    elif pretrained:
        model = convert_norm_to_detectron2_format(model, norm)
    else:
        model = convert_norm_to_detectron2_format(model, norm, init_default=True)

    # USE TENSORFLOW EPS, MOMENTUM defaults if model is tf pretrained
    if "tf" in model_name:
        model = convert_norm_eps_momentum_to_tf_defaults(model)

    # FREEZE FIRST 2 LAYERS
    max_block_number = int(model.feature_info[1]["module"][7:8])
    # max_block_number = int(model.feature_info[1]['name'][7:8])
    print(f"Freezing stem and first {max_block_number + 1} backbone blocks")
    for p in model.conv_stem.parameters():
        p.requires_grad = False
    model.bn1 = FrozenBatchNorm2d.convert_frozen_batchnorm(model.bn1)
    for block_number in range(0, max_block_number + 1):
        for p in model.blocks[block_number].parameters():
            p.requires_grad = False
        model.blocks[block_number] = FrozenBatchNorm2d.convert_frozen_batchnorm(model.blocks[block_number])

    return model


@BACKBONE_REGISTRY.register()
def build_timm_bifpn_backbone(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode

    Returns:
        modeling (Backbone): modeling module, must be a subclass of :class:`Backbone`.
    """
    bottom_up = build_timm_backbone(cfg, input_shape)
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    backbone = BiFPN(
        bottom_up=bottom_up,
        out_channels=out_channels,
        top_block=LastLevelMaxPool(),
    )
    return backbone


def convert_norm_to_detectron2_format(module, norm: Union[str, Callable], init_default: bool = False):
    module_output = module
    if isinstance(module, torch.nn.BatchNorm2d):
        module_output = get_norm(norm, out_channels=module.num_features)
        if init_default:
            module_output.weight.data.fill_(1.0)
            module_output.bias.data.zero_()
        else:
            module_output.load_state_dict(module.state_dict())
    for name, child in module.named_children():
        new_child = convert_norm_to_detectron2_format(child, norm, init_default)
        if new_child is not child:
            module_output.add_module(name, new_child)
    return module_output


def convert_norm_eps_momentum_to_tf_defaults(module):
    module_output = module
    if isinstance(module, (nn.BatchNorm2d, BatchNorm2d, NaiveSyncBatchNorm, nn.SyncBatchNorm)):
        module_output.momentum = BN_MOMENTUM_TF_DEFAULT
        module_output.eps = BN_EPS_TF_DEFAULT
    elif isinstance(module, FrozenBatchNorm2d):
        module_output.eps = BN_EPS_TF_DEFAULT
    for name, child in module.named_children():
        new_child = convert_norm_eps_momentum_to_tf_defaults(child)
        module_output.add_module(name, new_child)
    del module
    return module_output
