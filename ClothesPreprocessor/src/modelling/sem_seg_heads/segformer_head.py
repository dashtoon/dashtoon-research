from typing import Callable, Dict, Optional, Tuple, Union

import fvcore.nn.weight_init as weight_init
import numpy as np
import torch
from detectron2.config import configurable
from detectron2.layers import Conv2d, ShapeSpec, get_norm
from detectron2.modeling import SEM_SEG_HEADS_REGISTRY
from detectron2.structures import ImageList
from detectron2.utils.registry import Registry
from torch import nn
from torch.nn import functional as F

from .deeplab import resize


@SEM_SEG_HEADS_REGISTRY.register()
class SegformerHead(nn.Module):
    """The all mlp Head of segformer.

    This head is the implementation of
    `Segformer <https://arxiv.org/abs/2105.15203>` _.
    """

    @configurable
    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        *,
        in_features: Tuple[int],
        num_classes: int,
        conv_dims: int,
        common_stride: int,
        loss_weight: float = 1.0,
        norm: Optional[Union[str, Callable]] = None,
        ignore_value: int = -1,
        dropout_ratio: float = 0.0,
    ):
        """
        Args:
            input_shape: shapes (channels and stride) of the input features
            num_classes: number of classes to predict
            conv_dims: number of output channels for the intermediate conv layers.
            common_stride: the common stride that all features will be upscaled to
            loss_weight: loss weight
            norm (str or callable): normalization for all conv layers
            ignore_value: category id to be ignored during training.
        """

        super().__init__()
        input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)
        if not len(input_shape):
            raise ValueError("SegformerHead(input_shape=) cannot be empty!")
        self.in_features = [k for k, v in input_shape]
        feature_channels = [v.channels for k, v in input_shape]

        self.ignore_value = ignore_value
        self.common_stride = common_stride
        self.loss_weight = loss_weight

        self.convs = []

        for i in range(len(feature_channels)):
            norm_module = get_norm(norm, conv_dims)
            conv = Conv2d(
                feature_channels[i],
                conv_dims,
                kernel_size=1,
                stride=1,
                bias=not norm,
                norm=norm_module,
                activation=nn.SiLU(inplace=True),
            )
            weight_init.c2_msra_fill(conv)
            self.convs.append(conv)
        self.convs = nn.ModuleList(self.convs)

        norm_module = get_norm(norm, conv_dims)
        self.fusion_conv = Conv2d(
            conv_dims * len(self.in_features),
            conv_dims,
            kernel_size=1,
            bias=not norm,
            norm=norm_module,
            activation=F.relu,
        )
        weight_init.c2_msra_fill(self.fusion_conv)

        self.conv_seg = nn.Conv2d(conv_dims, num_classes, kernel_size=1, bias=True)
        weight_init.c2_msra_fill(self.conv_seg)

        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        return {
            "input_shape": {k: v for k, v in input_shape.items() if k in cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES},
            "ignore_value": cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
            "num_classes": cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
            "conv_dims": cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM,
            "common_stride": cfg.MODEL.SEM_SEG_HEAD.COMMON_STRIDE,
            "norm": cfg.MODEL.SEM_SEG_HEAD.NORM,
            "loss_weight": cfg.MODEL.SEM_SEG_HEAD.LOSS_WEIGHT,
            "dropout_ratio": cfg.MODEL.SEM_SEG_HEAD.DROPOUT_RATIO,
            "in_features": cfg.MODEL.SEM_SEG_HEAD.IN_CHANNELS,
        }

    def cls_seg(self, feat):
        """Classify each pixel."""
        if self.dropout is not None:
            feat = self.dropout(feat)
        output = self.conv_seg(feat)
        return output

    def forward(self, features, targets=None):
        """
        Returns:
            In training, returns (None, dict of losses)
            In inference, returns (CxHxW logits, {})
        """
        outs = []
        features = list(features.values())
        for idx, feat in enumerate(features):
            x = feat
            conv = self.convs[idx]
            x = conv(x)
            x = resize(x, size=features[0].shape[-2:], mode="bilinear", align_corners=False)
            outs.append(x)
        out = self.fusion_conv(torch.cat(outs, dim=1))
        out = self.cls_seg(out)

        if self.training:
            return None, self.losses(out, targets)
        else:
            out = F.interpolate(out, scale_factor=self.common_stride, mode="bilinear", align_corners=False)
            return out, {}

    def losses(self, predictions, targets):
        predictions = predictions.float()  # https://github.com/pytorch/pytorch/issues/48163
        predictions = resize(predictions, size=targets.shape[-2:], mode="bilinear", align_corners=False)
        # ic(torch.unique(targets))
        # ic(predictions.shape, targets.shape)
        loss = nn.CrossEntropyLoss(reduction="mean", ignore_index=self.ignore_value)(predictions, targets)
        losses = {"loss_sem_seg": loss * self.loss_weight}
        return losses
