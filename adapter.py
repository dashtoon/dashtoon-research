# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from enum import Enum, unique
from collections import OrderedDict

from typing import List, Optional, Dict, Union

import torch
import torch.nn as nn

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.resnet import Downsample2D


class MultiAdapter(ModelMixin):
    r"""
    MultiAdapter is a wrapper model that contains multiple adapter models and merges their outputs according to
    user-assigned weighting.

    This model inherits from [`ModelMixin`]. Check the superclass documentation for the generic methods the library
    implements for all the model (such as downloading or saving, etc.)

    Parameters:
        adapters (`List[T2IAdapter]`, *optional*, defaults to None):
            A list of `T2IAdapter` model instances.
    """

    def __init__(self, adapters: List["T2IAdapter"]):
        super(MultiAdapter, self).__init__()

        self.num_adapter = len(adapters)
        self.adapters = nn.ModuleList(adapters)

    def forward(
        self, xs: torch.Tensor, adapter_weights: Optional[List[float]] = None
    ) -> List[torch.Tensor]:
        r"""
        Args:
            xs (`torch.Tensor`):
                (batch, channel, height, width) input images for multiple adapter models concated along dimension 1,
                `channel` should equal to `num_adapter` * "number of channel of image".
            adapter_weights (`List[float]`, *optional*, defaults to None):
                List of floats representing the weight which will be multiply to each adapter's output before adding
                them together.
        """
        if adapter_weights is None:
            adapter_weights = torch.tensor([1 / self.num_adapter] * self.num_adapter, device=self.device)
        else:
            adapter_weights = torch.tensor(adapter_weights, device=self.device)

        if xs.shape[1] % self.num_adapter != 0:
            raise ValueError(
                f"Expecting multi-adapter's input have number of channel that cab be evenly divisible "
                f"by num_adapter: {xs.shape[1]} % {self.num_adapter} != 0"
            )
        x_list = torch.chunk(xs, self.num_adapter, dim=1)
        accume_state = None
        for x, w, adapter in zip(x_list, adapter_weights, self.adapters):
            features = adapter(x)
            if accume_state is None:
                accume_state = features
            else:
                for i in range(len(features)):
                    # accume_state[i] += w * features[i]
                    accume_state[i] = accume_state[i] + w * features[i]
        return accume_state


class T2IAdapter(ModelMixin, ConfigMixin):
    r"""
    A simple ResNet-like model that accepts images containing control signals such as keyposes and depth. The model
    generates multiple feature maps that are used as additional conditioning in [`UNet2DConditionModel`]. The model's
    architecture follows the original implementation of
    [Adapter](https://github.com/TencentARC/T2I-Adapter/blob/686de4681515662c0ac2ffa07bf5dda83af1038a/ldm/modules/encoders/adapter.py#L97)
     and
     [AdapterLight](https://github.com/TencentARC/T2I-Adapter/blob/686de4681515662c0ac2ffa07bf5dda83af1038a/ldm/modules/encoders/adapter.py#L235).

    This model inherits from [`ModelMixin`]. Check the superclass documentation for the generic methods the library
    implements for all the model (such as downloading or saving, etc.)

    Parameters:
        in_channels (`int`, *optional*, defaults to 3):
            Number of channels of Aapter's input(*control image*). Set this parameter to 1 if you're using gray scale
            image as *control image*.
        channels (`List[int]`, *optional*, defaults to `(320, 640, 1280, 1280)`):
            The number of channel of each downsample block's output hidden state. The `len(block_out_channels)` will
            also determine the number of downsample blocks in the Adapter.
        num_res_blocks (`int`, *optional*, defaults to 2):
            Number of ResNet blocks in each downsample block
    """

    @register_to_config
    def __init__(
        self,
        in_channels: int = 3,
        channels: List[int] = [320, 640, 1280, 1280],
        num_res_blocks: int = 2,
        downscale_factor: int = 8,
        adapter_type: str = "full_adapter",
    ):
        super().__init__()

        if adapter_type == "full_adapter":
            self.adapter = FullAdapterXL(
                in_channels, channels, num_res_blocks, downscale_factor
            )
        elif adapter_type == "light_adapter":
            self.adapter = LightAdapterXL(
                in_channels, channels, num_res_blocks, downscale_factor
            )
        else:
            raise ValueError(
                f"unknown adapter_type: {type}. Choose either 'full_adapter' or 'simple_adapter'"
            )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        return self.adapter(x)

    @property
    def total_downscale_factor(self):
        return self.adapter.total_downscale_factor


# full adapter


class FullAdapterXL(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        channels: List[int] = [320, 640, 1280],
        num_res_blocks: int = 2,
        downscale_factor: int = 8,
        downs: List[bool] = [True, False, True],
    ):
        super().__init__()

        in_channels = in_channels * downscale_factor**2

        self.unshuffle = nn.PixelUnshuffle(downscale_factor)
        self.conv_in = nn.Conv2d(in_channels, channels[0], kernel_size=3, padding=1)
        self.downs = downs
        self.body = nn.ModuleList(
            [
                AdapterBlock(
                    channels[0], channels[0], num_res_blocks, down=self.downs[0]
                ),
                AdapterBlock(
                    channels[0], channels[1], num_res_blocks, down=self.downs[1]
                ),
                AdapterBlock(
                    channels[1], channels[2], num_res_blocks, down=self.downs[2]
                ),
            ]
        )

        self.total_downscale_factor = (
            downscale_factor * 2 * (len([d for d in self.downs if d]) + 1)
        )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.unshuffle(x)
        x = self.conv_in(x)

        features = []

        for block in self.body:
            x = block(x)
            features.append(x)

        return features


class AdapterBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_res_blocks, down=False):
        super().__init__()

        self.downsample = None
        if down:
            self.downsample = Downsample2D(in_channels)

        self.in_conv = None
        if in_channels != out_channels:
            self.in_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        self.resnets = nn.Sequential(
            *[AdapterResnetBlock(out_channels) for _ in range(num_res_blocks)],
        )

    def forward(self, x):
        if self.downsample is not None:
            x = self.downsample(x)

        if self.in_conv is not None:
            x = self.in_conv(x)

        x = self.resnets(x)

        return x


class AdapterResnetBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.act = nn.ReLU()
        self.block2 = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        h = x
        h = self.block1(h)
        h = self.act(h)
        h = self.block2(h)

        return h + x


# light adapter


class LightAdapterXL(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        channels: List[int] = [320, 640, 1280],
        num_res_blocks: int = 2,
        downscale_factor: int = 8,
        downs: List[bool] = [True, False, True],
    ):
        super().__init__()

        in_channels = in_channels * downscale_factor**2

        self.unshuffle = nn.PixelUnshuffle(downscale_factor)

        self.downs = downs

        self.body = nn.ModuleList(
            [
                LightAdapterBlock(
                    in_channels, channels[0], num_res_blocks, down=downs[0]
                ),
                LightAdapterBlock(
                    channels[0], channels[1], num_res_blocks, down=downs[1]
                ),
                LightAdapterBlock(
                    channels[1], channels[2], num_res_blocks, down=downs[2]
                ),
            ]
        )

        self.total_downscale_factor = (
            downscale_factor * 2 * (len([d for d in self.downs if d]) + 1)
        )

    def forward(self, x):
        x = self.unshuffle(x)

        features = []

        for block in self.body:
            x = block(x)
            features.append(x)

        return features


class LightAdapterBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_res_blocks, down=False):
        super().__init__()
        mid_channels = out_channels // 4

        self.downsample = None
        if down:
            self.downsample = Downsample2D(in_channels)

        self.in_conv = nn.Conv2d(in_channels, mid_channels, kernel_size=1)
        self.resnets = nn.Sequential(
            *[LightAdapterResnetBlock(mid_channels) for _ in range(num_res_blocks)]
        )
        self.out_conv = nn.Conv2d(mid_channels, out_channels, kernel_size=1)

    def forward(self, x):
        if self.downsample is not None:
            x = self.downsample(x)
        x = self.in_conv(x)
        x = self.resnets(x)
        x = self.out_conv(x)
        return x


class LightAdapterResnetBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.act = nn.ReLU()
        self.block2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        h = x
        h = self.block1(h)
        h = self.act(h)
        h = self.block2(h)

        return h + x


# ----------------------------------------------------------------------------
#                              addding coadapter classes
# ----------------------------------------------------------------------------

@unique
class ExtraCondition(Enum):
    '''
    from original T2I adatper
    
    a = ExtraCondition.sketch
    print(a.value) --> will give 0
    
    Attributes
    ----------
    sketch : int
        Represents a sketch condition.
    keypose : int
        Represents a keypose condition.
    seg : int
        Represents a segmentation condition.
    depth : int
        Represents a depth condition.
    canny : int
        Represents a canny condition.
    style : int
        Represents a style condition.
    color : int
        Represents a color condition.
    openpose : int
        Represents an openpose condition.

   
    '''    
    sketch = 0
    keypose = 1
    seg = 2
    depth = 3
    canny = 4
    style = 5
    color = 6
    openpose = 7



class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)




class QuickGELU(nn.Module):

    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class ResidualAttentionBlock(nn.Module):

    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict([("c_fc", nn.Linear(d_model, d_model * 4)), ("gelu", QuickGELU()),
                         ("c_proj", nn.Linear(d_model * 4, d_model))]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module
    

class CoAdapterFuser(nn.Module):
    # def __init__(self, unet_channels=[320, 640, 1280, 1280], width=768, num_head=8, n_layes=3):
    def __init__(self, unet_channels=[320, 640, 1280], width=768, num_head=8, n_layes=3):
        super(CoAdapterFuser, self).__init__()
        scale = width ** 0.5
        # 16, maybe large enough for the number of adapters?
        self.task_embedding = nn.Parameter(scale * torch.randn(16, width))
        self.positional_embedding = nn.Parameter(scale * torch.randn(len(unet_channels), width))
        self.spatial_feat_mapping = nn.ModuleList()
        for ch in unet_channels:
            self.spatial_feat_mapping.append(nn.Sequential(
                nn.SiLU(),
                nn.Linear(ch, width),
            ))
        self.transformer_layes = nn.Sequential(*[ResidualAttentionBlock(width, num_head) for _ in range(n_layes)])
        self.ln_post = LayerNorm(width)
        self.ln_pre = LayerNorm(width)
        self.spatial_ch_projs = nn.ModuleList()
        for ch in unet_channels:
            self.spatial_ch_projs.append(zero_module(nn.Linear(width, ch)))
        self.seq_proj = nn.Parameter(torch.zeros(width, width))

    def forward(self, features : Dict[str, torch.Tensor]):
        '''
        the key should be one of the ExtraCondition enum that is :
        sketch, keypose, seg, depth, canny, style, color, openpose
        
        and the value should be a list of 3 feature maps , each feature map is a 4D tensor of shape [N, C, H, W]

        Parameters
        ----------
        features : Dict[str, torch.Tensor]
            combined features from adapters of different types, key shows the adpter name

        Returns
        -------
        Tuple
            List[torch.Tensor], None
        '''        
        if len(features) == 0:
            return None, None
        inputs = []
        for cond_name in features.keys():
            # a = ExtraCondition.sketch
            # a.value --> 0
            task_idx = getattr(ExtraCondition, cond_name).value # 👈 this will be integer
            if not isinstance(features[cond_name], list):
                inputs.append(features[cond_name] + self.task_embedding[task_idx])
                continue

            feat_seq = []
            for idx, feature_map in enumerate(features[cond_name]):
                feature_vec = torch.mean(feature_map, dim=(2, 3))
                feature_vec = self.spatial_feat_mapping[idx](feature_vec)
                feat_seq.append(feature_vec)
            feat_seq = torch.stack(feat_seq, dim=1)  # Nx4xC
            feat_seq = feat_seq + self.task_embedding[task_idx]
            feat_seq = feat_seq + self.positional_embedding
            inputs.append(feat_seq)

        x = torch.cat(inputs, dim=1)  # NxLxC
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer_layes(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_post(x)

        ret_feat_map = None
        ret_feat_seq = None
        cur_seq_idx = 0
        for cond_name in features.keys():
            if not isinstance(features[cond_name], list):
                length = features[cond_name].size(1)
                transformed_feature = features[cond_name] * ((x[:, cur_seq_idx:cur_seq_idx+length] @ self.seq_proj) + 1)
                if ret_feat_seq is None:
                    ret_feat_seq = transformed_feature
                else:
                    ret_feat_seq = torch.cat([ret_feat_seq, transformed_feature], dim=1)
                cur_seq_idx += length
                continue

            length = len(features[cond_name])
            transformed_feature_list = []
            for idx in range(length):
                alpha = self.spatial_ch_projs[idx](x[:, cur_seq_idx+idx])
                alpha = alpha.unsqueeze(-1).unsqueeze(-1) + 1
                transformed_feature_list.append(features[cond_name][idx] * alpha)
            if ret_feat_map is None:
                ret_feat_map = transformed_feature_list
            else:
                ret_feat_map = list(map(lambda x, y: x + y, ret_feat_map, transformed_feature_list))
            cur_seq_idx += length

        assert cur_seq_idx == x.size(1)

        # return ret_feat_map, ret_feat_seq
        return ret_feat_map

