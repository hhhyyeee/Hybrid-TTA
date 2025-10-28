# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Sequence, Tuple, Union
from functools import partial

import torch
import torch.nn as nn

from mmengine.model import BaseModel
from mmengine.model.weight_init import trunc_normal_

from .mix_transformer import MixVisionTransformer
from mmseg.models.builder import BACKBONES


@BACKBONES.register_module()
class SimMIMMixVisionTransformer(MixVisionTransformer):
    def __init__(self, **cfg):
        super(SimMIMMixVisionTransformer, self).__init__(**cfg)
        self.embed_dims = cfg['embed_dims']
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dims[0]))

    def forward(self, x: torch.Tensor,
                mask: Optional[torch.Tensor]) -> Sequence[torch.Tensor]:
        if mask is None:
            return super().forward(x)

        else:
            B = x.shape[0]
            outs = []

            # stage 1
            x, H, W = self.patch_embed1(x)

            # MIM
            B, L, _ = x.shape
            mask_token = self.mask_token.expand(B, L, -1)
            w = mask.flatten(1).unsqueeze(-1).type_as(mask_token)
            x = x * (1. - w) + mask_token * w

            for i, blk in enumerate(self.block1):
                x = blk(x, H, W)
            x = self.norm1(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(x)

            # stage 2
            x, H, W = self.patch_embed2(x)
            for i, blk in enumerate(self.block2):
                x = blk(x, H, W)
            x = self.norm2(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(x)

            # stage 3
            x, H, W = self.patch_embed3(x)
            for i, blk in enumerate(self.block3):
                x = blk(x, H, W)
            x = self.norm3(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(x)

            # stage 4
            x, H, W = self.patch_embed4(x)
            for i, blk in enumerate(self.block4):
                x = blk(x, H, W)
            x = self.norm4(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(x)

            return outs


@BACKBONES.register_module()
class mit_b1_simmim(SimMIMMixVisionTransformer):
    def __init__(self, **kwargs):
        super(mit_b1_simmim, self).__init__(
            patch_size=4,
            embed_dims=[64, 128, 320, 512],
            num_heads=[1, 2, 5, 8],
            mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=[2, 2, 2, 2],
            sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0,
            drop_path_rate=0.1)

@BACKBONES.register_module()
class mit_b4_simmim(SimMIMMixVisionTransformer):
    def __init__(self, **kwargs):
        super(mit_b4_simmim, self).__init__(
            patch_size=4,
            embed_dims=[64, 128, 320, 512],
            num_heads=[1, 2, 5, 8],
            mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=[3, 8, 27, 3],
            sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0,
            drop_path_rate=0.1,
            **kwargs)

@BACKBONES.register_module()
class mit_b5_simmim(SimMIMMixVisionTransformer):
    def __init__(self, **kwargs):
        super(mit_b5_simmim, self).__init__(
            patch_size=4,
            embed_dims=[64, 128, 320, 512],
            num_heads=[1, 2, 5, 8],
            mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=[3, 6, 40, 3],
            sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0,
            drop_path_rate=0.1)


