# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Sequence, Tuple, Union
from functools import partial

import torch
import torch.nn as nn

from mmengine.model import BaseModel
from mmengine.model.weight_init import trunc_normal_

# from mmpretrain.models import MixTransformer #!DEBUG
# from mmseg.models import MixVisionTransformerCVP
from .mix_transformer_cvp import MixVisionTransformerCVP
from mmseg.models.builder import BACKBONES

# from mmpretrain.registry import MODELS
# from mmpretrain.structures import DataSample
# from .base import BaseSelfSupervisor


@BACKBONES.register_module()
class CVPSimMIMMixVisionTransformer(MixVisionTransformerCVP):
    def __init__(self, **cfg):
        super(CVPSimMIMMixVisionTransformer, self).__init__(**cfg)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dims[0]))

    def forward(self, x: torch.Tensor,
                mask: Optional[torch.Tensor]) -> Sequence[torch.Tensor]:

        if mask is None:
            return super().forward(x)

        else:
            B = x.shape[0]
            outs = []

            if self.handcrafted_tune:
                handcrafted1, handcrafted2, handcrafted3, handcrafted4 = self.prompt_generator.init_handcrafted(x)
            else:
                handcrafted1, handcrafted2, handcrafted3, handcrafted4 = None, None, None, None
            if self.conv_tune:
                conv1, conv2, conv3, conv4 = self.prompt_generator.spm(x)
            else:
                conv1, conv2, conv3, conv4 = None, None, None, None

            # stage 1
            x, H, W = self.patch_embed1(x)

            # MIM
            B, L, _ = x.shape
            mask_token = self.mask_token.expand(B, L, -1)
            w = mask.flatten(1).unsqueeze(-1).type_as(mask_token)
            x = x * (1. - w) + mask_token * w

            if '1' in self.tuning_stage:
                prompt1 = self.prompt_generator.init_prompt(x, handcrafted1, conv1, block_num=1)
                # prompt1 : (handcrated_feature, embedding_feature, conv_feature)
            for i, blk in enumerate(self.block1):
                if '1' in self.tuning_stage:
                    x = self.prompt_generator.get_prompt(x, prompt1, block_num=1, depth_num=i)
                x = blk(x, H, W)
            x = self.norm1(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(x)

            # stage 2
            x, H, W = self.patch_embed2(x)
            if '2' in self.tuning_stage:
                prompt2 = self.prompt_generator.init_prompt(x, handcrafted2, conv2, block_num=2)
            for i, blk in enumerate(self.block2):
                if '2' in self.tuning_stage:
                    x = self.prompt_generator.get_prompt(x, prompt2, 2, i)
                x = blk(x, H, W)
            x = self.norm2(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(x)

            # stage 3
            x, H, W = self.patch_embed3(x)
            if '3' in self.tuning_stage:
                prompt3 = self.prompt_generator.init_prompt(x, handcrafted3, conv3, block_num=3)
            for i, blk in enumerate(self.block3):
                if '3' in self.tuning_stage:
                    x = self.prompt_generator.get_prompt(x,prompt3, 3, i)
                x = blk(x, H, W)
            x = self.norm3(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(x)

            # stage 4
            x, H, W = self.patch_embed4(x)
            if '4' in self.tuning_stage:
                prompt4 = self.prompt_generator.init_prompt(x, handcrafted4, conv4, block_num=4)
            for i, blk in enumerate(self.block4):
                if '4' in self.tuning_stage:
                    x = self.prompt_generator.get_prompt(x, prompt4, 4, i)
                x = blk(x, H, W)
            x = self.norm4(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(x)

            return outs

            # # stage 1
            # # x, hw_shape = self.patch_embed(x)
            # x, H, W = self.patch_embed1(x)
            # B, L, _ = x.shape

            # mask_token = self.mask_token.expand(B, L, -1)
            # w = mask.flatten(1).unsqueeze(-1).type_as(mask_token)
            # x = x * (1. - w) + mask_token * w

            # if self.use_abs_pos_embed:
            #     x = x + self.absolute_pos_embed

            # x = self.drop_after_pos(x)

            # outs = []
            # for i, stage in enumerate(self.stages):
            #     x, hw_shape = stage(x, hw_shape)
            #     if i in self.out_indices:
            #         norm_layer = getattr(self, f'norm{i}')
            #         out = norm_layer(x)
            #         out = out.view(-1, *hw_shape,
            #                        stage.out_channels).permute(0, 3, 1,
            #                                                    2).contiguous()
            #         outs.append(out)

            # return tuple(outs)


@BACKBONES.register_module()
class mit_b1_cvp_simmim(CVPSimMIMMixVisionTransformer):
    def __init__(self, **kwargs):
        super(mit_b1_cvp_simmim, self).__init__(
            patch_size=4,
            embed_dims=[64, 128, 320, 512],
            num_heads=[1, 2, 5, 8],
            mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=[2, 2, 2, 2],
            sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0,
            drop_path_rate=0.1,
            **kwargs)


@BACKBONES.register_module()
class mit_b4_cvp_simmim(CVPSimMIMMixVisionTransformer):
    def __init__(self, **kwargs):
        super(mit_b4_cvp_simmim, self).__init__(
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
class mit_b5_cvp_simmim(CVPSimMIMMixVisionTransformer):
    def __init__(self, **kwargs):
        super(mit_b5_cvp_simmim, self).__init__(
            patch_size=4,
            embed_dims=[64, 128, 320, 512],
            num_heads=[1, 2, 5, 8],
            mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=[3, 6, 40, 3],
            sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0,
            drop_path_rate=0.1,
            **kwargs)


