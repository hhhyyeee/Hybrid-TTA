"""Vision Transformer Adapter for Dense Predictions
"""

import logging
from functools import partial

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
# from mmseg.models.pet.ms_deform_attn import MSDeformAttn
from timm.models.layers import DropPath

# from mmcv.cnn.bricks.transformer import build_attention
# from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttention

_logger = logging.getLogger(__name__)


def get_reference_points(spatial_shapes, device):
    reference_points_list = []
    for lvl, (H_, W_) in enumerate(spatial_shapes):
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
            torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
        ref_y = ref_y.reshape(-1)[None] / H_
        ref_x = ref_x.reshape(-1)[None] / W_
        ref = torch.stack((ref_x, ref_y), -1)
        reference_points_list.append(ref)
    reference_points = torch.cat(reference_points_list, 1)
    reference_points = reference_points[:, :, None]
    return reference_points


def deform_inputs(x):
    bs, c, h, w = x.shape
    spatial_shapes = torch.as_tensor([(h // 8, w // 8),
                                      (h // 16, w // 16),
                                      (h // 32, w // 32)],
                                     dtype=torch.long, device=x.device)
    level_start_index = torch.cat((spatial_shapes.new_zeros(
        (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
    reference_points = get_reference_points([(h // 16, w // 16)], x.device)
    deform_inputs1 = [reference_points, spatial_shapes, level_start_index]
    
    spatial_shapes = torch.as_tensor([(h // 16, w // 16)], dtype=torch.long, device=x.device)
    level_start_index = torch.cat((spatial_shapes.new_zeros(
        (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
    reference_points = get_reference_points([(h // 8, w // 8),
                                             (h // 16, w // 16),
                                             (h // 32, w // 32)], x.device)
    deform_inputs2 = [reference_points, spatial_shapes, level_start_index]
    
    return deform_inputs1, deform_inputs2


class ConvFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        n = N // 21
        x1 = x[:, 0:16 * n, :].transpose(1, 2).view(B, C, H * 2, W * 2).contiguous()
        x2 = x[:, 16 * n:20 * n, :].transpose(1, 2).view(B, C, H, W).contiguous()
        x3 = x[:, 20 * n:, :].transpose(1, 2).view(B, C, H // 2, W // 2).contiguous()
        x1 = self.dwconv(x1).flatten(2).transpose(1, 2)
        x2 = self.dwconv(x2).flatten(2).transpose(1, 2)
        x3 = self.dwconv(x3).flatten(2).transpose(1, 2)
        x = torch.cat([x1, x2, x3], dim=1)
        return x


# class Extractor(nn.Module):
#     def __init__(self, dim, num_heads=6, n_points=4, n_levels=1, deform_ratio=1.0,
#                  with_cffn=True, cffn_ratio=0.25, drop=0., drop_path=0.,
#                  norm_layer=partial(nn.LayerNorm, eps=1e-6), with_cp=False):
#         super().__init__()
#         self.query_norm = norm_layer(dim)
#         self.feat_norm = norm_layer(dim)
#         self.attn = MSDeformAttn(d_model=dim, n_levels=n_levels, n_heads=num_heads,
#                                  n_points=n_points, ratio=deform_ratio)
#         self.with_cffn = with_cffn
#         self.with_cp = with_cp
#         if with_cffn:
#             self.ffn = ConvFFN(in_features=dim, hidden_features=int(dim * cffn_ratio), drop=drop)
#             self.ffn_norm = norm_layer(dim)
#             self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

#     def forward(self, query, reference_points, feat, spatial_shapes, level_start_index, H, W):
        
#         def _inner_forward(query, feat):

#             attn = self.attn(self.query_norm(query), reference_points,
#                              self.feat_norm(feat), spatial_shapes,
#                              level_start_index, None)
#             query = query + attn
    
#             if self.with_cffn:
#                 query = query + self.drop_path(self.ffn(self.ffn_norm(query), H, W))
#             return query
        
#         if self.with_cp and query.requires_grad:
#             query = cp.checkpoint(_inner_forward, query, feat)
#         else:
#             query = _inner_forward(query, feat)
            
#         return query


class Injector(nn.Module):
    """240130 ver.
    """
    def __init__(self, dim, num_heads=6, n_points=4, n_levels=4, dropout=0.1, deform_ratio=1.0,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), init_values=0., with_cp=False):
        super().__init__()
        self.with_cp = with_cp
        self.query_norm = norm_layer(dim)
        self.feat_norm = norm_layer(dim)
        self.attn = MultiScaleDeformableAttention(embed_dims=dim,
                                                  num_heads=num_heads,
                                                  num_levels=n_levels,
                                                  num_points=n_points,
                                                  dropout=dropout,
                                                  batch_first=True)
        self.gamma = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)

    def forward(self, query, reference_points, feat, spatial_shapes, level_start_index):
        def _inner_forward(query, feat):
            attn = self.attn(self.query_norm(query),
                             self.feat_norm(feat),
                             self.feat_norm(feat),
                             reference_points=reference_points,
                             spatial_shapes=spatial_shapes,
                             level_start_index=level_start_index)
            return query + self.gamma * attn
        
        if self.with_cp and query.requires_grad:
            query = cp.checkpoint(_inner_forward, query, feat)
        else:
            query = _inner_forward(query, feat)

        return query


# class InteractionBlock(nn.Module):
#     def __init__(self, dim, num_heads=6, n_points=4, norm_layer=partial(nn.LayerNorm, eps=1e-6),
#                  drop=0., drop_path=0., with_cffn=True, cffn_ratio=0.25, init_values=0.,
#                  deform_ratio=1.0, extra_extractor=False, with_cp=False):
#         super().__init__()

#         self.injector = Injector(dim=dim, n_levels=3, num_heads=num_heads, init_values=init_values,
#                                  n_points=n_points, norm_layer=norm_layer, deform_ratio=deform_ratio,
#                                  with_cp=with_cp)
#         self.extractor = Extractor(dim=dim, n_levels=1, num_heads=num_heads, n_points=n_points,
#                                    norm_layer=norm_layer, deform_ratio=deform_ratio, with_cffn=with_cffn,
#                                    cffn_ratio=cffn_ratio, drop=drop, drop_path=drop_path, with_cp=with_cp)
#         if extra_extractor:
#             self.extra_extractors = nn.Sequential(*[
#                 Extractor(dim=dim, num_heads=num_heads, n_points=n_points, norm_layer=norm_layer,
#                           with_cffn=with_cffn, cffn_ratio=cffn_ratio, deform_ratio=deform_ratio,
#                           drop=drop, drop_path=drop_path, with_cp=with_cp)
#                 for _ in range(2)
#             ])
#         else:
#             self.extra_extractors = None

#     def forward(self, x, c, blocks, deform_inputs1, deform_inputs2, H, W):
#         x = self.injector(query=x, reference_points=deform_inputs1[0],
#                           feat=c, spatial_shapes=deform_inputs1[1],
#                           level_start_index=deform_inputs1[2])
#         for idx, blk in enumerate(blocks):
#             x = blk(x, H, W)
#         c = self.extractor(query=c, reference_points=deform_inputs2[0],
#                            feat=x, spatial_shapes=deform_inputs2[1],
#                            level_start_index=deform_inputs2[2], H=H, W=W)
#         if self.extra_extractors is not None:
#             for extractor in self.extra_extractors:
#                 c = extractor(query=c, reference_points=deform_inputs2[0],
#                               feat=x, spatial_shapes=deform_inputs2[1],
#                               level_start_index=deform_inputs2[2], H=H, W=W)
#         return x, c


# class InteractionBlockWithCls(nn.Module):
#     def __init__(self, dim, num_heads=6, n_points=4, norm_layer=partial(nn.LayerNorm, eps=1e-6),
#                  drop=0., drop_path=0., with_cffn=True, cffn_ratio=0.25, init_values=0.,
#                  deform_ratio=1.0, extra_extractor=False, with_cp=False):
#         super().__init__()

#         self.injector = Injector(dim=dim, n_levels=3, num_heads=num_heads, init_values=init_values,
#                                  n_points=n_points, norm_layer=norm_layer, deform_ratio=deform_ratio,
#                                  with_cp=with_cp)
#         self.extractor = Extractor(dim=dim, n_levels=1, num_heads=num_heads, n_points=n_points,
#                                    norm_layer=norm_layer, deform_ratio=deform_ratio, with_cffn=with_cffn,
#                                    cffn_ratio=cffn_ratio, drop=drop, drop_path=drop_path, with_cp=with_cp)
#         if extra_extractor:
#             self.extra_extractors = nn.Sequential(*[
#                 Extractor(dim=dim, num_heads=num_heads, n_points=n_points, norm_layer=norm_layer,
#                           with_cffn=with_cffn, cffn_ratio=cffn_ratio, deform_ratio=deform_ratio,
#                           drop=drop, drop_path=drop_path, with_cp=with_cp)
#                 for _ in range(2)
#             ])
#         else:
#             self.extra_extractors = None

#     def forward(self, x, c, cls, blocks, deform_inputs1, deform_inputs2, H, W):
#         x = self.injector(query=x, reference_points=deform_inputs1[0],
#                           feat=c, spatial_shapes=deform_inputs1[1],
#                           level_start_index=deform_inputs1[2])
#         x = torch.cat((cls, x), dim=1)
#         for idx, blk in enumerate(blocks):
#             x = blk(x, H, W)
#         cls, x = x[:, :1, ], x[:, 1:, ]
#         c = self.extractor(query=c, reference_points=deform_inputs2[0],
#                            feat=x, spatial_shapes=deform_inputs2[1],
#                            level_start_index=deform_inputs2[2], H=H, W=W)
#         if self.extra_extractors is not None:
#             for extractor in self.extra_extractors:
#                 c = extractor(query=c, reference_points=deform_inputs2[0],
#                               feat=x, spatial_shapes=deform_inputs2[1],
#                               level_start_index=deform_inputs2[2], H=H, W=W)
#         return x, c, cls


class SpatialPriorModule(nn.Module):
    def __init__(self, inplanes=64, embed_dim=384, with_cp=False):
        super().__init__()
        self.with_cp = with_cp

        self.stem = nn.Sequential(*[
            nn.Conv2d(3, inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(inplanes),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        ])
        self.conv2 = nn.Sequential(*[
            nn.Conv2d(inplanes, 2 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(2 * inplanes),
            nn.ReLU(inplace=True)
        ])
        self.conv3 = nn.Sequential(*[
            nn.Conv2d(2 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(4 * inplanes),
            nn.ReLU(inplace=True)
        ])
        self.conv4 = nn.Sequential(*[
            nn.Conv2d(4 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(4 * inplanes),
            nn.ReLU(inplace=True)
        ])
        self.fc1 = nn.Conv2d(inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc2 = nn.Conv2d(2 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc3 = nn.Conv2d(4 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc4 = nn.Conv2d(4 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        
        def _inner_forward(x):
            c1 = self.stem(x)
            c2 = self.conv2(c1)
            c3 = self.conv3(c2)
            c4 = self.conv4(c3)
            c1 = self.fc1(c1)
            c2 = self.fc2(c2)
            c3 = self.fc3(c3)
            c4 = self.fc4(c4)
    
            bs, dim, _, _ = c1.shape
            # c1 = c1.view(bs, dim, -1).transpose(1, 2)  # 4s
            c2 = c2.view(bs, dim, -1).transpose(1, 2)  # 8s
            c3 = c3.view(bs, dim, -1).transpose(1, 2)  # 16s
            c4 = c4.view(bs, dim, -1).transpose(1, 2)  # 32s
    
            return c1, c2, c3, c4
        
        if self.with_cp and x.requires_grad:
            outs = cp.checkpoint(_inner_forward, x)
        else:
            outs = _inner_forward(x)
        return outs


class CustomSpatialPriorModule(nn.Module):
    def __init__(self, inplanes=64, embed_dim=384, with_cp=False, **kwargs):
        super().__init__()
        self.with_cp = with_cp
        embed_dims = kwargs["embed_dims"]
        inplanes = embed_dims[0]

        self.stem = nn.Sequential(*[
            nn.Conv2d(3, inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(inplanes),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        ])
        self.conv2 = nn.Sequential(*[
            nn.Conv2d(inplanes, embed_dims[1], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(embed_dims[1]),
            nn.ReLU(inplace=True)
        ])
        self.conv3 = nn.Sequential(*[
            nn.Conv2d(embed_dims[1], embed_dims[2], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(embed_dims[2]),
            nn.ReLU(inplace=True)
        ])
        self.conv4 = nn.Sequential(*[
            nn.Conv2d(embed_dims[2], embed_dims[3], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(embed_dims[3]),
            nn.ReLU(inplace=True)
        ])
        self.fc1 = nn.Conv2d(inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc2 = nn.Conv2d(embed_dims[1], embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc3 = nn.Conv2d(embed_dims[2], embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc4 = nn.Conv2d(embed_dims[3], embed_dim, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        
        def _inner_forward(x):
            c1 = self.stem(x)
            c2 = self.conv2(c1)
            c3 = self.conv3(c2)
            c4 = self.conv4(c3)
            c1 = self.fc1(c1)
            c2 = self.fc2(c2)
            c3 = self.fc3(c3)
            c4 = self.fc4(c4)
    
            bs, dim, _, _ = c1.shape
            # c1 = c1.view(bs, dim, -1).transpose(1, 2)  # 4s
            c2 = c2.view(bs, dim, -1).transpose(1, 2)  # 8s
            c3 = c3.view(bs, dim, -1).transpose(1, 2)  # 16s
            c4 = c4.view(bs, dim, -1).transpose(1, 2)  # 32s
    
            return c1, c2, c3, c4

        def _custom_inner_forward(x):
            c1 = self.stem(x)
            c2 = self.conv2(c1)
            c3 = self.conv3(c2)
            c4 = self.conv4(c3)

            bs = c1.shape[0]
            c1 = c1.view(bs, c1.shape[1], -1).transpose(1, 2)
            c2 = c2.view(bs, c2.shape[1], -1).transpose(1, 2)
            c3 = c3.view(bs, c3.shape[1], -1).transpose(1, 2)
            c4 = c4.view(bs, c4.shape[1], -1).transpose(1, 2)

            # _c1 = self.fc1(c1)
            # _c2 = self.fc2(c2)
            # _c3 = self.fc3(c3)
            # _c4 = self.fc4(c4)

            # bs, dim, _, _ = c1.shape
            # # c1 = c1.view(bs, dim, -1).transpose(1, 2)  # 4s
            # _c2 = _c2.view(bs, dim, -1).transpose(1, 2)  # 8s
            # _c3 = _c3.view(bs, dim, -1).transpose(1, 2)  # 16s
            # _c4 = _c4.view(bs, dim, -1).transpose(1, 2)  # 32s

            return c1, c2, c3, c4

        if self.with_cp and x.requires_grad:
            outs = cp.checkpoint(_inner_forward, x)
        else:
            # outs = _inner_forward(x)
            outs = _custom_inner_forward(x)
        return outs


# ----- deprecated (2312??) -----
class CrossAttention(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f'dim {dim} should be divided by ' \
                                     f'num_heads {num_heads}.'

        self.dim = dim                              #512
        self.num_heads = num_heads                  #8
        head_dim = dim // num_heads                 #64
        self.scale = qk_scale or head_dim**-0.5     #8

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(
                dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, query, feat, H, W):
        """
        query: Transformer-originated
        feat: Stem-originated
        """
        B, N, C = query.shape
        q = self.q(query).reshape(B, N, self.num_heads,
                                  C // self.num_heads).permute(0, 2, 1,
                                                               3).contiguous()

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).contiguous().reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1).contiguous()
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads,
                                     C // self.num_heads).permute(
                                         2, 0, 3, 1, 4).contiguous()
        else:
            # kv = self.kv(x).reshape(B, -1, 2, self.num_heads,
            #                         C // self.num_heads).permute(
            #                             2, 0, 3, 1, 4).contiguous()
            kv = self.kv(feat).reshape(B, -1, 2, self.num_heads,
                                       C // self.num_heads).permute(
                                           2, 0, 3, 1, 4).contiguous()

        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1).contiguous()) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).contiguous().reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

