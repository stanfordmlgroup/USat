# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn

import timm.models.vision_transformer
from timm.models.vision_transformer import PatchEmbed
from usat.utils.satmae_helpers import get_2d_sincos_pos_embed
from usat.utils.builder import MODEL


class BasicVitLargeMultispectral(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, channel_groups=None, **kwargs):
        super(BasicVitLargeMultispectral, self).__init__(**kwargs)

        self.channel_groups = channel_groups
        self.patch_embed = nn.ModuleList([PatchEmbed(kwargs['img_size'], kwargs['patch_size'], len(group), kwargs['embed_dim'])
                                          for group in channel_groups])

        num_patches = self.patch_embed[0].num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, kwargs['embed_dim']))

        # Not sure if we want to use learned embeddings here
        #self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim) * .02)

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, kwargs['embed_dim']), requires_grad=False)
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(num_patches ** .5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))


        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

    def forward_features(self, x):
        B, _, _, _ = x.shape

        x_c_embed = []
        current = 0
        for i, group in enumerate(self.channel_groups):
            # The order of each group is just the order of x
            interval = torch.arange(current, current+len(group))
            current += len(group)
            x_c_embed.append(self.patch_embed[i](x[:,interval,:,:]))  # (N, L, D)

        x = torch.stack(x_c_embed, dim=1)  # (N, G, L, D)
        _, G, L, D = x.shape

        pos_embed = self.pos_embed[:, 1:, :].unsqueeze(1)  # (1, 1, L, pD)
        pos_embed = pos_embed.expand(-1, G, -1, -1)  # (1, c, L, pD)

        x = x + pos_embed
        x = x.view(B, -1, D)  # (N, G*L, D)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x

    def forward(self, imgs):
        x_c = []
        for i, group in enumerate(self.channel_groups):
            for band in group:
                x_c.append(imgs[band])
        x = torch.cat(x_c, dim=1)

        x = self.forward_features(x)
        return x
    

class BasicVitLargeRGB(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, channel_groups=None, **kwargs):
        super(BasicVitLargeRGB, self).__init__(**kwargs)

        self.channel_groups = channel_groups

        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, kwargs['embed_dim']) * .02, requires_grad=False)
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(num_patches ** .5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x

    def forward(self, imgs):
        x_c = []
        for i, group in enumerate(self.channel_groups):
            for band in group:
                x_c.append(imgs[band])
        x = torch.cat(x_c, dim=1)

        x = self.forward_features(x)
        return x



@MODEL.register_module()
def basic_vit_large_encoder(input_parmas, **kwargs):
    channel_groups = input_parmas.get("groups", [['S2:Red', 'S2:Green', 'S2:Blue', 'S2:NIR'], ['S2:RE1', 'S2:RE2', 'S2:RE3', 'S2:RE4'], ['S2:SWIR1', 'S2:SWIR2']])
    patch_size = input_parmas.get("patch_size", 16)
    model_dict = dict(
        img_size=input_parmas.get("img_size", 224),
        channel_groups=channel_groups,
        patch_size=patch_size,
        num_classes=0,      # don't need classification head
        embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return BasicVitLargeMultispectral(**model_dict)


@MODEL.register_module()
def basic_vit_large_rgb_encoder(input_parmas, **kwargs):
    channel_groups = input_parmas.get("groups", [['NAIP:Red'], ['NAIP:Green'], ['NAIP:Green'], ['NAIP:NIR']])
    patch_size = input_parmas.get("patch_size", 16)
    model_dict = dict(
        img_size=input_parmas.get("img_size", 224),
        channel_groups=channel_groups,
        num_classes=0,      # don't need classification head
        patch_size=patch_size,
        in_chans=input_parmas.get("num_channels", 4),
        embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return BasicVitLargeRGB(**model_dict)
