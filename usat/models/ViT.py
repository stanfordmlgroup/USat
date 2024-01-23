# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# port from https://github.com/facebookresearch/moco-v3

import warnings
import math
import torch
import torch.nn as nn
from functools import partial, reduce
from operator import mul
from typing import Dict

from timm.models.vision_transformer import VisionTransformer, _cfg, checkpoint_filter_fn
from timm.models.helpers import build_model_with_cfg, resolve_pretrained_cfg
from timm.models.layers.helpers import to_2tuple
from timm.models.layers import PatchEmbed

from usat.utils.builder import MODEL
from usat.utils.constants import IN_RESENET_CHANNEL_MAP

__all__ = [
    'vit_small_encoder', 
    'vit_base_encoder',
    'vit_conv_small_encoder',
    'vit_conv_base_encoder',
]


class VisionTransformerMoCo(VisionTransformer):
    def __init__(self, stop_grad_conv1=False, **kwargs):
        super().__init__(**kwargs)
        # Use fixed 2D sin-cos position embedding
        self.build_2d_sincos_position_embedding()

        # weight initialization
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                if 'qkv' in name:
                    # treat the weights of Q, K, V separately
                    val = math.sqrt(6. / float(m.weight.shape[0] // 3 + m.weight.shape[1]))
                    nn.init.uniform_(m.weight, -val, val)
                else:
                    nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.normal_(self.cls_token, std=1e-6)

        if isinstance(self.patch_embed, PatchEmbed):
            # xavier_uniform initialization
            val = math.sqrt(6. / float(3 * reduce(mul, self.patch_embed.patch_size, 1) + self.embed_dim))
            nn.init.uniform_(self.patch_embed.proj.weight, -val, val)
            nn.init.zeros_(self.patch_embed.proj.bias)

            if stop_grad_conv1:
                self.patch_embed.proj.weight.requires_grad = False
                self.patch_embed.proj.bias.requires_grad = False

    def build_2d_sincos_position_embedding(self, temperature=10000.):
        h, w = self.patch_embed.grid_size
        grid_w = torch.arange(w, dtype=torch.float32)
        grid_h = torch.arange(h, dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='ij')
        assert self.embed_dim % 4 == 0, 'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = self.embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature**omega)
        out_w = torch.einsum('m,d->md', [grid_w.flatten(), omega])
        out_h = torch.einsum('m,d->md', [grid_h.flatten(), omega])
        pos_emb = torch.cat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], dim=1)[None, :, :]

        assert self.num_prefix_tokens == 1, 'Assuming one and only one token, [cls]'
        pe_token = torch.zeros([1, 1, self.embed_dim], dtype=torch.float32)
        self.pos_embed = nn.Parameter(torch.cat([pe_token, pos_emb], dim=1))
        self.pos_embed.requires_grad = False


class ConvStem(nn.Module):
    """ 
    ConvStem, from Early Convolutions Help Transformers See Better, Tete et al. https://arxiv.org/abs/2106.14881
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True, **kwargs):
        super().__init__()

        assert patch_size == 16, 'ConvStem only supports patch size of 16'
        assert embed_dim % 8 == 0, 'Embed dimension must be divisible by 8 for ConvStem'

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        # build stem, similar to the design in https://arxiv.org/abs/2106.14881
        stem = []
        input_dim, output_dim = in_chans, embed_dim // 8
        for l in range(4):
            stem.append(nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=2, padding=1, bias=False))
            stem.append(nn.BatchNorm2d(output_dim))
            stem.append(nn.ReLU(inplace=True))
            input_dim = output_dim
            output_dim *= 2
        stem.append(nn.Conv2d(input_dim, embed_dim, kernel_size=1))
        self.proj = nn.Sequential(*stem)

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


@MODEL.register_module()
def vit_small_encoder (pretrained: bool = False, num_channels=3,
                       rgb_map: Dict[str, int] = None, **kwargs):
    variant = 'vit_small_patch16_224'

    # config for ViT small
    model_kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

    pretrained_cfg = resolve_pretrained_cfg(variant, pretrained_cfg=kwargs.pop('pretrained_cfg', None))
    model = build_model_with_cfg(
        VisionTransformerMoCo, variant, pretrained,
        pretrained_cfg=pretrained_cfg,
        pretrained_filter_fn=checkpoint_filter_fn,
        pretrained_custom_load='npz' in pretrained_cfg['url'],
        **model_kwargs)
    
    original_patch_proj_weights = model.patch_embed.proj.weight.clone()
    if num_channels != 3:
        model.patch_embed.proj = nn.Conv2d(num_channels, 384,
                                           kernel_size=16, stride=16)
    if pretrained:
        with torch.no_grad():
            for band, new_ind in rgb_map.items():
                old_ind = IN_RESENET_CHANNEL_MAP.get(band)
                new_ind = new_ind if isinstance(new_ind, list) else [new_ind]
                for ind in new_ind:
                    model.patch_embed.proj.weight[:,ind,:,:] = original_patch_proj_weights[:,old_ind,:,:]

    if not pretrained:
        model.default_cfg = _cfg()

    # remove head
    model.head = nn.Identity()

    return model


@MODEL.register_module()
def vit_base_encoder (pretrained: bool =False, **kwargs):
    variant = 'vit_base_patch16_224'

    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

    pretrained_cfg = resolve_pretrained_cfg(variant, pretrained_cfg=kwargs.pop('pretrained_cfg', None))
    model = build_model_with_cfg(
        VisionTransformerMoCo, variant, pretrained,
        pretrained_cfg=pretrained_cfg,
        pretrained_filter_fn=checkpoint_filter_fn,
        pretrained_custom_load='npz' in pretrained_cfg['url'],
        **model_kwargs)
    
    if not pretrained:
        model.default_cfg = _cfg()
    
    # remove head
    model.head = nn.Identity()
    
    return model


@MODEL.register_module()
def vit_large_encoder (pretrained: bool = False, num_channels=3,
                      rgb_map: Dict[str, int] = None, **kwargs):
    variant = 'vit_large_patch16_224'

    model_kwargs = dict(patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

    pretrained_cfg = resolve_pretrained_cfg(variant, pretrained_cfg=kwargs.pop('pretrained_cfg', None))
    model = build_model_with_cfg(
        VisionTransformerMoCo, variant, pretrained,
        pretrained_cfg=pretrained_cfg,
        pretrained_filter_fn=checkpoint_filter_fn,
        pretrained_custom_load='npz' in pretrained_cfg['url'],
        **model_kwargs)
    
    original_patch_proj_weights = model.patch_embed.proj.weight.clone()
    if num_channels != 3:
        model.patch_embed.proj = nn.Conv2d(num_channels, 1024,
                                           kernel_size=16, stride=16)
    if pretrained:
        with torch.no_grad():
            for band, new_ind in rgb_map.items():
                old_ind = IN_RESENET_CHANNEL_MAP.get(band)
                new_ind = new_ind if isinstance(new_ind, list) else [new_ind]
                for ind in new_ind:
                    model.patch_embed.proj.weight[:,ind,:,:] = original_patch_proj_weights[:,old_ind,:,:]
    if not pretrained:
        model.default_cfg = _cfg()
    
    # remove head
    model.head = nn.Identity()
    
    return model


@MODEL.register_module()
def vit_huge_encoder (pretrained: bool =False, **kwargs):
    variant = 'vit_huge_patch14_224'

    model_kwargs = dict(patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

    pretrained_cfg = resolve_pretrained_cfg(variant, pretrained_cfg=kwargs.pop('pretrained_cfg', None))
    model = build_model_with_cfg(
        VisionTransformerMoCo, variant, pretrained,
        pretrained_cfg=pretrained_cfg,
        pretrained_filter_fn=checkpoint_filter_fn,
        pretrained_custom_load='npz' in pretrained_cfg['url'],
        **model_kwargs)
    
    if not pretrained:
        model.default_cfg = _cfg()
    
    # remove head
    model.head = nn.Identity()
    
    return model


@MODEL.register_module()
def vit_conv_small_encoder (**kwargs):
    if kwargs.pop('pretrained', False):
        warnings.warn('ViT conv does not have ImageNet pre-trained weight. Initialize with random.')
    # minus one ViT block
    model = VisionTransformerMoCo(
        patch_size=16, embed_dim=384, depth=11, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), embed_layer=ConvStem, **kwargs)
    model.default_cfg = _cfg()
    # remove head
    model.head = nn.Identity()
    return model


@MODEL.register_module()
def vit_conv_base_encoder (**kwargs):
    if kwargs.pop('pretrained', False):
        warnings.warn('ViT conv does not have ImageNet pre-trained weight. Initialize with random.')
    # minus one ViT block
    model = VisionTransformerMoCo(
        patch_size=16, embed_dim=768, depth=11, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), embed_layer=ConvStem, **kwargs)
    model.default_cfg = _cfg()
    # remove head
    model.head = nn.Identity()
    return model