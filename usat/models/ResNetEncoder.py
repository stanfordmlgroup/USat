# Partially adpot from timm library https://github.com/rwightman/pytorch-image-models 
import copy
from typing import Dict

import torch
import torch.nn as nn
from timm.models.resnet import ResNet, Bottleneck, BasicBlock
from timm.models.helpers import build_model_with_cfg

from usat.utils.builder import MODEL
from usat.utils.constants import IN_RESENET_CHANNEL_MAP

def update_first_layer(
        model,
        pretrained: bool = False,
        num_channels: int = 3,
        rgb_map: Dict[str, int] = None
    ):
    """
    Modify the first layer of the model to accomadate a different
    number of channels
    """
    original_conv1_weights = model.conv1.weight.clone()

    if num_channels != 3:
        model.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7,
                                stride=2, padding=3, bias=False)

    # Update weights with pretrained weights in corresponding channel
    # based on rgb_map of the dataset
    if pretrained and rgb_map:
        with torch.no_grad():
            for band, new_ind in rgb_map.items():
                old_ind = IN_RESENET_CHANNEL_MAP.get(band)
                new_ind = new_ind if isinstance(new_ind, list) else [new_ind]
                for ind in new_ind:
                    model.conv1.weight[:,ind,:,:] = original_conv1_weights[:,old_ind,:,:]

    # remove head
    model.fc = nn.Identity()
    return model

@MODEL.register_module()
def resnet_50_encoder (pretrained: bool = False, num_channels: int = 3,
                       rgb_map: Dict[str, int] = None, **kwargs) -> ResNet:
    model_args = dict(block=Bottleneck, layers=[3, 4, 6, 3],  **kwargs)
    model = build_model_with_cfg(ResNet, 'resnet50', pretrained, **model_args)

    model = update_first_layer(model, pretrained, num_channels, rgb_map)
    return model

@MODEL.register_module()
def resnet_18_encoder (pretrained: bool = False, num_channels: int = 3,
                       rgb_map: Dict[str, int] = None, **kwargs) -> ResNet:
    model_args = dict(block=BasicBlock, layers=[2, 2, 2, 2],  **kwargs)
    model = build_model_with_cfg(ResNet, 'resnet18', pretrained, **model_args)

    model = update_first_layer(model, pretrained, num_channels, rgb_map)

    return model