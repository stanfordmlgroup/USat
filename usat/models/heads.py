import typing as T

import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

from usat.utils.builder import MODEL

@MODEL.register_module()
class MLP_head (nn.Module):
    def __init__ (self, input_dim: int, output_dim: int, hidden_dim: int,  num_proj_layer: int, 
                                            last_bn: bool = True, use_bias: bool = True) -> None:
        super().__init__()
        self.mlp  = {}
        self.num_proj_layer = num_proj_layer
        for l in range(num_proj_layer):
            dim1 = input_dim if l == 0 else hidden_dim
            dim2 = output_dim if l == num_proj_layer - 1 else hidden_dim
            layer_module = []
            layer_module.append(nn.Linear(dim1, dim2, bias=use_bias))
            if l < num_proj_layer - 1:
                layer_module.append(nn.BatchNorm1d(dim2))
                layer_module.append(nn.ReLU(inplace=True))
            elif last_bn:
                layer_module.append(nn.BatchNorm1d(dim2, affine=False))
            self.mlp[f'layer{l}'] = nn.Sequential(*layer_module)
        self.mlp = nn.ModuleDict(self.mlp)
    

    def forward(self, x: torch.Tensor) -> T.Tuple[torch.Tensor]:
        for l in range(self.num_proj_layer):
            x = self.mlp[f'layer{l}'](x)
        return x

@MODEL.register_module()
class ClassifierHead (nn.Module):
    def __init__(self, input_size: int, num_classes: int, drop_rate: float = 0., use_bias: bool = True, conv_head: bool = False) -> None:
        super().__init__()
        self.num_classes = num_classes
        if conv_head:
            self.head = nn.Conv2d(input_size, num_classes, 1, bias=use_bias)
        else:
            self.head = nn.Linear(input_size, num_classes, bias=use_bias)        
        self.drop_rate = drop_rate


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_rate > 0:
            x = F.dropout(x, p=float(self.drop_rate), training=self.training)
        logit = self.head(x)
        return logit


@MODEL.register_module()
class DummyModel (nn.Module):
    def __init__ (self, input_shape: T.Sequence, out_size: int):
        super().__init__()
        n_feature = np.product(input_shape)
        self.ln = nn.Linear(n_feature, out_size)
    

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.ln(x)
        return x