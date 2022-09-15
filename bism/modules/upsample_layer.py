import torch
from torch import Tensor
from .layer_norm import LayerNorm
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


class UpSampleLayer3D(nn.Module):
    def __init__(self,
                 in_channels: int, out_channels: int,
                 method='nearest',
                 kernel_size: Optional[Tuple[int]] = (3, 3, 3),
                 padding: Optional[Tuple[int]] = (1, 1, 1)):
        super(UpSampleLayer3D, self).__init__()
        self.method = method
        self.norm = LayerNorm(in_channels, eps=1e-6, data_format="channels_first")
        self.compress = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding)

    def forward(self, x: Tensor, shape: List[int]) -> Tensor:
        'Upsample layer'
        x = F.interpolate(x, shape[2::], mode=self.method)
        x = self.norm(x)
        x = self.compress(x)
        return x


class UpSampleLayer2D(nn.Module):
    def __init__(self,
                 in_channels: int, out_channels: int,
                 method='nearest',
                 kernel_size: Optional[Tuple[int]] = (3, 3),
                 padding: Optional[Tuple[int]] = (1, 1)):
        super(UpSampleLayer2D, self).__init__()
        self.method = method
        self.norm = LayerNorm(in_channels, eps=1e-6, data_format="channels_first")
        self.compress = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding)

    def forward(self, x: Tensor, shape: List[int]) -> Tensor:
        'Upsample layer'
        x = F.interpolate(x, shape[2::], mode=self.method)
        x = self.norm(x)
        x = self.compress(x)
        return x