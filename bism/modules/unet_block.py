import torch
from torch import Tensor
from typing import List, Tuple, Optional, Union
import torch.nn as nn
from .layer_norm import LayerNorm
from .drop_path import DropPath
"""
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
"""

class Block3D(nn.Module):
    """
    Unet Block.
    """
    def __init__(self, in_channels: int, out_channels: int,  *,
                 kernel_size: int = 7,
                 dilation: int = 1,
                 activation: Optional = None) -> None:
        super().__init__()

        kernel_size = (kernel_size, ) * 3 if isinstance(kernel_size, int) else kernel_size
        padding = tuple([k // 2 for k in kernel_size])

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, groups=1, dilation=dilation)  # depthwise conv
        self.norm = nn.BatchNorm3d(out_channels)
        self.act = activation() if activation else nn.GELU()

        self.in_channels = in_channels
        self.out_channels = out_channels

    def __repr__(self):
        return f'UNet Block[in_channels: {self.in_channels}, out_channels: {self.out_channels}]'

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)

        return x


class Block2D(nn.Module):
    """
    Unet Block.
    """

    def __init__(self, in_channels: int, out_channels: int, *,
                 kernel_size: int = 7,
                 dilation: int = 1,
                 activation: Optional = None) -> None:
        super().__init__()

        kernel_size = (kernel_size,) * 2 if isinstance(kernel_size, int) else kernel_size
        padding = tuple([k // 2 for k in kernel_size])

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, groups=1,
                              dilation=dilation)  # depthwise conv
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = activation() if activation else nn.GELU()

        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)

        return x