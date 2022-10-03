import torch
from torch import Tensor
from typing import List, Tuple, Optional, Union
import torch.nn as nn
from .layer_norm import LayerNorm
from .drop_path import DropPath
import warnings
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
    def __init__(self,
                 *,
                 in_channels: Optional[int] = None,
                 out_channels: Optional[int] = None,
                 dim: Optional[int] = None,
                 kernel_size: int = 7,
                 dilation: int = 1,
                 activation: Optional[nn.Module] = None,
                 drop_path: Optional[float] = None,
                 layer_scale_init_value: Optional[float] = None,
                 ) -> None:
        super().__init__()

        if dim is None:
            self.in_channels = in_channels
            self.out_channels = out_channels
        if dim is not None and in_channels is None and out_channels is None:
            self.in_channels = dim
            self.out_channels = dim
        if dim is not None and in_channels is not None and out_channels is not None:
            raise RuntimeError('Setting kwarg dim with kwarg in_channels and out_channels is ambiguous. Please either set dim OR in_channels and out_channels')

        if drop_path is not None or layer_scale_init_value is not None:
            warnings.warn('Drop Path or Layer Scaling is not supported. Values will be ignored')

        kernel_size = (kernel_size, ) * 3 if isinstance(kernel_size, int) else kernel_size
        padding = tuple([k // 2 for k in kernel_size])

        self.conv = nn.Conv3d(self.in_channels, self.out_channels, kernel_size=kernel_size, padding=padding, groups=1, dilation=dilation)  # depthwise conv
        self.norm = nn.BatchNorm3d(self.out_channels)
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

    def __init__(self,
                 *,
                 in_channels: Optional[int] = None,
                 out_channels: Optional[int] = None,
                 dim: Optional[int] = None,
                 kernel_size: int = 7,
                 dilation: int = 1,
                 activation: Optional[nn.Module] = None,
                 drop_path: Optional[float] = None,
                 layer_scale_init_value: Optional[float] = None,
                 ) -> None:
        super().__init__()

        if dim is None:
            self.in_channels = in_channels
            self.out_channels = out_channels
        if dim is not None and in_channels is None and out_channels is None:
            self.in_channels = dim
            self.out_channels = dim
        if dim is not None and in_channels is not None and out_channels is not None:
            raise RuntimeError('Setting kwarg dim with kwarg in_channels and out_channels is ambiguous. Please either set dim OR in_channels and out_channels')

        if drop_path is not None or layer_scale_init_value is not None:
            warnings.warn('Drop Path or Layer Scaling is not supported. Values will be ignored')

        kernel_size = (kernel_size,) * 2 if isinstance(kernel_size, int) else kernel_size
        padding = tuple([k // 2 for k in kernel_size])

        self.conv = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=kernel_size, padding=padding, groups=1,
                              dilation=dilation)  # depthwise conv
        self.norm = nn.BatchNorm2d(self.out_channels)
        self.act = activation() if activation else nn.GELU()

        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)

        return x