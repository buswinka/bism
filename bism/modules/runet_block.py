import torch
from torch import Tensor
from typing import List, Tuple, Optional, Union
import torch.nn as nn
from .layer_norm import LayerNorm
from .drop_path import DropPath
class RBlock3D(nn.Module):
    """
    Unet Block.
    """
    def __init__(self, dim: int, N: int, *,
                 kernel_size: int = 7,
                 dilation: int = 1,
                 activation: Optional = None) -> None:
        super(RBlock3D, self).__init__()

        self.kernel_size = (kernel_size, ) * 3 if isinstance(kernel_size, int) else kernel_size
        padding = tuple([k // 2 for k in self.kernel_size])

        self.conv = nn.Conv3d(dim, dim, kernel_size=self.kernel_size, padding=padding, groups=1, dilation=dilation)  # depthwise conv
        self.norm = nn.BatchNorm3d(dim)
        self.act = activation() if activation else nn.GELU()

        self.dim = dim
        self.n = N


    def __repr__(self):
        return f'Recurrent UNet Block[dim: {self.dim}, kernel_size: {self.kernel_size}, N: {self.n}]'

    def forward(self, x: Tensor) -> Tensor:
        for _ in range(self.n):
            x = self.conv(x)
            x = self.norm(x)
            x = self.act(x)

        return x


class RBlock2D(nn.Module):
    """
    Unet Block.
    """

    def __init__(self, dim: int, N: int, *,
                 kernel_size: int = 7,
                 dilation: int = 1,
                 activation: Optional = None) -> None:
        super(RBlock2D, self).__init__()

        self.kernel_size = (kernel_size,) * 2 if isinstance(kernel_size, int) else kernel_size
        padding = tuple([k // 2 for k in kernel_size])

        self.conv = nn.Conv2d(dim, dim, kernel_size=self.kernel_size, padding=padding, groups=1,
                              dilation=dilation)  # depthwise conv
        self.norm = nn.BatchNorm3d(dim)
        self.act = activation() if activation else nn.GELU()

        self.dim = dim
        self.n = N

    def __repr__(self):
        return f'Recurrent UNet Block[dim: {self.dim}, kernel_size: {self.kernel_size}, N: {self.n}]'

    def forward(self, x: Tensor) -> Tensor:
        for _ in range(self.n):
            x = self.conv(x)
            x = self.norm(x)
            x = self.act(x)

        return x