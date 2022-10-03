import warnings

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Union, Tuple, Optional
from functools import partial

from bism.modules.upsample_layer import UpSampleLayer2D
from bism.modules.concat import ConcatConv2D

"""
Re-implementation of Cellpose Net with some changes

- Each residual block only has 1 set of convolutions. 
Now you just string together multiple blocks

- Each block now has adjustable normalization and activation params

- Upsample now works slightly differently

- Downsample now uses strided convolutions instead of MaxPool
"""


class ResidualBlock2D(nn.Module):
    """
    Residual Block: Adds 4 blocks into a squential module
    """
    def __init__(self,
                 *,
                 dim: int,
                 kernel_size: Union[Tuple[int, int], int],
                 normalization: Optional[nn.Module] = partial(nn.BatchNorm2d, eps=1e-5),
                 activation: Optional[nn.Module] = nn.ReLU,
                 drop_path: Optional[float] = None,
                 layer_scale_init_value: Optional[float] = None,
                 ):
        super().__init__()

        if drop_path is not None or layer_scale_init_value is not None:
            warnings.warn('Drop Path or Layer Scaling is not supported. Values will be ignored')


        self.kernel_size: Tuple[int] = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, ) * 2
        self.padding = tuple([ks // 2 for ks in self.kernel_size])

        self.norm = normalization(dim)
        self.conv = nn.Conv2d(in_channels=dim, out_channels=dim,
                              kernel_size=kernel_size, padding=self.padding)

        self.activation = activation()


    def forward(self, x: Tensor) -> Tensor:
        input = x

        x = self.norm(x)
        x = self.activation(x)
        x = self.conv(x)

        x = input + x

        return x

class ResidualBlock3D(nn.Module):
    """
    Residual Block: Adds 4 blocks into a squential module
    """
    def __init__(self,
                 *,
                 dim: int,
                 kernel_size: Union[Tuple[int, int], int],
                 normalization: Optional[nn.Module] = partial(nn.BatchNorm3d, eps=1e-5),
                 activation: Optional[nn.Module] = nn.ReLU,
                 drop_path: Optional[float] = None,
                 layer_scale_init_value: Optional[float] = None,):
        super().__init__()

        if drop_path is not None or layer_scale_init_value is not None:
            warnings.warn('Drop Path or Layer Scaling is not supported. Values will be ignored')

        self.dim = dim
        self.kernel_size: Tuple[int] = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, ) * 3
        self.padding = tuple([ks // 2 for ks in self.kernel_size])


        self.norm = normalization(dim)
        self.conv = nn.Conv3d(in_channels=dim, out_channels=dim,
                              kernel_size=kernel_size, padding=self.padding)

        self.activation = activation()


    def forward(self, x: Tensor) -> Tensor:
        input = x

        x = self.norm(x)
        x = self.activation(x)
        x = self.conv(x)

        x = input + x

        return x