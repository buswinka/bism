import torch
from torch import Tensor
from typing import List, Tuple, Optional, Union
import torch.nn as nn
from .layer_norm import LayerNorm
from .drop_path import DropPath


class Block3D(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self,
                 *,
                 dim: int,
                 drop_path: Optional[float] = 0.0,
                 kernel_size: int = 7,
                 dilation: int = 1,
                 layer_scale_init_value: Optional[float] = 1e-2,
                 activation: Optional = None) -> None:
        super().__init__()

        kernel_size = (kernel_size,) * 3 if isinstance(kernel_size, int) else kernel_size
        padding = tuple([k // 2 for k in kernel_size])

        self.dwconv = nn.Conv3d(dim, dim, kernel_size=kernel_size, padding=padding, groups=dim,
                                dilation=dilation)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = activation() if activation else nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path: Union[DropPath, nn.Identity] = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.dim = dim

    def __repr__(self):
        out = f'Block[In={self.dim}, Out={self.dim}]' \
              f'\n\tDepthWiseConv[In={self.dwconv.in_channels}, Out={self.dwconv.out_channels}, KernelSize={self.dwconv.kernel_size}]' \
              f'\n\tLayerNorm[Shape={self.norm.normalized_shape}]' \
              f'\n\tPointWiseConv[In={self.pwconv1.in_features}, Out={self.pwconv1.out_features}]' \
              f'\n\tGELU[]' \
              f'\n\tPointWiseConv[In={self.pwconv2.in_features}, Out={self.pwconv2.out_features}]' \
              f'\n\tScale[]' \
              f'\n\tDropPath[]'

        return out

    def forward(self, x: Tensor) -> Tensor:  # Image of shape (B, C, H, W) -> (10, 3, 100, 100)
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 4, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 4, 1, 2, 3)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class Block2D(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, *,
                 dim: int,
                 drop_path: Optional[float] = 0.0,
                 kernel_size: int = 7,
                 dilation: int = 1,
                 layer_scale_init_value: Optional[float] = 1e-2,
                 activation: Optional = None) -> None:

        super().__init__()

        kernel_size = (kernel_size,) * 2 if isinstance(kernel_size, int) else kernel_size
        padding = tuple([k // 2 for k in kernel_size])

        self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=padding, groups=dim,
                                dilation=dilation)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = activation() if activation else nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path: Union[DropPath, nn.Identity] = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.dim = dim

    def __repr__(self):
        out = f'Block[In={self.dim}, Out={self.dim}]' \
              f'\n\tDepthWiseConv[In={self.dwconv.in_channels}, Out={self.dwconv.out_channels}, KernelSize={self.dwconv.kernel_size}]' \
              f'\n\tLayerNorm[Shape={self.norm.normalized_shape}]' \
              f'\n\tPointWiseConv[In={self.pwconv1.in_features}, Out={self.pwconv1.out_features}]' \
              f'\n\tGELU[]' \
              f'\n\tPointWiseConv[In={self.pwconv2.in_features}, Out={self.pwconv2.out_features}]' \
              f'\n\tScale[]' \
              f'\n\tDropPath[]'

        return out

    def forward(self, x: Tensor) -> Tensor:  # Image of shape (B, C, H, W) -> (10, 3, 100, 100)
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (B, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (B, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x
