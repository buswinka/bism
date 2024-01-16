import torch
from torch import Tensor
from typing import List, Tuple, Optional, Union
import torch.nn as nn
from bism.modules.spade import SPADE2D, SPADE3D
from bism.modules.drop_path import DropPath
import warnings
import torch.utils.checkpoint

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


class DoubleSpadeBlock3D(nn.Module):
    """
    Unet Block.
    """

    def __init__(
        self,
        *,
        mask_channels: int,
        in_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        dim: Optional[int] = None,
        kernel_size: int = 7,
        dilation: int = 1,
        activation: Optional[nn.Module] = None,
        drop_path: Optional[float] = None,
        layer_scale_init_value: Optional[float] = None,
    ) -> None:
        super(DoubleSpadeBlock3D, self).__init__()

        self.mask_channels = mask_channels

        if dim is None:
            self.in_channels = in_channels
            self.out_channels = out_channels
        if dim is not None and in_channels is None and out_channels is None:
            self.in_channels = dim
            self.out_channels = dim
        if dim is not None and in_channels is not None and out_channels is not None:
            raise RuntimeError(
                "Setting kwarg dim with kwarg in_channels and out_channels is ambiguous. Please either set dim OR in_channels and out_channels"
            )

        if drop_path is not None or layer_scale_init_value is not None:
            warnings.warn(
                "Drop Path or Layer Scaling is not supported. Values will be ignored"
            )

        kernel_size = (
            (kernel_size,) * 3 if isinstance(kernel_size, int) else kernel_size
        )
        padding = tuple([k // 2 for k in kernel_size])

        self.conv_0 = nn.Conv3d(
            self.in_channels,
            self.out_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=1,
            dilation=dilation,
            bias=False
        )  # depthwise conv

        self.conv_1 = nn.Conv3d(
            self.out_channels,
            self.out_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=1,
            dilation=dilation,
            bias=False
        )  # depthwise conv

        self.conv_s = nn.Conv3d(self.in_channels, self.out_channels, kernel_size=1, bias=False)

        self.norm_0 = SPADE3D(mask_channels=self.mask_channels, tensor_channels=self.in_channels)
        self.norm_1 = SPADE3D(mask_channels=self.mask_channels, tensor_channels=self.out_channels)
        self.norm_s = SPADE3D(mask_channels=self.mask_channels, tensor_channels=self.in_channels)

        self.act = activation() if activation else nn.GELU()
        # self.act = torch.utils.checkpoint.checkpoint()

        self.in_channels = in_channels
        self.out_channels = out_channels

    def __repr__(self):
        return f"UNetSpadeDoubleBlock[in_channels: {self.in_channels}, out_channels: {self.out_channels}, mask_channels: {self.mask_channels}]"

    def forward(self, x_and_mask: Tensor) -> Tensor:
        """
        accepts concatenated input tensor and mask. We expect the last N channels to be the mask

        :param x_and_mask: self.in_channels + self.
        :return:
        """
        x, mask = x_and_mask[:, 0:self.in_channels, ...], x_and_mask[:, self.in_channels::, ...]

        x = x.contiguous(memory_format=torch.channels_last_3d)
        mask = mask.contiguous(memory_format=torch.channels_last_3d)

        skip = self.conv_s(self.norm_s(x, mask))
        x = self.conv_0(self.act(self.norm_0(x, mask)))
        x = self.conv_1(self.act(self.norm_1(x, mask)))

        return torch.concat((x + skip, mask), dim=1)


class DoubleSpadeBlock2D(nn.Module):
    """
    Unet Block.
    """

    def __init__(
        self,
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
        super(DoubleSpadeBlock2D, self).__init__()

        if dim is None:
            self.in_channels = in_channels
            self.out_channels = out_channels
        if dim is not None and in_channels is None and out_channels is None:
            self.in_channels = dim
            self.out_channels = dim
        if dim is not None and in_channels is not None and out_channels is not None:
            raise RuntimeError(
                "Setting kwarg dim with kwarg in_channels and out_channels is ambiguous. Please either set dim OR in_channels and out_channels"
            )

        if drop_path is not None or layer_scale_init_value is not None:
            warnings.warn(
                "Drop Path or Layer Scaling is not supported. Values will be ignored"
            )

        kernel_size = (
            (kernel_size,) * 2 if isinstance(kernel_size, int) else kernel_size
        )
        padding = tuple([k // 2 for k in kernel_size])

        self.conv = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=1,
            dilation=dilation,
            bias=False,
        )  # depthwise conv
        self.norm = nn.BatchNorm2d(self.out_channels)
        self.act = activation() if activation else nn.GELU()
        # self.act = torch.utils.checkpoint.checkpoint(self.act)

        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError('doublespadeblock2d not implemented.')
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)

        return x


if __name__ == "__main__":

    b = torch.compile(DoubleSpadeBlock3D(in_channels=1, out_channels=3).cuda(), mode="max-autotune", disable=True)
    b(torch.rand((1, 1, 300, 300, 20)).cuda(),
      torch.rand((1, 1, 300, 300, 20)).cuda())

