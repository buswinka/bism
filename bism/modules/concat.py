import torch
from torch import Tensor
import torch.nn as nn


class ConcatConv3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(ConcatConv3D, self).__init__()
        self._in_channels = in_channels
        self._out_channels = out_channels

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        x = torch.cat((x, y), dim=1)
        x = self.conv(x)
        return x

    def __repr__(self):
        return f'ConcatConv3d[in_channels: {self._in_channels}, out_channels: {self._out_channels}]'

class ConcatConv2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(ConcatConv2D, self).__init__()
        self._in_channels = in_channels
        self._out_channels = out_channels

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        x = torch.cat((x, y), dim=1)
        x = self.conv(x)
        return x

    def __repr__(self):
        return f'ConcatConv2d[in_channels: {self._in_channels}, out_channels: {self._out_channels}]'
