import torch
import torch.nn as nn
from torch import Tensor


class SPADE3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(SPADE3D, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.activation = nn.GELU()

        self.conv1 = nn.Conv3d(
            in_channels=self.in_channels,
            out_channels=self.out_channels // 2,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.conv_gamma = nn.Conv3d(
            in_channels=self.out_channels // 2,
            out_channels=self.out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.conv_beta = nn.Conv3d(
            in_channels=self.out_channels // 2,
            out_channels=self.out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, x: Tensor, mask: Tensor):
        """ forward pass """
        mask = torch.nn.functional.interpolate(mask, size=x.shape[2::])
        mask = self.activation(self.conv1(mask))
        gamma = self.activation(self.conv_gamma(mask))
        beta = self.activation(self.conv_beta(mask))

        return x * gamma + beta

    def __repr__(self):
        return f'SPADE3D[in_channels: {self.in_channels}, out_channels: {self.out_channels}]'

class SPADE2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(SPADE2D, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.activation = nn.GELU()

        self.conv1 = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels // 2,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.conv_gamma = nn.Conv2d(
            in_channels=self.out_channels // 2,
            out_channels=self.out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.conv_beta = nn.Conv2d(
            in_channels=self.out_channels // 2,
            out_channels=self.out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, x: Tensor):
        """ forward pass """

        x = self.activation(self.conv1(x))
        gamma = self.activation(self.conv_gamma(x))
        beta = self.activation(self.conv_beta(x))
        return gamma, beta


if __name__ == "__main__":

    b = torch.compile(SPADE3D(1, 3).cuda(), mode='max-autotune')
    b(torch.rand((1,1,300,300,20)).cuda(), torch.rand((1,1,300,300,20)).cuda())


