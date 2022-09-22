import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


class MakeStyle2D(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()

    def forward(self, x: Tensor) -> Tensor:
        """
        What is the in shape? [B, C, X, Y?]
        """
        style = F.avg_pool2d(x, kernel_size=(x.shape[-2], x.shape[-1]))
        style = self.flatten(style)
        style = style / torch.sum(style ** 2, dim=1, keepdim=True) ** .5

        return style

class MakeStyle3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()

    def forward(self, x: Tensor) -> Tensor:
        """
        What is the in shape? [B, C, X, Y?]
        """
        style = F.avg_pool3d(x, kernel_size=(x.shape[-3], x.shape[-2], x.shape[-1]))
        style = self.flatten(style)
        style = style / torch.sum(style ** 2, dim=1, keepdim=True) ** .5

        return style


class MergeStyle(nn.Module):
    def __init__(self, dim: int, style_channels: int):

        super(MergeStyle, self).__init__()

        self.dim = dim
        self.style_channels = style_channels

        self.linear = nn.Linear(style_channels, dim)

    def forward(self, x: Tensor, style: Tensor) -> Tensor:
        shape = x.shape
        newshape = [shape[0], shape[1]] + [1 for _ in range(x.ndim - 2)]

        features = self.linear(style)

        x = x + features.view(*newshape)

        return x
