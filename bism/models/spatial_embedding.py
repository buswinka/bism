import torch
from torch import Tensor
import torch.nn as nn
from typing import List, Union, Tuple

class SpatialEmbedding(nn.Module):
    def __init__(self, backbone: nn.Module):
        """
        A model architecture for skoots-based spatial embeddings.

        :param backbone:
        :type backbone:
        """
        super(SpatialEmbedding, self).__init__()

        self.backbone = backbone

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

        # Flatten Channels to out...
        self.out_embed = nn.Conv3d(self.backbone.out_channels, 3, kernel_size=3, stride=1, dilation=1, padding=1)
        self.out_prob = nn.Conv3d(self.backbone.out_channels, 1, kernel_size=3, stride=1, dilation=1, padding=1)
        self.out_skeleton = nn.Conv3d(self.backbone.out_channels, 1, kernel_size=3, stride=1, dilation=1, padding=1)

    def forward(self, x: Tensor) -> Tensor:

        x = self.backbone(x)

        x = torch.cat((
            self.tanh(self.out_embed(x)),
            self.sigmoid(self.out_skeleton(x)),
            self.sigmoid(self.out_prob(x))
        ), dim=1)

        return x

class GraphableSpatialEmbedding(nn.Module):
    def __init__(self, backbone: nn.Module):
        super(GraphableSpatialEmbedding, self).__init__()

        self.backbone = backbone

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

        # Flatten Channels to out...
        self.out_embed = nn.Conv3d(self.backbone.out_channels, 3, kernel_size=3, stride=1, dilation=1, padding=1)
        self.out_prob = nn.Conv3d(self.backbone.out_channels, 1, kernel_size=3, stride=1, dilation=1, padding=1)
        self.out_skeleton = nn.Conv3d(self.backbone.out_channels, 1, kernel_size=3, stride=1, dilation=1, padding=1)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:

        x = self.backbone(x)

        # x = torch.cat((
        #     self.tanh(self.out_embed(x)),
        #     self.sigmoid(self.out_skeleton(x)),
        #     self.sigmoid(self.out_prob(x))
        # ), dim=1)

        return self.tanh(self.out_embed(x)), self.sigmoid(self.out_skeleton(x)), self.sigmoid(self.out_prob(x))


