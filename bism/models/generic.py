import torch
from torch import Tensor
import torch.nn as nn
from typing import List, Union, Tuple


class Generic(nn.Module):
    def __init__(self, backbone: nn.Module, activations: List[nn.Module], channel_indicies: List[List[int]] = []):
        """
        A generic wrapper to a segmentation model. Often, it is desirable to have multiple activations
        be applied to different output channels. This wrapper allows a generic backend to have custom activations
        applied to each layer

        Note - this is best for flexibility and prototyping. For optimal speed, it's probably best to create
        your own constructor for this.

        :param backbone: nn.Module from bism.backbone
        :param activation: List[nn.Module]
        :param channel_indicies: List[List[int]] of channels to apply each activation to.
        """
        super(Generic, self).__init__()

        self.backbone = backbone
        self.channel_indicies = channel_indicies
        self.activation = activations


    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass applying each activation to each channel.

        :param x: input tensor
        :return: output of backbone with activations applied
        """

        x: Tensor = self.backbone(x)

        for activation, ind in zip(self.activation, self.channel_indicies):
            x[:, ind, ...] = activation(x[:, ind, ...])

        return x


if __name__=="__main__":
    x = torch.rand((1, 1, 100, 100, 10))
    backbone = nn.Conv3d(1, 3, 3, 1, 1)
    activations = [nn.ReLU(), nn.Sigmoid(), nn.Tanh()]
    channel_inds = [[0], [1], [2]]

    model = Generic(backbone=backbone, activations=activations, channel_indicies=channel_inds)

    out = model(x)