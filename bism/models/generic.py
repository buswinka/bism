import torch
from torch import Tensor
import torch.nn as nn
from typing import *


class Generic(nn.Module):
    def __init__(self, backbone: nn.Module, activations: List[nn.Module]):
        """
        A generic wrapper to a segmentation model. Often, it is desirable to have multiple activations
        be applied to different output channels. This wrapper allows a generic backend to have custom activations
        applied to each layer

        Note - this is best for flexibility and prototyping. For optimal speed, it's probably best to create
        your own constructor for this.

        :param backbone: nn.Module from bism.backbone
        :param activation: List[nn.Module] of a
        """
        super(Generic, self).__init__()

        self.backbone = backbone
        self.activation: List[Callable[[Tensor], Tensor]] = activations

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass applying each activation to each channel.

        :param x: input tensor
        :return: output of backbone with activations applied
        """

        x: Tensor = self.backbone(x)

        for ind, activation in enumerate(self.activation):
            if activation is not None:
                x[:, ind, ...] = activation(x[:, ind, ...])

        return x

        # outputs: List[Tensor] = []

        # for ind, activation in enumerate(self.activation):
        #     if activation is not None:
        #         outputs.append(activation(x[:, [ind], ...]))
        #
        # return torch.concat(outputs, dim=1)


if __name__=="__main__":
    x = torch.rand((1, 1, 100, 100, 10))
    backbone = nn.Conv3d(1, 3, 3, 1, 1)
    activations = [nn.ReLU(), nn.Sigmoid(), nn.Tanh()]

    model = Generic(backbone=backbone, activations=activations)

    out = model(x)