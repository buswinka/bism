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

    def forward(self, *args) -> Tensor:
        """
        Forward pass applying each activation to each channel.

        :param x: input tensor
        :return: output of backbone with activations applied
        """

        x: Tensor = self.backbone(*args)

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
    from bism.backends.unet_conditional_difusion import UNet_SPADE_3D
    x = torch.rand((1, 2, 300, 300, 20)).cuda()
    y = torch.rand((1, 1, 300, 300, 20)).cuda()
    backbone = UNet_SPADE_3D(in_channels=2, out_channels=1, mask_channels=1)
    activations = [nn.Tanh()]


    model = torch.compile(Generic(backbone=backbone, activations=activations).cuda(), mode='max-autotune')
    out = model(x, y)