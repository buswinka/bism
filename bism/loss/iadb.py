import torch
from torch import Tensor
import torch.nn as nn

# based on iterative alpha-deblending: a Minimalist deterministic diffusion model

class iadb(nn.Module):
    def __init__(self):
        super(iadb, self).__init__()

    def forward(self, out: Tensor, image: Tensor, noise: Tensor) -> Tensor:
        """
        very simple loss - simply sums the distance from the outpu tof the network to the image minus noise

        Shapes:
            - out: [B, 1, X, Y, Z]
            - image: [B, 1, X, Y, Z]
            - noise: [B, 1, X, Y, Z]

        :param out: output of network
        :param image: original image
        :param noise: noise used as fully diffused image

        :return: loss val
        """

        return torch.sum((out - (image - noise)) ** 2)
