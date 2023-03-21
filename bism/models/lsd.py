import torch
from torch import Tensor
import torch.nn as nn
from typing import List, Union, Tuple



class LSDModel(nn.Module):
    def __init__(self, backbone: nn.Module):
        """
        A model for predicting LSD's or Affinites for segmentation. Applies a sigmoid after the
        output of the backbone!

        :param backbone: nn.Module form bism.backends
        """
        super(LSDModel, self).__init__()

        self.backbone = backbone
        self.sigmoid = nn.Sigmoid()


    def forward(self, x: Tensor) -> Tensor:
        """ forward pass of model """

        x: Tensor = self.sigmoid(self.backbone(x))

        return x
