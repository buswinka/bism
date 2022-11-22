import torch
from torch import Tensor
import torch.nn.functional as F
from typing import List, Tuple, Optional, Union
import torch.nn as nn
import math


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, num_features, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (num_features,)

        # small optimization. Avoids repeated branching in forward call...
        self.norm_functon = F.layer_norm if self.data_format == 'channels_last' else self.layer_norm_channels_fist

    def forward(self, x: Tensor) -> Tensor:
        return self.norm_functon(x, self.normalized_shape, self.weight, self.bias, self.eps)

    @staticmethod
    def layer_norm_channels_fist(x: Tensor, shape: Tuple[int], weight: Tensor, bias: Tensor, eps: float) -> Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        # x = (x - u).div(torch.sqrt(s + eps))

        # new_shape = [1 for _ in range(x.ndim)]
        # new_shape[1] = -1

        return weight.reshape((1, -1, 1, 1, 1)) * (x - u).div(torch.sqrt(s + eps)) + bias.reshape((1, -1, 1, 1, 1))
