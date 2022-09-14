import torch
from torch import Tensor
import torch.nn as nn


def drop_path(x: Tensor, drop_prob: float = 0.0, training: bool = False) -> Tensor:
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    From TIMM

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x

    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x: Tensor) -> Tensor:
        """
        Applies Stochastic Depth per sample (when applied in main path of residual blocks).

        :param x: Tensor of shape [B, C, ...]
        :return: Tensor of shape [B, C, ...]
        """
        return drop_path(x, self.drop_prob, self.training)
