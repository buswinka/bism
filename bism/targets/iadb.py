import random
from typing import *

import torch
import torch.nn as nn
from torch import Tensor
from yacs.config import CfgNode


class IADBTarget(nn.Module):
    def __init__(self):
        super(IADBTarget, self).__init__()

        self.cached_noise_tensor = None
        self.cached_alpha_tensor = None
        self.cached_out_tensor = None

        self._last_used_shape = None

        self._device = "cpu"
        self._dtype = torch.float

    def set_device(self, device):
        self._device = device
        return self

    def set_dtype(self, dtype):
        self._dtype = dtype
        return self

    @torch.no_grad()
    def forward(self, image: Tensor, masks: Tensor):

        self._last_used_shape = image.shape

        # Preallocate tensors for filling later. Saves on memory and is faster!!!
        if (
            self.cached_alpha_tensor is None
            or self.cached_alpha_tensor.shape != image.shape
        ):
            self.cached_alpha_tensor = torch.empty(
                self._last_used_shape,
                dtype=self._dtype,
                device=self._device,
                memory_format=torch.channels_last_3d,
            )

        if (
            self.cached_noise_tensor is None
            or self.cached_noise_tensor.shape != image.shape
        ):
            self.cached_noise_tensor = torch.empty(
                self._last_used_shape,
                dtype=self._dtype,
                device=self._device,
                memory_format=torch.channels_last_3d,
            )

        if (
            self.cached_out_tensor is None
            or self.cached_out_tensor.shape[1] != image.shape[1] - 1
        ):
            b, c, x, y, z = image.shape
            self.cached_out_tensor = torch.empty(
                (b, c + 1, x, y, z),
                dtype=self._dtype,
                device=self._device,
                memory_format=torch.channels_last_3d,
            )

        # Prevents new memory allocation by setting out
        torch.randn(
            self._last_used_shape,
            out=self.cached_noise_tensor,
            dtype=self._dtype,
            device=self._device,
        )

        for b in range(self._last_used_shape[0]):
            self.cached_alpha_tensor[b, ...].fill_(random.random())  # random.random()

        torch.cat(
            (
                self.cached_noise_tensor.mul(1 - self.cached_alpha_tensor)
                + image.mul(self.cached_alpha_tensor),
                self.cached_alpha_tensor,
            ),
            out=self.cached_out_tensor,
            dim=1,
        )
        return (
            self.cached_out_tensor,
            self.cached_noise_tensor,
            masks.to(self._dtype, memory_format=torch.channels_last_3d),
        )


# def iadb_target(
#     image: Tensor, masks: Tensor, cfg: CfgNode
# ) -> Tuple[Tensor, Tensor, Tensor]:
#     """
#     Creates a diffused image from an input and a random alpha
#
#     concatenates the alpha to the diffused image
#
#     Configuration:
#         - currently does not use cfg
#
#     Shapes:
#         - image: :math:`(B, C=1, X, Y, Z)`
#         - returns: :math:` (B, C=2, X, Y, Z)`, :math:`(B, C=1, X, Y, Z)`
#
#     :param instance_mask: input instance mask...
#     :param cfg: YACS Cfg Node....
#     :return: diffused image: Tensor, noise: Tensor
#     """
#     # raise RuntimeError(
#     #     """
#     #     1. Convert everything to "channels-last-3d"
#     #     2. convert merged_transform to a class with cached hyperparams
#     #     3. turn iadb target to a class with cached memory (just fill noise when needed)
#     #     4. avoid casting
#     #
#     #
#     #     """
#     # )
#
#     b, c, x, y, z = image.shape
#     noise = torch.randn(
#         (b, c, x, y, z),
#         dtype=torch.half,
#         device=image.device,
#         memory_format=torch.channels_last_3d,
#     )
#
#     alpha = torch.rand(
#         (b, 1, 1, 1, 1),
#         device=image.half,
#         dtype=torch.float16,
#         memory_format=torch.channels_last_3d,
#     )
#     out = noise.mul(1 - alpha) + image.mul(alpha)
#
#     alpha = torch.ones_like(image).mul_(alpha)
#     out = torch.cat((out, alpha), dim=1)
#
#     return out, noise, masks.half()
