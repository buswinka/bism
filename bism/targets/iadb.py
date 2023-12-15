from typing import *

import torch
from torch import Tensor
from yacs.config import CfgNode


@torch.no_grad()
def iadb_target(image: Tensor, masks: Tensor, cfg: CfgNode) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Creates a diffused image from an input and a random alpha

    concatenates the alpha to the diffused image

    Configuration:
        - currently does not use cfg

    Shapes:
        - image: :math:`(B, C=1, X, Y, Z)`
        - returns: :math:` (B, C=2, X, Y, Z)`, :math:`(B, C=1, X, Y, Z)`

    :param instance_mask: input instance mask...
    :param cfg: YACS Cfg Node....
    :return: diffused image: Tensor, noise: Tensor
    """
    raise RuntimeError(
        """
        1. Convert everything to "channels-last-3d"
        2. convert merged_transform to a class with cached hyperparams
        3. turn iadb target to a class with cached memory (just fill noise when needed)
        4. avoid casting
        
        
        """
    )
    b, c, x, y, z = image.shape
    noise = torch.randn((b, c, x, y, z), dtype=torch.half, device=image.device, memory_format=torch.channels_last_3d)

    alpha = torch.rand((b, 1, 1, 1, 1), device=image.half, dtype=torch.float16, memory_format=torch.channels_last_3d)
    out = noise.mul(1 - alpha) + image.mul(alpha)

    alpha = torch.ones_like(image).mul_(alpha)
    out = torch.cat((out, alpha), dim=1)

    return out, noise, masks.half()