from torch import Tensor
from yacs.config import CfgNode
from typing import *


def semantic(instance_mask: Tensor, cfg: CfgNode) -> Tuple[Tensor, Tensor]:
    """
    Auto Context LSD Target. A single target for predicting both LSD's and Affinities, instead of concatenating, returns
    each in a tuple...

    Configuration:
        - cfg.TARGET.SEMANTIC.THR: threshold for semantic mask generation

    Shapes:
        - instance_mask: :math:`(B, C=1, X, Y, Z)`
        - returns: :math:` (B, C=1, X, Y, Z)`

    :param instance_mask: input instance mask...
    :param cfg: YACS Cfg Node....
    :return: Tensor, returns input image with all pixels > 0.5 set to one
    """

    return instance_mask.gt(cfg.TARGET.SEMANTIC.THR).float()
