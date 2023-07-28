from bism.targets.local_shape_descriptors import lsd
from bism.targets.affinities import affinities
import torch
from torch import Tensor
from yacs.config import CfgNode
from typing import *


def aclsd(instance_mask: Tensor, cfg: CfgNode) -> Tuple[Tensor, Tensor]:
    """
    Auto Context LSD Target. A single target for predicting both LSD's and Affinities, instead of concatenating, returns
    each in a tuple...

    Configuration:
        - cfg.TARGET.LSD.SIGMA: sigma of the lsd calculation algorithm, defualt (8, 8, 8)
        - cfg.TARGET.LSD.VOXEL_SIZE: voxel anisotropy of the input image, defualt (1, 1, 5)
        - cfg.TARGET.AFFINITIES.N_ERODE
        - cfg.TARGET.AFFINITIES.PAD
        - cfg.TARGET.AFFINITIES.NHOOD

    Shapes:
        - instance_mask: :math:`(B, C=1, X, Y, Z)`
        - returns: Tuple[LSD :math:` (B, C=3, X, Y, Z)` Affinities :math:`(B, C=3, X, Y, Z)`]

    :param instance_mask: input instance mask...
    :param cfg: YACS Cfg Node....
    :return: Tuple[Tensor, Tensor], (LSD's, Affinities)
    """

    _lsd = lsd(instance_mask, cfg)
    _aff = affinities(instance_mask, cfg)

    return _lsd, _aff
