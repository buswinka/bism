from bism.targets.local_shape_descriptors import lsd
from bism.targets.affinities import affinities
import torch
from torch import Tensor
from yacs.config import CfgNode

def mtlsd(instance_mask: Tensor, cfg: CfgNode):
    """
    A single target for predicting both LSD's and Affinities in the same go

    :param instance_mask:
    :param cfg:
    :return:
    """

    _lsd = lsd(instance_mask, cfg)
    _aff = affinities(instance_mask, cfg)

    return torch.concatenate((_lsd, _aff), dim=1)
