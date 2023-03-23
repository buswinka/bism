import torch
from torch import Tensor
from bism.utils.morphology import binary_erosion
from yacs.config import CfgNode
import numpy as np


def _make_neighborhood(radius: int = 1, device='cpu'):
    """
    Creates a matrix of pixel connected components.

    :param radius: distance of connected components
    :param device: torch device to return neighborhood on

    :return: neighborhood matrix
    """

    radius: Tensor = torch.ceil(torch.tensor(radius, device=device))  # round up always

    x = torch.arange(-radius, radius + 1, 1, device=device)
    y = torch.arange(-radius, radius + 1, 1, device=device)
    z = torch.arange(-radius, radius + 1, 1, device=device)

    [i, j, k] = torch.meshgrid(z, y, x, indexing='xy')

    ind = (i ** 2 + j ** 2 + k ** 2) <= radius ** 2  # only keep indices within radius
    i = i[ind].ravel()
    j = j[ind].ravel()
    k = k[ind].ravel()

    zero_index = torch.tensor(len(i) // 2, device=device).int()

    neighborhood = torch.vstack(
        (k[:zero_index], i[:zero_index], j[:zero_index])).T.int()

    return torch.flipud(neighborhood).contiguous()


def instances_to_affinities(instance_mask: Tensor,
                            neighborhood_radius: int = None,
                            pad: str = 'replicate') -> Tensor:
    """
    Calculates affinities from an instance segmentation mask.
    Adapted from pytorch connectomics.

    Shapes:
        - instance_mask: (X, Y, Z)
        - returns: (3, X, Y, Z)

    :param instance_mask: instance segmentation mask
    :param neighborhood_radius: distance of connected components
    :param pad: padding method
    :return: affinities of input instance_mask
    """
    neighborhood_radius: int = neighborhood_radius if neighborhood_radius is not None else 1
    neighborhood: Tensor = _make_neighborhood(neighborhood_radius, instance_mask.device)

    x, y, z = instance_mask.shape
    n_dim = neighborhood.shape[0]
    affinities = torch.zeros((n_dim, x, y, z), device=neighborhood.device)

    for n in range(n_dim):
        _left = instance_mask[max(0, -neighborhood[n, 0]):min(x, x - neighborhood[n, 0]),
                max(0, -neighborhood[n, 1]):min(y, y - neighborhood[n, 1]),
                max(0, -neighborhood[n, 2]):min(z, z - neighborhood[n, 2])]

        _right = instance_mask[max(0, neighborhood[n, 0]):min(x, x + neighborhood[n, 0]),
                 max(0, neighborhood[n, 1]):min(y, y + neighborhood[n, 1]),
                 max(0, neighborhood[n, 2]):min(z, z + neighborhood[n, 2])]

        affinities[n,
        max(0, -neighborhood[n, 0]):min(x, x - neighborhood[n, 0]),
        max(0, -neighborhood[n, 1]):min(y, y - neighborhood[n, 1]),
        max(0, -neighborhood[n, 2]):min(z, z - neighborhood[n, 2])] = _left.eq(_right) * _left.gt(
            0) * _right.gt(0)

    if pad == 'replicate':  # pad the boundary affinity
        affinities[0, 0] = (instance_mask[0] > 0).to(affinities.dtype)
        affinities[1, :, 0] = (instance_mask[:, 0] > 0).to(affinities.dtype)
        affinities[2, :, :, 0] = (instance_mask[:, :, 0] > 0).to(affinities.dtype)

    return affinities


def affinities(instance_mask: Tensor, cfg: CfgNode):
    """
    A target function for prediction affinities from an instance mask.

    Configuration:
        - cfg.TARGET.AFFINITIES.N_ERODE
        - cfg.TARGET.AFFINITIES.PAD
        - cfg.TARGET.AFFINITIES.NHOOD

    Shapes:
        - instance_mask: :math:`(B, C=1, X, Y, Z)`
        - returns: :math:`(B, C=3, X, Y, Z)`

    :param instance_mask: (B, 1, X, Y, Z) instance mask
    :param cfg: a configuration file defining a training experiment
    :return: (B, 3, X, Y, Z) Affinities
    """
    assert instance_mask.ndim == 5

    b, c, x, y, z = instance_mask.shape
    device = instance_mask.device
    out = torch.empty((b, 3, x, y, z), dtype=torch.float, device=device)

    # Perform this on each batch!
    for i in range(b):

        # erode each instance...
        _mask = torch.zeros((x, y, z), device=device)
        for u in torch.unique(instance_mask[i, 0, ...]):
            if u == 0: continue
            for _ in range(cfg.TARGET.AFFINITIES.N_ERODE):
                _mask += binary_erosion(instance_mask[i, 0, ...].view(1, 1, x, y, z).eq(u).float()).view(x, y, z)

        instance_mask[i, 0, ...] *= _mask.eq(cfg.TARGET.AFFINITIES.N_ERODE)
        out[i, ...] = instances_to_affinities(instance_mask[i, 0, ...], cfg.TARGET.AFFINITIES.NHOOD, cfg.TARGET.AFFINITIES.PAD)

    return out


if __name__ == '__main__':
    import skimage.io as io

    inst_mask = io.imread(
        '/Users/chrisbuswinka/Dropbox (Partners HealthCare)/skoots-experiments/data/hair-cell/chris/train/image-1.labels.tif')
    inst_mask = torch.from_numpy(inst_mask)
    inst_mask = inst_mask

    aff: Tensor = instances_to_affinities(inst_mask, _make_neighborhood(1)).permute(0, 2, 3, 1)

    aff = aff.float().permute(3, 1, 2, 0).mul(255).round().int()

    io.imsave('/Users/chrisbuswinka/Desktop/afftest.tif', aff.cpu().numpy())
