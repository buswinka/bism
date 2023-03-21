import torch
from torch import Tensor
import numpy as np


def mknhood3d(radius: int=1, device='cpu'):
    # Makes nhood structures for some most used dense graphs.
    # Janelia pyGreentea: https://github.com/naibaf7/PyGreentea

    # The neighborhood reference for the dense graph representation we use
    # nhood(1,:) is a 3 vector that describe the node that conn(:,:,:,1) connects to
    # so to use it: conn(23,12,42,3) is the edge between node [23 12 42] and [23 12 42]+nhood(3,:)
    # See? It's simple! nhood is just the offset vector that the edge corresponds to.

    ceilrad = torch.ceil(torch.tensor(radius, device=device))
    x = torch.arange(-ceilrad, ceilrad+1, 1)
    y = torch.arange(-ceilrad, ceilrad+1, 1)
    z = torch.arange(-ceilrad, ceilrad+1, 1)
    [i, j, k] = torch.meshgrid(z, y, x, indexing='xy')

    idxkeep = (i**2+j**2+k**2) <= radius**2
    i = i[idxkeep].ravel()
    j = j[idxkeep].ravel()
    k = k[idxkeep].ravel()
    zeroIdx = torch.tensor(len(i) // 2, device=device).int()

    nhood = torch.vstack(
        (k[:zeroIdx], i[:zeroIdx], j[:zeroIdx])).T.int()
    return torch.flipud(nhood).contiguous()


def seg_to_aff(instance_mask, nhood: Tensor = None, pad: str = 'replicate'):
    """
    Calculates affinities from an instance segmentation mask.
    Adapted from pytorch connectomics.

    Shapes:
        - seg: (X, Y, Z)
        - returns: (3, X, Y, Z)

    :param instance_mask: instance segmentation mask
    :param nhood: neighborhood tensor
    :param pad: padding method
    :return: Affinities
    """
    nhood: Tensor = nhood if nhood is not None else mknhood3d(1, instance_mask.device)

    x, y, z = instance_mask.shape
    n_dim = nhood.shape[0]
    aff = torch.zeros((n_dim, x, y, z), device=nhood.device)

    for n in range(n_dim):

        _left = instance_mask[max(0, -nhood[n, 0]):min(x, x - nhood[n, 0]),
             max(0, -nhood[n, 1]):min(y, y-nhood[n, 1]),
             max(0, -nhood[n, 2]):min(z, z-nhood[n, 2])]

        _right = instance_mask[max(0, nhood[n, 0]):min(x, x + nhood[n, 0]),
             max(0, nhood[n, 1]):min(y, y+nhood[n, 1]),
             max(0, nhood[n, 2]):min(z, z+nhood[n, 2])]

        aff[n,
        max(0, -nhood[n, 0]):min(x, x-nhood[n, 0]),
        max(0, -nhood[n, 1]):min(y, y-nhood[n, 1]),
        max(0, -nhood[n, 2]):min(z, z-nhood[n, 2])] = _left.eq(_right) * _left.gt(0) * _right.gt(0)

    if pad == 'replicate':  # pad the boundary affinity
        aff[0, 0] = (instance_mask[0] > 0).to(aff.dtype)
        aff[1, :, 0] = (instance_mask[:, 0] > 0).to(aff.dtype)
        aff[2, :, :, 0] = (instance_mask[:, :, 0] > 0).to(aff.dtype)

    return aff


if __name__=='__main__':
    import skimage.io as io

    inst_mask = io.imread('/Users/chrisbuswinka/Dropbox (Partners HealthCare)/skoots-experiments/data/hair-cell/chris/train/image-1.labels.tif')
    inst_mask = torch.from_numpy(inst_mask)
    inst_mask = inst_mask

    aff: Tensor = seg_to_aff(inst_mask, mknhood3d(1)).permute(0, 2, 3, 1)

    aff = aff.float().permute(3, 1, 2, 0).mul(255).round().int()

    io.imsave('/Users/chrisbuswinka/Desktop/afftest.tif', aff.cpu().numpy())
