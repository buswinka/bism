from typing import Tuple, List, Dict, Optional

import torch
from torch import Tensor
import torch.nn.functional as F

from yacs.config import CfgNode


def make_gaussian_kernel_1d(sigma: float, device: torch.device):
    """
    Generates a one dimmensional guassian kernel based on a single parameter - sigma.
    The kernel size will be calculated at: 2 * (3.0 * sigma + 0.5) + 1

    This implmentation was heavily adapted from scipy.ndimage

    :param sigma:
    :param radius:
    :return:
    """
    TRUNCATE = 3.0
    radius = int(TRUNCATE * sigma + 0.5)

    sigma2 = sigma * sigma
    x: Tensor = torch.arange(-radius, radius + 1, device=device, dtype=torch.float64)
    kernel: Tensor = torch.exp(-0.5 / sigma2 * x ** 2)

    # Normalize to total
    kernel = kernel / kernel.sum()

    return kernel


def __outer_product(input_array: Tensor) -> Tensor:
    """
    Computes the unique values of the outer products of the first dim (coord dim) for input.
    Implementation adapted from original implementation: https://github.com/funkelab/lsd
    but in torch...

    Internally converts the input to float64 for numerical precision. Therefore, this function takes a lot
    of memory to run, and may cause issues with cuda.

    Shapes:
        - input_array: :math:`(C, Z, X, Y)`
        - returns: :math:`(C*C, Z, X, Y)`

    :param: input array
    :returns: torch.float64 outer product of input array
    """

    c = input_array.shape[0]  # number of channels
    _high_precision_input_array = input_array.to(torch.float64)
    outer = torch.einsum("i...,j...->ij...", _high_precision_input_array, _high_precision_input_array)
    out = outer.reshape((c ** 2,) + input_array.shape[1:])
    return out


def __get_stats(coords: Tensor, mask: Tensor,
                sigma_voxel: Tuple[float, ...],
                sigma: Tuple[float, ...]) -> Tensor:
    """
    Function computes unscaled shape statistics.
    Credit goes to Jan and Arlo: https://github.com/funkelab/lsd

    The calculated statistics are as follows:
        - Stats [0, 1, 2] are for the mean offset
        - Stats [3, 4, 5] are the variance
        - Stats [6, 7, 8] are the pearson covariance
        - Stats [9] is the distance

    None are normalized!

    Shapes:
        - coords: :math:`(3, Z_{in}, X_{in}, Y_{in})`
        - mask: :math:`(Z_{in}, X_{in}, Y_{in})`
        - sigma_voxel: :math:`(3)`
        - sigma: :math:`(3)`
        - returns: :math:`(10, Z_{in}, X_{in}, Y_{in})`

    :param coords: Meshgrid of indicies
    :param mask: torch.int instance segmentation mask
    :param sigma_voxel: sigma / voxel for each dim
    :param sigma: standard deviation for bluring at each spatial dim
    :return: Statistics for each instance of the instance segmentation mask
    """

    assert coords.ndim == 4
    assert mask.ndim == 3
    assert coords.device == mask.device

    masked_coords: Tensor = coords * mask

    count: Tensor = __aggregate(mask, sigma_voxel)

    count[count == 0] = 1  # done for numerical stability.

    n_spatial_dims = len(count.shape)  # should always be 3

    mean: List[Tensor] = [__aggregate(masked_coords[d], sigma_voxel).unsqueeze(0) for d in range(n_spatial_dims)]
    mean: Tensor = torch.concatenate(mean, dim=0).div(count)

    mean_offset: Tensor = mean - coords

    # covariance
    coords_outer: Tensor = __outer_product(masked_coords)  # [9, X, Y, Z]

    entries = [0, 4, 8, 1, 2, 5]
    covariance: Tensor = torch.concatenate([__aggregate(coords_outer[d], sigma_voxel).unsqueeze(0) for d in entries],
                                           dim=0).float()

    # print(f'{covariance.isnan().sum()=}, {covariance.div(count).isnan().sum()=}, {count.min()=}, {__outer_product(mean)[entries].isnan().sum()=}')
    # covariance.div_(count).sub_()  # in place for memory
    covariance /= count
    covariance -= __outer_product(mean)[entries]

    # print('covariance nan: ', covariance.isnan().sum(), covariance.dtype, __outer_product(mean)[entries].max())

    variance = covariance[[0, 1, 2], ...]

    # Pearson coefficients of zy, zx, yx
    pearson = covariance[[3, 4, 5], ...]

    # normalize Pearson correlation coefficient
    variance[variance < 1e-3] = 1e-3  # numerical stability

    pearson[0, ...] /= torch.sqrt(variance[0, ...] * variance[1, ...])
    pearson[1, ...] /= torch.sqrt(variance[0, ...] * variance[2, ...])
    pearson[2, ...] /= torch.sqrt(variance[1, ...] * variance[2, ...])

    # normalize variances to interval [0, 1]
    variance[0, ...] /= sigma[0] ** 2
    variance[1, ...] /= sigma[1] ** 2
    variance[2, ...] /= sigma[2] ** 2

    return torch.concatenate((mean_offset, variance, pearson, count.unsqueeze(0)), dim=0)


def __aggregate(array: Tensor, sigma: Tuple[float, ...]) -> Tensor:
    """
    Performs a 3D gaussian blur on the input tensor using repeated 1D convolutions.
    Needs to run on 64bit floats in order for numerical identity with the original LSD paper.

    This should give identical results to the original LSD implementation... should....
    But it all runs in torch and on the GPU.

    Credit goes to Jan and Arlo: https://github.com/funkelab/lsd
    Credit also goes to SCIPY NDIMAGE

    Shapes:
        - array: :math:`(Z_{in}, X_{in}, Y_{in})`
        - sigma: :math: `(3)`
        - returns :math:`(Z_{in}, X_{in}, Y_{in})`

    :param array: input array to blur
    :param sigma: tuple of standard deviations for each dimension
    :return: blurred array
    """

    assert array.ndim == 3, 'Array must be 3D with shape (Z, X, Y)'
    assert len(sigma) == 3, 'Must provide 3 sigma values, one for each spatial dimension'

    z, x, y = array.shape
    dtype = array.dtype

    array = array.reshape(1, 1, z, x, y).to(torch.float64)  ## need this for numerical precision
    device = array.device

    # Separable 1D convolution
    for i in range(3):
        # 3d convolution need 5D tensor (B, C, X, Y, Z)
        kernel: Tensor = make_gaussian_kernel_1d(sigma=sigma[i], device=device).view(1, 1, -1, 1, 1)
        kernel = kernel.to(torch.float64)
        pad: Tuple[int] = tuple(int((k - 1) // 2) for k in kernel.shape[2::])

        array = F.conv3d(array, kernel, stride=1, padding=pad)
        array = array.permute(0, 1, 4, 2, 3)  # shuffle the tensor dims to ap

    return array.squeeze(0).squeeze(0).to(dtype)


@torch.jit.ignore()
def instances_to_lsd(segmentation: Tensor, sigma: Tuple[float, float, float], voxel_size: Tuple[int, int, int]):
    """
    Pytorch reimplementation of local-shape-descriptors without gunpowder.
    Credit goes to Jan and Arlo: https://github.com/funkelab/lsd

    Never downsamples, always computes the lsd's for every label. Uses a Guassian instead of sphere.

    Base implementation assumes numpy ordering (Z, X, Y), therefore all code uses this ordering, however we
    expect inputs to be in the form (1, X, Y, Z) and outputs to be in the form: (10, X, Y, Z)

    Shapes:
        - segmentation: (1, X, Y, Z)
        - sigma: (3)
        - voxel_size: (3)
        - returns: (C=10, X, Y, Z)

    :param segmentation:  label array to compute the local shape descriptors for
    :param sigma: The radius to consider for the local shape descriptor.
    :param voxel_size: anisotropy of the (X, Y, Z) dimension of the input array

    :return: local shape descriptors
    """

    # convert from pytorch channel ordering, to numpy channel ordering... (1, X, Y, Z) -> (Z, X, Y)
    segmentation = segmentation.squeeze(0).permute(2, 0, 1)
    voxel_size: Tuple[float, float, float] = (voxel_size[2], voxel_size[0], voxel_size[1])
    sigma: Tuple[float, float, float] = (sigma[2], sigma[0], sigma[1])

    # store device of input tensor
    device = segmentation.device

    shape = segmentation.shape
    labels = torch.unique(segmentation)

    descriptors = torch.zeros((10, shape[0], shape[1], shape[2]), dtype=torch.float, device=device)
    sigma_voxel = [s / v for s, v in zip(sigma, voxel_size)]

    # Grid of indexes for computing the descriptors. Can be cached, we dont.
    grid = torch.meshgrid(
        torch.arange(0, shape[0] * voxel_size[0], voxel_size[0], device=device),
        torch.arange(0, shape[1] * voxel_size[1], voxel_size[1], device=device),
        torch.arange(0, shape[2] * voxel_size[2], voxel_size[2], device=device),
        indexing='ij')
    grid = [g.unsqueeze(0) for g in grid]
    grid = torch.concatenate(grid, dim=0)

    for label in labels:  # do this for each instance
        if label == 0: continue

        mask = (segmentation == label).float()
        descriptor: Tensor = __get_stats(coords=grid, mask=mask, sigma_voxel=sigma_voxel, sigma=sigma)
        descriptors.add_(descriptor * mask)

    max_distance = torch.tensor(sigma, dtype=torch.float, device=device)

    # correct descriptors for proper scaling
    descriptors[[0, 1, 2], ...] = (
            descriptors[[0, 1, 2], ...] / max_distance[:, None, None, None] * 0.5
            + 0.5
    )
    # pearsons in [0, 1]
    descriptors[[6, 7, 8], ...] = descriptors[[6, 7, 8], ...] * 0.5 + 0.5

    # reset background to 0
    descriptors[[0, 1, 2, 6, 7, 8], ...] *= segmentation != 0

    # Clamp to reasonable values
    torch.clamp(descriptors, 0, 1, out=descriptors)

    return descriptors.permute(0, 2, 3, 1)


def lsd(instance_mask: Tensor, cfg: CfgNode):
    """
    Called directly by training function. Makes it explicit that we're calling on a BATCHED
    tensor, not just a random 3d image.

    Configuration:
        - cfg.TARGET.LSD.SIGMA: sigma of the lsd calculation algorithm, defualt (8, 8, 8)
        - cfg.TARGET.LSD.VOXEL_SIZE: voxel anisotropy of the input image, defualt (1, 1, 5)

    Shapes:
        - instance_mask: :math:`(B, C=1, X, Y, Z)`
        - returns: :math:`(B, C=3, X, Y, Z)`

    :param instance_mask: instance mask
    :param cfg: a configuration file defining a training experiment
    :return: Local shape descriptors
    """
    assert instance_mask.ndim == 5

    b, c, x, y, z = instance_mask.shape
    device = instance_mask.device
    out = torch.empty((b, 10, x, y, z), dtype=torch.float32, device=device)
    for i in range(b):
        out[i, ...] = instances_to_lsd(instance_mask[i, 0, ...].to(torch.float32),
                                       cfg.TARGET.LSD.SIGMA,
                                       cfg.TARGET.LSD.VOXEL_SIZE)

    return out.half()


if __name__ == "__main__":
    from scipy.ndimage import gaussian_filter

    _size = 50
    a = torch.rand((_size ** 3)).reshape((_size, _size, _size)).mul(2).float()
    print(a)

    b = torch.from_numpy(gaussian_filter(a.numpy(), sigma=(8, 8, 8), truncate=3.0, mode="constant", cval=0.0))
    print(b)

    c = __aggregate(a, sigma=(8, 8, 8))
    print(b)

    # print(b - c)
    # print(c)
    # print(b.flatten(), '\n', c.flatten(), '\n', b.sub(c))

    print(f'Max Difference: {b.sub(c).abs().max()}')
    assert torch.allclose(b, c)
    # print(f'Max Difference: {b.sub(c).abs().max()}')
