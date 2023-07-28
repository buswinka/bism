import torch
from torch import Tensor
import torch.nn.functional as F
from typing import Dict, Tuple, Union, Sequence, List
from typing import Tuple

@torch.jit.script
def _compute_zero_padding(kernel_size: List[int]) -> List[int]:
    r"""Utility function that computes zero padding tuple.
    Adapted from Kornia
    """
    return [(k - 1) // 2 for k in kernel_size]

@torch.jit.script
def _get_binary_kernel3d(window_size: int, device: str) -> Tensor:
    r"""Creates a symmetric binary kernel to extract the patches. If the window size
    is HxWxD will create a (H*W)xHxW kernel.

    Adapted from a 2D Kornia implementation

    """
    window_range: int = int(window_size ** 3)
    kernel: Tensor = torch.zeros((window_range, window_range, window_range), device=device)
    for i in range(window_range):
        kernel[i, i, i] += 1.0
    kernel = kernel.view(-1, 1, window_size, window_size, window_size)

    # get rid of all zero kernels
    ind = torch.nonzero(kernel.view(kernel.shape[0], -1).sum(1))
    return kernel[ind[:, 0], ...]

@torch.jit.script
def _get_binary_kernel2d(window_size: int, device: str) -> Tensor:
    r"""Creates a symmetric binary kernel to extract the patches. If the window size
    is HxW will create a (H*W)xHxW kernel.

    Adapted from a 2D Kornia implementation
    """
    window_range: int = int(window_size ** 2)
    kernel: Tensor = torch.zeros((window_range, window_range), device=device)
    for i in range(window_range):
        kernel[i, i] += 1.0
    kernel = kernel.view(-1, 1, window_size, window_size)

    # get rid of all zero kernels
    ind = torch.nonzero(kernel.view(kernel.shape[0], -1).sum(1))
    return kernel[ind[:, 0], ...]


# re-implemented from torchvision.tensor.functional
def _get_gaussian_kernel1d(kernel_size: int, sigma: float) -> Tensor:
    ksize_half = (kernel_size - 1) * 0.5

    x = torch.linspace(-ksize_half, ksize_half, steps=kernel_size)
    pdf = torch.exp(-0.5 * (x / sigma).pow(2))
    kernel1d = pdf / pdf.sum()

    return kernel1d

# re-implemented from torchvision.tensor.functional
def _get_gaussian_kernel2d(
    kernel_size: List[int], sigma: List[float], dtype: torch.dtype, device: torch.device
) -> Tensor:
    kernel1d_x = _get_gaussian_kernel1d(kernel_size[0], sigma[0]).to(device, dtype=dtype)
    kernel1d_y = _get_gaussian_kernel1d(kernel_size[1], sigma[1]).to(device, dtype=dtype)
    kernel2d = torch.mm(kernel1d_y[:, None], kernel1d_x[None, :])
    return kernel2d

# expanded to 3D
def _get_gaussian_kernel3d(
        kernel_size: List[int], sigma: List[float], dtype: torch.dtype, device: torch.device
) -> Tensor:
    kernel1d_x = _get_gaussian_kernel1d(kernel_size[0], sigma[0]).to(device, dtype=dtype)
    kernel1d_y = _get_gaussian_kernel1d(kernel_size[1], sigma[1]).to(device, dtype=dtype)
    kernel1d_z = _get_gaussian_kernel1d(kernel_size[2], sigma[2]).to(device, dtype=dtype)

    kernel3d = (kernel1d_x[:, None] @ kernel1d_y[None, :]).unsqueeze(-1) @ kernel1d_z[None, :]
    return kernel3d


@torch.jit.script
def gauss_filter(input: Tensor, kernel: List[int], sigma: List[float]) -> Tensor:
    """
    gaussian filter of a 3D tensor

    :param input: (B, C, X, Y, Z)
    :param kernel: [int, int, int]
    :param sigma: [float, float, float]
    :return: blured image
    """
    padding: List[int] = _compute_zero_padding(kernel)
    kernel: Tensor = _get_gaussian_kernel3d(kernel, sigma, input.dtype, input.device)
    kernel = kernel.expand(input.shape[1], 1, kernel.shape[0], kernel.shape[1], kernel.shape[2])

    features: Tensor = F.conv3d(input, kernel, padding=padding, stride=(1, 1, 1), groups=input.shape[1])

    return features


def binary_erosion(image: Tensor) -> Tensor:
    """
    Performs binary erosion on a 5D Tensor.

    Shapes:
        - input: :math:`(B, C, X, Y, Z)`
        - output: :math:`(C, C, X, Y, Z)`


    :param image: binary image
    :return: eroded image
    """
    device = str(image.device)
    kernel = _get_binary_kernel3d(3, device)
    padding = _compute_zero_padding((3, 3, 3))

    b, c, h, w, d = image.shape
    # map the local window to single vector
    features: Tensor = F.conv3d(image.reshape(b * c, 1, h, w, d), kernel, padding=padding, stride=1)
    return features.min(dim=1)[0].unsqueeze(0)


@torch.jit.script
def binary_dilation(image: Tensor) -> Tensor:
    """
    Performs binary dilation on a 5D Tensor.

    Shapes:
        - input: :math:`(B, C, X, Y, Z)`
        - output: :math:`(C, C, X, Y, Z)`

    :param image: binary image
    :return: dilated image
    """
    padding: List[int] = _compute_zero_padding((3, 3, 3))
    kernel: Tensor = _get_binary_kernel3d(3, str(image.device))

    b, c, h, w, d = image.shape
    # map the local window to single vector
    features = F.conv3d(image.reshape(b * c, 1, h, w, d), kernel,
                        padding=padding, stride=1)
    return torch.max(features.view(b, c, -1, h, w, d), dim=2)[0]

def median_filter(input: Tensor) -> Tensor:
    padding: Tuple[int, int, int] = _compute_zero_padding((3, 3, 3))
    kernel: Tensor = _get_binary_kernel3d(3, input.dtype, input.device)
    b, c, h, w, d = input.shape
    # map the local window to single vector
    features: Tensor = F.conv3d(input.reshape(b * c, 1, h, w, d), kernel,
                                padding=padding, stride=1)
    return torch.median(features.view(b, c, -1, h, w, d), dim=2)[0]


def mean_filter(input: Tensor) -> Tensor:
    padding: Tuple[int, int, int] = _compute_zero_padding((3, 3, 3))
    kernel: Tensor = _get_binary_kernel3d(3, input.dtype, input.device)
    b, c, h, w, d = input.shape
    # map the local window to single vector
    features: Tensor = F.conv3d(input.reshape(b * c, 1, h, w, d), kernel,
                                padding=padding, stride=1)
    return torch.mean(features.view(b, c, -1, h, w, d), dim=2)[0]


def affinity_convolution3d(input: Tensor) -> Tensor:
    """
    returns the nearest neighbors along a new axis

    Shapes:
     - input: :math:`(B, C, X, Y, Z)`
     - returns: :math:`(B, C, 27, X, Y, Z)`

    """
    padding: List[int] = _compute_zero_padding((3, 3, 3))
    kernel: Tensor = _get_binary_kernel3d(3, input.dtype, input.device)
    b, c, h, w, d = input.shape
    # map the local window to single vector
    features: Tensor = F.conv3d(input.reshape(b * c, 1, h, w, d), kernel,
                                padding=padding, stride=1)
    return features.view(b, c, -1, h, w, d)

def _binary_convolution2d(input: Tensor,  padding_mode: str = 'constant') -> Tensor:
    """
    returns the nearest neighbors along a new axis

    padding_mode accepted values
    'constant', 'reflect', 'replicate' or 'circular'. Default: 'constant'

    Shapes:
     - input: :math:`(B, C, X, Y)`
     - returns: :math:`(B, C, 9, X, Y)`

    :param input:
    :param padding_mode:
    :return:
    """
    # Pad the image...
    padding: List[int] = _compute_zero_padding([3, 3])
    input = F.pad(input=input, pad=padding+padding, mode=padding_mode)

    kernel: Tensor = _get_binary_kernel2d(3, str(input.device))
    b, c, h, w = input.shape

    # map the local window to single vector
    features: Tensor = F.conv2d(input.reshape(b * c, 1, h, w), kernel,
                                padding=padding, stride=1)
    features = features.view(b, c, -1, h, w)

    padding_slice = slice(padding[0], -padding[0], 1)

    return features[..., padding_slice, padding_slice]


def _binary_convolution3d(input: Tensor,  padding_mode: str = 'constant') -> Tensor:
    """
    returns the nearest neighbors along a new axis

    padding_mode accepted values


    Shapes:
     - input: :math:`(B, C, X, Y, Z)`
     - returns: :math:`(B, C, 27, X, Y, Z)`

    :param input: input tensor
    :param padding_mode: one of the following: 'constant', 'reflect', 'replicate' or 'circular'. Default: 'constant'
    :return: affinities of the input tensor
    """
    # Pad the image...
    padding: List[int] = _compute_zero_padding([3, 3])
    input = F.pad(input=input, pad=padding+padding+padding, mode=padding_mode)

    kernel: Tensor = _get_binary_kernel3d(3, str(input.device))
    b, c, h, w, d = input.shape

    # map the local window to single vector
    features: Tensor = F.conv3d(input.reshape(b * c, 1, h, w, d), kernel,
                                padding=[1, 1, 1], stride=1)
    features = features.view(b, c, -1, h, w, d)

    padding_slice = slice(padding[0], -padding[0], 1)

    return features[..., padding_slice, padding_slice, padding_slice]


def binary_convolution(input: Tensor, padding_mode: str = 'constant') -> Tensor:
    """
    Returns the nearest neighbors of each pixel along a nex axis of a 2D or 3D tensor.

    Shapes:
     - input: :math:`(B, C, X, Y)` or :math:`(B, C, X, Y, Z)`
     - returns: :math:`(B, C, 9, X, Y)` or :math:`(B, C, 27, X, Y, Z)`

    :param input: input tensor
    :param padding_mode: one of the following: 'constant', 'reflect', 'replicate' or 'circular'. Default: 'constant'

    :return: affinities of the input tensor
    """

    if input.ndim == 4:
        out = _binary_convolution2d(input, padding_mode)
    elif input.ndim == 5:
        out = _binary_convolution3d(input, padding_mode)
    else:
        raise RuntimeError(f'Number of dimensions not supported: {input.shape}')

    return out




def dilate(input: Tensor) -> Tensor:
    padding: Tuple[int, int, int] = _compute_zero_padding((3, 3, 3))
    kernel: Tensor = _get_binary_kernel3d(3, input.dtype, input.device)
    b, c, h, w, d = input.shape
    # map the local window to single vector
    features = F.conv3d(input.reshape(b * c, 1, h, w, d), kernel,
                        padding=padding, stride=1)
    return torch.max(features.view(b, c, -1, h, w, d), dim=2)[0]
