import torch
from torch import Tensor
from typing import Tuple, List, Optional


@torch.jit.script
def _crop3d(img: Tensor, x: int, y: int, z: int, w: int, h: int, d: int) -> Tensor:
    """
    torch scriptable function which crops an image

    :param img: torch.Tensor image of shape [*, X, Y, Z]
    :param x: x coord of crop box
    :param y: y coord of crop box
    :param z: z coord of crop box
    :param w: width of crop box
    :param h: height of crop box
    :param d: depth of crop box
    :return:
    """
    return img[..., x:x + w, y:y + h, z:z + d]

@torch.jit.script
def _crop2d(img: Tensor, x: int, y: int, w: int, h: int) -> Tensor:
    """
    torch scriptable function which crops an image

    :param img: torch.Tensor image of shape [*, X, Y]
    :param x: x coord of crop box
    :param y: y coord of crop box
    :param w: width of crop box
    :param h: height of crop box
    :return:
    """

    return img[..., x:x + w, y:y + h]

def crop_to_identical_size(a: Tensor, b: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Crops Tensor a to the shape of Tensor b, then crops Tensor b to the shape of Tensor a.

    :param a: torch.Tensor with ndim 5, or ndim 4
    :param b: torch.Tensor with ndim equal to a
    :return: a, and b, each with identical shape
    """

    if a.ndim != b.ndim: raise RuntimeError(f'Cannot crop two tensors of differing dimensions {a.shape=}, {b.shape=}')

    if a.ndim == 5:
        a = _crop3d(a, x=0, y=0, z=0, w=b.shape[-3], h=b.shape[-2], d=b.shape[-1])
        b = _crop3d(b, x=0, y=0, z=0, w=a.shape[-3], h=a.shape[-2], d=a.shape[-1])
    elif a.ndim == 4:
        a = _crop3d(a, x=0, y=0, w=b.shape[-2], h=b.shape[-1])
        b = _crop3d(b, x=0, y=0, w=a.shape[-2], h=a.shape[-1])
    else:
        raise RuntimeError('Only supports tensors with minimum 3dimmensions and shape [..., X, Y, Z]')

    return a, b



def get_total_num_crops(image_shape: Tensor, crop_size: List[int], overlap: Optional[Tuple[int]]) -> int:
    total = 0

    for i, size in enumerate(crop_size):
        crop_size[i] = crop_size[i] if crop_size[i] < image_shape[i + 1] else image_shape[i + 1]
        # overlap[i] = overlap[i] if cropsize[i] < image_shape[i+1] else 0

    assert len(image_shape) - 1 == len(
        crop_size) == len(overlap), f'Image Shape must equal the shape of the crop.\n{image_shape}, {crop_size=}' \
                                        f'{overlap=}'
    dim = ['x', 'y', 'z'] if len(image_shape) == 3 else ['x', 'y']
    for c, o, d in zip(crop_size, overlap, dim):
        assert c - o*2 != 0, f'Overlap in {d} dimmension cannot be equal to or larger than crop size... {o*2=} < {c}'

    x = 0
    while x < image_shape[1]:
        _x = x if x + crop_size[0] <= image_shape[1] else image_shape[1] - crop_size[0]

        y = 0
        while y < image_shape[2]:
            _y = y if y + crop_size[1] <= image_shape[2] else image_shape[2] - crop_size[1]

            if len(image_shape) == 4:
                z = 0
                while z < image_shape[3]:
                    total += 1
                    z += (crop_size[2] - (overlap[2] * 2))
            else:
                total += 1

            y += (crop_size[1] - (overlap[1] * 2))
        x += (crop_size[0] - (overlap[0] * 2))

    return total


def crops(image: Tensor,
          crop_size: List[int],
          overlap: Optional[Tuple[int]] = (0, 0, 0)) -> Tuple[Tensor, List[int]]:
    """
    Generator which takes an image and sends out crops of a certain size with overlap pixels

    Shapes:
        -image: :math:`(C, X, Y, Z) or (C, X, Y)`
        -yeilds: :math:`(B, C, X, Y, Z) or (B, C, X, Y)`

    :param image: 4D torch.Tensor of shape [C, X, Y, Z] or 3D torch.Tensor of shape [C, X, Y]
    :param crop_size: Spatial dims of the resulting crops [X, Y, Z] or [X, Y]
    :param overlap: Overlap between each crop with identical shape as crop_size

    :yields: Crop of the image, and the indicies of the crop
    """

    image_shape = image.shape  # C, X, Y, Z

    for i, size in enumerate(crop_size):
        crop_size[i] = crop_size[i] if crop_size[i] < image_shape[i + 1] else image_shape[i + 1]


    assert len(image_shape) - 1 == len(
        crop_size) == len(overlap), f'Image Shape must equal the shape of the crop.\n{image.shape=}, {crop_size=}' \
                                        f'{overlap=}'

    dim = ['x', 'y', 'z'] if len(image_shape) == 3 else ['x', 'y']
    for c, o, d in zip(crop_size, overlap, dim):
        assert c - (o*2) != 0, f'Overlap in {d} dimmension cannot be equal to or larger than crop size... {c=} - {o*2=} = {c - (o*2)} < {c}'

    # for i in range(image_shape[1] // cropsize[1] + 1):

    x = 0
    while x < image_shape[1]:
        _x = x if x + crop_size[0] <= image_shape[1] else image_shape[1] - crop_size[0]

        y = 0
        while y < image_shape[2]:
            _y = y if y + crop_size[1] <= image_shape[2] else image_shape[2] - crop_size[1]

            if len(image_shape) == 4:
                z = 0
                while z < image_shape[3]:
                    _z = z if z + crop_size[2] <= image_shape[3] else image_shape[3] - crop_size[2]

                    yield image[:, _x:_x + crop_size[0], _y:_y + crop_size[1], _z:_z + crop_size[2]].unsqueeze(0), [_x, _y, _z]
                    z += (crop_size[2] - (overlap[2] * 2))
            else:
                yield image[:, _x:_x + crop_size[0], _y:_y + crop_size[1]].unsqueeze(0), [_x, _y]

            y += (crop_size[1] - (overlap[1] * 2))
        x += (crop_size[0] - (overlap[0] * 2))