import skimage.io as io
import numpy as np
import torch
from torch import Tensor
from typing import Optional
import logging

def imread(image_path: str,
           pin_memory: Optional[bool] = False, ndim=3) -> Tensor:
    """
    Imports an image from file and returns in torch format

    :param image_path: path to image
    :param pin_memory: saves torch tensor in pinned memory if true
    :return:
    """
    logging.info(f"Loading image from path: {image_path}")
    image: np.array = io.imread(image_path)  # [Z, X, Y, C]

    logging.debug(f"Loaded image with shape: {image.shape}")
    if ndim==3:
        logging.debug(f"Assuming image with shape {image.shape} as 3D image")

        image: np.array = image[..., np.newaxis] if image.ndim == 3 else image
        image: np.array = image.transpose(-1, 1, 2, 0)
        image: np.array = image[[2], ...] if image.shape[0] > 3 else image  # [C=1, X, Y, Z]

        image: Tensor = torch.from_numpy(image)
    else:
        logging.debug(f"Assuming image with shape {image.shape} as 2D image")

        image: np.array = image.transpose(-1, 0, 1)
        image: Tensor = torch.from_numpy(image)

    if pin_memory:
        image: Tensor = image.pin_memory()

    logging.debug(f"Returning loaded image with: {image.shape=}, {image.dtype=}")
    return image