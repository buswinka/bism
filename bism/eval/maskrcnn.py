import logging
import os
import os.path
from typing import List, Tuple, Callable, Union, OrderedDict, Optional, Dict

import torch
import torch.nn as nn
from torch import Tensor
from bism.models.construct import cfg_to_bism_model, cfg_to_torchvision_model
from bism.utils.io import imread

import skimage.io as io

from matplotlib import colormaps
import random
import torchvision.utils


@torch.no_grad()
def eval(image_path: str, model_file: str, device: str):
    """
    Runs an affinity segmentation on an image.

    :param model_file: Path to a pretrained bism model
    :return:
    """
    logging.info(f"Loading model file: {model_file}")
    checkpoint = torch.load(model_file, map_location="cpu")
    cfg = checkpoint["cfg"]

    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        device = 'mps' if torch.backends.mps.is_available() else device


    logging.info(f"Constructing BISM model")
    base_model: nn.Module = cfg_to_torchvision_model(cfg)

    state_dict = (
        checkpoint
        if not "model_state_dict" in checkpoint
        else checkpoint["model_state_dict"]
    )
    base_model.load_state_dict(state_dict)
    base_model = base_model.to(device)

    model = base_model
    model = model.eval()

    filename_without_extensions = os.path.splitext(image_path)[0]

    logging.info(f"Loading image from path: {image_path}")
    image = imread(
        image_path, pin_memory=False, ndim=3 if "3d" in cfg.MODEL.BACKBONE else 2
    )

    c, x, y = image.shape
    logging.info(
        f"Loaded an image with shape: {(c, x, y)}, dtype: {image.dtype}, min: {image.min()}, max: {image.max()}"
    )

    scale: float = 2 ** 16 if image.max() > 255 else 255
    logging.debug(f"image scaled by scale: {scale}")

    image = image.div(scale).to(device)

    # torchvision model does not support tileing, so we just run the image straight through
    out: Dict[str, Tensor] = model([image])[0]

    rgb = image.mul(255).round().to(torch.uint8)
    cmaps = ['Blues', 'Greens', 'Reds', 'Purples', 'Oranges', 'Greys']
    for cmap, i in zip(cmaps, range(1, cfg.MODEL.OUT_CHANNELS + 1)):
        l: Tensor = out['labels'].cpu().eq(i)
        n_instances: int = l.sum()
        colors = [tuple(int(c * 255) for c in colormaps[cmap](random.random() * 0.7 + 0.3)[0:3]) for _ in range(n_instances)]
        rgb = torchvision.utils.draw_segmentation_masks(rgb, out['masks'][l, 0, ...].gt(0.5), colors=colors, alpha=0.33)

    logging.info(f"Saving Output to: {filename_without_extensions}_lsd_.trch")
    io.imsave(f'{filename_without_extensions}_mask_overlay.png', rgb.permute(1, 2, 0).cpu().numpy())
    torch.save(out, f'{filename_without_extensions}_output_dict.trch')

    masks = out['masks'].gt(0.5) * torch.arange(1, out['masks'].shape[0]+1).view(-1, 1, 1, 1)

    return out