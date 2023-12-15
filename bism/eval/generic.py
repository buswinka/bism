import logging
import os
import os.path
from typing import List, Tuple, Callable, Union, OrderedDict, Optional, Dict

import bism.utils.cropping
import skimage.io as io
import torch
import torch.nn as nn
import torch.optim.swa_utils
from bism.models.construct import cfg_to_bism_model, cfg_to_torchvision_model
from bism.utils.io import imread
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

import tracemalloc
import time


@torch.no_grad()
def eval(image_path: str, model_file: str, device: str | None):
    """
    Executes a pretrained BISM model on an arbitrary image. Only should exectue if
    the train target determined by the configuration is 'generic', i.e. cfg.TRAIN.TARGET == 'generic'

    will silently save the output to {image_path}_out.tif

    :param image_path: path to image to analyze
    :param model_file: path to pretrained model file
    :param device: hardware accelerator

    :return: None
    """

    tracemalloc.start()
    start = time.time()

    logging.info(f"Loading model file: {model_file}")
    checkpoint = torch.load(model_file, map_location="cpu")
    cfg = checkpoint["cfg"]

    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        device = 'mps' if torch.backends.mps.is_available() else device

    logging.info(f"Constructing BISM model")

    base_model: nn.Module | List[nn.Module] = (
        cfg_to_bism_model(cfg)
        if "maskrcnn" not in cfg.MODEL.BACKBONE
        else cfg_to_torchvision_model(cfg)
    )

    state_dict = (
        checkpoint
        if not "model_state_dict" in checkpoint
        else checkpoint["model_state_dict"]
    )
    base_model.load_state_dict(state_dict)
    base_model = base_model.to(device)

    logging.info(f"Compiling BISM model with torch inductor")
    
    model = torch.compile(base_model) if 'cuda' in device else base_model
    
    testin = (
        torch.rand((1, cfg.MODEL.IN_CHANNELS, 300, 300, 20))
        if "3d" in cfg.MODEL.BACKBONE
        else torch.rand((1, cfg.MODEL.IN_CHANNELS, 300, 300))
    )

    _ = model(testin.to(device).float())

    filename_without_extensions = os.path.splitext(image_path)[0]

    logging.info(f"Loading image from path: {image_path}")
    image = imread(
        image_path, pin_memory=False, ndim=3 if "3d" in cfg.MODEL.BACKBONE else 2
    )

    if "3d" in cfg.MODEL.BACKBONE:
        logging.debug("assuming a 3d image from model construction")
        c, x, y, z = image.shape
    else:
        z = -1
        c, x, y = image.shape

    logging.info(
        f"Loaded an image with shape: {(c, x, y, z)}, dtype: {image.dtype}, min: {image.min()}, max: {image.max()}"
    )

    scale: float = 2**16 if image.max() > 255 else 255
    logging.debug(f"image scaled by scale: {scale}")

    modelout = (
        torch.zeros((cfg.MODEL.OUT_CHANNELS, x, y, z), dtype=torch.half)
        if "3d" in cfg.MODEL.BACKBONE
        else torch.zeros((cfg.MODEL.OUT_CHANNELS, x, y), dtype=torch.half)
    )
    cropsize = (
        [
            cfg.AUGMENTATION.CROP_WIDTH,
            cfg.AUGMENTATION.CROP_HEIGHT,
            cfg.AUGMENTATION.CROP_DEPTH,
        ]
        if "3d" in cfg.MODEL.BACKBONE
        else [cfg.AUGMENTATION.CROP_WIDTH, cfg.AUGMENTATION.CROP_HEIGHT]
    )
    overlap = [50, 50, 5] if "3d" in cfg.MODEL.BACKBONE else [50, 50]
    logging.debug(f"Creating cropper with {cropsize=}, {overlap=}")

    total = bism.utils.cropping.get_total_num_crops(image.shape, cropsize, overlap)
    iterator = tqdm(
        bism.utils.cropping.crops(image, cropsize, overlap), desc="", total=total
    )

    for slice, ind in iterator:
        with autocast(enabled=True) and torch.no_grad():  # Saves Memory!
            out = model(slice.div(scale).float().to(device))

        if "3d" in cfg.MODEL.BACKBONE:
            x, y, z = ind

            modelout[
                :,
                x + overlap[0] : x + cropsize[0] - overlap[0],
                y + overlap[1] : y + cropsize[1] - overlap[1],
                z + overlap[2] : z + cropsize[2] - overlap[2],
            ] = (
                out[
                    0,
                    :,
                    overlap[0] : -overlap[0],
                    overlap[1] : -overlap[1],
                    overlap[2] : -overlap[2] :,
                ]
                .half()
                .cpu()
            )
            iterator.desc = f"Evaluating Model on slice [x{x}:y{y}:z{z}]"
        else:
            x, y = ind
            modelout[
                :,
                x + overlap[0] : x + cropsize[0] - overlap[0],
                y + overlap[1] : y + cropsize[1] - overlap[1],
            ] = (
                out[0, :, overlap[0] : -overlap[0], overlap[1] : -overlap[1]]
                .half()
                .cpu()
            )
            iterator.desc = f"Evaluating Model on slice [x{x}:y{y}]"

    logging.info(f"Saving Output to: {filename_without_extensions}_out_.tif")

    model_filename_without_extension = model_file.rstrip('.yaml')

    with open(f"{model_filename_without_extension}_benchmark.txt", "w") as f:
        f.write(f"Generic Segmentation Benchmark:\n")
        f.write(f'Model File: {model_file}')
        f.write(f"Time: {time.time() - start} seconds\n")
        f.write(f"Current Memory Usage: {tracemalloc.get_traced_memory()[0] / (1024 * 1024)} [MB]\n\n")
        f.write(f"Max Memory Usage: {tracemalloc.get_traced_memory()[1] / (1024 * 1024)} [MB]\n\n")

    io.imsave(
        f"{filename_without_extensions}_out.tif",
        modelout[-1, ...].cpu().float().numpy(),
    )
    return modelout[-1, ...].cpu().float()
