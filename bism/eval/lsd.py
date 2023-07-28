import bism.utils.cropping
try:
    import waterz
except:
    raise RuntimeError('WaterZ must be installed via git to predict affinities: https://github.com/funkey/waterz')

import os.path
import os

import torch
import torch.nn as nn

from torch.cuda.amp import GradScaler, autocast
import torch.optim.swa_utils

from bism.models.construct import cfg_to_bism_model
from bism.utils.io import imread
from tqdm import tqdm
import fastremap

import numpy as np
import logging

from typing import List, Tuple, Callable, Union, OrderedDict, Optional, Dict

@torch.no_grad()
def eval(image_path: str, model_file: str):
    """
    Runs an affinity segmentation on an image.

    :param model_file: Path to a pretrained bism model
    :return:
    """
    logging.info(f'Loading model file: {model_file}')
    checkpoint = torch.load(model_file, map_location='cpu')
    cfg = checkpoint['cfg']

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    logging.info(f'Constructing BISM model')
    base_model: nn.Module = cfg_to_bism_model(cfg)  # This is our skoots torch model
    state_dict = checkpoint if not 'model_state_dict' in checkpoint else checkpoint['model_state_dict']
    base_model.load_state_dict(state_dict)
    base_model = base_model.to(device)

    logging.info(f'Compiling BISM model with torch inductor')
    model = torch.compile(base_model)  # compile using torch 2
    _ = model(torch.rand((1, 1, 300, 300, 20), device=device, dtype=torch.float))

    filename_without_extensions = os.path.splitext(image_path)[0]

    logging.info(f'Loading image from path: {image_path}')
    image = imread(image_path, pin_memory=False)
    c, x, y, z = image.shape

    logging.info(f'Loaded an image with shape: {(c, x, y, z)}, dtype: {image.dtype}, min: {image.min()}, max: {image.max()}')

    scale: float = 2**16 if image.max() > 255 else 255

    lsd = torch.zeros((10, x, y, z), dtype=torch.half)
    cropsize = [300, 300, 20]
    overlap = [50, 50, 5]

    total = bism.utils.cropping.get_total_num_crops(image.shape, cropsize, overlap)
    iterator = tqdm(bism.utils.cropping.crops(image, cropsize, overlap), desc='', total=total)

    for slice, (x, y, z) in iterator:
        with autocast(enabled=True) and torch.no_grad():  # Saves Memory!
            out = model(slice.div(scale).float().cuda())

        lsd[:,
        x + overlap[0]: x + cropsize[0] - overlap[0],
        y + overlap[1]: y + cropsize[1] - overlap[1],
        z + overlap[2]: z + cropsize[2] - overlap[2]] = out[0, :,
                                                        overlap[0]: -overlap[0],
                                                        overlap[1]: -overlap[1],
                                                        overlap[2]: -overlap[2]
                                                        :].half().cpu()

        iterator.desc = f'Evaluating UNet on slice [x{x}:y{y}:z{z}]'


    logging.info(f'Saving LSDs to: {filename_without_extensions}_lsd_.trch')

    torch.save(lsd[0:3, ...].cpu(), filename_without_extensions + '_lsd123.trch')
    torch.save(lsd[3:6, ...].cpu(), filename_without_extensions + '_lsd456.trch')
    torch.save(lsd[6:9, ...].cpu(), filename_without_extensions + '_lsd789.trch')
    torch.save(lsd[-1, ...].cpu(), filename_without_extensions + '_lsd10.trch')

if __name__ == '__main__':
    import numpy as np

    a = torch.rand((3, 100, 100, 100), dtype=torch.float32).numpy()

    out = waterz.agglomerate(a, thresholds=[0.2])
