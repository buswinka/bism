import os
import os.path
import tracemalloc

import fastremap
import torch
import torch.nn as nn
import torch.optim.swa_utils
import zarr
from torch.cuda.amp import autocast
from tqdm import tqdm

import bism.utils.cropping
from bism.models.construct import cfg_to_bism_model
from bism.utils.io import imread

try:
    import waterz
except:
    raise RuntimeError(
        "WaterZ must be installed via git to predict affinities: https://github.com/funkey/waterz"
    )

from typing import List

import logging
import time


@torch.inference_mode()
def eval(image_path: str, model_file: str):
    """
    Runs an affinity segmentation on an image.

    :param model_file: Path to a pretrained bism model
    :return:
    """
    tracemalloc.start()
    start = time.time()
    logging.info(f"Loading model file: {model_file}")
    checkpoint = torch.load(model_file, map_location="cpu")
    cfg = checkpoint["cfg"]

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    logging.info(f"Constructing BISM model")
    base_model: nn.Module = cfg_to_bism_model(cfg)  # This is our skoots torch model
    state_dict = (
        checkpoint
        if not "model_state_dict" in checkpoint
        else checkpoint["model_state_dict"]
    )
    base_model.load_state_dict(state_dict)
    base_model = base_model.to(device)

    logging.info(f"Compiling BISM model with torch inductor")
    model = torch.compile(base_model)  # compile using torch 2
    for _ in range(10):
        _ = model(torch.rand((1, 1, 300, 300, 20), device=device, dtype=torch.float))

    filename_without_extensions = os.path.splitext(image_path)[0]

    logging.info(f"Loading image from path: {image_path}")
    image = imread(image_path, pin_memory=False)
    c, x, y, z = image.shape

    scale: float = 2**16 if image.max() > 255 else 255

    logging.info(
        f"Loaded an image with shape: {(c, x, y, z)}, dtype: {image.dtype}, min: {image.min()}, max: {image.max()}"
    )

    affinities = torch.zeros((3, x, y, z), dtype=torch.half)
    cropsize = [300, 300, 20]
    overlap = [50, 50, 5]

    total = bism.utils.cropping.get_total_num_crops(image.shape, cropsize, overlap)
    iterator = tqdm(
        bism.utils.cropping.crops(image, cropsize, overlap), desc="", total=total
    )
    benchmark_start = time.time()

    logging.info(f"Predicting Affinities")
    for slice, (x, y, z) in iterator:
        with autocast(enabled=True) and torch.no_grad():  # Saves Memory!
            out = model(slice.div(scale).float().cuda())

        affinities[
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

        iterator.desc = f"Evaluating UNet on slice [x{x}:y{y}:z{z}]"

    # torch.save(affinities[[2, 0, 1], ...].cpu(), filename_without_extensions + '_affinities.trch')
    logging.info(f"Deleting Image File.")
    del image

    affinities = affinities.to(torch.float32).permute(0, 3, 1, 2)[[2, 0, 1], ...]
    thresholds = [0.95]

    logging.info(f"Agglomerating Affinities with {thresholds=}")

    instance_mask: List[np.ndarray] = waterz.agglomerate(
        affinities.contiguous().to(torch.float32).numpy(), thresholds=thresholds
    )
    mask = affinities.sum(0).gt(0.5 * 3).numpy()
    for _im, thr in zip(instance_mask, thresholds):
        # _im is a numpy array

        _im *= mask  # gt mask
        _im, remapping = fastremap.renumber(_im, in_place=True)

        # torch.save(_im.astype(np.int64).transpose(1, 2, 0), filename_without_extensions + f'_segmentaiton_at_{int(thr*100)}%.trch', pickle_protocol=4)
        with open(filename_without_extensions + "_affinities_benchmark.txt", "w") as f:
            f.write(f"Affinities Segmentation Benchmark:\n")
            f.write(f"----------------------------------\n")
            f.write(f"Time: {time.time() - benchmark_start} seconds\n")
            f.write(f"Memory (current/max): {tracemalloc.get_traced_memory()}\n\n")

        logging.info(
            f"Saving instance mask to: {filename_without_extensions}_segmentaiton_at_{int(thr * 100)}%.zarr"
        )
        zarr.save_array(
            filename_without_extensions + f"_segmentaiton_at_{int(thr * 100)}%.zarr",
            _im.transpose(1, 2, 0),
        )

    end = time.time()
    elapsed = end - start
    logging.info(
        f"DONE: Process took {elapsed} seconds, {elapsed / 60} minutes, {elapsed / (60 ** 2)}, hours"
    )


if __name__ == "__main__":
    import numpy as np

    a = torch.rand((3, 100, 100, 100), dtype=torch.float32).numpy()

    out = waterz.agglomerate(a, thresholds=[0.2])
