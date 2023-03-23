import bism.utils.cropping
import waterz
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

from typing import List, Tuple, Callable, Union, OrderedDict, Optional, Dict

@torch.no_grad()
def eval(image_path: str, model_file: str):
    """
    Runs an affinity segmentation on an image.

    :param model_file: Path to a pretrained bism model
    :return:
    """

    checkpoint = torch.load(model_file, map_location='cpu')
    cfg = checkpoint['cfg']

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    base_model: nn.Module = cfg_to_bism_model(cfg)  # This is our skoots torch model
    state_dict = checkpoint if not 'model_state_dict' in checkpoint else checkpoint['model_state_dict']
    base_model.load_state_dict(state_dict)
    base_model = base_model.to(device)

    model = torch.compile(base_model)  # compile using torch 2

    filename_without_extensions = os.path.splitext(image_path)[0]

    image = imread(image_path, pin_memory=False)
    c, x, y, z = image.shape

    scale: float = 2**16 if image.max() > 255 else 255

    affinities = torch.zeros((3, x, y, z), dtype=torch.half)
    cropsize = [300, 300, 20]
    overlap = [30, 30, 2]

    total = bism.utils.cropping.get_total_num_crops(image.shape, cropsize, overlap)
    iterator = tqdm(bism.utils.cropping.crops(image, cropsize, overlap), desc='', total=total)

    for slice, (x, y, z) in iterator:
        with autocast(enabled=True) and torch.no_grad():  # Saves Memory!
            out = model(slice.div(scale).float().cuda())

        affinities[:,
        x + overlap[0]: x + cropsize[0] - overlap[0],
        y + overlap[1]: y + cropsize[1] - overlap[1],
        z + overlap[2]: z + cropsize[2] - overlap[2]] = out[0, :,
                                                        overlap[0]: -overlap[0],
                                                        overlap[1]: -overlap[1],
                                                        overlap[2]: -overlap[2]
                                                        :].half().cpu()

        iterator.desc = f'Evaluating UNet on slice [x{x}:y{y}:z{z}]'


    torch.save(affinities[[2, 0, 1], ...].cpu(), filename_without_extensions + '_affinities.trch')

    affinities = affinities.to(torch.float32).permute(0, 3, 1, 2)[[2, 0, 1], ...]
    thresholds = [0.95]

    instance_mask: List[np.ndarray] = waterz.agglomerate(affinities.contiguous().to(torch.float32).numpy(),
                                       thresholds=thresholds)
    print(instance_mask)

    mask = affinities.sum(0).gt(0.5 * 3).numpy()
    for _im, thr in zip(instance_mask, thresholds):
        # _im is a numpy array

        _im, remapping = fastremap.renumber(_im, in_place=True)
        _im *= mask  # gt mask

        print(f'{_im.shape=}, {len(np.unique(_im))}, {_im.dtype}, {_im.max()}, {_im.min()}')
        torch.save(_im.astype(np.int64).transpose(1, 2, 0), filename_without_extensions + f'_segmentaiton_at_{int(thr*100)}%.trch')

if __name__ == '__main__':
    import numpy as np

    a = torch.rand((3, 100, 100, 100), dtype=torch.float32).numpy()

    out = waterz.agglomerate(a, thresholds=[0.2])
