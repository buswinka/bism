import glob
import logging
import os.path

import torch
from yacs.config import CfgNode
from typing import List


def run_model(model_file: str, image_path: str, device: str | None = None, log_level: int = 2) -> None:
    """
    Helper function to dispatch different eval scripts from one config file. Should infer which script to evaluate
    through the cfg file, i.e. the value of cfg.TRAIN.TARGET.

    :param model_file: (str) Pretrained model file (*.trch)
    :param image_path: (str) Image file (*.tif)
    :param device: (str) hardware accelerator ('Infer', 'cpu', 'cuda:0', 'cuda:1', 'cuda:N', 'mps', ...)
    :param log_level: (int) logging level
    :return: None
    """
    if device == 'Infer':  # weirdness with argparse requires this.
        device = None

    torch.manual_seed(101196)
    torch.set_float32_matmul_precision('high')

    # Set logging level
    _log_map = [
        logging.DEBUG,
        logging.INFO,
        logging.WARNING,
        logging.ERROR,
        logging.CRITICAL,
    ]
    logging.basicConfig(
        level=_log_map[log_level],
        format="[%(asctime)s] bism-eval [%(levelname)s]: %(message)s",
    )

    cfg: CfgNode = torch.load(model_file, map_location="cpu")["cfg"]

    files: List[str] = glob.glob(image_path + "/*.tif") if os.path.isdir(image_path) else [image_path]
    files.sort()

    if cfg.TRAIN.TARGET == "affinities":
        import bism.eval.affinities  # do it this way because waterz import is a nightmare
        for f in files:
            bism.eval.affinities.eval(f, model_file, device)

    elif cfg.TRAIN.TARGET == "lsd":
        import bism.eval.lsd
        for f in files:
            bism.eval.lsd.eval(f, model_file, device)

    elif cfg.TRAIN.TARGET == "torchvision":
        import bism.eval.maskrcnn
        for f in files:
            bism.eval.maskrcnn.eval(f, model_file, device)

    elif cfg.TRAIN.TARGET in ['generic', 'semantic']:
        import bism.eval.generic
        for f in files:
            bism.eval.generic.eval(f, model_file, device)
    else:
        raise RuntimeError(f'Cannot find evaluation script for target: {cfg.TRAIN.TARGET=}')