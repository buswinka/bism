import argparse
import glob
import logging
import os.path

import torch
from yacs.config import CfgNode


torch.set_float32_matmul_precision("high")


def run_model(model_file, image_path, log_level: int = 2):
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

    if cfg.TRAIN.TARGET == "affinities":
        import bism.eval.affinities # do it this way because waterz is a nightmare
        if os.path.isdir(image_path):
            files = glob.glob(image_path + "/*.tif")
            files.sort()
        else:
            files = [image_path]
        for f in files:
            bism.eval.affinities.eval(f, model_file)

    elif cfg.TRAIN.TARGET == "lsd":
        import bism.eval.lsd
        bism.eval.lsd.eval(image_path, model_file)

    else:
        import bism.eval.generic
        bism.eval.generic.eval(image_path, model_file)