import argparse
import glob
import logging
import os.path

import torch
from yacs.config import CfgNode

import bism.eval.affinities
import bism.eval.lsd

torch.set_float32_matmul_precision("high")


def main():
    parser = argparse.ArgumentParser(description="SKOOTS Training Parameters")
    parser.add_argument("--model_file", type=str, help="YAML config file for training")
    parser.add_argument("--image_path", type=str, help="Path to image")
    parser.add_argument(
        "--log",
        type=int,
        default=3,
        help="Log Level: 0-Debug, 1-Info, 2-Warning, 3-Error, 4-Critical",
    )

    args = parser.parse_args()

    # Set logging level
    _log_map = [
        logging.DEBUG,
        logging.INFO,
        logging.WARNING,
        logging.ERROR,
        logging.CRITICAL,
    ]
    logging.basicConfig(
        level=_log_map[args.log],
        format="[%(asctime)s] bism-eval [%(levelname)s]: %(message)s",
    )

    cfg: CfgNode = torch.load(args.model_file, map_location="cpu")["cfg"]

    if cfg.TRAIN.TARGET == "affinities":
        if os.path.isdir(args.image_path):
            files = glob.glob(args.image_path + "/*.tif")
            files.sort()
        else:
            files = [args.image_path]
        for f in files:
            bism.eval.affinities.eval(f, args.model_file)

    if cfg.TRAIN.TARGET == "lsd":
        bism.eval.lsd.eval(args.image_path, args.model_file)


if __name__ == "__main__":
    import sys

    sys.exit(main())
