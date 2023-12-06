import argparse
import glob
import logging
import os.path

import torch
from yacs.config import CfgNode
from bism.eval.run import run_model


torch.set_float32_matmul_precision("high")


def main():
    parser = argparse.ArgumentParser(description="BISM EVAL Parameters")
    parser.add_argument("-m", "--model_file", type=str, help="YAML config file for training")
    parser.add_argument("-i", "--image_path", type=str, help="Path to image")
    parser.add_argument(
        "--log",
        type=int,
        default=3,
        help="Log Level: 0-Debug, 1-Info, 2-Warning, 3-Error, 4-Critical",
    )

    args = parser.parse_args()

    run_model(args.model_file, args.image_path, args.log)

if __name__ == "__main__":
    import sys

    sys.exit(main())
