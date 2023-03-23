import argparse
import os.path
import sys
import warnings

import torch

import bism.eval.affinities
import bism.eval.lsd

from yacs.config import CfgNode


torch.set_float32_matmul_precision('high')


def main():
    parser = argparse.ArgumentParser(description='SKOOTS Training Parameters')
    parser.add_argument('--model_file', type=str, help='YAML config file for training')
    parser.add_argument('--image_path', type=str, help='Path to image')

    args = parser.parse_args()

    cfg: CfgNode = torch.load(args.model_file, map_location='cpu')['cfg']

    if cfg.TRAIN.TARGET == 'affinities':
        bism.eval.affinities.eval(args.image_path, args.model_file)

    if cfg.TRAIN.TARGET == 'lsd':
        bism.eval.lsd.eval(args.image_path, args.model_file)






if __name__ == '__main__':
    import sys
    sys.exit(main())
