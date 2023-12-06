import argparse
import os.path
from typing import *
import sys
import warnings

from bism.config.config import get_cfg_defaults

import torch
import torch.nn as nn
import torch.multiprocessing as mp

from bism.utils.distributed import setup_process, cleanup, find_free_port
from bism.train.default_engine import train
from bism.train.aclsd_engine import train as aclsd_train
from bism.models.construct import cfg_to_bism_model, cfg_to_torchvision_model

from yacs.config import CfgNode

from functools import partial
import logging

torch.set_float32_matmul_precision('high')

def load_cfg_from_file(args: argparse.Namespace):
    """Load configurations.
    """
    # Set configurations
    cfg = get_cfg_defaults()
    if os.path.exists(args.config_file):
        cfg.merge_from_file(args.config_file)
    else:
        raise ValueError('Could not find config file from path!')
    cfg.freeze()

    return cfg


def main():
    parser = argparse.ArgumentParser(description='SKOOTS Training Parameters')
    parser.add_argument('--config-file', type=str, help='YAML config file for training')
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
        format="[%(asctime)s] bism-train [%(levelname)s]: %(message)s",
        force=True,
    )

    cfg = load_cfg_from_file(args)
    if 'maskrcnn' not in cfg.MODEL.BACKBONE:
        model: nn.Module | List[nn.Module] = cfg_to_bism_model(cfg)  # This is our skoots torch model
    else:
        model: nn.Module = cfg_to_torchvision_model(cfg)


    if cfg.TRAIN.PRETRAINED_MODEL_PATH and not cfg.TRAIN.TARGET == 'aclsd':
        checkpoint = torch.load(cfg.TRAIN.PRETRAINED_MODEL_PATH)
        state_dict = checkpoint if not 'model_state_dict' in checkpoint else checkpoint['model_state_dict']
        model.load_state_dict(state_dict)

    port = find_free_port()
    world_size = cfg.SYSTEM.NUM_GPUS if cfg.TRAIN.DISTRIBUTED else 1
    if cfg.TRAIN.TARGET not in ['aclsd']:
        mp.spawn(train, args=(port, world_size, model, cfg, args.log), nprocs=world_size, join=True)

    elif cfg.TRAIN.TARGET == 'aclsd':
        mp.spawn(aclsd_train, args=(port, world_size, model, cfg), nprocs=world_size, join=True)



if __name__ == '__main__':
    import sys
    sys.exit(main())
