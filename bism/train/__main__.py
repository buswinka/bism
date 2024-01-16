import argparse
import glob
import logging
import os.path
from typing import *

import torch
import torch.multiprocessing as mp
import torch.nn as nn

from bism.config.config import get_cfg_defaults
from bism.config.validator import validate_config
from bism.models.construct import cfg_to_bism_model, cfg_to_torchvision_model
from bism.train.aclsd_engine import train as aclsd_train
from bism.train.default_engine import train
from bism.train.iadb_engine import train as iadb_engine
from bism.train.maskrcnn_engine import train as maskrcnn_engine
from bism.utils.distributed import find_free_port


torch.manual_seed(101196)  # repeatability
torch.set_float32_matmul_precision('high')

def load_cfg_from_filename(filename: str):
    """Load configurations.
    """
    # Set configurations
    cfg = get_cfg_defaults()
    if os.path.exists(filename):
        cfg.merge_from_file(filename)
    else:
        raise ValueError('Could not find config file from path!')
    cfg.freeze()

    return cfg


def main():
    parser = argparse.ArgumentParser(description='BISM Training Parameters')
    parser.add_argument('--config-file', type=str, help='YAML config file for training')
    parser.add_argument('-b', '--batch', action='store_true',
                        help='Batch execute a folder of training config files')
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

    configs: List[str] = glob.glob(os.path.join(args.config_file, '*.yaml')) if args.batch else [args.config_file]
    # Loop over configs and run the training.
    for f in configs:
        cfg = load_cfg_from_filename(f)
        validate_config(cfg)
        if 'maskrcnn' not in cfg.MODEL.BACKBONE:
            model: nn.Module | List[nn.Module] = cfg_to_bism_model(cfg)  # This is our skoots torch model
        else:
            model: nn.Module = cfg_to_torchvision_model(cfg)


        if cfg.TRAIN.PRETRAINED_MODEL_PATH and not cfg.TRAIN.TARGET == 'aclsd':
            checkpoint = torch.load(cfg.TRAIN.PRETRAINED_MODEL_PATH)
            state_dict = checkpoint if not 'model_state_dict' in checkpoint else checkpoint['model_state_dict']
            try:
                model.load_state_dict(state_dict)
            except RuntimeError:
                logging.error('could not map pretrained state_dict to instantiated model. Trying again with model.load_stat_dict(strict=False)')
                try:
                    model.load_state_dict(state_dict, strict=False)
                except Exception as e:
                    logging.critical('failed to map state_dict to instantiated model with model.load_stat_dict(strict=False). This may be due to an updated source file and should be reported as a bug.')
                    raise e



        port = find_free_port()
        world_size = cfg.SYSTEM.NUM_GPUS if cfg.TRAIN.DISTRIBUTED else 1

        if cfg.TRAIN.TARGET == 'aclsd':
            mp.spawn(aclsd_train, args=(port, world_size, model, cfg, args.log), nprocs=world_size, join=True)

        elif cfg.TRAIN.TARGET == 'iadb':
            mp.spawn(iadb_engine, args=(port, world_size, model, cfg, args.log), nprocs=world_size, join=True)

        elif cfg.TRAIN.TARGET == 'torchvision':
            mp.spawn(maskrcnn_engine, args=(port, world_size, model, cfg, args.log), nprocs=world_size, join=True)

        else:
            mp.spawn(train, args=(port, world_size, model, cfg, args.log), nprocs=world_size, join=True)



if __name__ == '__main__':
    import sys
    sys.exit(main())
