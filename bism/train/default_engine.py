from functools import partial
from typing import List, Tuple, Callable, Union, OrderedDict, Optional, Dict
import os.path
import os

import torch
import torch.nn as nn
from torch import Tensor
import torch.optim.lr_scheduler
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
import torch.optim.swa_utils

from tqdm import trange
from statistics import mean
from yacs.config import CfgNode

from bism.utils.distributed import setup_process
from bism.train.merged_transform import transform_from_cfg
from bism.train.dataloader import dataset, MultiDataset, generic_colate, torchvision_colate
from bism.utils.visualization import write_progress, write_torchvision_progress
from bism.targets import _valid_targets
import logging

Dataset = Union[Dataset, DataLoader]

from bism.config.valid import _valid_optimizers, _valid_loss_functions, _valid_lr_schedulers

logging.basicConfig(level=logging.DEBUG)

torch.manual_seed(101196)
torch.set_float32_matmul_precision('high')


def train(rank: str, port: str, world_size: int, base_model: nn.Module, cfg: CfgNode, logging_level):
    setup_process(rank, world_size, port, backend='nccl')
    device = f'cuda:{rank}'

    _log_map = [
        logging.DEBUG,
        logging.INFO,
        logging.WARNING,
        logging.ERROR,
        logging.CRITICAL,
    ]
    logging.basicConfig(
        level=_log_map[logging_level],
        format=f"[%(asctime)s] bism-train [%(levelname)s]: %(message)s",
        force=True,
    )

    base_model = base_model.to(device)
    base_model = torch.nn.parallel.DistributedDataParallel(base_model)

    if int(torch.__version__[0]) >= 2 and cfg.MODEL.COMPILE:
        logging.info(f'compiling model with torch.inductor')
        model = torch.compile(base_model)
    else:
        model = base_model

    colate = generic_colate if 'torchvision' != cfg.TRAIN.TARGET else torchvision_colate
    augmentations: Callable[[Dict[str, Tensor]], Dict[str, Tensor]] = partial(transform_from_cfg, cfg=cfg,
                                                                              device=device)

    # INIT DATA ----------------------------
    _datasets = []
    for path, N in zip(cfg.TRAIN.TRAIN_DATA_DIR, cfg.TRAIN.TRAIN_SAMPLE_PER_IMAGE):
        logging.info(f'Loading images sampled {N=} times for training from path: {path}')
        _device = device if cfg.TRAIN.STORE_DATA_ON_GPU else 'cpu'
        _datasets.append(dataset(path=path,
                                 transforms=augmentations,
                                 sample_per_image=N,
                                 device=device,
                                 pad_size=10).to(_device))


    merged_train = MultiDataset(*_datasets)

    train_sampler = torch.utils.data.distributed.DistributedSampler(merged_train)
    dataloader = DataLoader(merged_train, num_workers=0, batch_size=cfg.TRAIN.TRAIN_BATCH_SIZE,
                            sampler=train_sampler, collate_fn=colate)

    # Validation Dataset
    _datasets = []
    for path, N in zip(cfg.TRAIN.VALIDATION_DATA_DIR, cfg.TRAIN.VALIDATION_SAMPLE_PER_IMAGE):
        logging.info(f'Loading images sampled {N=} times for validation from path: {path}')
        _device = device if cfg.TRAIN.STORE_DATA_ON_GPU else 'cpu'
        _datasets.append(dataset(path=path,
                                 transforms=augmentations,
                                 sample_per_image=N,
                                 device=device,
                                 pad_size=10).to(_device))

    merged_validation = MultiDataset(*_datasets)
    test_sampler = torch.utils.data.distributed.DistributedSampler(merged_validation)
    if _datasets or cfg.TRAIN.VALIDATION_BATCH_SIZE >= 1:
        valdiation_dataloader = DataLoader(merged_validation, num_workers=0, batch_size=cfg.TRAIN.VALIDATION_BATCH_SIZE,
                                           sampler=test_sampler,
                                           collate_fn=colate)

    else:  # we might not want to run validation...
        valdiation_dataloader = None

    # INIT FROM CONFIG ----------------------------
    torch.backends.cudnn.benchmark = cfg.TRAIN.CUDNN_BENCHMARK
    torch.autograd.profiler.profile = cfg.TRAIN.AUTOGRAD_PROFILE
    torch.autograd.profiler.emit_nvtx(enabled=cfg.TRAIN.AUTOGRAD_EMIT_NVTX)
    torch.autograd.set_detect_anomaly(cfg.TRAIN.AUTOGRAD_DETECT_ANOMALY)

    epochs = cfg.TRAIN.NUM_EPOCHS

    writer = SummaryWriter() if rank == 0 else None
    if writer:
        print('SUMMARY WRITER LOG DIR: ', writer.get_logdir())


    optimizer = _valid_optimizers[cfg.TRAIN.OPTIMIZER](model.parameters(),
                                                       lr=cfg.TRAIN.LEARNING_RATE,
                                                       weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    scheduler = _valid_lr_schedulers[cfg.TRAIN.SCHEDULER](optimizer, T_0=cfg.TRAIN.SCHEDULER_T0)
    scaler = GradScaler(enabled=cfg.TRAIN.MIXED_PRECISION)

    swa_model = torch.optim.swa_utils.AveragedModel(model)
    swa_start = 100

    _kwarg = {k:v for k, v in zip(cfg.TRAIN.LOSS_KEYWORDS, cfg.TRAIN.LOSS_VALUES)}
    loss_fn: Callable = _valid_loss_functions[cfg.TRAIN.LOSS_FN](**_kwarg)

    # this is basically a fn that takes an instance mask, and creates a tensor represnetation
    # which is useful for segmentation. This might be a diffusion gradient (cellpose), distance map (omnipose),
    # affinities (boundry segmentation) or local shape descriptors.
    # Should always take a 5D tensor of instance masks (B, 1, X, Y, Z) and return a 5D Tensor (B, C, X, Y, Z)
    # of targets to train a model against
    target_fn: Callable[[Tensor], Tensor] = partial(_valid_targets[cfg.TRAIN.TARGET], cfg=cfg)

    # Save each loss value in a list...
    avg_epoch_loss = [-1]
    avg_val_loss = [-1]

    # WARMUP LOOP ----------------------------
    logging.info('Performing Warmup...')
    for images, masks in dataloader:
        # usually a Tensor, sometimes a List of Data Dicts for torchvision
        target: Union[Tensor, List[Dict[str, Tensor]]] = target_fn(masks)  # makes the target we want
        pass

    warmup_range = trange(cfg.TRAIN.N_WARMUP, desc='Warmup: {}')
    for w in warmup_range:
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=cfg.TRAIN.MIXED_PRECISION):  # Saves Memory!

            out: Tensor = model(images) if cfg.TRAIN.TARGET != 'torchvision' else model(images, target)
            loss: Tensor = loss_fn(out, target)

            warmup_range.desc = f'{loss.item()}'

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    # TRAIN LOOP ----------------------------
    logging.info('Training...')
    epoch_range = trange(epochs, desc=f'Loss = {1.0000000}') if rank == 0 else range(epochs)
    for e in epoch_range:
        logging.info(f'Starting training with epoch: {e}')
        _loss = []

        if cfg.TRAIN.DISTRIBUTED:
            logging.debug(f'Set train sampler to epoch {e}')
            train_sampler.set_epoch(e)

        for niter, (images, masks) in enumerate(dataloader):
            target: Tensor = target_fn(masks)  # makes the target we want
            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=cfg.TRAIN.MIXED_PRECISION):  # Saves Memory!
                out: Tensor = model(images) if cfg.TRAIN.TARGET != 'torchvision' else model(images, target)
                loss: Tensor = loss_fn(out, target)

                if torch.isnan(loss):
                    logging.warning(f'NAN value detected in loss at epoch: {e}/{epochs} : {niter}/{len(dataloader)}')
                    if isinstance(out, dict):
                        for k, v in out.items():
                            logging.warning(f'{k}={v}')
                    continue

                logging.debug(f'loss value at epoch {e}/{epochs} and batch {niter}/{len(dataloader)} -> {loss.item()}')

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # if e > swa_start:
            #     swa_model.update_parameters(model)

            _loss.append(loss.item())

        avg_epoch_loss.append(mean(_loss))
        scheduler.step()
        logging.info(f'Average training loss for epoch {e}: {avg_epoch_loss[-1]}')

        if writer and (rank == 0):
            logging.info(f'writing to train tensorboard for epoch: {e}')
            writer.add_scalar('lr', scheduler.get_last_lr()[-1], e)
            writer.add_scalar('Loss/train', avg_epoch_loss[-1], e)
            if cfg.TRAIN.TARGET == 'torchvision':
                for k, v in out.items():
                    logging.debug(f'writing torchvision train loss scalar: {k}:{v}')
                    writer.add_scalar(f'Loss/{k}', out[k], e)

            if isinstance(out, Tensor):  # A normal Bism Output will be a Tensor
                write_progress(writer=writer, tag='Train', cfg=cfg, epoch=e, images=images, masks=masks, target=target, out=out)
            elif isinstance(out, Dict):  # This is a torchvision output
                with autocast(enabled=cfg.TRAIN.MIXED_PRECISION):  # Saves Memory!
                    logging.debug('evaluating torchvision model for image output')
                    model.eval()
                    out = model(images)
                    logging.debug(f'model eval predicted {out[0]["scores"].shape} objects in a test image')
                    model.train()
                write_torchvision_progress(writer=writer, tag='Train', cfg=cfg, epoch=e, images=images, masks=masks, target=target, out=out)
            else:
                logging.error(f'Attempting to write an output of unknown type to tensorboard during training step: {type(out)=}')

        # # Validation Step
        if e % 10 == 0 and valdiation_dataloader:
            _loss = []
            for images, masks in valdiation_dataloader:
                target: Tensor = target_fn(masks)  # makes the target we want

                with autocast(enabled=cfg.TRAIN.MIXED_PRECISION):  # Saves Memory!
                    with torch.no_grad():
                        out: Tensor = model(images) if cfg.TRAIN.TARGET != 'torchvision' else model(images, target)
                        loss: Tensor = loss_fn(out, target)

                scaler.scale(loss)
                _loss.append(loss.item())

            avg_val_loss.append(mean(_loss))

            logging.info(f'Average validation loss for epoch {e}: {avg_val_loss[-1]}')
            if writer and (rank == 0):
                logging.info(f'writing to validation tensorboard for epoch: {e}')
                writer.add_scalar('Loss/validate', avg_epoch_loss[-1], e)
                if cfg.TRAIN.TARGET == 'torchvision':
                    for k, v in out.items():
                        logging.debug(f'writing torchvision validation loss scalar: {k}:{v}')
                        writer.add_scalar(f'Loss/validate_{k}', out[k], e)

                if isinstance(out, Tensor):  # A normal Bism Output will be a Tensor
                    write_progress(writer=writer, tag='Train', cfg=cfg, epoch=e, images=images, masks=masks,
                                   target=target, out=out)
                elif isinstance(out, Dict):  # This is a torchvision output
                    logging.debug('evaluating torchvision model for image output')
                    with autocast(enabled=cfg.TRAIN.MIXED_PRECISION):  # Saves Memory!
                        model.eval()
                        out = model(images)
                        model.train()        # now we write the loss to tqdm progress bar

                    write_torchvision_progress(writer=writer, tag='Train', cfg=cfg, epoch=e, images=images, masks=masks,
                                               target=target, out=out)
                else:
                    logging.error(
                        f'Attempting to write an output of unknown type to tensorboard during training step: {type(out)=}')
        if rank == 0:
            epoch_range.desc = f'lr={scheduler.get_last_lr()[-1]:.3e}, Loss (train | val): ' + f'{avg_epoch_loss[-1]:.5f} | {avg_val_loss[-1]:.5f}'

        # Save a state dict every so often
        if e % cfg.TRAIN.SAVE_INTERVAL == 0:
            logging.info(f'saving intermediate model state_dict to ./test_{e}.trch')
            state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
            torch.save(state_dict, cfg.TRAIN.SAVE_PATH + f'/test_{e}.trch')

    if rank == 0:
        state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
        constants = {'cfg': cfg,
                     'model_state_dict': state_dict,
                     'optimizer_state_dict': optimizer.state_dict(),
                     'avg_epoch_loss': avg_epoch_loss,
                     'avg_val_loss': avg_epoch_loss,
                     }
        try:
            torch.save(constants, f'{cfg.TRAIN.SAVE_PATH}/{os.path.split(writer.log_dir)[-1]}.trch')
            logging.info(
                f'saved final training model to: {cfg.TRAIN.SAVE_PATH}/{os.path.split(writer.log_dir)[-1]}.trch')
        except:
            logging.warning(
                f'Could not save final training model to: {cfg.TRAIN.SAVE_PATH}/{os.path.split(writer.log_dir)[-1]}.trch' +\
                f'Saving at {os.getcwd()}/{os.path.split(writer.log_dir)[-1]}.trch instead'
            )
            torch.save(constants, f'{os.getcwd()}/{cfg.TRAIN.TARGET}_{os.path.split(writer.log_dir)[-1]}.trch', )

    logging.info('Training has concluded')
