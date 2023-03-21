import warnings
from functools import partial
from typing import List, Tuple, Callable, Union, OrderedDict, Optional
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

Dataset = Union[Dataset, DataLoader]

from bism.config.valid import _valid_optimizers, _valid_loss_functions, _valid_lr_schedulers


torch.manual_seed(101196)
torch.set_float32_matmul_precision('high')

def train(rank: str, port: str, world_size: int, base_model: nn.Module, cfg: CfgNode):
    setup_process(rank, world_size, port, backend='nccl')
    device = f'cuda:{rank}'

    base_model = base_model.to(device)
    base_model = torch.nn.parallel.DistributedDataParallel(base_model)

    if int(torch.__version__[0]) >= 2:
        print('Comiled with Inductor')
        model = torch.compile(base_model)
    else:
        model = torch.jit.script(base_model)


    augmentations: Callable[[Dict[str, Tensor]], Dict[str, Tensor]] = partial(transform_from_cfg, cfg=cfg,
                                                                              device=device)
    background_agumentations: Callable[[Dict[str, Tensor]], Dict[str, Tensor]] = partial(background_transform_from_cfg,
                                                                                         cfg=cfg, device=device)
    # Training Dataset - MultiDataset[Mitochondria, Background]
    _datasets = []
    for path, N in zip(cfg.TRAIN.TRAIN_DATA_DIR, cfg.TRAIN.TRAIN_SAMPLE_PER_IMAGE):
        _device = device if cfg.TRAIN.STORE_DATA_ON_GPU else 'cpu'
        _datasets.append(dataset(path=path,
                                 transforms=augmentations,
                                 sample_per_image=N,
                                 device=device,
                                 pad_size=10).to(_device))

    for path, N in zip(cfg.TRAIN.BACKGROUND_DATA_DIR, cfg.TRAIN.BACKGROUND_SAMPLE_PER_IMAGE):
        _device = device if cfg.TRAIN.STORE_DATA_ON_GPU else 'cpu'
        _datasets.append(dataset(path=path,
                                 transforms=background_agumentations, sample_per_image=N,
                                 device=device,
                                 pad_size=100).to(_device))

    merged_train = MultiDataset(*_datasets)

    train_sampler = torch.utils.data.distributed.DistributedSampler(merged_train)
    dataloader = DataLoader(merged_train, num_workers=0, batch_size=cfg.TRAIN.TRAIN_BATCH_SIZE,
                            sampler=train_sampler, collate_fn=skeleton_colate)

    # Validation Dataset
    _datasets = []
    for path, N in zip(cfg.TRAIN.VALIDATION_DATA_DIR, cfg.TRAIN.VALIDATION_SAMPLE_PER_IMAGE):
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
                                           collate_fn=skeleton_colate)

    else:  # we might not want to run validation...
        valdiation_dataloader = None

    torch.backends.cudnn.benchmark = cfg.TRAIN.CUDNN_BENCHMARK
    torch.autograd.profiler.profile = cfg.TRAIN.AUTOGRAD_PROFILE
    torch.autograd.profiler.emit_nvtx(enabled=cfg.TRAIN.AUTOGRAD_EMIT_NVTX)
    torch.autograd.set_detect_anomaly(cfg.TRAIN.AUTOGRAD_DETECT_ANOMALY)

    sigma: Sigma = init_sigma(cfg, device)
    epochs = cfg.TRAIN.NUM_EPOCHS

    writer = SummaryWriter() if rank == 0 else None
    if writer:
        print('SUMMARY WRITER LOG DIR: ', writer.get_logdir())

    # TRAIN LOOP ----------------------------

    num = torch.tensor(cfg.SKOOTS.VECTOR_SCALING, device=device)

    optimizer = _valid_optimizers[cfg.TRAIN.OPTIMIZER](model.parameters(),
                                                       lr=cfg.TRAIN.LEARNING_RATE,
                                                       weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    scheduler = _valid_lr_schedulers[cfg.TRAIN.SCHEDULER](optimizer, T_0=cfg.TRAIN.SCHEDULER_T0)
    scaler = GradScaler(enabled=cfg.TRAIN.MIXED_PRECISION)

    swa_model = torch.optim.swa_utils.AveragedModel(model)
    swa_start = 100
    swa_scheduler = torch.optim.swa_utils.SWALR(optimizer, swa_lr=0.05)

    _kwarg = {k:v for k,v in zip(cfg.TRAIN.LOSS_AFFINITY_KEYWORDS, cfg.TRAIN.LOSS_AFFINITY_VALUES)}
    loss_fn: Callable = _valid_loss_functions[cfg.TRAIN.LOSS_AFFINITY](**_kwarg)

    # Save each loss value in a list...
    avg_epoch_loss = [-1]
    avg_val_loss = [-1]


    # skel_crossover_loss = skoots.train.loss.split(n_iter=3, alpha=2)

    # Warmup... Get the first from train_data
    for images, _, target in dataloader:
        pass
    assert images is not None, len(dataloader)


    warmup_range = trange(cfg.TRAIN.N_WARMUP, desc='Warmup: {}')
    for w in warmup_range:
        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=cfg.TRAIN.MIXED_PRECISION):  # Saves Memory!
            out: Tensor = model(images)
            loss: Tensor = loss_fn(out, target)

            warmup_range.desc = f'{loss.item()}'

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    # Train Step...
    epoch_range = trange(epochs, desc=f'Loss = {1.0000000}') if rank == 0 else range(epochs)
    for e in epoch_range:
        _loss  = [], [], [], []

        if cfg.TRAIN.DISTRIBUTED:
            train_sampler.set_epoch(e)

        for images, target in dataloader:
            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=cfg.TRAIN.MIXED_PRECISION):  # Saves Memory!
                out: Tensor = model(images)
                loss: Tensor = loss_fn(out, target)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if e > swa_start:
                swa_model.update_parameters(model)

            _loss.append(loss.item())
            _embed.append(_loss_embed.item())
            _prob.append(_loss_prob.item())
            _skele.append(_loss_skeleton.item())

        avg_epoch_loss.append(mean(_loss))
        scheduler.step()

        if writer and (rank == 0):
            writer.add_scalar('lr', scheduler.get_last_lr()[-1], e)
            writer.add_scalar('Loss/train', avg_epoch_loss[-1], e)

            write_progress(writer=writer, tag='Train', epoch=e, images=images, masks=masks,
                           probability_map=probability_map,
                           vector=vector, out=out, skeleton=skeleton,
                           predicted_skeleton=predicted_skeleton, gt_skeleton=skele_masks)

        # # Validation Step
        if e % 10 == 0 and valdiation_dataloader:
            _loss = []
            for images, masks, skeleton, skele_masks, baked in valdiation_dataloader:
                with autocast(enabled=cfg.TRAIN.MIXED_PRECISION):  # Saves Memory!
                    with torch.no_grad():
                        out: Tensor = model(images)
                        loss = loss_fn(out, target)


                scaler.scale(loss)
                _loss.append(loss.item())

            avg_val_loss.append(mean(_loss))


            if writer and (rank == 0):
                write_progress(writer=writer, tag='Validation', epoch=e, images=images, masks=masks,
                               probability_map=probability_map,
                               vector=vector, out=out, skeleton=skeleton,
                               predicted_skeleton=predicted_skeleton, gt_skeleton=skele_masks)

        if rank == 0:
            epoch_range.desc = f'lr={scheduler.get_last_lr()[-1]:.3e}, Loss (train | val): ' + f'{avg_epoch_loss[-1]:.5f} | {avg_val_loss[-1]:.5f}'

        state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
        if e % 100 == 0:
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
        except:
            print(f'Could not save at: {cfg.TRAIN.SAVE_PATH}/{os.path.split(writer.log_dir)[-1]}.trch'
                  f'Saving at {os.getcwd()}/{os.path.split(writer.log_dir)[-1]}.trch instead')

            torch.save(constants, f'{os.getcwd()}/{os.path.split(writer.log_dir)[-1]}.trch', )
