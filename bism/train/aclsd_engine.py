import os
import os.path
from functools import partial
from statistics import mean
from typing import *

import torch
import torch.nn as nn
import torch.optim.lr_scheduler
import torch.optim.swa_utils
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange
from copy import deepcopy
from yacs.config import CfgNode

from bism.config.valid import _valid_targets
from bism.train.dataloader import dataset, MultiDataset, generic_colate as colate
# from bism.train.merged_transform import transform_from_cfg
from bism.utils.distributed import setup_process
from bism.utils.visualization import write_progress

Dataset = Union[Dataset, DataLoader]

from bism.config.valid import (
    _valid_optimizers,
    _valid_loss_functions,
    _valid_lr_schedulers,
)

torch.manual_seed(101196)
torch.set_float32_matmul_precision("high")

""""
Auto Context LSD Predicts Affinities through LSD Directly!

I am assuming that each network was trained in tandom....

We therefore need to train two models independantly. Im gunna run backprop on each. 

"""


def train(rank: str, port: str, world_size: int, base_model: Tuple[nn.Module, nn.Module], cfg: CfgNode):
    setup_process(rank, world_size, port, backend="nccl")
    device = f"cuda:{rank}"

    lsd_base_model = base_model[0].to(device)
    lsd_base_model = torch.nn.parallel.DistributedDataParallel(lsd_base_model)

    affinities_base_model = base_model[1].to(device)
    affinities_base_model = torch.nn.parallel.DistributedDataParallel(affinities_base_model)

    if int(torch.__version__[0]) >= 2:
        print("Comiled with Inductor")
        lsd_model = torch.compile(lsd_base_model)
        aff_model = torch.compile(affinities_base_model)
    else:
        raise EnvironmentError('Must be running torch 2')

    augmentations: Callable[[Dict[str, Tensor]], Dict[str, Tensor]] = partial(
        transform_from_cfg, cfg=cfg, device=device
    )

    # INIT DATA ----------------------------
    _datasets = []
    for path, N in zip(cfg.TRAIN.TRAIN_DATA_DIR, cfg.TRAIN.TRAIN_SAMPLE_PER_IMAGE):
        _device = device if cfg.TRAIN.STORE_DATA_ON_GPU else "cpu"
        _datasets.append(
            dataset(
                path=path,
                transforms=augmentations,
                sample_per_image=N,
                device=device,
                pad_size=10,
            ).to(_device)
        )

    merged_train = MultiDataset(*_datasets)

    train_sampler = torch.utils.data.distributed.DistributedSampler(merged_train)
    dataloader = DataLoader(
        merged_train,
        num_workers=0,
        batch_size=cfg.TRAIN.TRAIN_BATCH_SIZE,
        sampler=train_sampler,
        collate_fn=colate,
    )

    # Validation Dataset
    _datasets = []
    for path, N in zip(
        cfg.TRAIN.VALIDATION_DATA_DIR, cfg.TRAIN.VALIDATION_SAMPLE_PER_IMAGE
    ):
        _device = device if cfg.TRAIN.STORE_DATA_ON_GPU else "cpu"
        _datasets.append(
            dataset(
                path=path,
                transforms=augmentations,
                sample_per_image=N,
                device=device,
                pad_size=10,
            ).to(_device)
        )

    merged_validation = MultiDataset(*_datasets)  # _datasets is List[Dataset]
    test_sampler = torch.utils.data.distributed.DistributedSampler(merged_validation)
    if _datasets or cfg.TRAIN.VALIDATION_BATCH_SIZE >= 1:
        valdiation_dataloader = DataLoader(
            merged_validation,
            num_workers=0,
            batch_size=cfg.TRAIN.VALIDATION_BATCH_SIZE,
            sampler=test_sampler,
            collate_fn=colate,
        )

    else:  # we might not want to run validation...
        valdiation_dataloader = None

    # INIT FROM CONFIG ----------------------------
    torch.backends.cudnn.benchmark = cfg.TRAIN.CUDNN_BENCHMARK
    torch.autograd.profiler.profile = cfg.TRAIN.AUTOGRAD_PROFILE
    torch.autograd.profiler.emit_nvtx(enabled=cfg.TRAIN.AUTOGRAD_EMIT_NVTX)
    torch.autograd.set_detect_anomaly(cfg.TRAIN.AUTOGRAD_DETECT_ANOMALY)

    epochs: int = cfg.TRAIN.NUM_EPOCHS

    writer = SummaryWriter() if rank == 0 else None
    if writer:
        print("SUMMARY WRITER LOG DIR: ", writer.get_logdir())

    # Optimizers and Schedulers for each model...
    lsd_optimizer: Optimizer = _valid_optimizers[cfg.TRAIN.OPTIMIZER](
        lsd_model.parameters(),
        lr=cfg.TRAIN.LEARNING_RATE,
        weight_decay=cfg.TRAIN.WEIGHT_DECAY,
    )
    lsd_scheduler: LRScheduler = _valid_lr_schedulers[cfg.TRAIN.SCHEDULER](
        lsd_optimizer, T_0=cfg.TRAIN.SCHEDULER_T0
    )

    aff_optimizer: Optimizer = _valid_optimizers[cfg.TRAIN.OPTIMIZER](
        aff_model.parameters(),
        lr=cfg.TRAIN.LEARNING_RATE,
        weight_decay=cfg.TRAIN.WEIGHT_DECAY,
    )
    aff_scheduler: LRScheduler = _valid_lr_schedulers[cfg.TRAIN.SCHEDULER](
        aff_optimizer, T_0=cfg.TRAIN.SCHEDULER_T0
    )

    lsd_scaler = GradScaler(enabled=cfg.TRAIN.MIXED_PRECISION)
    aff_scaler = GradScaler(enabled=cfg.TRAIN.MIXED_PRECISION)

    # swa_model = torch.optim.swa_utils.AveragedModel(model)
    # swa_start = 100

    _kwarg: Dict[str, Any] = {k: v for k, v in zip(cfg.TRAIN.LOSS_KEYWORDS, cfg.TRAIN.LOSS_VALUES)}
    loss_fn: Callable[[Tensor, Tensor], Tensor] = _valid_loss_functions[cfg.TRAIN.LOSS_FN](**_kwarg)

    # ACLSDs ONLY!!!
    # this is basically a fn that takes an instance mask, and creates a tensor representation of both LSDs and Aff...
    # Should always take a 5D tensor of instance masks (B, 1, X, Y, Z) and a YACS Cfg Node
    # and return a Tuple of Tensors...
    target_fn: Callable[[Tensor, CfgNode], Tuple[Tensor, Tensor]] = partial(
        _valid_targets[cfg.TRAIN.TARGET], cfg=cfg
    )

    # Save each loss value in a list...
    avg_aff_epoch_loss = [-1.0]
    avg_lsd_epoch_loss = [-1.0]
    avg_lsd_val_loss = [-1.0]
    avg_aff_val_loss = [-1.0]

    # WARMUP LOOP ----------------------------
    for images, masks in dataloader:
        lsd_target, aff_target = target_fn(masks)  # makes the target we want
        pass

    warmup_range = trange(cfg.TRAIN.N_WARMUP, desc="Warmup: {}")
    for w in warmup_range:
        lsd_optimizer.zero_grad(set_to_none=True)
        aff_optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=cfg.TRAIN.MIXED_PRECISION):  # Saves Memory!
            lsd: Tensor = lsd_model(images)
            lsd_loss: Tensor = loss_fn(lsd, lsd_target)

        lsd_scaler.scale(lsd_loss).backward(retain_graph=True)
        lsd_scaler.step(lsd_optimizer)
        lsd_scaler.update()

        lsd.detach()  # Already used this tensor. Detach to prevent double-dipping.

        with autocast(enabled=cfg.TRAIN.MIXED_PRECISION):  # Saves Memory!
            aff: Tensor = aff_model(lsd)
            aff_loss: Tensor = loss_fn(aff, aff_target)

        warmup_range.desc = f"{lsd_loss.item()}"
        aff_scaler.scale(aff_loss).backward()
        aff_scaler.step(aff_optimizer)
        aff_scaler.update()

    # TRAIN LOOP ----------------------------
    epoch_range = (
        trange(epochs, desc=f"Loss = {1.0000000}") if rank == 0 else range(epochs)
    )
    for e in epoch_range:
        _lsd_loss = []
        _aff_loss = []

        if cfg.TRAIN.DISTRIBUTED:
            train_sampler.set_epoch(e)

        for images, masks in dataloader:
            lsd_target, aff_target = target_fn(masks)  # makes the target we want
            lsd_optimizer.zero_grad(set_to_none=True)
            aff_optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=cfg.TRAIN.MIXED_PRECISION):  # Saves Memory!
                lsd: Tensor = lsd_model(images)
                lsd_loss: Tensor = loss_fn(lsd, lsd_target)

            lsd_scaler.scale(lsd_loss).backward(retain_graph=True)
            lsd_scaler.step(lsd_optimizer)
            lsd_scaler.update()

            lsd.detach()  # Already used this tensor. Detach to prevent double-dipping.

            with autocast(enabled=cfg.TRAIN.MIXED_PRECISION):  # Saves Memory!
                aff: Tensor = aff_model(lsd)
                aff_loss: Tensor = loss_fn(aff, aff_target)

            warmup_range.desc = f"{lsd_loss.item()}"
            aff_scaler.scale(aff_loss).backward()
            aff_scaler.step(aff_optimizer)
            aff_scaler.update()

            _lsd_loss.append(lsd_loss.item())
            _aff_loss.append(aff_loss.item())

        avg_lsd_epoch_loss.append(mean(_lsd_loss))
        avg_aff_epoch_loss.append(mean(_aff_loss))

        lsd_scheduler.step()
        aff_scheduler.step()

        if writer and (rank == 0):
            writer.add_scalar("lr", aff_scheduler.get_last_lr()[-1], e)
            writer.add_scalar("Loss/LSD train", avg_lsd_epoch_loss[-1], e)
            writer.add_scalar("Loss/AFF train", avg_aff_epoch_loss[-1], e)

            write_progress(
                writer=writer,
                tag="Train",
                cfg=cfg,
                epoch=e,
                images=images,
                masks=masks,
                target=torch.concatenate((lsd_target, aff_target), dim=1),
                out=torch.concatenate((lsd, aff), dim=1),
            )

        # # Validation Step
        if e % 10 == 0 and valdiation_dataloader:

            _lsd_loss = []
            _aff_loss = []

            for images, masks in dataloader:
                lsd_target, aff_target = target_fn(masks)  # makes the target we want

                with autocast(enabled=cfg.TRAIN.MIXED_PRECISION) and torch.no_grad():  # Saves Memory!
                    lsd: Tensor = lsd_model(images)
                    lsd_loss: Tensor = loss_fn(lsd, lsd_target)

                    aff: Tensor = aff_model(lsd)
                    aff_loss: Tensor = loss_fn(aff, aff_target)

                _lsd_loss.append(lsd_loss.item())
                _aff_loss.append(aff_loss.item())

            avg_lsd_val_loss.append(mean(_lsd_loss))
            avg_aff_val_loss.append(mean(_aff_loss))


            if writer and (rank == 0):
                writer.add_scalar("lr", aff_scheduler.get_last_lr()[-1], e)
                writer.add_scalar("Loss/LSD validate", avg_lsd_epoch_loss[-1], e)
                writer.add_scalar("Loss/AFF validate", avg_aff_epoch_loss[-1], e)

                write_progress(
                    writer=writer,
                    tag="Validation",
                    cfg=cfg,
                    epoch=e,
                    images=images,
                    masks=masks,
                    target=torch.concatenate((lsd_target, aff_target), dim=1),
                    out=torch.concatenate((lsd, aff), dim=1),
                )

        # now we write the loss to tqdm progress bar
        if rank == 0:
            epoch_range.desc = (
                f"lr={aff_scheduler.get_last_lr()[-1]:.3e}, LSD Loss (train | val): "
                + f"{avg_lsd_epoch_loss[-1]:.5f} | {avg_lsd_val_loss[-1]:.5f}"
                + f" Aff Loss (train | val): "
                + f"{avg_aff_epoch_loss[-1]:.5f} | {avg_aff_val_loss[-1]:.5f}"
            )

    if rank == 0:
        lsd_state_dict = (
            lsd_model.module.state_dict()
            if hasattr(lsd_model, "module")
            else lsd_model.state_dict()
        )

        aff_state_dict = (
            aff_model.module.state_dict()
            if hasattr(lsd_model, "module")
            else aff_model.state_dict()
        )
        constants = {
            "cfg": cfg,
            "lsd_model_state_dict": lsd_state_dict,
            "aff_model_state_dict": aff_state_dict,
            "lsd_optimizer_state_dict": lsd_optimizer.state_dict(),
            "aff_optimizer_state_dict": aff_optimizer.state_dict(),
            "avg_lsd_epoch_loss": avg_lsd_epoch_loss,
            "avg_aff_epoch_loss": avg_aff_epoch_loss,
            "avg_lsd_val_loss": avg_lsd_val_loss,
            "avg_aff_val_loss": avg_aff_val_loss,
        }

        try:
            torch.save(
                constants,
                f"{cfg.TRAIN.SAVE_PATH}/{os.path.split(writer.log_dir)[-1]}.trch",
            )
        except:
            print(
                f"Could not save at: {cfg.TRAIN.SAVE_PATH}/{os.path.split(writer.log_dir)[-1]}.trch"
                f"Saving at {os.getcwd()}/{os.path.split(writer.log_dir)[-1]}.trch instead"
            )

            torch.save(
                constants,
                f"{os.getcwd()}/{os.path.split(writer.log_dir)[-1]}.trch",
            )
