import logging
import os
import os.path
from functools import partial
from statistics import mean
from typing import Tuple, Callable, Union, Dict

import bism.loss.iadb
import bism.loss.iadb
import bism.utils
import torch
import torch.nn as nn
import torch.optim.lr_scheduler
import torch.optim.swa_utils
from bism.train.dataloader import (
    dataset,
    MultiDataset,
    generic_colate,
)
from bism.train.merged_transform import transform_from_cfg
from bism.utils.distributed import setup_process
from bism.utils.visualization import write_progress
from torch import Tensor
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange
from yacs.config import CfgNode

Dataset = Union[Dataset, DataLoader]

from bism.config.valid import (
    _valid_optimizers,
    _valid_lr_schedulers,
    _valid_targets,
)
import warnings

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.DEBUG)

torch.manual_seed(101196)
torch.set_float32_matmul_precision("high")

import torch._dynamo


@torch.no_grad()
def denoise(model: nn.Module, mask: Tensor, T: int = 128) -> Tensor:
    """
    Steered de-noising using a pretrained model and a semantic mask

    Shapes:
        - mask: :math:`(B, C=1, X, Y, Z)`
        - returns: :math:`(B, C=1, X, Y, Z)`

    :param model: nn.Module who's forward method accpets two arguments, img, and a mask
    :param mask: Semantic Mask used to steer image generation
    :param T: Number of de-noising steps
    :return: denoised image
    """
    model.eval()
    img = torch.randn_like(mask)
    for t in range(T):
        alpha = t / T * torch.ones_like(img)
        x = torch.cat((img, alpha), dim=1)
        img = img + 1 / T * model(x, mask)

    model.train()

    return img


def train(
    rank: int,
    port: str,
    world_size: int,
    base_model: nn.Module,
    cfg: CfgNode,
    logging_level: int,
):
    """
    This is never meant to be called directly. Instead it should be called as part of a DataDistributedParallel
    pipeline to train a model over multiple GPUs.

    :param rank: which gpu
    :param port: which port to host DDP on
    :param world_size: total number of devices to train on
    :param base_model: preinstantiated model which will be cast to a DDP model
    :param cfg:
    :param logging_level:
    :return:
    """
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "0"
    setup_process(rank, world_size, port, backend="nccl")
    device = f"cuda:{rank}"

    assert cfg.TRAIN.TARGET == "iadb"
    assert cfg.MODEL.MODEL == "generic"
    assert cfg.TRAIN.LOSS_FN == "iadb"

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

    if int(torch.__version__[0]) >= 2:
        logging.info(f"compiling model with torch.inductor")
        model = torch.compile(
            base_model,
            options={
                "epilogue_fusion": True,
                "max_autotune": True,
                "tune_layout": True,
                # "aggressive_fusion": True,
                # "max_fusion_size": 128,
                "triton.max_tiles": 3,
            },
            disable=not cfg.MODEL.COMPILE,
        )
        torch._dynamo.reset()
    else:
        model = base_model

    colate = generic_colate
    augmentations: Callable[[Dict[str, Tensor]], Dict[str, Tensor]] = partial(
        transform_from_cfg, cfg=cfg, device=device
    )

    # INIT DATA ----------------------------
    _datasets = []
    for path, N in zip(cfg.TRAIN.TRAIN_DATA_DIR, cfg.TRAIN.TRAIN_SAMPLE_PER_IMAGE):
        logging.info(
            f"Loading images sampled {N=} times for training from path: {path}"
        )
        _device = device if cfg.TRAIN.STORE_DATA_ON_GPU else "cpu"
        _datasets.append(
            dataset(
                path=path,
                transforms=augmentations,
                sample_per_image=N,
                device=device,
                pad_size=10,
            )
            .map(lambda x: x.mul(255).float().clamp(0, 255).round().to(torch.uint8))
            .to(_device)
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

    # INIT FROM CONFIG ----------------------------
    torch.backends.cudnn.benchmark = cfg.TRAIN.CUDNN_BENCHMARK
    torch.autograd.profiler.profile = cfg.TRAIN.AUTOGRAD_PROFILE
    torch.autograd.profiler.emit_nvtx(enabled=cfg.TRAIN.AUTOGRAD_EMIT_NVTX)
    torch.autograd.set_detect_anomaly(cfg.TRAIN.AUTOGRAD_DETECT_ANOMALY)
    torch.autograd.gradcheck = None
    torch.autograd.gradgradcheck = None

    epochs = cfg.TRAIN.NUM_EPOCHS

    writer = SummaryWriter() if rank == 0 else None

    if rank == 0:
        logging.info(f"SUMMARY WRITER LOG DIR: {writer.get_logdir()}")
        # writer.add_hparams({f"SYSTEM.{k}".upper():str(v) for k, v in dict(cfg.SYSTEM).items()})
        # writer.add_hparams({f"MODEL.{k}".upper():str(v) for k, v in dict(cfg.MODEL).items()})
        # writer.add_hparams({f"TRAIN.{k}".upper():str(v) for k, v in dict(cfg.TRAIN).items()})
        # writer.add_hparams({f"AUGMENTATION.{k}".upper():str(v) for k, v in dict(cfg.AUGMENTATION).items()})

    optimizer = _valid_optimizers[cfg.TRAIN.OPTIMIZER](
        base_model.parameters(),
        lr=cfg.TRAIN.LEARNING_RATE,
        weight_decay=cfg.TRAIN.WEIGHT_DECAY,
        fused=True,
    )
    scheduler = _valid_lr_schedulers[cfg.TRAIN.SCHEDULER](
        optimizer, T_0=cfg.TRAIN.SCHEDULER_T0
    )
    scaler = GradScaler(enabled=cfg.TRAIN.MIXED_PRECISION)

    # This is different from other engines as we want to compile and the other dict index thing threw an error
    _loss_fn = bism.loss.iadb.iadb()
    loss_fn = torch.compile(_loss_fn)

    target_fn: Callable[[Tensor, Tensor], Tuple[Tensor, Tensor, Tensor]] = partial(
        torch.compile(_valid_targets[cfg.TRAIN.TARGET]), cfg=cfg
    )

    # Save each loss value in a list...
    avg_epoch_loss = [-1]
    avg_val_loss = [-1]

    # WARMUP LOOP ----------------------------
    logging.info("Warmup...")
    for images, masks in dataloader:
        blended, noise, masks = target_fn(images, masks)
        break
    for _ in trange(cfg.TRAIN.N_WARMUP, desc="warmup: "):
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=cfg.TRAIN.MIXED_PRECISION):  # Saves Memory!
            # target_fn does the blending
            out: Tensor = model(blended, masks)
            loss: Tensor = loss_fn(out, images, noise)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()


    # TRAIN LOOP ----------------------------
    logging.info("Training...")
    epoch_range = (
        trange(epochs, desc=f"Loss = {1.0000000}") if rank == 0 else range(epochs)
    )


    # context manager handles saving of model in the event of a crash
    with bism.utils.train_context(
        cfg, model, optimizer, avg_epoch_loss, avg_val_loss, writer
    ):
        for e in epoch_range:

            logging.info(f"Starting training with epoch: {e}")
            _loss = []

            if cfg.TRAIN.DISTRIBUTED:
                logging.debug(f"Set train sampler to epoch {e}")
                train_sampler.set_epoch(e)

            for niter, (images, masks) in enumerate(dataloader):
                optimizer.zero_grad(set_to_none=True)
                with autocast(enabled=cfg.TRAIN.MIXED_PRECISION):  # Saves Memory!
                    # target_fn does the blending
                    blended, noise, masks = target_fn(images, masks)
                    out: Tensor = model(blended, masks)
                    loss: Tensor = loss_fn(out, images, noise)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                if torch.isnan(loss):
                    logging.critical(
                        f"NAN value detected in loss at epoch: {e}/{epochs} : {niter}/{len(dataloader)}"
                    )
                    raise RuntimeError(
                        f"NAN value detected in loss at epoch: {e}/{epochs} : {niter}/{len(dataloader)}"
                    )

                logging.debug(
                    f"loss value at epoch {e}/{epochs} and batch {niter}/{len(dataloader)} -> {loss.item()}"
                )
                _loss.append(loss.item())

            scheduler.step()
            avg_epoch_loss.append(mean(_loss) if _loss else -1)
            logging.info(f"Average training loss for epoch {e}: {avg_epoch_loss[-1]}")

            # write progress
            if writer and (rank == 0):
                logging.info(f"writing to train tensorboard for epoch: {e}")
                writer.add_scalar("lr", scheduler.get_last_lr()[-1], e)
                writer.add_scalar("Loss/train", avg_epoch_loss[-1], e)

            # # Validation Step
            if e % cfg.TRAIN.VALIDATE_EPOCH_SKIP == 0:
                with autocast(enabled=cfg.TRAIN.MIXED_PRECISION):  # Saves Memory!
                    if rank == 0:
                        img = denoise(model, masks.gt(0.5).float())
                        write_progress(
                            writer=writer,
                            tag="Denoise",
                            cfg=cfg,
                            epoch=e,
                            images=images,
                            masks=masks,
                            target=None,
                            out=img,
                        )

            # update tqdm progrss bar
            if rank == 0:
                epoch_range.desc = (
                    f"lr={scheduler.get_last_lr()[-1]:.3e}, Loss (train | val): "
                    + f"{avg_epoch_loss[-1]:.5f} | {avg_val_loss[-1]:.5f}"
                )

            # Save a state dict every so often
            if e % cfg.TRAIN.SAVE_INTERVAL == 0:
                logging.info(f"saving intermediate model state_dict to ./test_{e}.trch")
                state_dict = (
                    model.module.state_dict()
                    if hasattr(model, "module")
                    else model.state_dict()
                )
                torch.save(state_dict, cfg.TRAIN.SAVE_PATH + f"/test_{e}.trch")

    # ------------ saving should be handled by context manager ------------ #

    logging.info("Training has concluded")
