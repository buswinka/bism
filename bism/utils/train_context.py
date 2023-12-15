import contextlib
import logging
import os.path
from typing import List

import torch
import torch.distributed
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter
from yacs.config import CfgNode


def _save(
    cfg: CfgNode,
    model: nn.Module,
    optimizer: Optimizer,
    epoch_loss: List[float | Tensor],
    val_loss: List[float | Tensor],
    filename: str,
):
    """
    helper fun

    :param cfg:
    :param model:
    :param optimizer:
    :param epoch_loss:
    :param val_loss:
    :param filename:
    :return:
    """
    state_dict = (
        model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    )
    logging.debug("creating save dict in the train_context context manager")
    constants = {
        "cfg": cfg,
        "model_state_dict": state_dict,
        "optimizer_state_dict": optimizer.state_dict(),
        "avg_epoch_loss": epoch_loss,
        "avg_val_loss": val_loss,
    }
    torch.save(constants, filename)


@contextlib.contextmanager
def train_context(
    cfg: CfgNode,
    model: nn.Module,
    optimizer: Optimizer,
    epoch_loss: List[float | Tensor],
    val_loss: List[float | Tensor],
    writer: SummaryWriter,
):
    """
    Context manager to handle autosaving upon raised Exception in training loop

    :param cfg: yacs ConfigNode
    :param model: model being trained
    :param optimizer: optimizer training the model
    :param epoch_loss: list of all epoch losses
    :param val_loss: list of all validation losses
    :param writer: Summarywriter
    :return: None
    """
    has_dumped = False
    try:
        yield

    except Exception as e:
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        save_filepath = f"{cfg.TRAIN.SAVE_PATH}/{os.path.split(writer.log_dir)[-1]}_rank{rank}_errordump.trch"
        _save(cfg, model, optimizer, epoch_loss, val_loss, save_filepath)
        has_dumped = True

        logging.critical(f"Exception raised. Dumping model state to: {save_filepath}")

        raise e

    finally:

        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

        # have to save if rank == 0 or not in a distributed training run
        if not has_dumped and rank == 0:

            save_filepath = f"{cfg.TRAIN.SAVE_PATH}/{os.path.split(writer.log_dir)[-1]}.trch"

            logging.debug(f"attempting to save at {save_filepath}")

            _save(cfg, model, optimizer, epoch_loss, val_loss, save_filepath)

            logging.debug(f"saved successfully")
            logging.critical(
                f"Exited training context. Saved snapshot to: {save_filepath}"
            )
