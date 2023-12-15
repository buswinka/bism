from typing import List, Dict

import torch
import torchvision.ops
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from yacs.config import CfgNode


def write_torchvision_progress(
    writer: SummaryWriter,
    tag: str,
    epoch: int,
    images: List[Tensor],
    masks: List[Tensor],
    target: List[Dict[str, Tensor]],
    out: List[Dict[str, Tensor]],
    cfg: CfgNode = None,
):
    """
    Writes to torch the progress of a torchvision training model, which has a different output due to difereing API with
    BISM.

    :param writer:
    :param tag:
    :param epoch:
    :param images:
    :param masks:
    :param target:
    :param out:
    :param cfg:
    :return:
    """

    assert isinstance(
        images, list
    ), f" kwarg images is not of type list: {type(images)=}"
    assert isinstance(masks, list), f" kwarg masks is not of type list: {type(masks)=}"
    assert isinstance(
        target, list
    ), f" kwarg masks is not of type list: {type(target)=}"

    images = images[0].cpu()
    # print(images.shape, images.max(), images.min(), images.dtype, )
    images = images.float().mul(255).clamp(0, 255).to(torch.uint8)

    images = images[[0, 0, 0], ...]

    bool_mask = out[0]["masks"].gt(0.5)  # [N, X, Y] shape
    gt_mask = target[0]["masks"].gt(0.5)
    scores = out[0]["scores"].gt(0.25)
    boxes = out[0]["boxes"]

    # print(scores.shape, bool_mask.shape, gt_mask.shape, boxes.shape)
    # print(bool_mask[scores, ...].shape, gt_mask.squeeze(0).shape, boxes[scores, ...].shape)

    pred_masked = torchvision.utils.draw_segmentation_masks(
        images.cpu(), bool_mask[scores, 0, ...].cpu(), alpha=0.3
    )
    actual_masked = torchvision.utils.draw_segmentation_masks(
        images.cpu(), gt_mask.squeeze(0).cpu(), alpha=0.3
    )
    pred_boxes = torchvision.utils.draw_bounding_boxes(
        images.cpu(), boxes[scores, ...].cpu()
    )
    actual_boxes = torchvision.utils.draw_bounding_boxes(
        images.cpu(), target[0]["boxes"].cpu()
    )

    img_list = [
        images.to(torch.uint8),
        actual_boxes.to(torch.uint8),
        actual_masked.to(torch.uint8),
        pred_boxes.to(torch.uint8),
        pred_masked.to(torch.uint8),
    ]

    # for im in img_list:
    #     print(im.shape, im.dtype, im.max(), im.min())

    _img = make_grid(img_list, nrow=1)

    writer.add_image(tag, _img, epoch, dataformats="CWH")


def write_progress(
    writer: SummaryWriter,
    tag: str,
    epoch: int,
    images: Tensor,
    masks: Tensor,
    target: Tensor,
    out: Tensor,
    cfg: CfgNode = None,
):
    """
    for writing training images to tensorboard

    :param writer:
    :param tag:
    :param epoch:
    :param images:
    :param masks:
    :param lsd:
    :param out:
    :return:
    """
    if images.ndim == 4:
        images = images.unsqueeze(-1).repeat((1, 1, 1, 1, 8))
        out = out.unsqueeze(-1).repeat((1, 1, 1, 1, 8))
        target = target.unsqueeze(-1).repeat((1, 1, 1, 1, 8))
        masks = masks.unsqueeze(-1).repeat((1, 1, 1, 1, 8))

    _a = images[0, [0, 0, 0], :, :, 7].cpu()
    _b = masks[0, [0, 0, 0], :, :, 7].gt(0.5).float().cpu()

    img_list = [_a, _b]

    c = target.shape[1] if target else out.shape[1]

    if cfg is not None and cfg.TRAIN.TARGET in [
        "affinities",
        "mtlsd",
        "aclsd",
        "omnipose",
    ]:
        if target is not None:
            img_list.append(target[0, c - 3 : c, ..., 7].float().cpu())

        if out is not None:
            img_list.append(out[0, c - 3 : c, ..., 7].float().cpu())
        c -= 3


    if cfg is not None and cfg.TRAIN.TARGET in ["lsd", "mtlsd", "aclsd"]:
        img_list.append(target[0, 0:3, ..., 7].float().cpu())
        img_list.append(out[0, 0:3, ..., 7].float().cpu())

        img_list.append(target[0, 3:6, ..., 7].float().cpu())
        img_list.append(out[0, 3:6, ..., 7].float().cpu())

        img_list.append(target[0, 6:9, ..., 7].float().cpu())
        img_list.append(out[0, 6:9, ..., 7].float().cpu())

        img_list.append(target[0, 9, ..., 7].expand(3, -1, -1).float().cpu())
        img_list.append(out[0, 9, ..., 7].expand(3, -1, -1).float().cpu())
        c -= 9

    # catch anything thats left.
    for index in range(c):
        if target is not None:
            img_list.append(target.float().cpu()[0, [index], ..., 7].expand(3, -1, -1))
        if out is not None:
            img_list.append(out.float().cpu()[0, [index], ..., 7].expand(3, -1, -1))

    _img = make_grid(img_list, nrow=2, normalize=True, scale_each=True)

    writer.add_image(tag, _img, epoch, dataformats="CWH")
