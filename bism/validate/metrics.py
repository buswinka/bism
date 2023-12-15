from typing import Union, List, Tuple, Dict, Any, Callable
from functools import partial

import torch
from tqdm import tqdm, trange
from torch import Tensor
from functools import cache
import skimage.io
import numpy as np
import bism.utils.io
import logging
import matplotlib.pyplot as plt

import torchvision.ops

from typing import DefaultDict


class DataDict(DefaultDict):
    masks: Tensor
    boxes: Tensor
    labels: Tensor
    score: Tensor


def metric(func):
    """
    Decorator which validates the input of a validation metric. Wrapped function should
    have at minimum, two arguments: gt and pred.

    :return: wrapped funciton
    """

    def wrapper(gt: Tensor, pred: Tensor, *args, **kwargs):
        assert gt.device == pred.device, f"{gt.device=} != {pred.device}"
        assert gt.shape == pred.shape, f"{gt.shape=} != {pred.shape}"

        return func(gt, pred, *args, **kwargs)

    return wrapper


def _cast_as_tensor(a: Union[Tensor | float | List[float]]) -> Tensor:
    """casts a float or list of floats to a tensor"""
    if isinstance(a, float):
        a = torch.tensor([a])
    elif isinstance(a, list):
        a = torch.tensor(a)
    elif isinstance(a, Tensor):
        pass
    else:
        raise ValueError(
            f"casting of argument of type {type(a)} to tensor is not supported"
        )
    return a


# @cache
def mask_iou(gt: Tensor, pred: Tensor, verbose: bool = False):
    """
    Calculates the IoU of each object on a per-mask-basis.

    :param gt: mask 1 with N instances
    :param pred: mask 2 with M instances
    :return: NxM matrix of IoU's
    """
    assert gt.shape == pred.shape, "Input tensors must be the same shape"
    assert gt.device == pred.device, "Input tensors must be on the same device"

    logging.info(
        f"Calculating iou between ground truth and predicted segmentation mask: {gt.shape=}, {pred.shape=}"
    )

    a_unique = gt.unique()
    a_unique = a_unique[a_unique > 0]
    logging.debug(f"{a_unique.shape} unique gt: {a_unique}")

    b_unique = pred.unique()
    b_unique = b_unique[b_unique > 0]
    logging.debug(f"{b_unique.shape} unique pred: {b_unique}")

    iou = torch.zeros(
        (a_unique.shape[0], b_unique.shape[0]), dtype=torch.float, device=gt.device
    )
    logging.debug(
        f"Unique gt: {a_unique.shape}. Unique pred: {b_unique.shape}. IoU matrix shape: {iou.shape=}"
    )

    iterator = (
        tqdm(enumerate(a_unique), total=len(a_unique))
        if verbose
        else enumerate(a_unique)
    )
    for i, au in iterator:
        logging.debug(
            f"Instance {i}/{a_unique.shape[0]} ({i/a_unique.shape[0]*100:0.1f}%) with id: {au}"
        )

        _a = gt.eq(au).squeeze()

        # we only calculate iou of lables which have "contact with" our mask
        touching = pred[_a].unique()
        touching = touching[touching != 0]

        for j, bu in enumerate(b_unique):
            if torch.any(touching == bu):
                _b = pred.eq(bu).squeeze()

                intersection = torch.logical_and(_a, _b).sum()
                union = torch.logical_or(_a, _b).sum()

                iou[i, j] = intersection / union
            else:
                iou[i, j] = 0.0

    return iou


@metric
@cache
def confusion_matrix(
    gt: Tensor, pred: Tensor, thr: Union[Tensor | float | List[float]] = 0.75
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Calculates and returns the TP, FP, and FN rates of a ground_truth vs a prediction

    :param gt:
    :param pred:
    :param thr:

    :return: (TP, FP, FN)
    """
    thr: Tensor = _cast_as_tensor(thr)

    logging.info(f"Calculating TP, FP, FN confusion matrix with threshold {thr=}")

    true_positive = torch.zeros_like(thr)
    false_positive = torch.zeros_like(thr)
    false_negative = torch.zeros_like(thr)

    iou: Tensor = mask_iou(gt, pred, verbose=False)

    for i, t in enumerate(thr):
        logging.debug(
            f"Running Iou at thr:{t} - {i}/{thr.shape[0]} ({i/thr.shape[0]*100:0.1f}%)"
        )

        gt_max, gt_indicies = iou.max(dim=1)
        gt = torch.logical_not(gt_max.gt(t)) if iou.shape[1] > 0 else torch.ones(0)
        pred = (
            torch.logical_not(iou.max(dim=0)[0].gt(t))
            if iou.shape[0] > 0
            else torch.ones(0)
        )

        true_positive[i] = torch.sum(torch.logical_not(gt))
        false_negative[i] = torch.sum(pred)
        false_negative[i] = torch.sum(gt)

    out = (
        true_positive.cpu() / iou.shape[0],
        false_positive.cpu() / iou.shape[0],
        false_negative.cpu() / iou.shape[0],
    )
    logging.info(f"Confusion Matrix (TP, FP, FN): {out}")
    logging.info(f"TP: {out[0]}")
    logging.info(f"FP: {out[1]}")
    logging.info(f"FN: {out[2]}")

    return out


@metric
def average_precision(
    gt: Tensor, pred: Tensor, thr: Union[Tensor | float | List[float]] = 0.75
):
    """
    Calculates the average precision of two instance segmentation masks.

    :param gt:
    :param pred:
    :param thr:
    :return:
    """
    thr: Tensor = _cast_as_tensor(thr)

    logging.info(f"Calculating Average Precision at thr {thr=}")
    out = torch.mean(precision(gt, pred, thr))

    logging.info(f"AP@{thr=}: {out}")

    return out


@metric
def precision(
    gt: Tensor, pred: Tensor, thr: Union[Tensor | float | List[float]] = 0.75
):
    """

    TP / (TP + FP) @ a thr value

    :param gt:
    :param pred:
    :param thr:
    :return:
    """
    thr = _cast_as_tensor(thr)

    logging.info(f"Calculating Precision at thr {thr=}")
    tp, fp, fn = confusion_matrix(gt, pred, thr)
    out = tp / (tp + fp)

    out[tp == 0] = 0
    if len(thr) > 1:
        plt.plot(thr.numpy(), tp.numpy(), "-")
        plt.plot(thr.numpy(), fp.numpy(), "-")
        plt.plot(thr.numpy(), fn.numpy(), "-")
        plt.legend(["tp", "fp", "fn"])
        plt.xlabel("thr")
        plt.ylabel("metric")
        plt.show()

        plt.plot(thr.numpy(), out.numpy(), "-")
        plt.xlabel("thr")
        plt.ylabel("precision")
        plt.show()

    logging.info(f"Precision: {out}")
    return out


@metric
def recall(gt: Tensor, pred: Tensor, thr: Union[Tensor | float | List[float]] = 0.75):
    logging.info(f"Predicting recall at thr {thr=}")
    tp, fp, fn = confusion_matrix(gt, pred, thr)
    out = tp / (tp + fn)
    logging.info(f"Recall: {out}")
    return out


@metric
def accuracy(gt: Tensor, pred: Tensor, thr: Union[Tensor | float | List[float]] = 0.75):
    logging.info(f"Predicting accuracy at thr {thr=}")
    tp, fp, fn = confusion_matrix(gt, pred, thr)
    out = tp / (tp + fp + fn)
    logging.info(f"Accuracy: {out}")
    return out


@metric
@cache
def _iou_instance_dict(gt: Tensor, pred: Tensor) -> Dict[int, Tensor]:
    """
    Given two instance masks, compares each instance in b against a. Usually assumes A is the ground truth.

    :param a: Mask A
    :param b: Mask B
    :return:  Dict of instances and every IOU for each instance
    """
    logging.debug(f"Calculating an iou instance dict from two segmentation masks")
    gt_unique = gt.unique()
    gt_unique = gt_unique[gt_unique > 0]

    pred_unique = pred.unique()
    pred_unique = pred_unique[pred_unique > 0]

    iou = {}

    for i, au in tqdm(enumerate(gt_unique), total=len(gt_unique)):
        _gt = gt == au

        touching = pred[
            _gt
        ].unique()  # we only calculate iou of lables which have "contact with" our mask
        touching = touching[touching != 0]
        iou[au] = []

        for j, predu in enumerate(pred_unique):
            if torch.any(touching == predu):
                _pred = pred == predu

                intersection = torch.logical_and(_gt, _pred).sum()
                union = torch.logical_or(_gt, _pred).sum()
                iou[au].append((intersection / union).item())

    return iou


def get_segmentation_errors(ground_truth: Tensor, predicted: Tensor) -> float:
    """
    Calculates the IoU of each object on a per-mask-basis.

    :param ground_truth: mask 1 with N instances
    :param predicted: mask 2 with M instances
    :return: NxM matrix of IoU's
    """
    logging.info(f"Determining rate of segmentation errors")

    iou = _iou_instance_dict(ground_truth, predicted)
    for k, v in iou.items():
        iou[k] = torch.tensor(v)

    num_split = 0
    for k, v in iou.items():
        if v.gt(0.2).int().sum() > 1:
            num_split += 1

    over_segmentation_rate = num_split / len(iou)

    iou = _iou_instance_dict(predicted, ground_truth)
    for k, v in iou.items():
        iou[k] = torch.tensor(v)

    num_split = 0
    for k, v in iou.items():
        if v.gt(0.2).int().sum() > 1:
            num_split += 1

    under_segmentation_rate = num_split / len(iou)
    logging.info(f"over segmentation rate: {over_segmentation_rate}")
    logging.info(f"under segmentation rate: {under_segmentation_rate}")
    return over_segmentation_rate, under_segmentation_rate


@metric
def one_to_one(gt, pred):
    raise NotImplementedError


@metric
def over_segmentation_rate(gt, pred):
    raise NotImplementedError


@metric
def under_segmentation_rate(gt, pred):
    raise NotImplementedError


def _colapse_mask(mask, labels, thr):
    """ mask [N, 1, X, Y] """
    mask = mask.gt(thr).squeeze()  # N, X, Y
    mask = mask * torch.arange(1, mask.shape[0] + 1, device=mask.device).view(-1, 1, 1)

    out = []
    for u in labels.unique():
        _m = mask[labels == u, ...]
        _m = _m if _m.numel() > 0 else torch.zeros_like(mask[0, ...])
        _m = _m.max(0)[0].unsqueeze(0)  # [1, x, y]
        out.append(_m)
    return torch.cat(out, dim=0)


def calculate_summary_metrics_from_data_dicts(
    gt: DataDict,
    pred: DataDict,
    thr: List[float] | Tensor,
    mask_thr: float = 0.5,
    overlap_thr: float = 0.5,
):

    thr: Tensor = _cast_as_tensor(thr)

    pred["masks"] = pred["masks"].gt(mask_thr)

    # pred["masks"] = _colapse_mask(pred['masks'], pred['labels'], mask_thr)
    gt["masks"] = _colapse_mask(gt["masks"], gt["labels"], mask_thr)

    true_positive = torch.zeros_like(thr)
    false_positive = torch.zeros_like(thr)
    false_negative = torch.zeros_like(thr)

    for i, t in enumerate(thr):
        logging.info(
            f"Running Iou at thr:{t} - {i}/{thr.shape[0]} ({i/thr.shape[0]*100:0.1f}%)"
        )

        ind = pred["scores"].clone().gt(t)

        boxes = pred["boxes"].clone()[ind, :]
        labels = pred["labels"].clone()[ind]
        pred_masks = _colapse_mask(pred["masks"].clone()[ind, 0, ...], labels, mask_thr)

        # plt.imshow(pred_masks.permute(1,2,0).numpy())
        # plt.title('pred')
        # plt.show()
        # plt.imshow(gt['masks'].permute(1,2,0).numpy())
        # plt.title('gt')
        # plt.show()
        # raise RuntimeError

        iou: Tensor = mask_iou(gt["masks"].clone(), pred_masks, verbose=False) # N x M

        logging.debug(f"{iou.shape=} -> {iou.max()=}")

        _gt = (
            iou.max(dim=1)[0].gt(overlap_thr)
            if iou.shape[1] > 0
            else torch.ones(0)
        )
        _pred = (
            iou.max(dim=0)[0].lt(overlap_thr)
            if iou.shape[0] > 0
            else torch.ones(0)
        )

        true_positive[i] = _gt.sum() / _gt.shape[0]
        false_positive[i] = torch.sum(_pred) / _pred.shape[0]
        false_negative[i] = torch.sum(torch.logical_not(_gt)).sum() / _gt.shape[0]

    out = (
        true_positive.cpu(),
        false_positive.cpu(),
        false_negative.cpu(),
    )
    return out

def matrix_nms(seg_masks, cate_labels, cate_scores, kernel='gaussian', sigma=2.0, sum_masks=None):
    """Matrix NMS for multi-class masks.

    Args:
        seg_masks (Tensor): shape (n, h, w)
        cate_labels (Tensor): shape (n), mask labels in descending order
        cate_scores (Tensor): shape (n), mask scores in descending order
        kernel (str):  'linear' or 'gauss'
        sigma (float): std in gaussian method
        sum_masks (Tensor): The sum of seg_masks

    Returns:
        Tensor: cate_scores_update, tensors of shape (n)
    """
    n_samples = len(cate_labels)
    if n_samples == 0:
        return []
    if sum_masks is None:
        sum_masks = seg_masks.sum((1, 2)).float()
    seg_masks = seg_masks.reshape(n_samples, -1).float()
    # inter.
    inter_matrix = torch.mm(seg_masks, seg_masks.transpose(1, 0))
    # union.
    sum_masks_x = sum_masks.expand(n_samples, n_samples)
    # iou.
    iou_matrix = (inter_matrix / (sum_masks_x + sum_masks_x.transpose(1, 0) - inter_matrix)).triu(diagonal=1)
    # label_specific matrix.
    cate_labels_x = cate_labels.expand(n_samples, n_samples)
    label_matrix = (cate_labels_x == cate_labels_x.transpose(1, 0)).float().triu(diagonal=1)

    # IoU compensation
    compensate_iou, _ = (iou_matrix * label_matrix).max(0)
    compensate_iou = compensate_iou.expand(n_samples, n_samples).transpose(1, 0)

    # IoU decay
    decay_iou = iou_matrix * label_matrix

    # matrix nms
    if kernel == 'gaussian':
        decay_matrix = torch.exp(-1 * sigma * (decay_iou ** 2))
        compensate_matrix = torch.exp(-1 * sigma * (compensate_iou ** 2))
        decay_coefficient, _ = (decay_matrix / compensate_matrix).min(0)
    elif kernel == 'linear':
        decay_matrix = (1-decay_iou)/(1-compensate_iou)
        decay_coefficient, _ = decay_matrix.min(0)
    else:
        raise NotImplementedError

    # update the score.
    cate_scores_update = cate_scores * decay_coefficient
    return cate_scores_update


@metric
def calculate_summary_metrics_from_images(
    gt: Tensor, pred: Tensor, verbose: bool = False
) -> Dict[str, float]:
    thr = torch.linspace(0, 1, 50)

    logging.info("Calculating summary metrics")

    ap = average_precision(gt, pred, thr)
    tp75, fp75, fn75 = confusion_matrix(gt, pred, 0.75)
    over, under = get_segmentation_errors(gt, pred)

    summary = {
        "AP": ap,
        "AP-75": average_precision(gt, pred, 0.75).item(),
        "TP-75": tp75.item(),
        "FP-75": fp75.item(),
        "FN-75": fn75.item(),
        "Accuracy-0.5": accuracy(gt, pred, 0.5).item(),
        "Accuracy-0.75": accuracy(gt, pred, 0.75).item(),
        "Accuracy-0.95": accuracy(gt, pred, 0.95).item(),
        "Precision-0.75": precision(gt, pred, 0.75).item(),
        "Recall-0.75": recall(gt, pred, 0.75).item(),
        "over_segmentation_rate": over,
        "under_segmentation_rate": under,
    }
    if verbose:
        print(
            f"""
        Accuracy Statistics
        ===================
        AP             | {ap.item()}
        AP-75          | {average_precision(gt, pred, 0.75).item()}
        TP-75          | {tp75.item()}
        FP-75          | {fp75.item()}
        FN-75          | {fn75.item()}
        Accuracy-0.5   | {accuracy(gt, pred, 0.5).item()}
        Accuracy-0.75  | {accuracy(gt, pred, 0.75).item()}
        Accuracy-0.95  | {accuracy(gt, pred, 0.95).item()}
        Precision-0.75 | {precision(gt, pred, 0.75).item()}
        Recall-0.75    | {recall(gt, pred, 0.75).item()}
        OverSegRate    | {over}
        UnderSegRate   | {under}
        ===================
        """
        )

    return summary


def accuracy_from_images(ground_truth_path: str, predicted_path: str) -> None:
    """

    :param ground_truth_path:
    :param predicted_path:
    :return:
    """
    logging.info(f"GT Image Path: {ground_truth_path}")
    logging.info(f"Pred Image Path: {predicted_path}")

    gt = torch.from_numpy(skimage.io.imread(ground_truth_path).astype(np.int32))
    pred = torch.from_numpy(skimage.io.imread(predicted_path).astype(np.int32))

    summary = calculate_summary_metrics_from_images(gt, pred, verbose=True)


if __name__ == "__main__":
    from yacs.config import CfgNode
    import torch.nn as nn
    from bism.models.construct import cfg_to_torchvision_model
    from bism.train.dataloader import torchvision_colate, dataset, MultiDataset
    from torch.utils.data import DataLoader
    from bism.targets.torchvision import maskrcnn
    import logging
    import numpy as np
    from copy import deepcopy

    # logging.basicConfig(
    #     level=2,
    #     format="[%(asctime)s] bism-validate [%(levelname)s]: %(message)s",
    # )

    def cuda(a: Dict[str, Tensor]):
        for k, v in a.items():
            a[k] = v.cuda()
        return a

    device = "cuda:0"
    model_path = "trained_model_files/maskrcnn/Dec02_22-56-46_CHRISUBUNTU.trch"
    model_file: Dict[str, Any] = torch.load(model_path, map_location="cpu")

    cfg: CfgNode = model_file["cfg"]
    state_dict = model_file["model_state_dict"]

    model: nn.Module = cfg_to_torchvision_model(cfg)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    _datasets = []
    for path in ["data/stereocilia/val"]:
        _datasets.append(
            dataset(
                path=path,
                transforms=lambda x: x,
                sample_per_image=1,
                device=device,
                pad_size=10,
            ).to(device)
        )

    merged_train = MultiDataset(*_datasets)
    dataloader = DataLoader(merged_train, num_workers=0, collate_fn=torchvision_colate)

    target_fn: Callable[[Tensor], Tensor] = partial(maskrcnn, cfg=cfg)

    all_tp = []
    all_fp = []
    all_fn = []

    for images, masks in dataloader:

        images = [im.float().to(device) for im in images]

        target: DataDict = target_fn(masks)[0]
        out: DataDict = model(images)[0]


        out['scores'] =  matrix_nms(out['masks'].squeeze(1), out['labels'], out['scores'])

        tp, fp, fn = calculate_summary_metrics_from_data_dicts(
            target, out, thr=[i / 10 for i in range(10)], overlap_thr=0.66
        )

        all_tp.append(tp)
        all_fp.append(fp)
        all_fn.append(fn)

        ax0 = plt.subplot(221)
        ax0.imshow(target["masks"].gt(0.5).float().cpu().permute(1, 2, 0).numpy())

        mask = _colapse_mask(out["masks"], out["labels"], 0.5)
        ax1 = plt.subplot(222)
        ax1.imshow(mask.squeeze().gt(0.5).float().cpu().permute(1, 2, 0).numpy())

        ax2 = plt.subplot(212)
        ax2.plot(np.linspace(0, 1, 10), tp.cpu().numpy(), "-")
        ax2.plot(np.linspace(0, 1, 10), fp.cpu().numpy(), "-")
        ax2.plot(np.linspace(0, 1, 10), fn.cpu().numpy(), "-")
        ax2.legend(["tp", "fp", "fn"])

        plt.show()

    def plot(arr: List[Tensor], c):
        x = np.linspace(0, 1, 10)
        for tp in arr:
            plt.plot(x, tp.cpu().numpy(), c=c, alpha=0.2)
        arr = torch.cat([x.unsqueeze(0) for x in arr], dim=0).mean(0)
        plt.plot(x, arr.cpu().numpy(), c=c, alpha=1.0)

    plot(all_tp, "C0")
    plot(all_fp, "C1")
    plot(all_fn, "C2")
    plt.legend(['tp', 'fp', 'fn'])
    plt.show()

    precision = [tp / (tp + fp) for tp, fn, fp in zip(all_tp, all_fn, all_fp)]
    plot(precision, 'C3')
    plt.ylabel('precision')
    plt.show()

    recall = [tp / (tp + fn) for tp, fn, fp in zip(all_tp, all_fn, all_fp)]
    plot(recall , 'C4')
    plt.ylabel('recall')
    plt.show()

    accuracy = [ tp / (tp + fp + fn) for tp, fn, fp in zip(all_tp, all_fn, all_fp)]
    plot(accuracy , 'C5')
    plt.ylabel('accuracy')
    plt.show()

