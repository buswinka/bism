from typing import Union, List, Tuple, Dict

import torch
from tqdm import tqdm, trange
from torch import Tensor
from functools import cache


def metric(func):
    """
    Decorator which validates the input of a validation metric. Wrapped function should
    have at minimum, two arguments: gt and pred.

    :return: wrapped funciton
    """

    def wrapper(gt: Tensor, pred: Tensor, *args, **kwargs):
        assert gt.device() == pred.device(), f"{gt.device()=} != {pred.device()}"
        assert gt.shape() == pred.shape(), f"{gt.device()=} != {pred.device()}"
        func(gt, pred, *args, **kwargs)

    return wrapper


def _cast_as_tensor(a: Union[Tensor | float | List[float]]) -> Tensor:
    """casts a float or list of floats to a tensor"""
    if isinstance(a, float):
        a = torch.tensor(a)
    elif isinstance(a, list):
        a = torch.tensor(a)
    elif isinstance(a, Tensor):
        pass
    else:
        raise ValueError(
            f"casting of argument of type {type(a)} to tensor is not supported"
        )
    return a


@cache
def mask_iou(gt: Tensor, pred: Tensor, verbose: bool = False):
    """
    Calculates the IoU of each object on a per-mask-basis.

    :param gt: mask 1 with N instances
    :param pred: mask 2 with M instances
    :return: NxM matrix of IoU's
    """
    assert gt.shape == pred.shape, "Input tensors must be the same shape"
    assert gt.device == pred.device, "Input tensors must be on the same device"

    a_unique = gt.unique()
    a_unique = a_unique[a_unique > 0]

    b_unique = pred.unique()
    b_unique = b_unique[b_unique > 0]

    iou = torch.zeros(
        (a_unique.shape[0], b_unique.shape[0]), dtype=torch.float, device=gt.device
    )

    iterator = (
        tqdm(enumerate(a_unique), total=len(a_unique))
        if verbose
        else enumerate(a_unique)
    )
    for i, au in iterator:
        _a = gt.eq(au)

        # we only calculate iou of lables which have "contact with" our mask
        touching = pred[_a].unique()
        touching = touching[touching != 0]

        for j, bu in enumerate(b_unique):
            if torch.any(touching == bu):
                _b = pred == bu

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

    true_positive = torch.zeros_like(thr)
    false_positive = torch.zeros_like(thr)
    false_negative = torch.zeros_like(thr)

    iou: Tensor = mask_iou(gt, pred, verbose=False)

    for i, t in enumerate(thr):

        gt_max, gt_indicies = iou.max(dim=1)
        gt = torch.logical_not(gt_max.gt(thr)) if iou.shape[1] > 0 else torch.ones(0)
        pred = (
            torch.logical_not(iou.max(dim=0)[0].gt(thr))
            if iou.shape[0] > 0
            else torch.ones(0)
        )

        true_positive[i] = torch.sum(torch.logical_not(gt))
        false_negative[i] = torch.sum(pred)
        false_negative[i] = torch.sum(gt)

    return (
        true_positive.cpu(),
        false_positive.cpu(),
        false_negative.cpu(),
    )


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
    return torch.mean(precision(gt, pred, thr))

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
    tp, fp, fn = confusion_matrix(gt, pred, thr)

    return tp / (tp + fp)


@metric
def recall(gt: Tensor, pred: Tensor, thr: Union[Tensor | float | List[float]] = 0.75):
    tp, fp, fn = confusion_matrix(gt, pred, thr)
    return tp / (tp + fn)


@metric
def accuracy(gt: Tensor, pred: Tensor, thr: Union[Tensor | float | List[float]] = 0.75):
    tp, fp, fn = confusion_matrix(gt, pred, thr)
    return tp / (tp + fp + fn)

@metric
@cache
def _iou_instance_dict(gt: Tensor, pred: Tensor) -> Dict[int, Tensor]:
    """
    Given two instance masks, compares each instance in b against a. Usually assumes A is the ground truth.

    :param a: Mask A
    :param b: Mask B
    :return:  Dict of instances and every IOU for each instance
    """
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


def summary(gt, pred):

    thr = torch.linspace(0, 1, 25)

    ap = average_precision(gt, pred, thr)
    tp75, fp75, fn75 = confusion_matrix(gt, pred, 0.75)

    print(f"""
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
    ===================
    """)