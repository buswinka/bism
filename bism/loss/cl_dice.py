import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, List, Union

def soft_erode(img: Tensor) -> Tensor:
    """ approximates morphological operations through max_pooling for 2D and 3D """
    if len(img.shape) == 4:
        p1 = -F.max_pool2d(-img, (3, 1), (1, 1), (1, 0))
        p2 = -F.max_pool2d(-img, (1, 3), (1, 1), (0, 1))
        return torch.min(p1, p2)
    elif len(img.shape) == 5:
        p1 = -F.max_pool3d(-img, (3, 1, 1), (1, 1, 1), (1, 0, 0))
        p2 = -F.max_pool3d(-img, (1, 3, 1), (1, 1, 1), (0, 1, 0))
        p3 = -F.max_pool3d(-img, (1, 1, 3), (1, 1, 1), (0, 0, 1))
        return torch.min(torch.min(p1, p2), p3)


def soft_dilate(img: Tensor) -> Tensor:
    """ approximates morphological operations through max_pooling for 2D and 3D """
    if len(img.shape) == 4:
        return F.max_pool2d(img, (3, 3), (1, 1), (1, 1))
    elif len(img.shape) == 5:
        return F.max_pool3d(img, (3, 3, 3), (1, 1, 1), (1, 1, 1))


def soft_open(img: Tensor) -> Tensor:
    """ approximates morphological operations through max_pooling for 2D and 3D """
    return soft_dilate(soft_erode(img))


def soft_skeletonize(img: Tensor, iter_: int) -> Tensor:
    """
    Performs a soft-skeletonization by terativly performing "soft morphological operations"

    :param img: Image to perform operation on
    :param iter_: Number of times to perform the operation
    :return: Soft-skeleton
    """
    img1 = soft_open(img)
    skel = F.relu(img - img1)
    for j in range(iter_):
        img = soft_erode(img)
        img1 = soft_open(img)
        delta = F.relu(img - img1)
        skel = skel + F.relu(delta - skel * delta)
    return skel


def soft_dice(predicted: Tensor, ground_truth: Tensor, smooth: int = 1) -> Tensor:
    """
    Computes the soft dice metric

    :param ground_truth:
    :param predicted:
    :param smooth: smoothing factor to prevent division by zero
    :return:
    """
    intersection = torch.sum((ground_truth * predicted))
    coeff = (2. * intersection + smooth) / (
            torch.sum(ground_truth) + torch.sum(predicted) + smooth)



    return (1. - coeff)


class soft_cldice(nn.Module):
    def __init__(self, iter_=3, smooth=1.):
        super(soft_cldice, self).__init__()
        self.iter = iter_
        self.smooth = smooth

    def forward(self, predicted: Tensor, ground_truth: Tensor) -> Tensor:
        """
        Calculates the soft-clDice metric on a true and predicted value

        :param ground_truth:
        :param predicted:
        :return:
        """
        skeleton_predicted = soft_skeletonize(predicted, self.iter)
        skeleton_true = soft_skeletonize(ground_truth, self.iter)

        tprec = (torch.sum(torch.multiply(skeleton_predicted, ground_truth)[:, 1:, ...]) + self.smooth) / (
                torch.sum(skeleton_predicted[:, 1:, ...]) + self.smooth)
        tsens = (torch.sum(torch.multiply(skeleton_true, predicted)[:, 1:, ...]) + self.smooth) / (
                torch.sum(skeleton_true[:, 1:, ...]) + self.smooth)

        cl_dice = 1. - 2.0 * (tprec * tsens) / (tprec + tsens)

        return cl_dice


class soft_dice_cldice(nn.Module):
    def __init__(self, iter_=3, alpha=0.5, smooth=1.):
        super(soft_dice_cldice, self).__init__()
        self.iter = iter_
        self.smooth = smooth
        self.alpha = alpha

    def forward(self, predicted: Tensor, ground_truth: Tensor) -> Tensor:
        """
        Calculates a singular loss value combining soft-Dice and soft-clDice which can be used to train
        a neural network

        :param predicted: Input tensor
        :param ground_truth: Ground Truth Tensor
        :return: Single value which to perform a backwards pass
        """

        dice = soft_dice(ground_truth, predicted)

        skel_pred = soft_skeletonize(predicted, self.iter)
        skel_true = soft_skeletonize(ground_truth, self.iter)

        tprec = (torch.sum(skel_pred * ground_truth) + self.smooth) / (
                    torch.sum(skel_pred) + self.smooth)
        tsens = (torch.sum(skel_true * predicted) + self.smooth) / (
                    torch.sum(skel_true) + self.smooth)

        cl_dice = 1. - 2.0 * (tprec * tsens) / (tprec + tsens)


        return (1.0 - self.alpha) * dice + self.alpha * cl_dice


if __name__ == '__main__':
    lossfn = soft_dice_cldice()

    predicted = torch.rand((1, 1, 20, 20, 10), device='cpu')
    gt = torch.rand((1, 1, 20, 20, 10), device='cpu').round().float()
    a = lossfn(predicted, gt)
    print(a)
