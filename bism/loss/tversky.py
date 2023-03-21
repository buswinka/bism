import torch
from torch import Tensor
import torch.nn as nn
from typing import Union, List
from bism.utils.cropping import crop_to_identical_size


class tversky(nn.Module):
    def __init__(self, alpha, beta, eps):
        """
        Returns tversky index of two torch.Tensors

        :param smooth: float
                - Very small number to ensure numerical stability. Default 1e-10
        :param alpha: float
                - Value which penalizes False Positive Values
        :param beta: float
                - Value which penalizes False Negatives
        :param gamma: float
                - Focal loss term
        """
        super(tversky, self).__init__()

        self.alpha = torch.tensor(alpha)
        self.beta = torch.tensor(beta)
        self.eps = torch.tensor(eps)

    def forward(self, predicted: Union[Tensor, List[Tensor]], ground_truth: Tensor) -> Tensor:

        if self.alpha.device != predicted.device:  # silently caches device
            self.alpha.to(predicted.device)
            self.beta.to(predicted.device)
            self.eps.to(predicted.device)

        futures: List[torch.jit.Future[torch.Tensor]] = []

        # List of Tensors
        if isinstance(predicted, list):
            for i, pred in enumerate(predicted):
                futures.append(
                    torch.jit.fork(self._tversky, pred, ground_truth[i, ...], self.alpha, self.beta, self.eps))

        # Already Batched Tensor
        elif isinstance(predicted, Tensor):
            for i in range(predicted.shape[0]):
                futures.append(
                    torch.jit.fork(self._tversky, predicted[i, ...], ground_truth[i, ...], self.alpha, self.beta,
                                   self.eps))

        results: List[Tensor] = []
        for future in futures:
            results.append(torch.jit.wait(future))

        return torch.mean(torch.stack(results))

    @staticmethod
    def _tversky(pred: Tensor, gt: Tensor, alpha: Tensor, beta: Tensor, eps: float = 1e-8) -> Tensor:
        """
        tversky loss on per image basis.

        Args:
            pred: [N, X, Y, Z] Tensor of predicted segmentation masks (N instances)
            gt: [N, X, Y, Z] Tensor of ground truth segmentation masks (N instances)
            alpha: Penalty to false positives
            beta: Penalty to false negatives
            eps: stability parameter

        Returns:
        """

        # ------------------- Expand Masks
        unique = torch.unique(gt)
        unique = unique[unique != 0]

        # assert gt.ndim == 4, f'{gt.shape=}'

        _, x, y, z = gt.shape
        nd_masks = torch.zeros((unique.shape[0], x, y, z), device=pred.device)
        for i, id in enumerate(unique):
            nd_masks[i, ...] = (gt == id).float().squeeze(0)

        pred, nd_masks = crop_to_identical_size(pred, nd_masks)


        true_positive: Tensor = pred.mul(nd_masks).sum()
        false_positive: Tensor = torch.logical_not(nd_masks).mul(pred).sum().add(1e-10).mul(alpha)
        false_negative: Tensor = ((1 - pred) * nd_masks).sum() * beta

        tversky = (true_positive + eps) / (true_positive + false_positive + false_negative + eps)

        return 1 - tversky

    def __repr__(self):
        return f'LossFn[name=tversky, alpha={self.alpha.item()}, beta={self.beta.item()}, eps={self.eps.item()}'
