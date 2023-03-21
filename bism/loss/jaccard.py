import torch
from torch import Tensor
import torch.nn as nn
from typing import Union, List
from bism.utils.cropping import crop_to_identical_size


class jaccard(nn.Module):
    def __init__(self):
        super(jaccard, self).__init__()

    def forward(self, predicted: torch.Tensor, ground_truth: torch.Tensor,
                eps: float = 1e-10) -> torch.Tensor:
        """
        Returns jaccard index of two torch.Tensors

        :param predicted: [B, I, X, Y, Z] torch.Tensor
                - probabilities calculated from hcat.utils.embedding_to_probability
                  where B: is batch size, I: instances in image
        :param ground_truth: [B, I, X, Y, Z] torch.Tensor
                - segmentation mask for each instance (I).
        :param eps: float
                - Very small number to ensure numerical stability. Default 1e-10
        :return: jaccard_loss: [1] torch.Tensor
                - Result of Loss Function Calculation
        """

        # Crop both tensors to the same shape
        predicted, ground_truth = crop_to_identical_size(predicted, ground_truth)

        intersection = (predicted * ground_truth).sum().add(eps)
        union = (predicted + ground_truth).sum().sub(intersection).add(eps)

        return 1.0 - (intersection / union)


