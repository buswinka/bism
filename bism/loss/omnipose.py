import torch
import torch.nn as nn

from bism.utils.cropping import crop_to_identical_size


class omnipose_loss(nn.Module):
    def __init__(self, *args, **kwargs):
        super(omnipose_loss, self).__init__()
        self.mse_loss = torch.nn.MSELoss()

    def forward(
        self, predicted: torch.Tensor, ground_truth: torch.Tensor, smooth: float = 1e-10
    ) -> torch.Tensor:
        """
        Returns dice index for first channel, and L2 loss for all others.

        :param predicted: [B, 5, X, Y, Z] torch.Tensor
                - predicted tensor of a model trained on the output of the omnipose target function
        :param ground_truth: [B, 5, X, Y, Z] torch.Tensor
                - the output of the omnipose target function
        :param smooth: float
                - Very small number to ensure numerical stability. Default 1e-10
        :return: dice_loss: [1] torch.Tensor
                - Result of Loss Function Calculation
        """

        # Crop both tensors to the same shape
        predicted, ground_truth = crop_to_identical_size(predicted, ground_truth)

        # We'll just run dice on the semantic mask.
        intersection = (predicted[:, 0, ...] * ground_truth[:, 0, ...]).sum().add(smooth)
        denominator = (predicted[:, 0, ...] + ground_truth[:, 0, ...]).sum().add(smooth)
        loss = 1 - (2 * intersection / denominator)

        return loss + self.mse_loss(predicted[:, 1::, ...], ground_truth[:, 1::, ...])
