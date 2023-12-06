from typing import Tuple, Dict, List

import torch
import torch.nn as nn
from torch import Tensor


class sumloss(nn.Module):
    def __init__(self, loss_weights: List[float]):
        super(sumloss, self).__init__()
        self.loss_weights = loss_weights

    def forward(self, outputs_losses: Dict[str, Tensor], *args, **kwargs) -> Tuple[Tensor, Tensor]:
        """
        Helper target for summing the losses output by default from a torchvision segmentation model

        :param outputs_losses: Precomputed losses from torchvision
        :param cfg: YACS configuration Node
        :return: summed losses
        """
        for k, v in outputs_losses.items():
            device = v.device
            break

        total = torch.zeros((1,), device=device)

        assert len(self.loss_weights) == len(
            outputs_losses.keys()), f'{len(self.loss_weights)=},  {len(outputs_losses.keys())=}'

        for weight, k in zip(self.loss_weights, outputs_losses.keys()):
            total += outputs_losses[k] * torch.tensor([weight, ], device=device)

        return total
