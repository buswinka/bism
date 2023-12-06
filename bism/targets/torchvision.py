import torch
from torch import Tensor
from yacs.config import CfgNode
from typing import Tuple, Dict, List, TypedDict
import torchvision.ops


class DataDict(TypedDict):
    boxes: Tensor
    labels: Tensor
    masks: Tensor


def maskrcnn(batched_masks: List[Tensor], cfg: CfgNode) -> List[DataDict]:
    """
    Takes a batched set of 2D masks and converts them into a list of data dictonaries for training.

    Torchvision requires a list of data to handle batching.

    :param batched_masks: UInt8Tensor[B, C=3, X, Y] where C=0 is short stereocilia, C=1 is tall stereocilia, and C=3
    is large stereocilia
    :param cfg: YACS Config Node - none are used.
    :return: List[DataDict] a batch of training examples
    """

    # convert masks from a [3, X, Y] mask of N uint8 labels, to a [N, X, Y] tensor of uint8 binary segmentation masks
    # get a FloatTensor[N, 4] for each mask [x1, y1, x2, y2]
    # an Int64Tensor[N] for each label

    # assert len(batched_masks.shape) == 4, f'{batched_masks.shape=}'
    # assert batched_masks.shape[1] == 3, f'{batched_masks.shape=}'

    output: List[DataDict] = []

    for b, masks in enumerate(batched_masks):  # Loop over batched data
        c, x, y = masks.shape

        # masks = batched_masks[b, ...]

        # DataDict needs to have keys: 'labels', 'masks', 'boxes'
        total_instances = 0
        for i in range(masks.shape[0]):
            un = masks[i, ...].unique()
            un = un[un != 0]
            total_instances += len(un)  # will always be zero

        # print(batched_masks.shape, masks.shape, total_instances, masks[0, ...].max(), masks[1,...].max(), masks[2, ...].max())
        boxes = torch.zeros((total_instances, 4), device=masks.device)
        labels = torch.zeros((total_instances,), device=masks.device)
        binary_masks = torch.zeros((total_instances, x, y), device=masks.device)

        i = 0
        for l in range(masks.shape[0]):
            for u in masks[l, ...].unique():
                if u == 0:
                    continue

                _mask = masks[l, ...].eq(u).float()

                # BBOX
                nonzero = _mask.nonzero()  # [N, 3]

                if nonzero.shape[0] == 0:
                    continue

                # y0, y1 = nonzero[:, 0].min(), nonzero[:, 0].max()
                # x0, x1 = nonzero[:, 1].min(), nonzero[:, 1].max()
                # boxes[i, :] = torch.tensor([x0, y0, x1, y1])
                #
                # Masks
                binary_masks[i, ...] = _mask

                # Labels
                labels[i] = l+1

                i += 1 # next index


        boxes = torchvision.ops.masks_to_boxes(binary_masks)
        inds = torchvision.ops.remove_small_boxes(boxes, min_size=3)
        binary_masks = binary_masks[inds, ...]
        labels = labels[inds]
        boxes = boxes[inds, :]

        output.append({'boxes': boxes, 'labels': labels.to(torch.int64), 'masks': binary_masks})

    return output