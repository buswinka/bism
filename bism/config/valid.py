import torch
import lion_pytorch

import bism.loss.cl_dice
import bism.loss.tversky
import bism.loss.dice
import bism.loss.jaccard
import bism.loss.omnipose
import bism.loss.torchvision

_valid_optimizers = {
    'adamw': torch.optim.AdamW,
    'adam': torch.optim.Adam,
    'sgd': torch.optim.SGD,
    'lion': lion_pytorch.Lion,
    'adamax': torch.optim.Adamax
}

_valid_loss_functions = {
    'soft_dice_cldice': bism.loss.cl_dice.soft_dice_cldice,
    'soft_cldice': bism.loss.cl_dice.soft_cldice,
    'tversky': bism.loss.tversky.tversky,
    'dice': bism.loss.dice.dice,
    'jaccard': bism.loss.jaccard.jaccard,
    'mse': torch.nn.MSELoss,
    'omnipose': bism.loss.omnipose.omnipose_loss,
    'torchvision': bism.loss.torchvision.sumloss,


}

_valid_lr_schedulers = {
    'cosine_annealing_warm_restarts': torch.optim.lr_scheduler.CosineAnnealingWarmRestarts
}
