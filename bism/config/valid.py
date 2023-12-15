import torch
import lion_pytorch

import bism.loss.cl_dice
import bism.loss.tversky
import bism.loss.dice
import bism.loss.jaccard
import bism.loss.omnipose
import bism.loss.torchvision
import bism.loss.iadb

from bism.targets.affinities import affinities
from bism.targets.local_shape_descriptors import lsd
from bism.targets.aclsd import aclsd
from bism.targets.mtlsd import mtlsd
from bism.targets.omnipose import omnipose
from bism.targets.semantic import semantic
from bism.targets.torchvision import maskrcnn
from bism.targets.iadb import iadb_target

"""
 --- Idea --- 
 This is not an awful way to do it, but requires us to import everything, also makes it hard to validate 
 Could implement a class to do this. Worth the added complexity? Probably not.
"""

_valid_targets = {
    'lsd': lsd,
    'affinities': affinities,
    'mtlsd': mtlsd,
    'aclsd': aclsd,
    'omnipose': omnipose,
    'semantic': semantic,
    'torchvision': maskrcnn,
    'iadb': iadb_target
}

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
    'iadb': bism.loss.iadb.iadb


}

_valid_lr_schedulers = {
    'cosine_annealing_warm_restarts': torch.optim.lr_scheduler.CosineAnnealingWarmRestarts
}
