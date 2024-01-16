import torch
import lion_pytorch
from functools import partial

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
from bism.targets.maskrcnn import maskrcnn
from bism.targets.iadb import IADBTarget

import bism.backends
import bism.backends.unet_conditional_difusion
from bism.models.generic import Generic
from bism.models.lsd import LSDModel
from bism.models.spatial_embedding import SpatialEmbedding


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
    'iadb': IADBTarget,
    'identity': lambda x: x
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
_valid_backbone_constructors = {
    "bism_unext": bism.backends.unext.UNeXT_3D,
    "bism_unext2d": bism.backends.unext.UNeXT_2D,
    "bism_unet": bism.backends.unet.UNet_3D,
    "bism_unet2d": bism.backends.unet.UNet_2D,
    "bism_unet2d_spade": bism.backends.unet_conditional_difusion.UNet_SPADE_2D,
    "bism_unet3d_spade": bism.backends.unet_conditional_difusion.UNet_SPADE_3D,
}

_valid_model_blocks = {
    "block3d": bism.modules.convnext_block.Block3D,
    "block2d": bism.modules.convnext_block.Block2D,
    'unet_block3d': bism.modules.unet_block.Block3D,
    'unet_block2d': bism.modules.unet_block.Block2D,
    'double_spade_block3d': bism.modules.unet_and_spade_block.DoubleSpadeBlock3D
}

_valid_upsample_layers = {
    "upsamplelayer3d": bism.modules.upsample_layer.UpSampleLayer3D,
    "upsamplelayer2d": bism.modules.upsample_layer.UpSampleLayer2D,
}

_valid_normalization = {
    "layernorm": partial(
        bism.modules.layer_norm.LayerNorm, data_format="channels_first"
    )
}

_valid_activations = {
    "gelu": torch.nn.GELU,
    "relu": torch.nn.ReLU,
    "silu": torch.nn.SiLU,
    "selu": torch.nn.SELU,
    "tanh": torch.nn.Tanh,
    "sigmoid": torch.nn.Sigmoid,
}

_valid_concat_blocks = {
    "concatconv3d": bism.modules.concat.ConcatConv3D,
    "concatconv2d": bism.modules.concat.ConcatConv2D,
}

_valid_models = {
    "spatial_embedding": SpatialEmbedding,
    "lsd": LSDModel,
    "generic": Generic,
}