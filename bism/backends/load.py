from bism.backends.unet import UNet_2D, UNet_3D
from bism.backends.unext import UNeXT_2D, UNeXT_3D
from bism.backends.r_unet import RUNeT_2D, RUNeT_3D
from bism.backends.cellpose_net import CPnet_2D, CPnet_3D
from bism.backends.unetplusplus import UNetPlusPlus_3D, UNetPlusPlus_2D

import torch.nn as nn
import torch

__available_models__ = ['unet', 'unet++', 'runet', 'cellpose_net', 'unext']
def get_constructor(model_name, spatial_dim: int = 2):
    if spatial_dim not in [2, 3]:
        raise RuntimeError(f'Spatial dimmension of {spatial_dim} is not supported')

    constructors = {'unet2d': UNet_2D,
                    'unet3d': UNet_3D,
                    'unet++2d': UNetPlusPlus_2D,
                    'unet++3d': UNetPlusPlus_3D,
                    'runet2d': RUNeT_2D,
                    'runet3d': RUNeT_3D,
                    'cellpose_net2d': CPnet_2D,
                    'cellpose_net3d': CPnet_3D,
                    'unext2d': UNeXT_2D,
                    'unext3d': UNeXT_3D
                    }

    return constructors[f'{model_name}{spatial_dim}d']