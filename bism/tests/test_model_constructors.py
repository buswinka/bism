import pytest

from bism.models.unet import UNet_2D, UNet_3D
from bism.models.unext import UNeXT_2D, UNeXT_3D
from bism.models.r_unet import RUNeT_2D, RUNeT_3D
from bism.models.cellpose_net import CPnet_2D, CPnet_3D

from bism.modules.runet_block import RBlock2D, RBlock3D
from bism.modules.residual_block import ResidualBlock2D, ResidualBlock3D
from bism.modules.convnext_block import Block2D, Block3D
from bism.modules.unet_block import Block2D as UNetBlock2D, Block3D as UNetBlock3D
from bism.models.unetplusplus import UNetPlusPlus_3D

from bism.models.spatial_embedding import SpatialEmbedding

import torch.nn as nn
import torch

from itertools import product


@pytest.mark.parametrize("model_constructor", [UNet_2D, UNeXT_2D, RUNeT_2D, CPnet_2D])
def test_2d_model_initalization(model_constructor: nn.Module):
    x = torch.rand((1, 1, 100, 100))
    model = model_constructor(in_channels=1,
                              out_channels=4,
                              depths=[2, 2, 2, 2, 2],
                              dims=[32, 64, 128, 64, 32],
                              kernel_size=7,
                              drop_path_rate=0.0,
                              layer_scale_init_value=1.,
                              activation=nn.GELU)
    y = model(x)
    assert y.shape[1] == 4


@pytest.mark.parametrize("model_constructor", [UNet_3D, UNeXT_3D, RUNeT_3D, CPnet_3D])
def test_3d_model_initalization(model_constructor: nn.Module):
    x = torch.rand((1, 1, 100, 100, 20))
    model = model_constructor(in_channels=1,
                              out_channels=4,
                              depths=[2, 2, 2, 2, 2],
                              dims=[32, 64, 128, 64, 32],
                              kernel_size=7,
                              drop_path_rate=0.0,
                              layer_scale_init_value=1.,
                              activation=nn.GELU)
    y = model(x)
    assert y.shape[1] == 4


#  These two tests only test against the two generic models... unext and cellpose_net
@pytest.mark.parametrize('model_constructor,block',
                         product([UNeXT_2D, CPnet_2D], [RBlock2D, ResidualBlock2D, Block2D, UNetBlock2D]))
def test_2d_model_block_compatibility(model_constructor, block):
    x = torch.rand((1, 1, 100, 100))
    model = model_constructor(block=block)
    y = model(x)
    assert y.shape[1] == 4


@pytest.mark.parametrize('model_constructor,block',
                         product([UNeXT_3D, CPnet_3D], [RBlock3D, ResidualBlock3D, Block3D, UNetBlock3D]))
def test_3d_model_block_compatibility(model_constructor, block):
    x = torch.rand((1, 1, 100, 100, 20))
    model = model_constructor(block=block)
    y = model(x)
    assert y.shape[1] == 4


@pytest.mark.parametrize('model_constructor',
                         [UNet_3D, UNeXT_3D, RUNeT_3D, CPnet_3D, UNet_2D, UNeXT_2D, RUNeT_2D, CPnet_2D])
def test_jit_compatibility(model_constructor):
    model = torch.jit.script(model_constructor())


@pytest.mark.parametrize('model_constructor', [UNet_3D, UNeXT_3D, RUNeT_3D, CPnet_3D, UNetPlusPlus_3D])
def test_3d_spatial_embedding_compatibility(model_constructor):

    model = SpatialEmbedding(
        backbone=model_constructor(in_channels=1, out_channels=16)).cuda()
    x = torch.rand(1, 1, 300, 300, 20).cuda()
    y = model(x)
    assert y.shape[1] == 5
