import torch
import torch.nn as nn
import bism.backends
from functools import partial
from yacs.config import CfgNode

from bism.models.spatial_embedding import SpatialEmbedding
from bism.models.lsd import LSDModel
from bism.models.generic import Generic

def cfg_to_bism_model(cfg: CfgNode) -> nn.Module:
    """ utility function to get a bism model from cfg """

    _valid_backbone_constructors = {
        'bism_unext': bism.backends.unext.UNeXT_3D,
        'bism_unet': bism.backends.unet.UNet_3D
    }

    _valid_model_blocks = {
        'block3d': bism.modules.convnext_block.Block3D
    }

    _valid_upsample_layers = {
        'upsamplelayer3d': bism.modules.upsample_layer.UpSampleLayer3D
    }

    _valid_normalization = {
        'layernorm': partial(bism.modules.layer_norm.LayerNorm, data_format='channels_first')
    }

    _valid_activations = {
        'gelu': torch.nn.GELU,
        'relu': torch.nn.ReLU,
        'silu': torch.nn.SiLU,
        'selu': torch.nn.SELU
    }

    _valid_concat_blocks = {
        'concatconv3d': bism.modules.concat.ConcatConv3D
    }

    _valid_models = {
        'spatial_embedding': SpatialEmbedding,
        'lsd': LSDModel,
        'generic': Generic
    }


    model_config = [
        cfg.MODEL.DIMS,
        cfg.MODEL.DEPTHS,
        cfg.MODEL.KERNEL_SIZE,
        cfg.MODEL.DROP_PATH_RATE,
        cfg.MODEL.LAYER_SCALE_INIT_VALUE,
        cfg.MODEL.ACTIVATION,
        cfg.MODEL.BLOCK,
        cfg.MODEL.CONCAT_BLOCK,
        cfg.MODEL.UPSAMPLE_BLOCK,
        cfg.MODEL.NORMALIZATION,
    ]

    model_kwargs = [
        'dims', 'depths', 'kernel_size', 'drop_path_rate',
        'layer_scale_init_value', 'activation', 'block',
        'concat_conv', 'upsample_layer', 'normalization'
    ]

    valid_dicts = [
        None, None, None, None, None, _valid_activations, _valid_model_blocks, _valid_concat_blocks, _valid_upsample_layers, _valid_normalization
    ]

    kwarg = {}
    for kw, config, vd in zip(model_kwargs, model_config, valid_dicts):
        if vd is not None:
            if config in vd:
                kwarg[kw] = vd[config]
            else:
                raise RuntimeError(f'{config} is not a valid config option for {kw}. Valid options are: {vd.keys()}')
        else:
            kwarg[kw] = config

    if cfg.MODEL.BACKBONE in _valid_backbone_constructors:
        backbone = _valid_backbone_constructors[cfg.MODEL.BACKBONE]
    else:
        raise RuntimeError(f'{cfg.MODEL.ARCHITECTURE} is not a valid model constructor, valid options are: {_valid_backbone_constructors.keys()}')

    backbone = backbone(cfg.MODEL.IN_CHANNELS, cfg.MODEL.OUT_CHANNELS, **kwarg)
    model = _valid_models[cfg.MODEL.MODEL](backbone)

    return model
