import torch.nn as nn
import bism.backends
from functools import partial
from yacs.config import CfgNode


def cfg_to_bism_model(cfg: CfgNode) -> nn.Module:
    """ utility function to get a bism model from cfg """

    _valid_model_constructors = {
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
        'gelu': nn.GELU,
        'relu': nn.ReLU,
        'silu': nn.SiLU,
        'selu': nn.SELU
    }

    _valid_concat_blocks = {
        'concatconv3d': bism.modules.concat.ConcatConv3D
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

    if cfg.MODEL.ARCHITECTURE in _valid_model_constructors:
        backbone = _valid_model_constructors[cfg.MODEL.ARCHITECTURE]
    else:
        raise RuntimeError(f'{cfg.MODEL.ARCHITECTURE} is not a valid model constructor, valid options are: {_valid_model_constructors.keys()}')

    backbone = backbone(cfg.MODEL.IN_CHANNELS, cfg.MODEL.OUT_CHANNELS, **kwarg)
    model = SpatialEmbedding(backbone)

    return model

