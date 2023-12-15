from copy import copy
from functools import partial
from typing import *

import torch
import torch.nn as nn
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.mask_rcnn import (
    MaskRCNN,
    _resnet_fpn_extractor,
    RPNHead,
    FastRCNNConvFCHead,
    MaskRCNNHeads,
)
from torchvision.models.resnet import ResNet50_Weights, resnet50
from yacs.config import CfgNode

import bism.backends
import bism.backends.unet_conditional_difusion
from bism.models.generic import Generic
from bism.models.lsd import LSDModel
from bism.models.spatial_embedding import SpatialEmbedding


def cfg_to_bism_model(cfg: CfgNode) -> nn.Module:
    """utility function to get a bism model from cfg"""

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
        "dims",
        "depths",
        "kernel_size",
        "drop_path_rate",
        "layer_scale_init_value",
        "activation",
        "block",
        "concat_conv",
        "upsample_layer",
        "normalization",
    ]

    valid_dicts = [
        None,
        None,
        None,
        None,
        None,
        _valid_activations,
        _valid_model_blocks,
        _valid_concat_blocks,
        _valid_upsample_layers,
        _valid_normalization,
    ]

    kwarg = {}
    for kw, config, vd in zip(model_kwargs, model_config, valid_dicts):
        if vd is not None:
            if config in vd:
                kwarg[kw] = vd[config]
            else:
                raise RuntimeError(
                    f"{config} is not a valid config option for {kw}. Valid options are: {vd.keys()}"
                )
        else:
            kwarg[kw] = config

    if cfg.MODEL.BACKBONE in _valid_backbone_constructors:
        backbone = _valid_backbone_constructors[cfg.MODEL.BACKBONE]
    else:
        raise RuntimeError(
            f"{cfg.MODEL.ARCHITECTURE} is not a valid model constructor, valid options are: {_valid_backbone_constructors.keys()}"
        )

    if cfg.TRAIN.TARGET == "iadb":
        backbone = backbone(cfg.MODEL.IN_CHANNELS, cfg.MODEL.OUT_CHANNELS, cfg.TARGET.IADB.MASK_CHANNELS, **kwarg)
        assert cfg.MODEL.MODEL == 'generic', f'iadb backbone can only be used with generic model, not {cfg.MODEL.MODEL=}'
        keys: List[str] = copy(cfg.MODEL.OUTPUT_ACTIVATIONS)
        if len(keys) != cfg.MODEL.OUT_CHANNELS:
            raise RuntimeError(
                "The number of output activations must equal the number of output channels"
            )
        activations: List[nn.Module | None] = [
            _valid_activations[a]() if a is not None else None
            for a in cfg.MODEL.OUTPUT_ACTIVATIONS
        ]
        model = _valid_models[cfg.MODEL.MODEL](backbone, activations)
        return model

    elif cfg.TRAIN.TARGET == "aclsd":
        lsd_backbone = backbone(cfg.MODEL.IN_CHANNELS, 10, **kwarg)
        aff_backbone = backbone(10, 3, **kwarg)
        print("cfg.MODEL.MODEL")
        lsd_model = _valid_models[cfg.MODEL.MODEL](lsd_backbone)
        aff_model = _valid_models[cfg.MODEL.MODEL](aff_backbone)

        return lsd_model, aff_model

    else:
        backbone = backbone(cfg.MODEL.IN_CHANNELS, cfg.MODEL.OUT_CHANNELS, **kwarg)
        if cfg.MODEL.MODEL != "generic":
            model = _valid_models[cfg.MODEL.MODEL](backbone)
        else:
            keys: List[str] = copy(cfg.MODEL.OUTPUT_ACTIVATIONS)
            if len(keys) != cfg.MODEL.OUT_CHANNELS:
                raise RuntimeError(
                    "The number of output activations must equal the number of output channels"
                )
            activations: List[nn.Module | None] = [
                _valid_activations[a]() if a is not None else None
                for a in cfg.MODEL.OUTPUT_ACTIVATIONS
            ]
            model = _valid_models[cfg.MODEL.MODEL](backbone, activations)

        return model


def cfg_to_torchvision_model(cfg: CfgNode) -> nn.Module:
    __valid_torchvision_weights__ = {
        "maskrcnn_resnet50_fpn_v2": MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1,
    }

    weights = __valid_torchvision_weights__[cfg.MODEL.BACKBONE]

    assert (
        cfg.MODEL.IN_CHANNELS == 3
    ), f"Torchvision models assume images in RGB format! Configuration value {cfg.MODEL.IN_CHANNELS=} must equal 3"

    weights = MaskRCNN_ResNet50_FPN_V2_Weights.verify(weights)
    weights_backbone = ResNet50_Weights.IMAGENET1K_V1

    is_trained = weights is not None or weights_backbone is not None
    trainable_backbone_layers = 5

    backbone = resnet50(weights=weights_backbone, progress=False)
    backbone = _resnet_fpn_extractor(
        backbone, trainable_backbone_layers, norm_layer=nn.BatchNorm2d
    )

    anchor_sizes = tuple(tuple(a) for a in cfg.MODEL.ANCHOR_SIZES)
    print(anchor_sizes, cfg.MODEL.ANCHOR_ASPECT_RATIOS)

    rpn_anchor_generator = AnchorGenerator(
        anchor_sizes, ((0.5, 1.0, 2.0),) * len(anchor_sizes),
    )
    rpn_head = RPNHead(
        backbone.out_channels,
        rpn_anchor_generator.num_anchors_per_location()[0],
        conv_depth=2,
    )
    box_head = FastRCNNConvFCHead(
        (backbone.out_channels, 7, 7),
        [256, 256, 256, 256],
        [1024],
        norm_layer=nn.BatchNorm2d,
    )
    mask_head = MaskRCNNHeads(
        backbone.out_channels, [256, 256, 256, 256], 1, norm_layer=nn.BatchNorm2d
    )

    model = MaskRCNN(
        backbone,
        num_classes=cfg.MODEL.OUT_CHANNELS + 1,
        rpn_anchor_generator=rpn_anchor_generator,
        rpn_head=rpn_head,
        box_head=box_head,
        mask_head=mask_head,
        box_nms_thresh=0.7,
        box_detections_per_img=500,
        # min_size=100,
        # max_size=4000,
    )
    return model
