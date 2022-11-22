import warnings

import torch.nn as nn
import torch
from torch import Tensor
from typing import List, Tuple, Optional, Union

from bism.modules.unet_block import Block2D, Block3D
from bism.modules.concat import ConcatConv2D, ConcatConv3D
from bism.modules.upsample_layer import UpSampleLayer3D, UpSampleLayer2D

from functools import partial

# This can be scripted!!!
class UNetND(nn.Module):
    """
    Generic Constructor for a UNet architecture of variable size and shape.
    """

    def __init__(self,
                 in_channels: Optional[int] = 1,
                 out_channels: int = 1,
                 spatial_dim: int = 2,
                 *,
                 dims: Optional[List[int]] = (32, 64, 128, 64, 32),  # [16, 32, 64, 32, 16],
                 depths: Optional[List[int]] = (2, 2, 2, 2, 2),  # [1, 2, 3, 2, 1],
                 kernel_size: Optional[Union[Tuple[int], int]] = 3,
                 drop_path_rate: Optional[float] = None,
                 layer_scale_init_value: Optional[float] = None,
                 activation: Optional[nn.Module] = nn.ReLU,
                 block: Optional[nn.Module] = Block2D,
                 concat_conv: Optional[nn.Module] = ConcatConv2D,
                 upsample_layer: Optional[nn.Module] = UpSampleLayer2D,
                 normalization: Optional[nn.Module] = None,
                 downsample: Optional[nn.Module] = nn.MaxPool2d,
                 name: Optional[str] = 'UNet'
                 ):
        """

        :param in_channels:
        :param out_channels:
        :param spatial_dim:
        :param dims:
        :param depths:
        :param kernel_size:
        :param drop_path_rate:
        :param layer_scale_init_value:
        :param activation:
        :param block:
        :param concat_conv:
        :param upsample_layer:
        :param normalization:
        :param downsample:
        :param name:
        """

        super(UNetND, self).__init__()

        assert len(depths) == len(dims), f'Number of depths should equal number of dims: {depths} != {dims}'

        # ----------------- Save Init Params...
        self.dims = dims
        self.depths = depths
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.activation = activation

        # ----------------- Model INIT
        self.downsample_layers = nn.ModuleList()
        self.down_blocks = nn.ModuleList()
        self.upsample_layers = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        self.concat = nn.ModuleList()

        # For repr
        self._activation = str(activation)
        self._block = str(block)
        self._concat_conv = str(concat_conv)
        self._upsample_layer = str(upsample_layer)
        self._normalizatoin = str(normalization)

        self._name = name

        self.spatial_dim = spatial_dim
        assert spatial_dim in (2, 3), f'Spatial Dimmension of {spatial_dim} is not supported'
        convolution = nn.Conv2d if spatial_dim == 2 else nn.Conv3d

        # 2D or 3D
        Block = block
        ConcatConv = concat_conv
        UpSampleLayer = upsample_layer

        DownsampleBlock = downsample

        if drop_path_rate is not None:
            warnings.warn(f'Drop path is not available for this model: {self._name}...')

        if layer_scale_init_value is not None:
            warnings.warn(f'layer scaling for computational blocks is not available for this model: {self._name}...')

        # ----------------- Downsample layers
        for i in range(len(dims) // 2):
            self.downsample_layers.append(downsample(2))

        # ----------------- Down Blocks
        _dims = [in_channels] + list(dims)
        for i in range(len(depths) // 2 + 1):
            stage = []
            for j in range(depths[i]):
                stage.append(
                    Block(in_channels=_dims[i] if j == 0 else _dims[i + 1],
                          out_channels=_dims[i + 1],
                          kernel_size=kernel_size,
                          activation=activation)
                )
            self.down_blocks.append(nn.Sequential(*stage))

        # ----------------- Upsample layers
        for i in range(len(dims) // 2):
            upsample_layer = UpSampleLayer(dims[i + len(dims) // 2], dims[i + len(dims) // 2])
            self.upsample_layers.append(upsample_layer)

        # ----------------- Up Blocks
        _dims = [in_channels] + list(dims)
        for i in range(1, len(dims) // 2):
            stage = []
            for j in range(depths[i + len(dims) // 2 + 1]):
                stage.append(
                    Block(
                        out_channels=_dims[i + len(_dims) // 2],
                        in_channels=_dims[i + len(_dims) // 2],
                        kernel_size=kernel_size,
                        activation=activation)
                )
            self.up_blocks.append(nn.Sequential(*stage))

        # -----------------  Concat Layers
        for i in range(len(dims) // 2):
            self.concat.append(ConcatConv(in_channels=dims[i + len(dims) // 2 + 1] + dims[i + len(dims) // 2],
                                          out_channels=dims[i + len(dims) // 2 + 1]))

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

        self.out_conv = convolution(dims[-1], out_channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        """
        Applies the UNet model to an input tensor
        :param x: 5D tensor [B, in_channels, X, Y, Z]
        :return: x: 5D tensor [B, out_channels, X, Y, Z]
        """

        steps: List[Tensor] = []
        shapes: List[List[int]] = []
        shapes.append(x.shape)

        # Down Stage of the Unet
        for i, (down, stage) in enumerate(zip(self.downsample_layers, self.down_blocks)):
            x: Tensor = stage(x)

            shapes.append(x.shape)  # Save shape for upswing of Unet
            steps.append(x)  # Save shape for upswing of Unet

            x: Tensor = down(x)

        x: Tensor = self.down_blocks[-1](x)  # bottom of the U

        shapes.append(x.shape)
        shapes.reverse()

        # Up Stage of the Unet
        for i, (up, cat, stage) in enumerate(zip(self.upsample_layers, self.concat, self.up_blocks)):
            x: Tensor = up(x, shapes[i + 1])
            y: Tensor = steps.pop(-1)
            x: Tensor = cat(x, y)
            x: Tensor = stage(x)

        x: Tensor = self.upsample_layers[-1](x, shapes[-1])
        y: Tensor = steps.pop(-1)
        x: Tensor = self.concat[-1](x, y)
        x: Tensor = self.out_conv(x)

        return x

    def __repr__(self):
        return f'{self._name}{self.spatial_dim}D[dims={self.dims}, depths={self.depths}, ' \
               f'in_channels={self.in_channels}, out_channels={self.out_channels}, block={self._block}, ' \
               f'normalization={self._normalizatoin}, activation={self._activation}, upsample={self._upsample_layer}, ' \
               f'concat={self._concat_conv}]'


class UNet_3D(UNetND):
    def __init__(self,
                 in_channels: Optional[int] = 1,
                 out_channels: int = 1,
                 *,
                 depths: Optional[List[int]] = [2, 2, 2, 2, 2],
                 dims: Optional[List[int]] = [32, 64, 128, 64, 32],
                 kernel_size: Optional[int] = 7,
                 drop_path_rate: Optional[float] = None,
                 layer_scale_init_value: Optional[float] = None,
                 activation: Optional[nn.Module] = nn.GELU,
                 block: Optional[nn.Module] = Block3D,
                 concat_conv: Optional[nn.Module] = ConcatConv3D,
                 upsample_layer: Optional[nn.Module] = UpSampleLayer3D,
                 normalization: Optional[nn.Module] = None,
                 name: Optional[str] = 'UNeXT'
                 ):
        super(UNet_3D, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            spatial_dim=3,
            depths=depths,
            dims=dims,
            drop_path_rate=drop_path_rate,
            layer_scale_init_value=layer_scale_init_value,
            kernel_size=kernel_size,
            activation=activation,
            block=block,
            concat_conv=concat_conv,
            upsample_layer=upsample_layer,
            normalization=normalization,
            downsample=nn.MaxPool3d,
            name=name
        )


class UNet_2D(UNetND):
    def __init__(self,
                 in_channels: Optional[int] = 1,
                 out_channels: int = 1,
                 *,
                 depths: Optional[List[int]] = [2, 2, 2, 2, 2],
                 dims: Optional[List[int]] = [32, 64, 128, 64, 32],
                 kernel_size: Optional[int] = 7,
                 drop_path_rate: Optional[float] = None,
                 layer_scale_init_value: Optional[float] = None,
                 activation: Optional[nn.Module] = nn.GELU,
                 block: Optional[nn.Module] = Block2D,
                 concat_conv: Optional[nn.Module] = ConcatConv2D,
                 upsample_layer: Optional[nn.Module] = UpSampleLayer2D,
                 normalization: Optional[nn.Module] = None,
                 name: Optional[str] = 'UNeXT'
                 ):
        super(UNet_2D, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            spatial_dim=2,
            depths=depths,
            dims=dims,
            drop_path_rate=drop_path_rate,
            layer_scale_init_value=layer_scale_init_value,
            kernel_size=kernel_size,
            activation=activation,
            block=block,
            concat_conv=concat_conv,
            upsample_layer=upsample_layer,
            normalization=normalization,
            downsample=nn.MaxPool2d,
            name=name
        )


if __name__ == '__main__':
    model = torch.jit.script(UNet_3D(in_channels=1, out_channels=1, dims=(4, 8, 16, 8, 4)))
    a = torch.rand((1, 1, 100, 100, 20))
    out = model(a)
