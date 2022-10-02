import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Union, Tuple, Optional
from functools import partial

from bism.modules.upsample_layer import UpSampleLayer2D, UpSampleLayer3D
from bism.modules.concat import ConcatConv2D, ConcatConv3D
from bism.modules.residual_block import ResidualBlock2D, ResidualBlock3D
from bism.modules.style import MergeStyle, MakeStyle2D, MakeStyle3D

"""
Re-implementation of Cellpose Net with some changes

- Each residual block only has 1 set of convolutions. 
Now you just string together multiple blocks

- Each block now has adjustable normalization and activation params

- Upsample now works slightly differently

- Downsample now uses strided convolutions instead of MaxPool
"""

class CPnetND(nn.Module):
    def __init__(self,
                 in_channels: Optional[int] = 1,
                 out_channels: int = 4,
                 spatial_dim: int = 2,
                 *,
                 depths: Optional[List[int]] = [2, 2, 2, 2, 2],
                 dims: Optional[List[int]] = [32, 64, 128, 64, 32],
                 kernel_size: Optional[int] = 7,
                 drop_path_rate: Optional[float] = 0.0,
                 layer_scale_init_value: Optional[float] = 1.,
                 activation: Optional[nn.Module] = nn.ReLU,
                 block: Optional[nn.Module] = ResidualBlock2D,
                 concat_conv: Optional[nn.Module] = ConcatConv2D,
                 upsample_layer: Optional[nn.Module] = UpSampleLayer2D,
                 normalization: Optional[nn.Module] = nn.BatchNorm2d,
                 name: Optional[str] = 'CellPoseNet'
                 ):
        super(CPnetND, self).__init__()

        assert len(depths) == len(dims), f'Number of depths should equal number of dims: {depths} != {dims}'

        self.spatial_dim = spatial_dim
        assert spatial_dim in (2, 3), f'Spatial Dimmension of {spatial_dim} is not supported'
        convolution = nn.Conv2d if spatial_dim == 2 else nn.Conv3d

        # ----------------- Save Init Params for Preheating...
        self.dims = dims
        self.depths = depths
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size,

        # For repr
        self._activation = str(activation)
        self._block = str(block)
        self._concat_conv = str(concat_conv)
        self._upsample_layer = str(upsample_layer)
        self._normalizatoin = str(normalization)

        self._name = name

        # 2D or 3D
        Block = block
        ConcatConv = concat_conv
        UpSampleLayer = upsample_layer


        # ----------------- Model INIT
        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        self.upsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers

        # Drop path should not decrease to nothing over time...
        dp_rates = [drop_path_rate for _ in range(sum(depths))]

        self.down_stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        self.up_stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks

        self.down_norms = nn.ModuleList()
        self.up_norms = nn.ModuleList()

        # ----------------- Stage 1
        self.init_stage = nn.Sequential(
            convolution(in_channels, dims[0], kernel_size=(7,) * spatial_dim, padding=(3,) * spatial_dim),
            Block(dim=dims[0], kernel_size=kernel_size, activation=activation)
        )

        # ----------------- Unlike original conv next we cannot reduce the stem by that much...
        stem = nn.Sequential(
            convolution(dims[0], dims[0], kernel_size=(2,) * spatial_dim, stride=(2,) * spatial_dim),
            normalization(dims[0])
        )
        self.downsample_layers.append(stem)

        # ----------------- Downsample layers
        for i in range(len(dims) // 2):
            downsample_layer = nn.Sequential(
                normalization(dims[i]),
                convolution(dims[i], dims[i + 1], kernel_size=(2,) * spatial_dim, stride=(2,) * spatial_dim),
            )
            self.downsample_layers.append(downsample_layer)

        # Down Stages
        cur = 0
        for i in range(len(dims) // 2 + 1):
            num_blocks = depths[i]
            stage = nn.Sequential(
                *[Block(dim=dims[i], kernel_size=kernel_size, activation=activation) for j in range(num_blocks)]
            )
            self.down_stages.append(stage)
            cur += depths[i]

        # Down Layer Normalization
        for i_layer in range(len(dims) // 2 + 1):
            layer = normalization(dims[i_layer])
            self.down_norms.append(layer)

        # Up Layer Normalization
        for i_layer in range(len(dims) // 2):
            layer = normalization(dims[i_layer + len(dims) // 2 + 1])
            self.up_norms.append(layer)

        # Upsample layers
        for i in range(len(dims) // 2):
            upsample_layer = UpSampleLayer(dims[i + len(dims) // 2], dims[i + 1 + len(dims) // 2])
            self.upsample_layers.append(upsample_layer)

        # Up Stages
        cur = 0
        for i in range(len(dims) // 2):
            num_blocks = depths[i + len(dims) // 2 + 1]
            stage = nn.Sequential(
                *[Block(dim=dims[i + len(dims) // 2 + 1],  kernel_size=kernel_size,
                        activation=activation) for j in range(num_blocks)]
            )
            self.up_stages.append(stage)
            cur += depths[i]

        # Up MergeStyle
        self.merge_style_blocks = nn.ModuleList()
        for i in range(len(dims) // 2):
            stage = MergeStyle(dim=dims[i + len(dims) // 2 + 1], style_channels=dims[len(dims) // 2])
            self.merge_style_blocks.append(stage)

        # Concat Layers
        self.concat = nn.ModuleList()
        for i in range(len(dims) // 2):
            self.concat.append(ConcatConv(in_channels=dims[i + len(dims) // 2 + 1] + dims[len(dims) // 2 - (i + 1)],
                                          out_channels=dims[i + len(dims) // 2 + 1])
            )

        self.final_concat = ConcatConv(in_channels=dims[0] + dims[-1], out_channels=dims[-1])

        # Final Upscale
        self.upscale_out = UpSampleLayer(dims[-1], dims[-1])

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()


        # CP net ***STYLE***
        self.make_style = MakeStyle2D() if spatial_dim == 2 else MakeStyle3D()

        # Flatten Channels to out...
        self.out_conv = convolution(dims[-1], out_channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        """

        NEEDS INPUT TO BE NORMALIZED TO GET ANYWHERE! WAS JACK SHIT BEFORE NORMALIZATION!!!!

        Plan is to preinitalize steps to include the necessary shit with torch.empty(devie=cuda)
        I think that allocating and dealocating is expensive
        See significant speedup in epoch time if batch size is 1

        I dont know why... Figure out tomorrow. Could be data bus and distributed data?

        Time per epoch with batch size of 24: ~30 seconds
        Time per epoch with batch size of 2: ~10-15seconds

        :param x:
        :return:
        """
        x = self.init_stage(x)

        shapes: List[List[int]] = [x.shape]
        steps: List[Tensor] = [x]

        assert len(self.upsample_layers) == len(self.concat)
        assert len(self.upsample_layers) == len(self.up_stages)
        assert len(self.upsample_layers) == len(self.up_norms)

        for i, (down, stage, norm) in enumerate(zip(self.downsample_layers, self.down_stages, self.down_norms)):

            # Save input for upswing of Unet
            if i != 0:
                steps.append(x)
            x = down(x)
            x = norm(x)
            x = stage(x)

            shapes.append(x.shape)

        style = self.make_style(x)

        shapes.reverse()

        for i, (up, cat, stage, norm, merge_style) in enumerate(
                zip(self.upsample_layers, self.concat, self.up_stages, self.up_norms, self.merge_style_blocks)):

            x = up(x, shapes[i + 1])
            y = steps.pop(-1)
            x = cat(x, y)
            x = merge_style(x, style)
            x = stage(x)
            x = norm(x)

        # Out
        x = self.upscale_out(x, shapes[-1])
        y = steps.pop(-1)
        x = self.final_concat(x, y)
        x = self.out_conv(x)

        return x

    def __repr__(self):
        return f'{self._name}{self.spatial_dim}D[dims={self.dims}, depths={self.depths}, ' \
               f'in_channels={self.in_channels}, out_channels={self.out_channels}, block={self._block}, ' \
               f'normalization={self._normalizatoin}, activation={self._activation}, upsample={self._upsample_layer}, ' \
               f'concat={self._concat_conv}]'


class CPnet_2D(CPnetND):
    def __init__(self,
                 in_channels: Optional[int] = 1,
                 out_channels: int = 4,
                 *,
                 depths: Optional[List[int]] = [2, 2, 2, 2, 2],
                 dims: Optional[List[int]] = [32, 64, 128, 64, 32],
                 kernel_size: Optional[int] = 7,
                 drop_path_rate: Optional[float] = 0.0,
                 layer_scale_init_value: Optional[float] = 1.,
                 activation: Optional[nn.Module] = nn.ReLU,
                 block: Optional[nn.Module] = ResidualBlock2D,
                 concat_conv: Optional[nn.Module] = ConcatConv2D,
                 upsample_layer: Optional[nn.Module] = UpSampleLayer2D,
                 normalization: Optional[nn.Module] = nn.BatchNorm2d,
                 ):

        super(CPnet_2D, self).__init__(in_channels, out_channels,
                                       spatial_dim=2,
                                       dims=dims,
                                       depths=depths,
                                       kernel_size=kernel_size,
                                       drop_path_rate=drop_path_rate,
                                       layer_scale_init_value=layer_scale_init_value,
                                       activation=activation,
                                       block=block,
                                       concat_conv=concat_conv,
                                       upsample_layer=upsample_layer,
                                       normalization=normalization)


class CPnet_3D(CPnetND):
    def __init__(self,
                 in_channels: Optional[int] = 1,
                 out_channels: int = 4,
                 *,
                 depths: Optional[List[int]] = [2, 2, 2, 2, 2],
                 dims: Optional[List[int]] = [32, 64, 128, 64, 32],
                 kernel_size: Optional[int] = 7,
                 drop_path_rate: Optional[float] = 0.0,
                 layer_scale_init_value: Optional[float] = 1.,
                 activation: Optional[nn.Module] = nn.ReLU,
                 block: Optional[nn.Module] = ResidualBlock3D,
                 concat_conv: Optional[nn.Module] = ConcatConv3D,
                 upsample_layer: Optional[nn.Module] = UpSampleLayer3D,
                 normalization: Optional[nn.Module] = nn.BatchNorm3d,
                 ):

        super(CPnet_3D, self).__init__(in_channels, out_channels,
                                       spatial_dim=3,
                                       dims=dims,
                                       depths=depths,
                                       kernel_size=kernel_size,
                                       drop_path_rate=drop_path_rate,
                                       layer_scale_init_value=layer_scale_init_value,
                                       activation=activation,
                                       block=block,
                                       concat_conv=concat_conv,
                                       upsample_layer=upsample_layer,
                                       normalization=normalization)


if __name__=='__main__':
    model = CPnet_3D(depths=[2, 2, 2, 2, 2], dims=[8,16,32,16,8], out_channels=16)
    x = torch.rand((1, 1, 300, 300, 20))
    y = model(x)
