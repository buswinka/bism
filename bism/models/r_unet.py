import torch.nn as nn
import torch
from torch import Tensor
from typing import List, Tuple, Optional

from bism.modules.runet_block import RBlock2D, RBlock3D
from bism.modules.layer_norm import LayerNorm
from bism.modules.concat import ConcatConv2D, ConcatConv3D
from bism.modules.upsample_layer import UpSampleLayer3D, UpSampleLayer2D
from bism.modules.drop_path import DropPath

from functools import partial


@torch.jit.script
def _crop(img: Tensor, x: int, y: int, z: int, w: int, h: int, d: int) -> Tensor:
    return img[..., x:x + w, y:y + h, z:z + d]


class RUNeTND(nn.Module):
    def __init__(self,
                 in_channels: Optional[int] = 1,
                 out_channels: int = 4,
                 spatial_dim: int = 2,
                 recurrent_steps: Optional[int] = 2,
                 *,
                 depths: Optional[List[int]] = (1, 2, 3, 2, 1),  # [1, 2, 3, 2, 1],
                 dims: Optional[List[int]] = (32, 64, 128, 64, 32),  # [16, 32, 64, 32, 16],
                 drop_path_rate: Optional[float] = 0.0,
                 layer_scale_init_value: Optional[float] = 1.,
                 kernel_size: int = 7,
                 activation: Optional[nn.Module] = nn.GELU,
                 block: Optional[nn.Module] = RBlock2D,
                 concat_conv: Optional[nn.Module] = ConcatConv2D,
                 upsample_layer: Optional[nn.Module] = UpSampleLayer2D,
                 normalization: Optional[nn.Module] = partial(LayerNorm, data_format='channels_first'),
                 name: Optional[str] = 'RecurrentUNet'
                 ):
        super(RUNeTND, self).__init__()

        assert len(depths) == len(dims), f'Number of depths should equal number of dims: {depths} != {dims}'

        # ----------------- Save Init Params for Preheating...
        self.dims = dims
        self.depths = depths
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.N = recurrent_steps
        self.spatial_dim = spatial_dim

        self._name = name

        # For repr
        self._activation = str(activation)
        self._block = str(block)
        self._concat_conv = str(concat_conv)
        self._upsample_layer = str(upsample_layer)
        self._normalizatoin = str(normalization)

        assert spatial_dim in (2, 3), f'Spatial Dimmension of {spatial_dim} is not supported'
        convolution = nn.Conv2d if spatial_dim == 2 else nn.Conv3d

        # 2D or 3D
        Block = partial(block, N=recurrent_steps)
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
            convolution(in_channels, dims[0], kernel_size=7, padding=3),
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
                *[Block(dim=dims[i + len(dims) // 2 + 1], kernel_size=kernel_size, activation=activation) for j in range(num_blocks)]
            )
            self.up_stages.append(stage)
            cur += depths[i]

        # Concat Layers
        self.concat = nn.ModuleList()
        for i in range(len(dims) // 2):
            self.concat.append(ConcatConv(2 * dims[i + len(dims) // 2 + 1], dims[i + len(dims) // 2 + 1]))

        self.final_concat = ConcatConv(2 * dims[-1], dims[-1])

        # Final Upscale
        self.upscale_out = UpSampleLayer(dims[-1], dims[-1])

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

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

        for i, (down, stage, norm) in enumerate(zip(self.downsample_layers, self.down_stages, self.down_norms)):

            # Save input for upswing of Unet
            if i != 0:
                steps.append(x)

            x = down(x)
            x = norm(x)
            x = stage(x)
            shapes.append(x.shape)

        shapes.reverse()
        for i, (up, cat, stage, norm) in enumerate(
                zip(self.upsample_layers, self.concat, self.up_stages, self.up_norms)):
            x = up(x, shapes[i + 1])
            y = steps.pop(-1)
            x = cat(x, y)
            x = stage(x)
            x = norm(x)

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

class RUNeT_2D(RUNeTND):
    def __init__(self,
                 in_channels: Optional[int] = 1,
                 out_channels: int = 4,
                 recurrent_steps: Optional[int] = 2,
                 *,
                 depths: Optional[List[int]] = (1, 2, 3, 2, 1),  # [1, 2, 3, 2, 1],
                 dims: Optional[List[int]] = (32, 64, 128, 64, 32),  # [16, 32, 64, 32, 16],
                 drop_path_rate: Optional[float] = 0.0,
                 layer_scale_init_value: Optional[float] = 1.,
                 kernel_size: int = 7,
                 activation: Optional[nn.Module] = nn.GELU,
                 block: Optional[nn.Module] = RBlock2D,
                 concat_conv: Optional[nn.Module] = ConcatConv2D,
                 upsample_layer: Optional[nn.Module] = UpSampleLayer2D,
                 normalization: Optional[nn.Module] = partial(LayerNorm, data_format='channels_first'),
                 name: Optional[str] = 'RecurrentUNet'
                 ):
        super(RUNeT_2D, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            spatial_dim=2,
            recurrent_steps=recurrent_steps,
            depths=depths,
            dims=dims,
            kernel_size=kernel_size,
            drop_path_rate=drop_path_rate,
            layer_scale_init_value=layer_scale_init_value,
            activation=activation,
            block=block,
            concat_conv=concat_conv,
            upsample_layer=upsample_layer,
            normalization=normalization,
            name=name
        )

class RUNeT_3D(RUNeTND):
    def __init__(self,
                 in_channels: Optional[int] = 1,
                 out_channels: int = 4,
                 recurrent_steps: Optional[int] = 2,
                 *,
                 depths: Optional[List[int]] = (1, 2, 3, 2, 1),  # [1, 2, 3, 2, 1],
                 dims: Optional[List[int]] = (32, 64, 128, 64, 32),  # [16, 32, 64, 32, 16],
                 drop_path_rate: Optional[float] = 0.0,
                 layer_scale_init_value: Optional[float] = 1.,
                 kernel_size: int = 7,
                 activation: Optional[nn.Module] = nn.GELU,
                 block: Optional[nn.Module] = RBlock3D,
                 concat_conv: Optional[nn.Module] = ConcatConv3D,
                 upsample_layer: Optional[nn.Module] = UpSampleLayer3D,
                 normalization: Optional[nn.Module] = partial(LayerNorm, data_format='channels_first'),
                 name: Optional[str] = 'RecurrentUNet'
                 ):

        super(RUNeT_3D, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            spatial_dim=3,
            recurrent_steps=recurrent_steps,
            depths=depths,
            dims=dims,
            kernel_size=kernel_size,
            drop_path_rate=drop_path_rate,
            layer_scale_init_value=layer_scale_init_value,
            activation=activation,
            block=block,
            concat_conv=concat_conv,
            upsample_layer=upsample_layer,
            normalization=normalization,
            name=name
        )

if __name__ == '__main__':
    model = torch.jit.script(RUNeT_2D())
    x = torch.rand((1, 1, 100, 100))
    _ = model(x)

    model = RUNeT_3D()
    x = torch.rand((1, 1, 100, 100, 20))
    _ = model(x)