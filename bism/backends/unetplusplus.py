import torch.nn as nn
import torch
from torch import Tensor
from typing import List, Tuple, Optional, Union
from functools import partial

from bism.modules.unet_block import Block2D, Block3D
from bism.modules.concat import ConcatConv2D, ConcatConv3D
from bism.modules.upsample_layer import UpSampleLayer3D, UpSampleLayer2D
from bism.modules.layer_norm import LayerNorm


class UNetPlusPlusND(nn.Module):
    """
    Generic Constructor for a UNet++ architecture of variable size and shape.
    """
    def __init__(self,
                 in_channels: Optional[int] = 1,
                 out_channels: int = 1,
                 L: int = 4,
                 spatial_dim: int = 2,
                 *,
                 dims: Optional[List[int]] = (16, 32, 64, 128),  # [16, 32, 64, 128],
                 depths: Optional[List[int]] = (2, 2, 2, 2),  # [1, 2, 3, 2, 1],
                 kernel_size: Optional[Union[Tuple[int], int]] = 3,
                 activation: Optional[nn.Module] = nn.ReLU,
                 block: Optional[nn.Module] = Block3D,
                 concat_conv: Optional[nn.Module] = ConcatConv3D,
                 upsample_layer: Optional[nn.Module] = UpSampleLayer3D,
                 normalization: Optional[nn.Module] = partial(LayerNorm, data_format='channels_first'),
                 name: Optional[str]='UNet++'
                 ):
        """
        Initialize the model with custom depth, and dimensions, kernel size, and activation function.

        :param in_channels: int - Input channels
        :param out_channels:  int - Output Channels
        :param dims: Optional[List[int]] Number of filter channels for each block at each stage of the UNet
        :param depths: Optional[List[int]] Number of repeating blocks at each level of the UNet
        :param kernel_size: Optional[Union[Tuple[int], int]] Kernel size for each convolution in a UNet block
        :param activation: Optional[nn.Module] Activation function for each block in the UNet
        """

        super(UNetPlusPlusND, self).__init__()

        assert len(depths) == len(dims) == L, f'Number of depths should equal number of dims and layers: {depths} != {dims} != {L}'

        # ----------------- Save Init Params...
        self.dims = dims
        self.depths = depths
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.activation = activation
        self.L = L

        self.spatial_dim = spatial_dim
        assert spatial_dim in (2, 3), f'Spatial Dimmension of {spatial_dim} is not supported'
        convolution = nn.Conv2d if spatial_dim == 2 else nn.Conv3d

        # For repr
        self._activation = str(activation)
        self._block = str(block)
        self._concat_conv = str(concat_conv)
        self._upsample_layer = str(upsample_layer)
        self._normalizatoin = str(normalization)
        self._name = name

        # 2D or 3D | Weird case rules because they're all nn.Module classes...
        Block: nn.Module = block
        ConcatConv: nn.Module = concat_conv
        UpSampleLayer: nn.Module = upsample_layer

        self.in_conv = convolution(in_channels=in_channels, out_channels=dims[0], kernel_size=self.kernel_size, padding=(self.kernel_size//2,) * spatial_dim)
        self.out_conv = convolution(in_channels=dims[0], out_channels=out_channels, kernel_size=self.kernel_size, padding=(self.kernel_size//2,) * spatial_dim)

        # All Computation Blocks indexed like a list of lists [L, I]
        self.blocks = nn.ModuleList()
        for l in range(self.L):
            layer_blocks = nn.ModuleList()
            for i in range(self.L - l):
                stage = []
                for j in range(self.depths[i]):
                    stage.append(
                        Block(in_channels=dims[i],
                              out_channels=dims[i],
                              kernel_size=kernel_size,
                              activation=activation)
                    )
                layer_blocks.append(nn.Sequential(*stage))

            self.blocks.append(layer_blocks)

        # All Upsample Blocks
        # only need upsamples for layers 1-l
        self.upsample = nn.ModuleList()
        for l in range(self.L - 1):
            upsample = nn.ModuleList()
            for i in range(self.L - (1 + l)):
                upsample.append(UpSampleLayer(dims[i+1], dims[i]))
            self.upsample.append(upsample)

        # All Concat Blocks
        # only need upsamples for stages 1->L
        self.concat = nn.ModuleList()
        for i in range(1, self.L):
            layer_cat = nn.ModuleList()
            for l in range(self.L - i):
                layer_cat.append(
                    ConcatConv(in_channels=(i + 2) * dims[l], out_channels=dims[l])
                )
            self.concat.append(layer_cat)

        # Downsample Blocks
        # only need Downsamples for Stages 0->(L-1)
        self.downsample = nn.ModuleList()
        for i in range(self.L - 1):
            downsample = nn.ModuleList()
            for l in range(self.L - (i+1)):
                downsample.append(
                    nn.Sequential(
                        convolution(dims[l], dims[l + 1], kernel_size=(2,) * spatial_dim, stride=(2,) * spatial_dim),
                        normalization(dims[l + 1],))
                    )
            self.downsample.append(downsample)

    @torch.jit.ignore()
    def forward(self, x: Tensor) -> Tensor:
        r"""
        NOT SCRITPABLE! In theory you could make it scriptable but shenanigans with the nn.Module list
        make it unikely... maybe worth doing if speeds up considerably.

        Applies the UNet++ model to an input tensor. Assumes a UNet++ structure of:

        ::
                      i =     0 1 2 3 4       --->      Schematized
                     -------------------------   ------------------------
              |C: 16 | in ->  ■ ■ ■ ■ ■ -> out |   in -> ■ ■ ■ ■ ■ -> out
              |C: 32 |        ■ ■ ■ ■          |          ■ ■ ■ ■
            l |C: 64 |        ■ ■ ■            |           ■ ■ ■
              |C:128 |        ■ ■              |            ■ ■
              |C:256 |        ■                |             ■

        Where *l* is the "Layer" and *i* is the "STAGE" and ■ is a convolutional "Block"

        (How we represent the relationships of the blocks are on the left)
        (How the paper represents the relationship between blocks are on the right)

        ---------
        A Block PRESERVES the number of dimmensions | C: N -> N
        Upsamples REDCUE the number of dimensions   | C: N -> M  [N > M]
        Downsample INCREASE the number of dimensions| C: M -> N  [N > M]

        Channels increase as l gets deeper according to self.dims
        --------

        :param x: 5D tensor [B, in_channels, X, Y, Z] INPUT
        :return: x: 5D tensor [B, out_channels, X, Y, Z] OUTPUT
        """
        # We have two caches for tensors.
        # Cache preserves every intermediary tensor after a block
        # upsample only preserves the results of the upsample modules so that we can input them to the next stage
        cache: List[List[Tensor]] = [[] for _ in range(self.L)]
        upsample: List[Tensor] = [torch.empty(0) for _ in range(self.L)]

        y: Tensor = self.in_conv(x)

        for i in range(self.L):  # This is the STAGE
            shapes: List[Tensor] = [torch.tensor(y.shape)]  # We need the shape of the tensor for upsampling later

            for l in range(self.L - i):  # This is the LAYER
                if i != 0:
                    previous_tensors: Tensor = torch.concat([upsample[l]] + cache[l], dim=1)
                    y: Tensor = self.concat[i-1][l](y, previous_tensors) if i != 0 else y

                y: Tensor = self.blocks[i][l](y)  # preserves channels

                cache[l].append(y)
                shapes.append(y.shape)

                y: Tensor = self.downsample[i][l](y) if l != (self.L - (i + 1)) else y  # Increases channels... kj

            #  Upsample all layers and cache the new tensors if all but the last stage...
            if i != self.L - 1:
                upsample: List[Tensor] = []
                for j, upsample_module in enumerate(self.upsample[i]):
                    upsample.append(
                        upsample_module(cache[j+1][i], shapes[j+1])  # dont upsample layer 1!!
                    )

                y = cache[0][i]  # reset y for the next i

        return self.out_conv(y)

    def __repr__(self):
        return f'{self._name}{self.spatial_dim}D[dims={self.dims}, depths={self.depths}, ' \
               f'in_channels={self.in_channels}, out_channels={self.out_channels}, block={self._block}, ' \
               f'normalization={self._normalizatoin}, activation={self._activation}, upsample={self._upsample_layer}, ' \
               f'concat={self._concat_conv}]'


class UNetPlusPlus_2D(UNetPlusPlusND):
    def __init__(self,
                 in_channels: Optional[int] = 1,
                 out_channels: int = 4,
                 L: int = 4,
                 *,
                 depths: Optional[List[int]] = [2, 2, 2, 2, 2],
                 dims: Optional[List[int]] = [32, 64, 128, 64, 32],
                 kernel_size: Optional[int] = 7,
                 activation: Optional[nn.Module] = nn.GELU,
                 block: Optional[nn.Module] = Block2D,
                 concat_conv: Optional[nn.Module] = ConcatConv2D,
                 upsample_layer: Optional[nn.Module] = UpSampleLayer2D,
                 normalization: Optional[nn.Module] = partial(LayerNorm, data_format='channels_first'),
                 name: Optional[str] = 'Unet++'
                 ):

        super(UNetPlusPlus_2D, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            L = L,
            spatial_dim=2,
            depths=depths,
            dims=dims,
            kernel_size=kernel_size,
            activation=activation,
            block=block,
            concat_conv=concat_conv,
            upsample_layer=upsample_layer,
            normalization=normalization,
            name=name
        )


class UNetPlusPlus_3D(UNetPlusPlusND):
    def __init__(self,
                 in_channels: Optional[int] = 1,
                 out_channels: int = 4,
                 L: int = 4,
                 *,
                 depths: Optional[List[int]] = (2, 2, 2, 2),
                 dims: Optional[List[int]] = (16, 32, 64, 128),  # [16, 32, 64, 128],
                 kernel_size: Optional[int] = 7,
                 activation: Optional[nn.Module] = nn.GELU,
                 block: Optional[nn.Module] = Block3D,
                 concat_conv: Optional[nn.Module] = ConcatConv3D,
                 upsample_layer: Optional[nn.Module] = UpSampleLayer3D,
                 normalization: Optional[nn.Module] = partial(LayerNorm, data_format='channels_first'),
                 name: Optional[str] = 'UNet++'
                 ):
        super(UNetPlusPlus_3D, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            L=L,
            spatial_dim=3,
            depths=depths,
            dims=dims,
            kernel_size=kernel_size,
            activation=activation,
            block=block,
            concat_conv=concat_conv,
            upsample_layer=upsample_layer,
            normalization=normalization,
            name=name
        )


if __name__ == '__main__':
    model = torch.jit.script(UNetPlusPlus_3D(dims=[1,1,1,1], depths=[1,1,1,1], L=4))
    x = torch.rand(1,1,300,300,20)
    y = model(x)
    print(y.shape)


    model = torch.jit.script(UNetPlusPlus_2D(dims=[1,1,1,1], depths=[1,1,1,1], L=4))
    x = torch.rand(1,1,300,300)
    y = model(x)
    print(y.shape)
