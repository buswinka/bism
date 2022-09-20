import torch.nn as nn
import torch
from torch import Tensor
from typing import List, Tuple, Optional, Union
from functools import partial

from bism.modules.unet_block import Block2D, Block3D
from bism.modules.concat import ConcatConv2D, ConcatConv3D
from bism.modules.upsample_layer import UpSampleLayer3D, UpSampleLayer2D
from bism.modules.layer_norm import LayerNorm


class UNetPlusPlus_3D(nn.Module):
    """
    Generic Constructor for a UNet architecture of variable size and shape.
    """
    def __init__(self,
                 in_channels: Optional[int] = 1,
                 out_channels: int = 1,
                 L: int = 4,
                 *,
                 dims: Optional[List[int]] = (16, 32, 64, 128),  # [16, 32, 64, 128],
                 depths: Optional[List[int]] = (2, 2, 2, 2),  # [1, 2, 3, 2, 1],
                 kernel_size: Optional[Union[Tuple[int], int]] = 3,
                 activation: Optional[nn.Module] = nn.ReLU,
                 block = Block3D,
                 concat_conv = ConcatConv3D,
                 upsample_layer = UpSampleLayer3D,
                 normalization = partial(LayerNorm, data_format='channels_first'),
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
        super(UNetPlusPlus_3D, self).__init__()

        assert len(depths) == len(dims) == L, f'Number of depths should equal number of dims and layers: {depths} != {dims} != {L}'

        # ----------------- Save Init Params...
        self.dims = dims
        self.depths = depths
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.activation = activation
        self.L = L

        # 2D or 3D | Weird case rules because they're all nn.Module classes...
        Block: nn.Module = block
        ConcatConv: nn.Module = concat_conv
        UpSampleLayer: nn.Module = upsample_layer

        self.in_conv = nn.Conv3d(in_channels=in_channels, out_channels=dims[0], kernel_size=self.kernel_size, padding=(1,1,1))
        self.out_conv = nn.Conv3d(in_channels=dims[0], out_channels=out_channels, kernel_size=self.kernel_size, padding=(1,1,1))

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
                        nn.Conv3d(dims[l], dims[l + 1], kernel_size=(2, 2, 2), stride=(2, 2, 2)),
                        normalization(dims[l + 1],))
                    )
            self.downsample.append(downsample)

    @torch.jit.ignore()
    def forward(self, x: Tensor) -> Tensor:
        """
        NOT SCRITPABLE!

        Applies the UNet++ model to an input tensor. Assumes a UNet++ structure of:
        (Note - this representation is not how the model was represented in the publication..."

                  i =     0 1 2 3 4       --->      Schematized
                 -------------------------   ------------------------
          |C: 16 | in ->  ■ ■ ■ ■ ■ -> out |   in -> ■ ■ ■ ■ ■ -> out
          |C: 32 |        ■ ■ ■ ■          |          ■ ■ ■ ■
        l |C: 64 |        ■ ■ ■            |           ■ ■ ■
          |C:128 |        ■ ■              |            ■ ■
          |C:256 |        ■                |             ■

        Where *l* is the "Layer" and *i* is the "STAGE" and ■ is a convolutional "Block"

        ---------
        A Block PRESERVES the number of dimmensions | C: N -> N
        Upsamples REDCUE the number of dimensions   | C: N -> M  [N > M]
        Downsample INCREASE the number of dimensions| C: M -> N  [N > M]

        Channels increase as l gets deeper according to self.dims
        --------

        :param x: 5D tensor [B, in_channels, X, Y, Z]
        :return: x: 5D tensor [B, out_channels, X, Y, Z]
        """

        cache: List[List[Tensor]] = [[] for _ in range(self.L)]  # This is where we cache the intermediate tensors...
        upsample: List[Tensor] = [torch.empty(0) for _ in range(self.L)]

        y = self.in_conv(x)

        for i in range(self.L):  # This is the STAGE
            shapes: List[Tensor] = [torch.tensor(y.shape)]

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
        return f'UNet++[in_channels={self.in_channels}, out_channels={self.out_channels}, ' \
               f'dims={self.dims}, depths={self.depths}, ' \
               f'kernel_size={self.kernel_size}, activation={self.activation}]'


if __name__ == '__main__':
    model = UNetPlusPlus_3D(dims=[1,1,1,1], depths=[1,1,1,1], L=4)
    x = torch.rand(1,1,300,300,20)
    _ = model(x)