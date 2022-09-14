import torch.nn as nn
import torch
from torch import Tensor
from typing import List, Tuple, Optional

from bism.modules.unet_block import Block2D, Block3D
from bism.modules.layer_norm import LayerNorm
from bism.modules.concat import ConcatConv2D, ConcatConv3D
from bism.modules.upsample_layer import UpSampleLayer3D, UpSampleLayer2D
from bism.modules.drop_path import DropPath

from functools import partial


@torch.jit.script
def _crop(img: Tensor, x: int, y: int, z: int, w: int, h: int, d: int) -> Tensor:
    return img[..., x:x + w, y:y + h, z:z + d]


class UNet_3D(nn.Module):
    def __init__(self, in_channels: Optional[int] = 1,
                 depths: Optional[List[int]] = (2, 2, 2, 2, 2),  # [1, 2, 3, 2, 1],
                 dims: Optional[List[int]] = (32, 64, 128, 64, 32),  # [16, 32, 64, 32, 16],
                 out_channels: int = 1,
                 kernel_size: int = 3,
                 activation: Optional = nn.ReLU,
                 ):
        super(UNet_3D, self).__init__()

        assert len(depths) == len(dims), f'Number of depths should equal number of dims: {depths} != {dims}'

        # ----------------- Save Init Params for Preheating...
        self.dims = dims
        self.depths = depths
        self.in_channels = in_channels
        self.out_channels = out_channels

        # 2D or 3D
        Block = Block3D
        ConcatConv = ConcatConv3D
        UpSampleLayer = UpSampleLayer3D

        # ----------------- Model INIT
        self.downsample_layers = nn.ModuleList()
        self.upsample_layers = nn.ModuleList()

        self.down_blocks = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        self.up_blocks = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks

        self.down_norms = nn.ModuleList()
        self.up_norms = nn.ModuleList()

        # ----------------- Downsample layers
        for i in range(len(dims) // 2):
            self.downsample_layers.append(nn.MaxPool3d(2))

        # ----------------- Down Stages
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

        # Upsample layers
        for i in range(len(dims) // 2):
            upsample_layer = UpSampleLayer(dims[i + len(dims) // 2], dims[i + 1 + len(dims) // 2])
            self.upsample_layers.append(upsample_layer)

        # Up Stages
        _dims = [in_channels] + list(dims)  # [1, 16, 32, 64, 32, 16] | [2, 2, 2, 2, 2]
        for i in range(1, len(dims) // 2):
            stage = []
            for j in range(depths[i + len(dims) // 2 + 1]):
                stage.append(
                    Block(out_channels=_dims[i + 1 + len(_dims) // 2] if j == depths[i + len(dims) // 2] - 1 else _dims[i + len(_dims) // 2],
                          in_channels=_dims[i + len(_dims) // 2],
                          kernel_size=kernel_size,
                          activation=activation)
                )

            self.up_blocks.append(nn.Sequential(*stage))

        # Concat Layers
        self.concat = nn.ModuleList()
        for i in range(len(dims) // 2):
            self.concat.append(ConcatConv(dims[i + len(dims) // 2 + 1]))

        self.final_concat = ConcatConv(in_channels=dims[-1])

        # Final Upscale
        self.upscale_out = UpSampleLayer(dims[-1], dims[-1])

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

        # Flatten Channels to out...
        self.out_conv = nn.Conv3d(dims[-1], out_channels, kernel_size=1)


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

        steps = []
        shapes = []

        # assert len(self.upsample_layers) == len(self.concat)
        # assert len(self.upsample_layers) == len(self.up_blocks)
        # assert len(self.upsample_layers) == len(self.up_norms), f'{len(self.up_norms)=}, {len(self.upsample_layers)=}'

        shapes.append(x.shape)

        for i, (down, stage) in enumerate(zip(self.downsample_layers, self.down_blocks)):
            # Save input for upswing of Unet
            x = stage(x)
            shapes.append(x.shape)
            steps.append(x)
            x = down(x)

        x = self.down_blocks[-1](x)  # bottom of the U

        shapes.append(x.shape)
        shapes.reverse()

        for i, (up, cat, stage) in enumerate(
                zip(self.upsample_layers, self.concat, self.up_blocks)):
            x = up(x, shapes[i + 1])
            y = steps.pop(-1)
            x = cat(x, y)
            x = stage(x)

        # Out

        x = self.upscale_out(x, shapes[-1])
        y = steps.pop(-1)
        x = self.final_concat(x, y)
        x = self.out_conv(x)

        return x


    def __repr__(self):
        return f'nn.Module[name=UNeXt, dims={self.dims}, depths={self.depths}, ' \
               f'in_channels={self.in_channels}, out_channels={self.out_channels}]'


class UNet_2D(nn.Module):
    def __init__(self, in_channels: Optional[int] = 1,
                 depths: Optional[List[int]] = (2, 2, 2, 2, 2),  # [1, 2, 3, 2, 1],
                 dims: Optional[List[int]] = (32, 64, 128, 64, 32),  # [16, 32, 64, 32, 16],
                 out_channels: int = 1,
                 kernel_size: int = 3,
                 activation: Optional = nn.ReLU,
                 ):
        super(UNet_2D, self).__init__()

        assert len(depths) == len(dims), f'Number of depths should equal number of dims: {depths} != {dims}'

        # ----------------- Save Init Params for Preheating...
        self.dims = dims
        self.depths = depths
        self.in_channels = in_channels
        self.out_channels = out_channels

        # 2D or 3D
        Block = Block2D
        ConcatConv = ConcatConv2D
        UpSampleLayer = UpSampleLayer2D

        # ----------------- Model INIT
        self.downsample_layers = nn.ModuleList()
        self.upsample_layers = nn.ModuleList()

        self.down_blocks = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        self.up_blocks = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks

        self.down_norms = nn.ModuleList()
        self.up_norms = nn.ModuleList()

        # ----------------- Downsample layers
        for i in range(len(dims) // 2):
            self.downsample_layers.append(nn.MaxPool3d(2))

        # ----------------- Down Stages
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

        # Upsample layers
        for i in range(len(dims) // 2):
            upsample_layer = UpSampleLayer(dims[i + len(dims) // 2], dims[i + 1 + len(dims) // 2])
            self.upsample_layers.append(upsample_layer)

        # Up Stages
        _dims = [in_channels] + list(dims)  # [1, 16, 32, 64, 32, 16] | [2, 2, 2, 2, 2]
        for i in range(1, len(dims) // 2):
            stage = []
            for j in range(depths[i + len(dims) // 2 + 1]):
                stage.append(
                    Block(out_channels=_dims[i + 1 + len(_dims) // 2] if j == depths[i + len(dims) // 2] - 1 else _dims[i + len(_dims) // 2],
                          in_channels=_dims[i + len(_dims) // 2],
                          kernel_size=kernel_size,
                          activation=activation)
                )

            self.up_blocks.append(nn.Sequential(*stage))

        # Concat Layers
        self.concat = nn.ModuleList()
        for i in range(len(dims) // 2):
            self.concat.append(ConcatConv(dims[i + len(dims) // 2 + 1]))

        self.final_concat = ConcatConv(in_channels=dims[-1])

        # Final Upscale
        self.upscale_out = UpSampleLayer(dims[-1], dims[-1])

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

        # Flatten Channels to out...
        self.out_conv = nn.Conv3d(dims[-1], out_channels, kernel_size=1)


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

        steps = []
        shapes = []

        # assert len(self.upsample_layers) == len(self.concat)
        # assert len(self.upsample_layers) == len(self.up_blocks)
        # assert len(self.upsample_layers) == len(self.up_norms), f'{len(self.up_norms)=}, {len(self.upsample_layers)=}'

        shapes.append(x.shape)

        for i, (down, stage) in enumerate(zip(self.downsample_layers, self.down_blocks)):
            # Save input for upswing of Unet
            x = stage(x)
            shapes.append(x.shape)
            steps.append(x)
            x = down(x)

        x = self.down_blocks[-1](x)  # bottom of the U

        shapes.append(x.shape)
        shapes.reverse()

        for i, (up, cat, stage) in enumerate(
                zip(self.upsample_layers, self.concat, self.up_blocks)):
            x = up(x, shapes[i + 1])
            y = steps.pop(-1)
            x = cat(x, y)
            x = stage(x)

        # Out

        x = self.upscale_out(x, shapes[-1])
        y = steps.pop(-1)
        x = self.final_concat(x, y)
        x = self.out_conv(x)

        return x


    def __repr__(self):
        return f'nn.Module[name=UNeXt, dims={self.dims}, depths={self.depths}, ' \
               f'in_channels={self.in_channels}, out_channels={self.out_channels}]'



if __name__ == '__main__':
    model = UNet_2D(dims=(4,8,16,8,4))
    a = torch.rand((1, 1, 100, 100))
    out = model(a)