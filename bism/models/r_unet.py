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


class RUNeT_3D(nn.Module):
    def __init__(self,
                 in_channels: Optional[int] = 1,
                 N: Optional[int] = 2,
                 depths: Optional[List[int]] = (1, 2, 3, 2, 1),  # [1, 2, 3, 2, 1],
                 dims: Optional[List[int]] = (32, 64, 128, 64, 32),  # [16, 32, 64, 32, 16],
                 drop_path_rate: Optional[float] = 0.0,
                 layer_scale_init_value: Optional[float] = 1.,
                 out_channels: int = 4,
                 kernel_size: int = 7,
                 activation: Optional = nn.GELU,
                 ):
        super(RUNeT_3D, self).__init__()

        assert len(depths) == len(dims), f'Number of depths should equal number of dims: {depths} != {dims}'

        # ----------------- Save Init Params for Preheating...
        self.dims = dims
        self.depths = depths
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.N = N

        # 2D or 3D
        Block = partial(RBlock3D, N=N)
        ConcatConv = ConcatConv3D
        UpSampleLayer = UpSampleLayer3D


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
            nn.Conv3d(in_channels, dims[0], kernel_size=7, padding=3),
            Block(dim=dims[0], kernel_size=kernel_size, activation=activation)
        )

        # ----------------- Unlike original conv next we cannot reduce the stem by that much...
        stem = nn.Sequential(
            nn.Conv3d(dims[0], dims[0], kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)

        # ----------------- Downsample layers
        for i in range(len(dims) // 2):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv3d(dims[i], dims[i + 1], kernel_size=(2, 2, 2), stride=(2, 2, 2)),
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
        norm_layer = partial(LayerNorm, eps=1e-6, data_format="channels_first")
        for i_layer in range(len(dims) // 2 + 1):
            layer = norm_layer(dims[i_layer])
            self.down_norms.append(layer)

        # Up Layer Normalization
        norm_layer = partial(LayerNorm, eps=1e-6, data_format="channels_first")
        for i_layer in range(len(dims) // 2):
            layer = norm_layer(dims[i_layer + len(dims) // 2 + 1])
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
        return f'nn.Module[name=UNeXt, dims={self.dims}, depths={self.depths}, ' \
               f'in_channels={self.in_channels}, out_channels={self.out_channels}]'


if __name__ == '__main__':
    model = RUNeT_3D()
    x = torch.rand((1, 1, 100, 100, 20))
    _ = model(x)