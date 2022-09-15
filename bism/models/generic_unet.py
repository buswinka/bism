import torch.nn as nn
import torch
from torch import Tensor
from typing import List, Tuple, Optional, Union

from bism.modules.unet_block import Block2D, Block3D
from bism.modules.concat import ConcatConv2D, ConcatConv3D
from bism.modules.upsample_layer import UpSampleLayer3D, UpSampleLayer2D


@torch.jit.script
def _crop(img: Tensor, x: int, y: int, z: int, w: int, h: int, d: int) -> Tensor:
    return img[..., x:x + w, y:y + h, z:z + d]

class UNet_3D(nn.Module):
    """
    Generic Constructor for a UNet architecture of variable size and shape.
    """
    def __init__(self,
                 in_channels: Optional[int] = 1,
                 out_channels: int = 1,
                 *,
                 dims: Optional[List[int]] = (32, 64, 128, 64, 32),  # [16, 32, 64, 32, 16],
                 depths: Optional[List[int]] = (2, 2, 2, 2, 2),  # [1, 2, 3, 2, 1],
                 kernel_size: Optional[Union[Tuple[int], int]] = 3,
                 activation: Optional[nn.Module] = nn.ReLU,
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
        super(UNet_3D, self).__init__()

        assert len(depths) == len(dims), f'Number of depths should equal number of dims: {depths} != {dims}'

        # ----------------- Save Init Params...
        self.dims = dims
        self.depths = depths
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.activation = activation

        # 2D or 3D
        Block = Block3D
        ConcatConv = ConcatConv3D
        UpSampleLayer = UpSampleLayer3D

        # ----------------- Model INIT
        self.downsample_layers = nn.ModuleList()
        self.down_blocks = nn.ModuleList()

        self.upsample_layers = nn.ModuleList()
        self.up_blocks = nn.ModuleList()

        self.concat = nn.ModuleList()

        # ----------------- Downsample layers
        for i in range(len(dims) // 2):
            self.downsample_layers.append(nn.MaxPool3d(2))

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
                        out_channels=_dims[i  +len(_dims) // 2],
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

        self.out_conv = nn.Conv3d(dims[-1], out_channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        """
        Applies the UNet model to an input tensor
        :param x: 5D tensor [B, in_channels, X, Y, Z]
        :return: x: 5D tensor [B, out_channels, X, Y, Z]
        """

        steps: List[Tensor] = []
        shapes: List[Tensor] = []
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
        return f'UNet[in_channels={self.in_channels}, out_channels={self.out_channels}, ' \
               f'dims={self.dims}, depths={self.depths}, ' \
               f'kernel_size={self.kernel_size}, activation={self.activation}]'

class UNet_2D(nn.Module):
    """
    Generic Constructor for a UNet architecture of variable size and shape.
    """
    def __init__(self,
                 in_channels: Optional[int] = 1,
                 out_channels: int = 1,
                 *,
                 dims: Optional[List[int]] = (32, 64, 128, 64, 32),  # [16, 32, 64, 32, 16],
                 depths: Optional[List[int]] = (2, 2, 2, 2, 2),  # [1, 2, 3, 2, 1],
                 kernel_size: Optional[Union[Tuple[int], int]] = 3,
                 activation: Optional[nn.Module] = nn.ReLU,
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
        super(UNet_2D, self).__init__()

        assert len(depths) == len(dims), f'Number of depths should equal number of dims: {depths} != {dims}'

        # ----------------- Save Init Params...
        self.dims = dims
        self.depths = depths
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.activation = activation

        # 2D or 3D
        Block = Block2D
        ConcatConv = ConcatConv2D
        UpSampleLayer = UpSampleLayer2D

        # ----------------- Model INIT
        self.downsample_layers = nn.ModuleList()
        self.down_blocks = nn.ModuleList()

        self.upsample_layers = nn.ModuleList()
        self.up_blocks = nn.ModuleList()

        self.concat = nn.ModuleList()

        # ----------------- Downsample layers
        for i in range(len(dims) // 2):
            self.downsample_layers.append(nn.MaxPool2d(2))

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
                        out_channels=_dims[i  +len(_dims) // 2],
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

        self.out_conv = nn.Conv2d(dims[-1], out_channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        """
        Applies the UNet model to an input tensor
        :param x: 5D tensor [B, in_channels, X, Y, Z]
        :return: x: 5D tensor [B, out_channels, X, Y, Z]
        """

        steps: List[Tensor] = []
        shapes: List[Tensor] = []
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
        return f'UNet2D[in_channels={self.in_channels}, out_channels={self.out_channels}, ' \
               f'dims={self.dims}, depths={self.depths}, ' \
               f'kernel_size={self.kernel_size}, activation={self.activation}]'


if __name__ == '__main__':
    model = UNet_2D(dims=(4,8,16,8,4))
    a = torch.rand((1, 1, 100, 100))
    out = model(a)