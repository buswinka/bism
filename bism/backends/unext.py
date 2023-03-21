import torch.nn as nn
import torch
from torch import Tensor
from typing import List, Tuple, Optional

from bism.modules.convnext_block import Block2D, Block3D
from bism.modules.layer_norm import LayerNorm
from bism.modules.concat import ConcatConv2D, ConcatConv3D
from bism.modules.upsample_layer import UpSampleLayer3D, UpSampleLayer2D

from functools import partial

# Scriptable!!!
class UNeXTND(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 spatial_dim: int,
                 *,
                 depths: Optional[List[int]] = [2, 2, 2, 2, 2],
                 dims: Optional[List[int]] = [32, 64, 128, 64, 32],
                 kernel_size: Optional[int] = 7,
                 drop_path_rate: Optional[float] = 0.0,
                 layer_scale_init_value: Optional[float] = 1.,
                 activation: Optional[nn.Module] = nn.GELU,
                 block: Optional[nn.Module] = Block3D,
                 concat_conv: Optional[nn.Module] = ConcatConv3D,
                 upsample_layer: Optional[nn.Module] = UpSampleLayer3D,
                 normalization: Optional[nn.Module] = partial(LayerNorm, data_format='channels_first'),
                 name: Optional[str] = 'UNeXT'
                 ):
        """
        Initalizes a (2/3)D UNeXT Model.

        :param in_channels: Model input channels
        :type in_channels: int
        :param out_channels: Model output channels
        :type out_channels: int
        :param spatial_dim: Spatial dimmensions; either 2 or 3
        :type spatial_dim: int
        :param depths: The number of computational blocks at each part of the U. Must have an odd number of elements
        :type depths: List[int]
        :param dims: Channels at each stage of the U. Must be the same length as depths
        :type dims: List[int]
        :param kernel_size: Kernel size of all convolutions in computational blocks
        :type kernel_size: int
        :param drop_path_rate: Drop path probability
        :type drop_path_rate: float
        :param layer_scale_init_value: Initializes residual path scaling parameter. If 0.0, does nothing.
        :type layer_scale_init_value: float
        :param activation: Activation function of computational blocks if applicable
        :type activation: nn.Module
        :param block: Convolution computational block.
        :type block: nn.Module
        :param concat_conv: Concatentation block
        :type concat_conv: nn.Module
        :param upsample_layer: Upsample block
        :type upsample_layer: nn.Module
        :param normalization: Normalization layer
        :type normalization: nn.Module
        :param name: Name of the model which is shown in the repr
        :type name: str
        """

        super(UNeXTND, self).__init__()


        assert len(depths) == len(dims), f'Number of depths should equal number of dims: {depths} != {dims}'

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

        self.spatial_dim = spatial_dim
        assert spatial_dim in (2, 3), f'Spatial Dimmension of {spatial_dim} is not supported'
        convolution = nn.Conv2d if spatial_dim == 2 else nn.Conv3d

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
            Block(dim=dims[0], drop_path=0.0, kernel_size=kernel_size, activation=activation)
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
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                convolution(dims[i], dims[i + 1], kernel_size=(2,) * spatial_dim, stride=(2,) * spatial_dim),
            )
            self.downsample_layers.append(downsample_layer)

        # Down Stages
        cur = 0
        for i in range(len(dims) // 2 + 1):
            num_blocks = depths[i]
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j], kernel_size=kernel_size, activation=activation,
                        layer_scale_init_value=layer_scale_init_value) for j in range(num_blocks)]
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
                *[Block(dim=dims[i + len(dims) // 2 + 1], drop_path=dp_rates[cur + j], kernel_size=kernel_size,
                        activation=activation,
                        layer_scale_init_value=layer_scale_init_value) for j in range(num_blocks)]
            )
            self.up_stages.append(stage)
            cur += depths[i]

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

        # Flatten Channels to out...
        self.out_conv = convolution(dims[-1], out_channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        """
        Performs the forward pass of the model.

        :param x: Input 4/5D tensor [B, C, X, Y, Z?]
        :type x: Tensor
        :return: Output tensor of identical size as input
        :rtype: Tensor
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


class UNeXT_3D(UNeXTND):
    def __init__(self,
                 in_channels: Optional[int] = 1,
                 out_channels: int = 4,
                 *,
                 depths: Optional[List[int]] = [2, 2, 2, 2, 2],
                 dims: Optional[List[int]] = [32, 64, 128, 64, 32],
                 kernel_size: Optional[int] = 7,
                 drop_path_rate: Optional[float] = 0.0,
                 layer_scale_init_value: Optional[float] = 1.,
                 activation: Optional[nn.Module] = nn.GELU,
                 block: Optional[nn.Module] = Block3D,
                 concat_conv: Optional[nn.Module] = ConcatConv3D,
                 upsample_layer: Optional[nn.Module] = UpSampleLayer3D,
                 normalization: Optional[nn.Module] = partial(LayerNorm, data_format='channels_first'),
                 name: Optional[str] = 'UNeXT'
                 ):
        """
        Initalizes a 3D UNeXT Model.
        ::
           dims = [32, 64, 128, 64, 32]
           depths = [2, 2, 4, 2, 2]

           ■ : ConvNextBlock
           ─ : Skip Connections

                      Depths:    2   2     4     2   2
                      ┌──────────────────────────────────────
           Dims:   32 │   in -> ■ ■─┬─────────────┬─■ ■ -> out
                   64 │             ■ ■─┬─────┬─■ ■
                  128 │                 ■ ■ ■ ■


        :param in_channels: Model input channels
        :type in_channels: int
        :param out_channels: Model output channels
        :type out_channels: int
        :param depths: The number of computational blocks at each part of the U. Must have an odd number of elements
        :type depths: List[int]
        :param dims: Channels at each stage of the U. Must be the same length as depths
        :type dims: List[int]
        :param kernel_size: Kernel size of all convolutions in computational blocks
        :type kernel_size: int
        :param drop_path_rate: Drop path probability
        :type drop_path_rate: float
        :param layer_scale_init_value: Initializes residual path scaling parameter. If 0.0, does nothing.
        :type layer_scale_init_value: float
        :param activation: Activation function of computational blocks if applicable
        :type activation: nn.Module
        :param block: 3D compatible convolution computational block.
        :type block: nn.Module
        :param concat_conv: 3D compatible concatentation block
        :type concat_conv: nn.Module
        :param upsample_layer: 3D compatible upsample block
        :type upsample_layer: nn.Module
        :param normalization: 3D compatible normalization layer
        :type normalization: nn.Module
        :param name: Name of the model which is shown in the repr
        :type name: str
        """

        super(UNeXT_3D, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            spatial_dim=3,
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

        


class UNeXT_2D(UNeXTND):
    def __init__(self,
                 in_channels: Optional[int] = 1,
                 out_channels: int = 4,
                 *,
                 depths: Optional[List[int]] = [2, 2, 2, 2, 2],
                 dims: Optional[List[int]] = [32, 64, 128, 64, 32],
                 kernel_size: Optional[int] = 7,
                 drop_path_rate: Optional[float] = 0.0,
                 layer_scale_init_value: Optional[float] = 1.,
                 activation: Optional[nn.Module] = nn.GELU,
                 block: Optional[nn.Module] = Block2D,
                 concat_conv: Optional[nn.Module] = ConcatConv2D,
                 upsample_layer: Optional[nn.Module] = UpSampleLayer2D,
                 normalization: Optional[nn.Module] = partial(LayerNorm, data_format='channels_first'),
                 name: Optional[str] = 'UNeXT'
                 ):
        """
        Initalizes a 2D UNeXT Model.

        :param in_channels: Model input channels
        :type in_channels: int
        :param out_channels: Model output channels
        :type out_channels: int
        :param depths: The number of computational blocks at each part of the U. Must have an odd number of elements
        :type depths: List[int]
        :param dims: Channels at each stage of the U. Must be the same length as depths
        :type dims: List[int]
        :param kernel_size: Kernel size of all convolutions in computational blocks
        :type kernel_size: int
        :param drop_path_rate: Drop path probability
        :type drop_path_rate: float
        :param layer_scale_init_value: Initializes residual path scaling parameter. If 0.0, does nothing.
        :type layer_scale_init_value: float
        :param activation: Activation function of computational blocks if applicable
        :type activation: nn.Module
        :param block: Convolution computational block.
        :type block: nn.Module
        :param concat_conv: Concatentation block
        :type concat_conv: nn.Module
        :param upsample_layer: Upsample block
        :type upsample_layer: nn.Module
        :param normalization: Normalization layer
        :type normalization: nn.Module
        :param name: Name of the model which is shown in the repr
        :type name: str
        """
        super(UNeXT_2D, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            spatial_dim=2,
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


if __name__=='__main__':
    print(torch.cuda.is_available())
    # model = torch.compile(UNeXT_3D(depths=[1, 1, 1, 1, 1], dims=[1,2,3,4,5], out_channels=16).to('cuda:1'), backend='inductor')

    # x = torch.rand((1,1,300,300,20))
    # y = model(x)

    # model = torch.jit.script(UNeXT_2D(depths=[1, 1, 1, 1, 1], dims=[1,2,3,4,5], out_channels=16))
    # x = torch.rand((1,1,300,300))
    # y = model(x)

