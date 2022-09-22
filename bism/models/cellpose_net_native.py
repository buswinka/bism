import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Union, Tuple, Optional

kernel_size = 3

"""
The Native Implementation of CPnet from Cellpose! Refactored for clarity and best coding style
Will re-implement later... as of right now, its kinda hard to re-work to be similar to other UNet implementations...
Also they have a weird residual block architecture... 

Re-Implementation Strategy is
-----------------------------
1. Get rid of all of these fucked blocks n shit.. Use a standardized approach...
2. Standardize the residual block
3. 

"""

def ConvBatchReLU(in_channels: int, out_channels: int, kernel_size: int) -> nn.Module:
    """
    Returns a nn.Sequential of Conv2d -> BN -> ReLU
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
        nn.BatchNorm2d(out_channels, eps=1e-5),
        nn.ReLU(inplace=True),
    )


def BatchReLUConv(in_channels: int, out_channels: int, kernel_size: int) -> nn.Module:
    """
    Returns a nn.Sequential of BN -> ReLU -> Conv2d
    """
    return nn.Sequential(
        nn.BatchNorm2d(in_channels, eps=1e-5),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
    )


def BatchConvNoReLU(in_channels, out_channels, kernel_size):
    """
    Returns a nn.Sequential of BN -> Conv2d
    """
    return nn.Sequential(
        nn.BatchNorm2d(in_channels, eps=1e-5),
        nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
    )


class ResidualDownBlock(nn.Module):
    """
    Residual Block: Adds 4 blocks into a squential module
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.stem = BatchConvNoReLU(in_channels, out_channels, 1)


        self.blocks = nn.Sequential(  # Create a sequential of Convolutional Blocks
            *[BatchReLUConv(in_channels if t == 0 else out_channels, out_channels, kernel_size) for t in range(4)]
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x) + self.blocks[1](self.blocks[0](x))
        x = x + self.blocks[3](self.blocks[2](x))

        return x


class ConvDownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        self.blocks = nn.Sequential(
            *[BatchReLUConv(in_channels if t == 0 else out_channels, out_channels, kernel_size) for t in range(2)]
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.blocks[0](x)
        x = self.blocks[1](x)
        return x


class DownsampleBlock(nn.Module):
    def __init__(self, dims: List[int], kernel_size: int, residual_on=True):
        super().__init__()
        self.down = nn.Sequential()
        self.maxpool = nn.MaxPool2d(2, 2)
        self.dims = dims

        for n in range(len(dims) - 1):  # Down Half of the U
            if residual_on:
                self.down.add_module('res_down_%d' % n, ResidualDownBlock(dims[n], dims[n + 1], kernel_size))
            else:
                self.down.add_module('conv_down_%d' % n, ConvDownBlock(dims[n], dims[n + 1], kernel_size))

    def forward(self, x: Tensor) -> List[Tensor]:
        """ Returns a List of Tensors at each stage of a UNet """
        xd = []
        for n in range(len(self.down)):
            y = self.maxpool(xd[n - 1]) if n > 0 else x
            xd.append(self.down[n](y))
        return xd


class BatchConvolutionStyleBlock(nn.Module):
    """ Incorporates style into the conv? I think this is the upswing... """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 style_channels: int,
                 kernel_size: int,
                 concatenation: Optional[bool] = False):

        super().__init__()
        self.concatenation: bool = concatenation
        if concatenation:
            self.conv = BatchReLUConv(in_channels * 2, out_channels, kernel_size)
            self.linear = nn.Linear(style_channels, out_channels * 2)
        else:
            self.conv = BatchReLUConv(in_channels, out_channels, kernel_size)
            self.linear = nn.Linear(style_channels, out_channels)

    def forward(self, style: Tensor, x: Tensor, y: Optional[Union[Tensor, None]] = None) -> Tensor:
        if y is not None:
            x = torch.cat((y, x), dim=1) if self.concatenation else x + y

        features: Tensor = self.linear(style)  # Something like [B, STYLE]
        y = x + features.unsqueeze(-1).unsqueeze(-1)
        y = self.conv(y)
        return y


class ResidualUpBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 style_channels: int,
                 kernel_size: int,
                 concatenation: Optional[bool] = False):
        super().__init__()

        self.batch_conv = BatchReLUConv(in_channels, out_channels, kernel_size)

        self.conv0 = BatchConvolutionStyleBlock(out_channels, out_channels, style_channels, kernel_size, concatenation=concatenation)
        self.conv1 = BatchConvolutionStyleBlock(out_channels, out_channels, style_channels, kernel_size)
        self.conv2 = BatchConvolutionStyleBlock(out_channels, out_channels, style_channels, kernel_size)

        self.proj = BatchConvNoReLU(in_channels, out_channels, 1)

    def forward(self, x: Tensor, y: Tensor, style: Tensor) -> Tensor:

        x = self.proj(x) + self.conv0(style, self.batch_conv(x), y=y)
        x = x + self.conv2(style, self.conv1(style, x))
        return x


class ConvolutionUpBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 style_channels: int,
                 kernel_size: int,
                 concatenation: Optional[bool] = False):
        super().__init__()
        self.batch_conv = BatchReLUConv(in_channels, out_channels, kernel_size)
        self.style_conv = BatchConvolutionStyleBlock(out_channels, out_channels, style_channels, kernel_size,
                                                                  concatenation=concatenation)

    def forward(self, x: Tensor, y: Tensor, style: Tensor) -> Tensor:
        x = self.style_conv(style=style, x=self.batch_conv(x), y=y)
        return x


class MakeStyle(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()

    def forward(self, x: Tensor) -> Tensor:
        """
        What is the in shape? [B, C, X, Y?]
        """
        style = F.avg_pool2d(x, kernel_size=(x.shape[-2], x.shape[-1]))
        style = self.flatten(style)
        style = style / torch.sum(style ** 2, dim=1, keepdim=True) ** .5

        return style


class UpsampleBlock(nn.Module):
    def __init__(self,
                 dims: List[int],
                 kernel_size: int,
                 residual_on: Optional[bool] = True,
                 concatenation: Optional[bool] = False):
        super().__init__()

        self.upsampling = nn.Upsample(scale_factor=2, mode='nearest')
        Block = ResidualUpBlock if residual_on else ConvolutionUpBlock

        self.up = nn.Sequential(
            *[Block(dims[n], dims[n - 1], dims[-1], kernel_size, concatenation) for n in range(1, len(dims))]
        )

    def forward(self, style: Tensor, xd: List[Tensor]) -> Tensor:
        x = self.up[-1](xd[-1], xd[-1], style)

        # Skip connections are very subtle, but they're there... :(
        for n in range(len(self.up) - 2, -1, -1):
            x = self.upsampling(x) # upsamples x
            x = self.up[n](x, xd[n], style)  # Concatenation, Style Addition, and whatever else is here...
        return x


class CPnet(nn.Module):
    def __init__(self,
                 dims: List[int],
                 out_channels: int,
                 kernel_size: int,
                 residual_on: Optional[bool] = True,
                 style_on: Optional[bool] = True,
                 concatenation: Optional[bool] = False,
                 diam_mean: float = 30.):
        super(CPnet, self).__init__()

        self.dims = dims  # I think the number of channels at each stage?
        self.out_channels = out_channels  # out_channels
        self.kernel_size = kernel_size # kernel_size
        self.residual_on = residual_on
        self.style_on = style_on
        self.concatenation = concatenation

        self.downsample = DownsampleBlock(dims, kernel_size, residual_on=residual_on)
        print(self.downsample)

        nbaseup = dims[1:]
        nbaseup.append(nbaseup[-1])

        self.upsample = UpsampleBlock(nbaseup, kernel_size, residual_on=residual_on, concatenation=concatenation)
        self.make_style = MakeStyle()

        self.output = BatchReLUConv(nbaseup[0], out_channels, 1)

        self.diam_mean = nn.Parameter(data=torch.ones(1) * diam_mean, requires_grad=False)
        self.diam_labels = nn.Parameter(data=torch.ones(1) * diam_mean, requires_grad=False)
        self.style_on = style_on

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Basically Computes all Down Blocks of a UNet
        From the bottom block, makes some 'style' tensor that idk what it does tbh

        Ads this style to the upswing

        """
        x: List[Tensor] = self.downsample(x)          # List of each tensor along the encoder
        style: Tensor = self.make_style(x[-1])        # Make the style from the output of the last block

        style0 = style
        style = style if self.style_on else style * 0  # Zero the style if you don't want to use it

        x: Tensor = self.upsample(style, x)            # Upsample based on each down stage and style...
        x: Tensor = self.output(x)

        return x, style0

    def save_model(self, filename):
        torch.save(self.state_dict(), filename)

    def load_model(self, filename, cpu=False):
        if not cpu:
            state_dict = torch.load(filename)
        else:
            self.__init__(self.nbase,
                          self.nout,
                          self.kernel_size,
                          self.residual_on,
                          self.style_on,
                          self.concatenation,
                          self.mkldnn,
                          self.diam_mean)
            state_dict = torch.load(filename, map_location=torch.device('cpu'))
        self.load_state_dict(dict([(name, param) for name, param in state_dict.items()]), strict=False)


if __name__ == '__main__':
    model = CPnet([4,8,16,32], 3, 3)
    x = torch.rand((1,4,300,300))
    out = model(x)
    print(out.shape)