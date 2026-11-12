"""Building blocks of the AerialSegNet encoder-decoder.

The whole network is assembled from four pieces:

``SingleConv``   depthwise 3x3 -> pointwise 1x1 -> BatchNorm -> ReLU
``StagedLayer``  two ``SingleConv`` in series, first output concatenated onto
                 the second, so the block doubles the channel count
``Down``         2x2 max pool followed by a ``StagedLayer``
``Up``           transposed conv to a quarter of the channels, pad to the skip,
                 concatenate, then a ``StagedLayer``
"""

import torch
import torch.nn.functional as F
from torch import nn


class SingleConv(nn.Module):
    """Depthwise separable convolution with BatchNorm and ReLU.

    Splitting the 3x3 spatial filter from the 1x1 channel mixer is what keeps
    the full network at 2.49M parameters.

    Args:
        in_chnls: input channels.
        out_chnls: output channels.
    """

    def __init__(self, in_chnls: int, out_chnls: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_chnls, in_chnls, kernel_size=3, padding=1, groups=in_chnls
        )
        self.pointwise = nn.Conv2d(in_chnls, out_chnls, kernel_size=1)
        self.BN = nn.BatchNorm2d(out_chnls)
        self.RLU = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.depthwise(x)
        y = self.pointwise(y)
        y = self.BN(y)
        return self.RLU(y)


class StagedLayer(nn.Module):
    """Two separable convolutions with a dense-style concatenation.

    The output has ``2 * out_chnls`` channels: the first convolution's feature
    map concatenated with the second's.

    Args:
        in_chnls: input channels.
        out_chnls: channels produced by each of the two convolutions.
    """

    def __init__(self, in_chnls: int, out_chnls: int) -> None:
        super().__init__()
        self.conv1 = SingleConv(in_chnls, out_chnls)
        self.conv2 = SingleConv(out_chnls, out_chnls)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(x)
        y1 = self.conv2(y)
        return torch.cat([y, y1], dim=1)


class Down(nn.Module):
    """Downsampling stage: 2x2 max pool then a :class:`StagedLayer`.

    Args:
        in_chnls: input channels.
        out_chnls: channels per convolution; the stage emits ``2 * out_chnls``.
    """

    def __init__(self, in_chnls: int, out_chnls: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            StagedLayer(in_chnls, out_chnls),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Up(nn.Module):
    """Upsampling stage with a learned transposed convolution.

    The transposed convolution cuts the channel count to ``in_chnls // 4``
    before the skip connection is concatenated, which keeps the decoder
    narrower than the encoder at every matching resolution.

    Args:
        in_chnls: channels arriving from the layer below.
        out_chnls: the stage emits ``out_chnls`` channels in total
            (``StagedLayer`` doubles ``out_chnls // 2``).
    """

    def __init__(self, in_chnls: int, out_chnls: int) -> None:
        super().__init__()
        self.trans = nn.ConvTranspose2d(
            in_chnls, in_chnls // 4, kernel_size=2, stride=2
        )
        self.conv = StagedLayer(in_chnls * 3 // 4, out_chnls // 2)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x1: feature map from the layer below (gets upsampled).
            x2: skip connection from the matching encoder stage.
        """
        x1 = self.trans(x1)

        # odd input sizes leave the upsampled map one pixel short of the skip
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        x1 = F.pad(
            x1,
            [
                diff_x // 2,
                diff_x - diff_x // 2,
                diff_y // 2,
                diff_y - diff_y // 2,
            ],
        )

        return self.conv(torch.cat([x2, x1], dim=1))


class OutConv(nn.Module):
    """1x1 convolution mapping decoder channels to class logits."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)
