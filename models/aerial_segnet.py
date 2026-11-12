"""AerialSegNet: a lightweight separable encoder-decoder for UAV imagery.

Channel ladder at 480x480 input (``base_channels=32``)::

    res        encoder                       out ch.   skip   decoder            out ch.
    480x480    StagedLayer(3 -> 32)               64    -->   Up(128 -> 64)           64
    240x240    Down(64 -> 64)                    128    -->   Up(256 -> 128)         128
    120x120    Down(128 -> 128)                  256    -->   Up(512 -> 256)         256
     60x60     Down(256 -> 256)                  512    -->   Up(1024 -> 512)        512
     30x30     Down(512 -> 512)                 1024     *    bottleneck

A final 1x1 ``OutConv`` maps the 64 full-resolution decoder channels to
``n_classes`` logits.
"""

import torch
from torch import nn

from models.modules.activation import Activation
from models.modules.blocks import Down, OutConv, StagedLayer, Up


class AerialSegNet(nn.Module):
    """Separable staged-dense U-shaped segmentation network.

    Args:
        in_ch: input image channels (3 for RGB).
        n_classes: number of segmentation classes.
        base_channels: channels produced by the first stage. The published
            configuration uses 32, which gives 2,486,562 parameters at
            ``n_classes=4``.
        head_activation: activation applied to the output logits. The original
            notebook applies ``relu`` here; see the note below.

    Note:
        ``head_activation="relu"`` clamps negative logits to zero before the
        softmax inside ``CrossEntropyLoss``. That is what the reported runs
        used, so it is the default for reproducibility, but ``"none"`` is the
        conventional choice and is worth trying for new experiments.
    """

    def __init__(
        self,
        in_ch: int = 3,
        n_classes: int = 4,
        base_channels: int = 32,
        head_activation: str = "relu",
    ) -> None:
        super().__init__()
        self.in_ch = in_ch
        self.n_classes = n_classes
        self.base_channels = base_channels

        c = base_channels
        self.activation = Activation(head_activation)

        # encoder, each stage doubles the channels through the concatenation
        self.conv1 = StagedLayer(in_ch, c)  # -> 2c   @ 1/1
        self.down1 = Down(2 * c, 2 * c)  # -> 4c   @ 1/2
        self.down2 = Down(4 * c, 4 * c)  # -> 8c   @ 1/4
        self.down3 = Down(8 * c, 8 * c)  # -> 16c  @ 1/8
        self.down4 = Down(16 * c, 16 * c)  # -> 32c  @ 1/16

        # decoder
        self.up1 = Up(32 * c, 16 * c)  # -> 16c  @ 1/8
        self.up2 = Up(16 * c, 8 * c)  # -> 8c   @ 1/4
        self.up3 = Up(8 * c, 4 * c)  # -> 4c   @ 1/2
        self.up4 = Up(4 * c, 2 * c)  # -> 2c   @ 1/1

        self.out_final = OutConv(2 * c, n_classes)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: float tensor of shape ``(N, in_ch, H, W)``.

        Returns:
            Logits of shape ``(N, n_classes, H, W)``.
        """
        out1 = self.conv1(image)
        out2 = self.down1(out1)
        out3 = self.down2(out2)
        out4 = self.down3(out3)
        out5 = self.down4(out4)

        out = self.up1(out5, out4)
        out = self.up2(out, out3)
        out = self.up3(out, out2)
        out = self.up4(out, out1)
        out = self.out_final(out)

        return self.activation(out)

    @torch.no_grad()
    def num_parameters(self, trainable_only: bool = True) -> int:
        """Count parameters. Returns 2,486,562 for the published settings."""
        params = self.parameters()
        if trainable_only:
            params = (p for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in params)


# Backwards-compatible alias: the notebook calls this class L_Densed_UNet.
L_Densed_UNet = AerialSegNet


def build_model(
    in_ch: int = 3,
    n_classes: int = 4,
    base_channels: int = 32,
    head_activation: str = "relu",
) -> AerialSegNet:
    """Factory used by ``train.py`` so the model can be built from a config."""
    return AerialSegNet(
        in_ch=in_ch,
        n_classes=n_classes,
        base_channels=base_channels,
        head_activation=head_activation,
    )
