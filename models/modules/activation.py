"""Activation functions used by AerialSegNet."""

import torch
import torch.nn.functional as F
from torch import nn


class Mish(nn.Module):
    """Mish activation: x * tanh(softplus(x)).

    Reference: Misra, "Mish: A Self Regularized Non-Monotonic Activation
    Function", BMVC 2020.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.tanh(F.softplus(x))


class Activation(nn.Module):
    """Thin dispatcher so the activation can be selected from a config file.

    Args:
        name: one of ``relu``, ``prelu``, ``mish`` or ``none``.
    """

    SUPPORTED = ("relu", "prelu", "mish", "none")

    def __init__(self, name: str = "relu") -> None:
        super().__init__()
        name = name.lower()
        if name not in self.SUPPORTED:
            msg = f"Unknown activation '{name}'. Expected one of {self.SUPPORTED}."
            raise ValueError(msg)

        self.name = name
        if name == "relu":
            self.act = nn.ReLU()
        elif name == "prelu":
            self.act = nn.PReLU()
        elif name == "mish":
            self.act = Mish()
        else:
            self.act = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x)
