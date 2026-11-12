"""Optimisers and schedulers."""

from __future__ import annotations

import torch
from torch.optim import lr_scheduler

from optim.radam import RAdam

__all__ = ["RAdam", "build_optimizer", "build_scheduler"]


def build_optimizer(
    params,
    name: str = "radam",
    lr: float = 1e-5,
    weight_decay: float = 2e-4,
    betas: tuple[float, float] = (0.9, 0.999),
):
    """Build an optimiser by name.

    Args:
        params: parameters to optimise.
        name: ``radam``, ``adam``, ``adamw`` or ``sgd``.
        lr: learning rate.
        weight_decay: weight decay.
        betas: Adam betas, ignored by SGD.
    """
    name = name.lower()
    if name == "radam":
        return RAdam(params, lr=lr, weight_decay=weight_decay, betas=betas)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay, betas=betas)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, betas=betas)
    if name == "sgd":
        return torch.optim.SGD(
            params, lr=lr, weight_decay=weight_decay, momentum=0.9
        )
    msg = f"Unknown optimizer '{name}'. Expected radam, adam, adamw or sgd."
    raise ValueError(msg)


def build_scheduler(optimizer, cfg: dict | None):
    """Build a scheduler from a ``{type: ..., ...}`` dict.

    Supported types: ``plateau`` (ReduceLROnPlateau), ``step`` (StepLR),
    ``cosine`` (CosineAnnealingLR) and ``none``.
    """
    if not cfg or cfg.get("type", "none") == "none":
        return None

    kind = cfg["type"].lower()
    kwargs = {k: v for k, v in cfg.items() if k != "type"}

    if kind == "plateau":
        return lr_scheduler.ReduceLROnPlateau(optimizer, **kwargs)
    if kind == "step":
        return lr_scheduler.StepLR(optimizer, **kwargs)
    if kind == "cosine":
        return lr_scheduler.CosineAnnealingLR(optimizer, **kwargs)

    msg = f"Unknown scheduler '{kind}'. Expected plateau, step, cosine or none."
    raise ValueError(msg)
