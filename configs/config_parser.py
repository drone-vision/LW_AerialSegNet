"""Config parsing.

Experiments are described by a YAML file under ``configs/exp``. Any field can
be overridden on the command line::

    python train.py --config configs/exp/aerialsegnet_nitrdrone.yaml \
        --train.epochs 10 --data.batch_size 2

Run ``python train.py -h`` for the full argument list.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from jsonargparse import ActionConfigFile, ArgumentParser


@dataclass
class ModelArgs:
    """Network geometry."""

    in_ch: int = 3
    n_classes: int = 4
    base_channels: int = 32
    # "relu" reproduces the published runs; "none" is the conventional choice
    head_activation: str = "relu"


@dataclass
class DataArgs:
    """Dataset location and loading."""

    data_path: str = "./datasets/NITRDrone"
    img_size: int = 480
    batch_size: int = 5
    val_batch_size: int = 2
    test_batch_size: int = 1
    num_workers: int = 4
    pin_memory: bool = False
    load_test: bool = False
    # geometric transforms only, see data/transforms.py
    train_transforms: list[dict] | None = None


@dataclass
class LossArgs:
    """Loss and class weighting."""

    name: str = "cross_entropy"
    class_weights: list[float] | None = None
    # when class_weights is null, derive them with 1 / log(c + p_class)
    auto_weight: bool = True
    weight_c: float = 1.02
    # False counts only the first batch, matching the published weights
    weight_full_pass: bool = False


@dataclass
class SchedulerArgs:
    """LR scheduler. ``type``: plateau, step, cosine or none."""

    type: str = "plateau"
    factor: float = 0.2
    patience: int = 2
    cooldown: int = 2


@dataclass
class TrainArgs:
    """Optimisation and loop control."""

    epochs: int = 150
    optimizer: str = "radam"
    lr: float = 1.0e-5
    weight_decay: float = 2.0e-4
    scheduler: SchedulerArgs = field(default_factory=SchedulerArgs)
    scheduler_every_n_epochs: int = 3
    # reproduce the notebook's next(iter(loader)) sampling
    legacy_step_sampling: bool = False
    steps_per_epoch: int | None = None


def get_parser() -> ArgumentParser:
    """Build the argument parser used by ``train.py`` and ``predict.py``."""
    parser = ArgumentParser(
        description="AerialSegNet: UAV aerial image semantic segmentation"
    )
    parser.add_argument(
        "-c",
        "--config",
        action=ActionConfigFile,
        help="Path to a configuration file in yaml or json format.",
    )

    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument(
        "--device", type=str, default="auto", help="cuda, cpu or auto"
    )
    parser.add_argument(
        "--config_tag", type=str, default="aerialsegnet", help="run name"
    )
    parser.add_argument(
        "--res_path", type=str, default="./results", help="where results are written"
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="./checkpoints",
        help="checkpoint directory, or the file to load with --eval_only",
    )
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="skip training; requires a checkpoint file in --ckpt_path",
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        help="fast dev run: one epoch over a handful of batches",
    )
    parser.add_argument(
        "--save_predictions",
        action="store_true",
        help="write image/mask/prediction triplets after evaluation",
    )

    parser.add_argument("--model", type=ModelArgs, default=ModelArgs())
    parser.add_argument("--data", type=DataArgs, default=DataArgs())
    parser.add_argument("--loss", type=LossArgs, default=LossArgs())
    parser.add_argument("--train", type=TrainArgs, default=TrainArgs())

    return parser


def resolve_device(name: str) -> str:
    """Turn ``auto`` into ``cuda`` or ``cpu``."""
    import torch

    if name != "auto":
        return name
    return "cuda" if torch.cuda.is_available() else "cpu"
