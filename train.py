"""Entry point for training and evaluating AerialSegNet.

Examples::

    python train.py --config configs/exp/aerialsegnet_nitrdrone.yaml
    python train.py --config configs/exp/aerialsegnet_nitrdrone.yaml --train.epochs 10
    python train.py --config configs/exp/aerialsegnet_nitrdrone.yaml \
        --eval_only --ckpt_path checkpoints/AerialSegNet_best.pth
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn import CrossEntropyLoss

from configs.config_parser import get_parser, resolve_device
from data.datamodule import SegDataModule, class_weight_tensor, compute_class_weights
from engine.trainer import Trainer, load_checkpoint
from engine.visualise import save_predictions
from models.aerial_segnet import build_model
from optim import build_optimizer, build_scheduler


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_criterion(config, datamodule, device):
    """Build the loss, resolving class weights if needed."""
    if config.loss.name != "cross_entropy":
        msg = (
            f"Unknown loss '{config.loss.name}'. Only cross_entropy is "
            "implemented; add new losses in train.py."
        )
        raise ValueError(msg)

    weights = None
    if config.loss.class_weights is not None:
        weights = np.asarray(config.loss.class_weights, dtype=np.float32)
    elif config.loss.auto_weight:
        weights = compute_class_weights(
            datamodule.train_dataloader(),
            config.model.n_classes,
            c=config.loss.weight_c,
            full_pass=config.loss.weight_full_pass,
        )

    if weights is None:
        print("class weights: none")
        return CrossEntropyLoss()

    print("class weights:", np.round(weights, 4).tolist())
    return CrossEntropyLoss(weight=class_weight_tensor(weights, device))


def main() -> None:
    parser = get_parser()
    config = parser.parse_args()

    seed_everything(config.seed)
    device = resolve_device(config.device)
    print(f"device: {device}")

    res_path = Path(config.res_path) / config.config_tag
    res_path.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ data
    datamodule = SegDataModule(
        data_path=config.data.data_path,
        img_size=config.data.img_size,
        batch_size=config.data.batch_size,
        val_batch_size=config.data.val_batch_size,
        test_batch_size=config.data.test_batch_size,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        train_transforms=config.data.train_transforms,
        load_test=config.data.load_test,
    )
    datamodule.setup()

    # ----------------------------------------------------------------- model
    model = build_model(
        in_ch=config.model.in_ch,
        n_classes=config.model.n_classes,
        base_channels=config.model.base_channels,
        head_activation=config.model.head_activation,
    ).to(device)
    print(f"parameters: {model.num_parameters():,}")

    criterion = build_criterion(config, datamodule, device)
    optimizer = build_optimizer(
        model.parameters(),
        name=config.train.optimizer,
        lr=config.train.lr,
        weight_decay=config.train.weight_decay,
    )
    scheduler = build_scheduler(optimizer, vars(config.train.scheduler))

    ckpt_dir = Path(config.ckpt_path)
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        scheduler=scheduler,
        num_classes=config.model.n_classes,
        ckpt_dir=str(ckpt_dir.parent if ckpt_dir.suffix else ckpt_dir),
        ckpt_prefix=config.config_tag,
        scheduler_every_n_epochs=config.train.scheduler_every_n_epochs,
        legacy_step_sampling=config.train.legacy_step_sampling,
        steps_per_epoch=config.train.steps_per_epoch,
    )

    # ------------------------------------------------------------------- run
    if config.eval_only:
        if not ckpt_dir.is_file():
            msg = (
                f"--eval_only needs a checkpoint file, got '{ckpt_dir}'. "
                "Pass the .pth path with --ckpt_path."
            )
            raise FileNotFoundError(msg)
        model, epoch, val_loss = load_checkpoint(ckpt_dir, model)
        model.to(device)
        print(f"loaded {ckpt_dir} (epoch {epoch}, val_loss {val_loss})")
    else:
        epochs = 1 if config.dev else config.train.epochs
        history = trainer.fit(
            datamodule.train_dataloader(),
            datamodule.val_dataloader(),
            epochs=epochs,
        )
        history.to_json(res_path / "history.json")
        history.to_csv(res_path / "history.csv")
        print(f"history written to {res_path}")

    # ------------------------------------------------------------------ eval
    eval_loader = (
        datamodule.test_dataloader()
        if config.data.load_test
        else datamodule.val_dataloader()
    )
    split = "test" if config.data.load_test else "val"
    stats, per_class = trainer.evaluate(eval_loader, split_name=split)

    (res_path / "metrics.json").write_text(
        json.dumps({"split": split, "mean": stats, "per_class": per_class}, indent=2)
    )

    if config.save_predictions:
        n = save_predictions(
            eval_loader, model, device, folder=str(res_path / "predictions")
        )
        print(f"wrote {n} prediction triplets to {res_path / 'predictions'}")


if __name__ == "__main__":
    main()
