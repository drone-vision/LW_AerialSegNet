"""Training and evaluation loop.

The loop keeps the structure of the original notebook (per-epoch train pass,
per-epoch validation pass, best-IoU checkpointing, scheduler stepped every N
epochs) but fixes three things that were accidents of notebook code rather
than design choices. Each one is behind a flag so the published run can still
be reproduced exactly:

``legacy_step_sampling``
    The notebook drew each training batch with ``next(iter(train_loader))``
    inside the step loop, which rebuilds the iterator every step. With
    ``shuffle=True`` that samples batches with replacement and respawns the
    worker processes once per step. Default here is a normal epoch pass; set
    the flag to reproduce the published behaviour.

``scheduler_every_n_epochs``
    The notebook's guard was ``if epoch % visualize_every * 4 == 0``, which
    Python parses as ``(epoch % visualize_every) * 4``, so the scheduler
    actually stepped every 3rd epoch. That interval is now explicit.

per-class accuracy
    The notebook's per-class printer looped over the recall dict when printing
    accuracy, so the two columns were identical. :class:`PerClassScore`
    computes them separately.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from tqdm import tqdm

from engine.metrics import (
    PerClassScore,
    mean_accuracy,
    mean_f1,
    mean_iou,
    mean_precision,
    mean_recall,
)

METRIC_KEYS = ("loss", "iou", "f1", "prec", "rec", "acc")


class History(dict):
    """Per-epoch metric history, ``{"train_iou": [...], "val_iou": [...]}``."""

    def __init__(self) -> None:
        super().__init__()
        for split in ("train", "val"):
            for key in METRIC_KEYS:
                self[f"{split}_{key}"] = []
        self["lr"] = []

    def append(self, split: str, values: dict[str, float]) -> None:
        for key in METRIC_KEYS:
            self[f"{split}_{key}"].append(values[key])

    def to_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self, indent=2))

    def to_csv(self, path: str | Path) -> None:
        import csv

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        cols = [k for k in self if self[k]]
        rows = zip(*[self[c] for c in cols])
        with path.open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["epoch", *cols])
            for i, row in enumerate(rows, start=1):
                writer.writerow([i, *row])


class Trainer:
    """Drives training and evaluation.

    Args:
        model: the network.
        criterion: loss, typically weighted ``CrossEntropyLoss``.
        optimizer: optimiser.
        device: ``cuda`` or ``cpu``.
        scheduler: optional LR scheduler, stepped on validation loss.
        num_classes: number of segmentation classes.
        ckpt_dir: directory for checkpoints; ``None`` disables saving.
        ckpt_prefix: filename prefix for checkpoints.
        scheduler_every_n_epochs: how often to step the scheduler.
        legacy_step_sampling: reproduce the notebook's batch sampling.
        steps_per_epoch: number of steps when ``legacy_step_sampling`` is on.
    """

    def __init__(
        self,
        model,
        criterion,
        optimizer,
        device,
        scheduler=None,
        num_classes: int = 4,
        ckpt_dir: str | None = "checkpoints",
        ckpt_prefix: str = "AerialSegNet",
        scheduler_every_n_epochs: int = 3,
        legacy_step_sampling: bool = False,
        steps_per_epoch: int | None = None,
    ) -> None:
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.num_classes = num_classes
        self.ckpt_dir = Path(ckpt_dir) if ckpt_dir else None
        self.ckpt_prefix = ckpt_prefix
        self.scheduler_every_n_epochs = scheduler_every_n_epochs
        self.legacy_step_sampling = legacy_step_sampling
        self.steps_per_epoch = steps_per_epoch

        self.history = History()
        self.best_val_iou = 0.0
        self.best_epoch = 0

        if self.ckpt_dir:
            self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- passes

    def _batches(self, loader, train: bool):
        """Yield batches, honouring ``legacy_step_sampling`` during training."""
        if train and self.legacy_step_sampling:
            steps = self.steps_per_epoch or len(loader)
            for _ in range(steps):
                yield next(iter(loader))
        else:
            yield from loader

    def _run_epoch(self, loader, train: bool, per_class: PerClassScore | None = None):
        self.model.train() if train else self.model.eval()
        totals = dict.fromkeys(METRIC_KEYS, 0.0)
        n_steps = 0

        if train and self.legacy_step_sampling:
            total = self.steps_per_epoch or len(loader)
        else:
            total = len(loader)

        desc = "train" if train else "val"
        with torch.set_grad_enabled(train):
            for x, y in tqdm(self._batches(loader, train), total=total, desc=desc):
                x = x.to(self.device)
                y = y.to(self.device)
                if y.dim() == 4:
                    y = y.squeeze(1)

                if train:
                    self.optimizer.zero_grad()

                out = self.model(x.float())
                loss = self.criterion(out, y.long())

                if train:
                    loss.backward()
                    self.optimizer.step()

                totals["loss"] += loss.item()
                totals["iou"] += mean_iou(y, out, self.num_classes)
                totals["f1"] += mean_f1(y, out, num_classes=self.num_classes)
                totals["prec"] += mean_precision(y, out, num_classes=self.num_classes)
                totals["rec"] += mean_recall(y, out, num_classes=self.num_classes)
                totals["acc"] += mean_accuracy(y, out, num_classes=self.num_classes)
                n_steps += 1

                if per_class is not None:
                    per_class.update(y, out)

        n_steps = max(n_steps, 1)
        return {k: v / n_steps for k, v in totals.items()}

    # ------------------------------------------------------------------ fit

    def fit(self, train_loader, val_loader, epochs: int, start_epoch: int = 1):
        """Train for ``epochs`` epochs, validating after each one.

        Returns:
            The :class:`History` of per-epoch metrics.
        """
        start = time.time()

        for epoch in range(start_epoch, start_epoch + epochs):
            print(f"\nEpoch {epoch}")

            train_stats = self._run_epoch(train_loader, train=True)
            self.history.append("train", train_stats)
            print("  train  " + _fmt(train_stats))

            val_pc = PerClassScore(self.num_classes)
            val_stats = self._run_epoch(val_loader, train=False, per_class=val_pc)
            self.history.append("val", val_stats)
            print("  val    " + _fmt(val_stats))

            if val_stats["iou"] > self.best_val_iou:
                self.best_val_iou = val_stats["iou"]
                self.best_epoch = epoch
                self._save_checkpoint(epoch, train_stats, val_stats)

            lr = self.optimizer.param_groups[0]["lr"]
            self.history["lr"].append(lr)

            if (
                self.scheduler is not None
                and epoch % self.scheduler_every_n_epochs == 0
            ):
                if isinstance(
                    self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                ):
                    self.scheduler.step(val_stats["loss"])
                else:
                    self.scheduler.step()
                new_lr = self.optimizer.param_groups[0]["lr"]
                if new_lr != lr:
                    print(f"  lr     {lr:.3g} -> {new_lr:.3g}")

        elapsed = time.time() - start
        print(
            f"\nTraining done in {elapsed:.1f}s "
            f"({elapsed / 60:.1f} min). "
            f"Best val mIoU {self.best_val_iou:.4f} at epoch {self.best_epoch}."
        )
        print("\nFinal-epoch per-class scores (validation):")
        print(val_pc.as_table())

        return self.history

    # ----------------------------------------------------------------- eval

    @torch.no_grad()
    def evaluate(self, loader, split_name: str = "test"):
        """Run one pass and return ``(mean_metrics, per_class_dict)``."""
        per_class = PerClassScore(self.num_classes)
        stats = self._run_epoch(loader, train=False, per_class=per_class)
        print(f"\n{split_name}: " + _fmt(stats))
        print(per_class.as_table())
        return stats, per_class.compute()

    # ---------------------------------------------------------- checkpoints

    def _save_checkpoint(self, epoch: int, train_stats: dict, val_stats: dict) -> None:
        if self.ckpt_dir is None:
            return
        payload = {
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "train": train_stats,
            "val": val_stats,
            "val_loss": val_stats["loss"],
            "val_iou": val_stats["iou"],
        }
        path = self.ckpt_dir / f"{self.ckpt_prefix}_best.pth"
        torch.save(payload, path)
        print(f"  saved  {path} (val mIoU {val_stats['iou']:.4f})")


def _fmt(stats: dict) -> str:
    return "  ".join(f"{k}={v:.4f}" for k, v in stats.items())


def load_checkpoint(path: str | Path, model, optimizer=None):
    """Restore a checkpoint written by :class:`Trainer`.

    Also accepts the notebook's checkpoint format, which stored ``state_dict``,
    ``epoch`` and ``val_loss`` at the top level.

    Returns:
        ``(model, start_epoch, val_loss)``.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return model, ckpt.get("epoch", 0), ckpt.get("val_loss")
