"""Segmentation metrics.

Two layers:

``mean_iou`` / ``mean_f1`` / ``mean_precision`` / ``mean_recall`` /
``mean_accuracy`` reproduce the notebook's batch-level scores exactly. They
average over the classes **present in the label batch only**, skipping absent
classes rather than scoring them zero. That is why a batch containing just
background and vegetation can report a mean IoU far above the dataset-wide
figure, and it is the reason the headline validation mIoU is 0.61 while the
per-class table shows two classes above 0.97.

:class:`PerClassScore` accumulates the same per-class quantities across a run
so you can read the dataset-wide numbers instead.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

CLASS_NAMES = ("background", "road", "occluded_road", "vegetation")


def _to_flat_pred(pred: torch.Tensor) -> torch.Tensor:
    """Softmax -> argmax -> flatten."""
    pred = F.softmax(pred, dim=1)
    pred = torch.argmax(pred, dim=1)
    return pred.reshape(-1)


def _confusion_parts(
    pred_flat: torch.Tensor, label_flat: torch.Tensor, sem_class: int
) -> tuple[int, int, int, int]:
    """Return (tp, fp, fn, tn) for one class."""
    pred_inds = pred_flat == sem_class
    target_inds = label_flat == sem_class

    tp = (pred_inds & target_inds).long().sum().item()
    fp = (pred_inds & ~target_inds).long().sum().item()
    fn = (target_inds & ~pred_inds).long().sum().item()
    tn = (~pred_inds & ~target_inds).long().sum().item()
    return tp, fp, fn, tn


def _mean_over_present(
    label: torch.Tensor,
    pred: torch.Tensor,
    num_classes: int,
    fn_score,
) -> float:
    """Average ``fn_score`` over the classes present in ``label``."""
    pred_flat = _to_flat_pred(pred)
    label_flat = label.reshape(-1)

    scores = []
    for sem_class in range(num_classes):
        if (label_flat == sem_class).long().sum().item() == 0:
            continue
        tp, fp, fn, tn = _confusion_parts(pred_flat, label_flat, sem_class)
        scores.append(fn_score(tp, fp, fn, tn))

    if not scores:
        return float("nan")
    return float(np.mean(scores))


def mean_iou(label, pred, num_classes: int = 4) -> float:
    """Mean intersection over union over the classes present in ``label``."""
    return _mean_over_present(
        label,
        pred,
        num_classes,
        lambda tp, fp, fn, tn: tp / (tp + fp + fn) if (tp + fp + fn) else 0.0,
    )


def mean_f1(label, pred, eps: float = 1e-7, num_classes: int = 4) -> float:
    """Mean Dice / F1 over the classes present in ``label``."""
    return _mean_over_present(
        label,
        pred,
        num_classes,
        lambda tp, fp, fn, tn: (2 * tp) / ((2 * tp) + fp + fn + eps),
    )


def mean_precision(label, pred, eps: float = 1e-7, num_classes: int = 4) -> float:
    """Mean precision over the classes present in ``label``."""
    return _mean_over_present(
        label, pred, num_classes, lambda tp, fp, fn, tn: (tp + eps) / (tp + fp + eps)
    )


def mean_recall(label, pred, eps: float = 1e-7, num_classes: int = 4) -> float:
    """Mean recall over the classes present in ``label``."""
    return _mean_over_present(
        label, pred, num_classes, lambda tp, fp, fn, tn: (tp + eps) / (tp + fn + eps)
    )


def mean_accuracy(label, pred, eps: float = 1e-7, num_classes: int = 4) -> float:
    """Mean one-vs-rest pixel accuracy over the classes present in ``label``.

    Note:
        Dominated by true negatives on a 4-class problem, so it sits near
        0.99 regardless of how the rare classes do. Quote IoU instead.
    """
    return _mean_over_present(
        label,
        pred,
        num_classes,
        lambda tp, fp, fn, tn: (tp + tn + eps) / (tp + tn + fp + fn + eps),
    )


class PerClassScore:
    """Accumulates per-class confusion counts across batches.

    Unlike the batch-level helpers above, this aggregates true/false
    positives and negatives first and computes the metrics once at the end,
    so the result is a dataset-wide figure rather than a mean of per-batch
    means.

    Args:
        num_classes: number of classes.
        class_names: display names, defaults to :data:`CLASS_NAMES`.
        eps: smoothing added to each ratio.
    """

    def __init__(
        self,
        num_classes: int = 4,
        class_names: tuple[str, ...] | None = None,
        eps: float = 1e-7,
    ) -> None:
        self.num_classes = num_classes
        self.class_names = class_names or CLASS_NAMES[:num_classes]
        self.eps = eps
        self.reset()

    def reset(self) -> None:
        self.tp = np.zeros(self.num_classes, dtype=np.int64)
        self.fp = np.zeros(self.num_classes, dtype=np.int64)
        self.fn = np.zeros(self.num_classes, dtype=np.int64)
        self.tn = np.zeros(self.num_classes, dtype=np.int64)
        self.seen = np.zeros(self.num_classes, dtype=np.int64)

    @torch.no_grad()
    def update(self, label: torch.Tensor, pred: torch.Tensor) -> None:
        """Add one batch.

        Args:
            label: ``(N, H, W)`` class indices.
            pred: ``(N, C, H, W)`` logits.
        """
        pred_flat = _to_flat_pred(pred)
        label_flat = label.reshape(-1)

        for sem_class in range(self.num_classes):
            present = (label_flat == sem_class).long().sum().item()
            tp, fp, fn, tn = _confusion_parts(pred_flat, label_flat, sem_class)
            self.tp[sem_class] += tp
            self.fp[sem_class] += fp
            self.fn[sem_class] += fn
            self.tn[sem_class] += tn
            self.seen[sem_class] += present

    def compute(self) -> dict[str, dict[str, float]]:
        """Return ``{class_name: {iou, dice, precision, recall, accuracy, pixels}}``.

        A class that never appears in any label gets ``nan`` rather than zero.
        """
        eps = self.eps
        out: dict[str, dict[str, float]] = {}
        for i, name in enumerate(self.class_names):
            tp, fp, fn, tn = (
                int(self.tp[i]),
                int(self.fp[i]),
                int(self.fn[i]),
                int(self.tn[i]),
            )
            if self.seen[i] == 0:
                out[name] = dict.fromkeys(
                    ("iou", "dice", "precision", "recall", "accuracy"), float("nan")
                )
                out[name]["pixels"] = 0
                continue
            out[name] = {
                "iou": (tp + eps) / (tp + fp + fn + eps),
                "dice": (2 * tp + eps) / ((2 * tp) + fp + fn + eps),
                "precision": (tp + eps) / (tp + fp + eps),
                "recall": (tp + eps) / (tp + fn + eps),
                "accuracy": (tp + tn + eps) / (tp + tn + fp + fn + eps),
                "pixels": int(self.seen[i]),
            }
        return out

    def mean_iou(self) -> float:
        """Dataset-wide mIoU over the classes that appeared at least once."""
        vals = [
            v["iou"] for v in self.compute().values() if not np.isnan(v["iou"])
        ]
        return float(np.mean(vals)) if vals else float("nan")

    def as_table(self) -> str:
        """Pretty-print the per-class table."""
        rows = self.compute()
        head = (
            f"{'class':<16}{'IoU':>9}{'Dice':>9}{'Prec':>9}"
            f"{'Recall':>9}{'Acc':>9}{'px':>14}"
        )
        lines = [head, "-" * len(head)]
        for name, m in rows.items():
            lines.append(
                f"{name:<16}{m['iou']:>9.4f}{m['dice']:>9.4f}"
                f"{m['precision']:>9.4f}{m['recall']:>9.4f}"
                f"{m['accuracy']:>9.4f}{m['pixels']:>14,}"
            )
        lines.append(f"{'mIoU':<16}{self.mean_iou():>9.4f}")
        return "\n".join(lines)


# Aliases matching the notebook's function names.
mIOU = mean_iou
mF_Score = mean_f1
mPrecision = mean_precision
mRecall = mean_recall
mAccuracy = mean_accuracy
