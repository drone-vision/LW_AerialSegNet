"""Parity checks against the original notebook.

These guard the refactor: the modules must produce the same numbers the
notebook did. Run with ``python -m pytest tests -q`` or just
``python tests/test_parity.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.datamodule import compute_class_weights  # noqa: E402
from engine.metrics import (  # noqa: E402
    mean_accuracy,
    mean_f1,
    mean_iou,
    mean_precision,
    mean_recall,
)
from models.aerial_segnet import build_model  # noqa: E402

PUBLISHED_PARAMS = 2_486_562
PUBLISHED_WEIGHTS = [1.8937, 18.9764, 48.0387, 3.7117]


# --------------------------------------------------------------------------
# the notebook's own metric implementations, kept verbatim as the reference
# --------------------------------------------------------------------------
def _nb_mean(label, pred, num_classes, score):
    pred = F.softmax(pred, dim=1)
    pred = torch.argmax(pred, dim=1).reshape(-1)
    label = label.reshape(-1)
    present = []
    for c in range(num_classes):
        pi, ti = (pred == c), (label == c)
        if ti.long().sum().item() == 0:
            continue
        npi, nli = (pred != c), (label != c)
        tp = (pi & ti).long().sum().item()
        fn = (ti & npi).long().sum().item()
        fp = (pi & nli).long().sum().item()
        tn = (npi & nli).long().sum().item()
        present.append(score(tp, fp, fn, tn))
    return float(np.mean(present))


def test_model_parameter_count():
    model = build_model(in_ch=3, n_classes=4, base_channels=32)
    assert model.num_parameters() == PUBLISHED_PARAMS, (
        f"expected {PUBLISHED_PARAMS:,}, got {model.num_parameters():,}"
    )


def test_forward_shape_and_odd_sizes():
    model = build_model()
    for h, w in [(480, 480), (250, 370), (128, 128)]:
        with torch.no_grad():
            out = model(torch.randn(1, 3, h, w))
        assert out.shape == (1, 4, h, w), f"{(h, w)} -> {tuple(out.shape)}"


def test_metrics_match_notebook():
    eps = 1e-7
    cases = [
        (mean_iou, lambda tp, fp, fn, tn: tp / (tp + fp + fn) if tp + fp + fn else 0.0),
        (mean_f1, lambda tp, fp, fn, tn: (2 * tp) / ((2 * tp) + fp + fn)),
        (mean_precision, lambda tp, fp, fn, tn: (tp + eps) / (tp + fp + eps)),
        (mean_recall, lambda tp, fp, fn, tn: (tp + eps) / (tp + fn + eps)),
        (
            mean_accuracy,
            lambda tp, fp, fn, tn: (tp + tn + eps) / (tp + tn + fp + fn + eps),
        ),
    ]

    torch.manual_seed(0)
    for trial in range(5):
        logits = torch.randn(2, 4, 32, 32)
        # alternate between batches that contain all 4 classes and only 3,
        # so the skip-absent-classes branch is exercised
        n_present = 4 if trial % 2 else 3
        label = torch.randint(0, n_present, (2, 32, 32)).float()

        for new_fn, score in cases:
            ref = _nb_mean(label, logits, 4, score)
            got = float(new_fn(label, logits))
            assert abs(ref - got) < 1e-9, f"{new_fn.__name__}: {ref} vs {got}"


def test_class_weights_match_published():
    """1 / log(1.02 + p) on the published class distribution."""
    p = [0.6757, 0.03411, 0.001035, 0.28921]
    counts = (np.array(p) * 1_152_000).astype(int)
    flat = np.concatenate([np.full(c, i) for i, c in enumerate(counts)])
    flat = flat[: (len(flat) // 1152) * 1152].reshape(-1, 1152)
    loader = [(None, torch.from_numpy(flat).float())]

    weights = compute_class_weights(loader, 4, c=1.02, full_pass=False)
    np.testing.assert_allclose(weights, PUBLISHED_WEIGHTS, rtol=1e-3)


def test_radam_steps():
    """RAdam runs on a modern torch without the deprecated signatures."""
    from optim import RAdam

    model = build_model(base_channels=4)
    opt = RAdam(model.parameters(), lr=1e-4, weight_decay=2e-4)
    before = next(model.parameters()).detach().clone()

    for _ in range(3):
        opt.zero_grad()
        out = model(torch.randn(1, 3, 32, 32))
        out.mean().backward()
        opt.step()

    after = next(model.parameters()).detach()
    assert not torch.allclose(before, after), "RAdam did not update the parameters"


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print(f"PASS  {name}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL  {name}: {exc}")
    raise SystemExit(1 if failures else 0)
