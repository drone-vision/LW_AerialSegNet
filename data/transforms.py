"""Transform construction.

Both the image and the mask go through the same transform object, so only
geometric and resizing operations belong here. Anything that changes pixel
intensity (normalisation, colour jitter, blur) would corrupt the mask and must
be applied to the image alone.
"""

from __future__ import annotations

from torchvision import transforms

# Geometric transforms that are safe to apply to a mask as well as an image.
_GEOMETRIC = {
    "Resize",
    "RandomCrop",
    "CenterCrop",
    "RandomHorizontalFlip",
    "RandomVerticalFlip",
    "RandomRotation",
    "RandomAffine",
    "RandomPerspective",
    "Pad",
}


def build_transforms(size: int, spec: list[dict] | None = None):
    """Build a transform pipeline from a config list.

    Args:
        size: target square side; used as the default argument for transforms
            that take a size (``Resize``, ``RandomCrop``, ``CenterCrop``).
        spec: list of single-key dicts, e.g.::

            - Resize: {}
            - RandomCrop: {}
            - RandomHorizontalFlip: {p: 0.5}

            An empty dict means "use the default arguments for ``size``".
            ``None`` reproduces the published pipeline.

    Returns:
        A ``torchvision.transforms.Compose``.

    Raises:
        ValueError: on an unknown or intensity-changing transform name.
    """
    if spec is None:
        # the published pipeline
        spec = [{"Resize": {}}, {"RandomCrop": {}}, {"CenterCrop": {}}]

    defaults = {
        "Resize": {"size": (size, size)},
        "RandomCrop": {"size": size},
        "CenterCrop": {"size": size},
    }

    ops = []
    for entry in spec:
        if len(entry) != 1:
            msg = f"Each transform entry needs exactly one key, got {entry}."
            raise ValueError(msg)
        (name, kwargs), = entry.items()
        kwargs = dict(kwargs or {})

        if name not in _GEOMETRIC:
            msg = (
                f"'{name}' is not a geometric transform. The same pipeline is "
                "applied to the mask, so intensity transforms are rejected. "
                f"Allowed: {sorted(_GEOMETRIC)}."
            )
            raise ValueError(msg)

        if not hasattr(transforms, name):
            msg = f"torchvision.transforms has no '{name}'."
            raise ValueError(msg)

        merged = {**defaults.get(name, {}), **kwargs}
        ops.append(getattr(transforms, name)(**merged))

    return transforms.Compose(ops)


def build_eval_transforms(size: int):
    """Deterministic pipeline used for validation and test."""
    return transforms.Compose(
        [transforms.Resize((size, size)), transforms.CenterCrop(size)]
    )
