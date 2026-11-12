"""NITRDrone dataset: paired aerial frames and 4-class label masks.

Expected directory layout (the one the published runs used)::

    <data_path>/
        train_images/   RGB frames
        train_masks/    single-channel masks, pixel value == class index
        val_images/
        val_masks/
        test_images/    optional
        test_masks/     optional

Images and masks are paired by sorted filename, so the two directories must
list in the same order.
"""

from __future__ import annotations

from glob import glob
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def list_split(data_path: str | Path, split: str) -> tuple[list[str], list[str]]:
    """Return sorted image and mask paths for one split.

    Args:
        data_path: dataset root.
        split: ``train``, ``val`` or ``test``.

    Raises:
        FileNotFoundError: if either directory is missing.
        ValueError: if the two directories hold a different number of files.
    """
    root = Path(data_path)
    img_dir = root / f"{split}_images"
    mask_dir = root / f"{split}_masks"

    for d in (img_dir, mask_dir):
        if not d.is_dir():
            msg = (
                f"Missing directory {d}. Expected "
                f"<data_path>/{{train,val,test}}_{{images,masks}}."
            )
            raise FileNotFoundError(msg)

    images = sorted(p for p in glob(str(img_dir / "*")) if _is_image(p))
    masks = sorted(p for p in glob(str(mask_dir / "*")) if _is_image(p))

    if len(images) != len(masks):
        msg = (
            f"{split}: {len(images)} images but {len(masks)} masks. "
            "Images and masks are paired by sorted filename."
        )
        raise ValueError(msg)
    if not images:
        msg = f"{split}: no images found under {img_dir}."
        raise ValueError(msg)

    return images, masks


def _is_image(path: str) -> bool:
    return Path(path).suffix.lower() in IMAGE_SUFFIXES


class NITRDroneDataset(Dataset):
    """Paired aerial image and segmentation mask.

    Args:
        images: list of image paths.
        labels: list of mask paths, aligned with ``images``.
        transform: torchvision transform applied to the image and the mask.
            The same transform object is applied to both, so only geometric
            and resizing transforms are safe here.

    Returns per item:
        ``(image, mask)`` where image is ``(3, H, W)`` uint8-valued float and
        mask is ``(H, W)`` holding class indices.

    Note:
        Images are returned in the raw 0-255 range, exactly as the published
        runs trained them. There is no normalisation step; if you add one,
        retrain rather than loading the released checkpoint.
    """

    def __init__(
        self,
        images: list[str],
        labels: list[str],
        transform=None,
    ) -> None:
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = Image.open(self.images[index]).convert("RGB")
        image = torch.from_numpy(np.array(image)).permute(2, 0, 1).contiguous()

        mask = Image.open(self.labels[index]).convert("L")
        mask = torch.from_numpy(np.array(mask, dtype=np.float32)).unsqueeze(0)

        if self.transform is not None:
            image = self.transform(image)
            mask = self.transform(mask)

        return image.squeeze(), mask.squeeze()
