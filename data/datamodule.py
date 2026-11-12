"""Dataloader construction and class weighting."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.dataset import NITRDroneDataset, list_split
from data.transforms import build_eval_transforms, build_transforms


class SegDataModule:
    """Builds the train, val and test loaders from a data config.

    Args:
        data_path: dataset root holding ``{train,val,test}_{images,masks}``.
        img_size: square side the frames are resized and cropped to.
        batch_size: training batch size.
        val_batch_size: validation batch size.
        test_batch_size: test batch size.
        num_workers: dataloader workers for train and val.
        pin_memory: pin host memory for faster host-to-device copies.
        train_transforms: transform spec, see :func:`data.transforms.build_transforms`.
        load_test: build a test loader as well. Set False if the split is absent.
    """

    def __init__(
        self,
        data_path: str,
        img_size: int = 480,
        batch_size: int = 5,
        val_batch_size: int = 2,
        test_batch_size: int = 1,
        num_workers: int = 4,
        pin_memory: bool = False,
        train_transforms: list[dict] | None = None,
        load_test: bool = False,
    ) -> None:
        self.data_path = data_path
        self.img_size = img_size
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.train_transforms = train_transforms
        self.load_test = load_test

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self) -> None:
        """Resolve file lists and build the datasets."""
        train_imgs, train_masks = list_split(self.data_path, "train")
        val_imgs, val_masks = list_split(self.data_path, "val")

        self.train_dataset = NITRDroneDataset(
            train_imgs,
            train_masks,
            transform=build_transforms(self.img_size, self.train_transforms),
        )
        self.val_dataset = NITRDroneDataset(
            val_imgs,
            val_masks,
            transform=build_eval_transforms(self.img_size),
        )

        if self.load_test:
            test_imgs, test_masks = list_split(self.data_path, "test")
            self.test_dataset = NITRDroneDataset(
                test_imgs,
                test_masks,
                transform=build_eval_transforms(self.img_size),
            )

        print(
            f"train: {len(self.train_dataset)} | val: {len(self.val_dataset)}"
            + (f" | test: {len(self.test_dataset)}" if self.test_dataset else "")
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            msg = "Test split was not loaded. Pass load_test=True."
            raise RuntimeError(msg)
        return DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=1,
            pin_memory=self.pin_memory,
        )


def compute_class_weights(
    loader: DataLoader,
    num_classes: int,
    c: float = 1.02,
    full_pass: bool = False,
) -> np.ndarray:
    """Inverse-log class weighting, ``1 / log(c + p_class)``.

    Args:
        loader: dataloader to sample label frequencies from.
        num_classes: number of classes.
        c: smoothing constant. Smaller values weight rare classes harder.
        full_pass: if True, count every mask in the loader. If False, count
            only the first batch, which is what the published runs did and
            is what reproduces the reported weights
            ``[1.894, 18.976, 48.039, 3.712]``.

    Returns:
        Array of ``num_classes`` weights.
    """
    counts = np.zeros(num_classes, dtype=np.int64)

    if full_pass:
        for _, y in loader:
            counts += np.bincount(
                y.flatten().cpu().numpy().astype(np.int64), minlength=num_classes
            )[:num_classes]
    else:
        _, y = next(iter(loader))
        counts = np.bincount(
            y.flatten().cpu().numpy().astype(np.int64), minlength=num_classes
        )[:num_classes]

    total = counts.sum()
    if total == 0:
        msg = "No label pixels found while computing class weights."
        raise ValueError(msg)

    p_class = counts / total
    return 1.0 / np.log(c + p_class)


def class_weight_tensor(weights: np.ndarray, device: str | torch.device):
    """Convert weights to a float tensor on the training device."""
    return torch.FloatTensor(weights).to(device)
