"""Mask colouring and prediction dumps.

The colour table below is the one the published figures use. Note that
:func:`decode_segmap` writes the channels back in BGR order, which is what the
original code did, so ``road`` with a table entry of ``(0, 0, 128)`` renders as
dark red rather than dark blue. The table and the writer are kept consistent
with the published figures on purpose; if you want true RGB, set
``bgr_order=False`` and expect different-looking (but equivalent) masks.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

COLOR_DICT = {
    "obj0": [0, 0, 0],  # background
    "obj1": [0, 0, 128],  # road
    "obj2": [0, 128, 0],  # occluded road
    "obj3": [0, 128, 128],  # vegetation
}


def decode_segmap(
    image, color_dict: dict | None = None, num_classes: int = 4, bgr_order: bool = True
) -> np.ndarray:
    """Turn a class-index map into a colour image.

    Args:
        image: ``(H, W)`` array or tensor of class indices.
        color_dict: ``{"obj0": [r, g, b], ...}``; defaults to :data:`COLOR_DICT`.
        num_classes: number of classes in the table.
        bgr_order: write channels back in BGR order, reproducing the published
            figures.

    Returns:
        ``(H, W, 3)`` uint8 array.
    """
    color_dict = color_dict or COLOR_DICT
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    image = np.asarray(image)

    label_colours = np.array(
        [color_dict[f"obj{i}"] for i in range(num_classes)], dtype=np.uint8
    )

    r = np.zeros_like(image, dtype=np.uint8)
    g = np.zeros_like(image, dtype=np.uint8)
    b = np.zeros_like(image, dtype=np.uint8)
    for label in range(num_classes):
        mask = image == label
        r[mask] = label_colours[label, 0]
        g[mask] = label_colours[label, 1]
        b[mask] = label_colours[label, 2]

    rgb = np.zeros((*image.shape, 3), dtype=np.uint8)
    if bgr_order:
        rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2] = b, g, r
    else:
        rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2] = r, g, b
    return rgb


@torch.no_grad()
def predict_rgb(model, tensor: torch.Tensor, color_dict: dict | None = None):
    """Run the model on one ``(1, 3, H, W)`` tensor and colour the prediction."""
    model.eval()
    out = model(tensor.float()).squeeze(0)
    out = out.data.max(0)[1].cpu().numpy()
    return decode_segmap(out, color_dict)


@torch.no_grad()
def save_predictions(
    loader,
    model,
    device,
    folder: str = "predictions",
    color_dict: dict | None = None,
    limit: int | None = None,
) -> int:
    """Write ``image_i.png``, ``mask_i.png`` and ``pred_i.png`` for a loader.

    Args:
        loader: dataloader to run over.
        model: trained model.
        device: device to run on.
        folder: output directory, created if missing.
        color_dict: colour table.
        limit: stop after this many frames.

    Returns:
        Number of frames written.
    """
    import torchvision
    from torchvision.transforms import ToPILImage, ToTensor

    out_dir = Path(folder)
    out_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    to_tensor = ToTensor()
    written = 0

    for x_batch, y_batch in loader:
        for i in range(x_batch.size(0)):
            if limit is not None and written >= limit:
                return written

            image = to_tensor(ToPILImage()(x_batch[i]))
            torchvision.utils.save_image(image, out_dir / f"image_{written}.png")

            mask = to_tensor(decode_segmap(y_batch[i], color_dict))
            torchvision.utils.save_image(mask, out_dir / f"mask_{written}.png")

            pred = predict_rgb(
                model, x_batch[i].unsqueeze(0).to(device), color_dict
            )
            torchvision.utils.save_image(
                to_tensor(pred), out_dir / f"pred_{written}.png"
            )
            written += 1

    return written


def plot_grid(
    model,
    batch,
    device,
    n_rows: int = 4,
    color_dict: dict | None = None,
    save_path: str | None = None,
):
    """Plot an image / ground-truth / prediction grid.

    Args:
        model: trained model.
        batch: ``(x, y)`` tuple from a dataloader.
        device: device to run the model on.
        n_rows: number of samples to show.
        color_dict: colour table.
        save_path: if given, write the figure here instead of showing it.
    """
    import matplotlib

    if save_path:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from torchvision.transforms import ToPILImage

    x_batch, y_batch = batch
    n_rows = min(n_rows, x_batch.size(0))

    fig, axes = plt.subplots(nrows=n_rows, ncols=3, figsize=(12, 4 * n_rows))
    axes = np.atleast_2d(axes)
    plt.setp(axes.flat, xticks=[], yticks=[])

    for row in range(n_rows):
        axes[row, 0].imshow(ToPILImage()(x_batch[row]))
        axes[row, 0].set_title(f"Image_{row + 1}")

        axes[row, 1].imshow(decode_segmap(y_batch[row], color_dict))
        axes[row, 1].set_title(f"Ground_Truth_{row + 1}")

        axes[row, 2].imshow(
            predict_rgb(model, x_batch[row].unsqueeze(0).to(device), color_dict)
        )
        axes[row, 2].set_title(f"Predicted_Mask_{row + 1}")

    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
    return fig
