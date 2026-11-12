"""Run a trained AerialSegNet over a folder of images and write colour masks.

Example::

    python predict.py --ckpt checkpoints/aerialsegnet_best.pth \
        --input path/to/frames --output predictions --img_size 480
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from data.dataset import IMAGE_SUFFIXES
from data.transforms import build_eval_transforms
from engine.visualise import decode_segmap
from models.aerial_segnet import build_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AerialSegNet inference")
    p.add_argument("--ckpt", required=True, help="path to a .pth checkpoint")
    p.add_argument("--input", required=True, help="image file or directory")
    p.add_argument("--output", default="predictions", help="output directory")
    p.add_argument("--img_size", type=int, default=480)
    p.add_argument("--n_classes", type=int, default=4)
    p.add_argument("--base_channels", type=int, default=32)
    p.add_argument("--head_activation", default="relu")
    p.add_argument("--device", default="auto", help="cuda, cpu or auto")
    p.add_argument(
        "--save_index",
        action="store_true",
        help="also write the raw class-index map as a single-channel PNG",
    )
    return p.parse_args()


def collect_inputs(path: str) -> list[Path]:
    p = Path(path)
    if p.is_file():
        return [p]
    files = sorted(f for f in p.iterdir() if f.suffix.lower() in IMAGE_SUFFIXES)
    if not files:
        msg = f"No images found in {p}."
        raise ValueError(msg)
    return files


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_model(
        n_classes=args.n_classes,
        base_channels=args.base_channels,
        head_activation=args.head_activation,
    )
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"] if "state_dict" in ckpt else ckpt)
    model.to(device).eval()

    transform = build_eval_transforms(args.img_size)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = collect_inputs(args.input)
    for f in files:
        image = Image.open(f).convert("RGB")
        tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1)
        tensor = transform(tensor).unsqueeze(0).float().to(device)

        logits = model(tensor)
        index_map = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        Image.fromarray(decode_segmap(index_map)).save(out_dir / f"{f.stem}_pred.png")
        if args.save_index:
            Image.fromarray(index_map).save(out_dir / f"{f.stem}_index.png")

    print(f"wrote {len(files)} predictions to {out_dir}")


if __name__ == "__main__":
    main()
