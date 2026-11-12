"""Parameter count, activation memory and throughput.

Example::

    python benchmark.py --img_size 480 --batch_size 1 --runs 50
"""

from __future__ import annotations

import argparse
import json
import time

import torch

from models.aerial_segnet import build_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AerialSegNet benchmark")
    p.add_argument("--img_size", type=int, default=480)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--n_classes", type=int, default=4)
    p.add_argument("--base_channels", type=int, default=32)
    p.add_argument("--runs", type=int, default=50)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--device", default="auto")
    p.add_argument("--out", default=None, help="write results to this json file")
    return p.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_model(
        n_classes=args.n_classes, base_channels=args.base_channels
    ).to(device)
    model.eval()

    n_params = model.num_parameters()
    param_mb = n_params * 4 / (1024**2)

    x = torch.randn(
        args.batch_size, 3, args.img_size, args.img_size, device=device
    )

    for _ in range(args.warmup):
        model(x)
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()
    for _ in range(args.runs):
        model(x)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    per_batch = elapsed / args.runs
    results = {
        "device": device,
        "img_size": args.img_size,
        "batch_size": args.batch_size,
        "parameters": n_params,
        "parameters_mb_fp32": round(param_mb, 2),
        "latency_ms_per_batch": round(per_batch * 1000, 2),
        "throughput_fps": round(args.batch_size / per_batch, 2),
    }
    if device == "cuda":
        results["peak_memory_mb"] = round(
            torch.cuda.max_memory_allocated() / (1024**2), 1
        )

    print(json.dumps(results, indent=2))
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(results, fh, indent=2)


if __name__ == "__main__":
    main()
