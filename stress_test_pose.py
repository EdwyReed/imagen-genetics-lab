"""Simple stress-test to exercise pose analytics in a tight loop.

The script produces synthetic images, runs them through the lightweight
pose/segmentation stack and ensures that lazy loading and device clean-up keep
memory stable over time.  It is intended to be lightweight so it can be used as
an integration smoke-test on machines with modest GPUs (such as a GTX 1060).
"""

from __future__ import annotations

import argparse
import time
from typing import Iterable

import numpy as np
import torch
from PIL import Image

from imagen_lab.pose import PoseAnalyzer


def _synthetic_frames(seed: int, count: int, size: int) -> Iterable[Image.Image]:
    rng = np.random.default_rng(seed)
    for _ in range(count):
        base = rng.random((size, size, 3), dtype=np.float32)
        gradients = np.linspace(0.0, 1.0, size, dtype=np.float32)
        grad_x = gradients.reshape(1, -1, 1)
        grad_y = gradients.reshape(-1, 1, 1)
        img = 0.6 * base + 0.25 * grad_x + 0.15 * grad_y
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        yield Image.fromarray(img, mode="RGB")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stress-test pose analytics for OOM issues")
    parser.add_argument("--iterations", type=int, default=25, help="Number of synthetic samples to process")
    parser.add_argument("--device", default="auto", help="Torch device to use (auto|cpu|cuda)")
    parser.add_argument("--precision", default="auto", choices=["auto", "fp16", "fp32"], help="Math precision")
    parser.add_argument("--resolution", type=int, default=384, help="Resolution of generated samples")
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducible synthetic frames")
    parser.add_argument("--verbose", action="store_true", help="Print metrics for each iteration")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analyzer = PoseAnalyzer(device=args.device, precision=args.precision, default_resolution=args.resolution)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    start_alloc = torch.cuda.memory_allocated(analyzer.device) if analyzer.device.type == "cuda" else 0
    start_time = time.perf_counter()

    for idx, frame in enumerate(_synthetic_frames(args.seed, args.iterations, args.resolution)):
        metrics = analyzer.analyze(frame)
        if args.verbose:
            payload = metrics.to_payload()
            print(f"[{idx:03d}] pose={payload['pose_class']} gaze={payload['gaze_direction']} "
                  f"coverage={payload['coverage']:.3f} skin={payload['skin_ratio']:.3f}")

    duration = time.perf_counter() - start_time
    end_alloc = torch.cuda.memory_allocated(analyzer.device) if analyzer.device.type == "cuda" else 0
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(
        "Stress test completed",
        f"iterations={args.iterations}",
        f"device={analyzer.device}",
        f"precision={analyzer.dtype}",
        f"time={duration:.2f}s",
        sep="\n",
    )
    if analyzer.device.type == "cuda":
        delta = end_alloc - start_alloc
        print(f"GPU memory delta: {delta / (1024 ** 2):.2f} MiB")


if __name__ == "__main__":
    main()
