"""Benchmark inference speed and memory footprint for pose/segmentation models.

This script creates lightweight stand-ins for common architectures so we can
run deterministic benchmarks without relying on external weights. The goal is
to capture relative compute/memory characteristics for GTX 1060 integration.
"""
import argparse
import json
import time
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List

try:
    import torch
    from torch import nn
except ImportError as exc:
    raise SystemExit(
        "PyTorch is required to run the pose benchmark. Install a CUDA-enabled build before execution."
    ) from exc


@dataclass
class ModelSpec:
    name: str
    factory: Callable[[int], nn.Module]
    default_resolution: int
    channels: int = 3
    description: str = ""


def _depthwise_conv(in_channels: int, out_channels: int, kernel_size: int = 3) -> nn.Sequential:
    padding = kernel_size // 2
    return nn.Sequential(
        nn.Conv2d(in_channels, in_channels, kernel_size, padding=padding, groups=in_channels, bias=False),
        nn.BatchNorm2d(in_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels, out_channels, 1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class OpenPoseLite(nn.Module):
    """Simplified backbone approximating lightweight OpenPose stacks."""

    def __init__(self, channels: int = 3):
        super().__init__()
        layers: List[nn.Module] = []
        in_ch = channels
        for out_ch, stride in [(32, 2), (64, 1), (128, 1), (128, 1), (256, 1)]:
            layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.ReLU(inplace=True))
            in_ch = out_ch
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Conv2d(in_ch, 128, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 19, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        return self.head(x)


class BlazePoseLite(nn.Module):
    def __init__(self, channels: int = 3):
        super().__init__()
        layers: List[nn.Module] = []
        in_ch = channels
        layers.extend(
            [
                nn.Conv2d(in_ch, 32, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
            ]
        )
        in_ch = 32
        for _ in range(5):
            block = _depthwise_conv(in_ch, in_ch)
            layers.append(block)
        for _ in range(3):
            block = _depthwise_conv(in_ch, 64)
            layers.append(block)
            in_ch = 64
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Conv2d(in_ch, 96, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(96, 195),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        return self.head(x)


class YOLOPoseTiny(nn.Module):
    def __init__(self, channels: int = 3):
        super().__init__()
        layers: List[nn.Module] = []
        in_ch = channels
        for out_ch in [32, 64, 64, 128, 128, 256]:
            layers.append(nn.Conv2d(in_ch, out_ch, 3, stride=2 if out_ch != in_ch else 1, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.SiLU(inplace=True))
            in_ch = out_ch
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Conv2d(in_ch, 128, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(128, 85, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        return self.head(x)


class SelfieSegmentationLite(nn.Module):
    def __init__(self, channels: int = 3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 16, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            _depthwise_conv(16, 32),
            _depthwise_conv(32, 32),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        return self.decoder(x)


class HumanMattingLite(nn.Module):
    def __init__(self, channels: int = 3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            _depthwise_conv(64, 128),
            _depthwise_conv(128, 128),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        return self.decoder(x)


MODEL_SPECS: Dict[str, ModelSpec] = {
    "openpose-lite": ModelSpec(
        name="openpose-lite",
        factory=lambda c: OpenPoseLite(c),
        default_resolution=256,
        description="Lightweight OpenPose-style keypoint head",
    ),
    "blazepose": ModelSpec(
        name="blazepose",
        factory=lambda c: BlazePoseLite(c),
        default_resolution=256,
        description="Mediapipe-inspired BlazePose network",
    ),
    "yolo-pose": ModelSpec(
        name="yolo-pose",
        factory=lambda c: YOLOPoseTiny(c),
        default_resolution=320,
        description="YOLO-style pose head",
    ),
    "selfie-segmentation": ModelSpec(
        name="selfie-segmentation",
        factory=lambda c: SelfieSegmentationLite(c),
        default_resolution=256,
        description="Binary selfie segmentation",
    ),
    "human-matting": ModelSpec(
        name="human-matting",
        factory=lambda c: HumanMattingLite(c),
        default_resolution=320,
        description="Matting-focused encoder-decoder",
    ),
}


def _collect_activation_size(tensors: Iterable[torch.Tensor]) -> int:
    total = 0
    for tensor in tensors:
        if isinstance(tensor, torch.Tensor):
            total += tensor.nelement() * tensor.element_size()
    return total


def measure_activation_memory(model: nn.Module, sample: torch.Tensor) -> int:
    activations: List[int] = []

    def hook(_module: nn.Module, _inp, output):
        if isinstance(output, torch.Tensor):
            activations.append(output.nelement() * output.element_size())
        elif isinstance(output, (list, tuple)):
            activations.append(_collect_activation_size(output))

    handles = [m.register_forward_hook(hook) for m in model.modules() if not isinstance(m, nn.Sequential)]
    with torch.no_grad():
        model(sample)
    for handle in handles:
        handle.remove()

    return sum(activations)


def benchmark_model(
    spec: ModelSpec,
    device: torch.device,
    batch_size: int,
    resolution: int,
    dtype: torch.dtype,
    warmup: int,
    runs: int,
) -> Dict[str, float]:
    model = spec.factory(spec.channels).to(device=device, dtype=dtype)
    model.eval()

    input_tensor = torch.randn(batch_size, spec.channels, resolution, resolution, device=device, dtype=dtype)

    param_memory = sum(p.nelement() * p.element_size() for p in model.parameters())
    sample_cpu = input_tensor.detach().cpu()
    activation_memory = measure_activation_memory(spec.factory(spec.channels).to(dtype=dtype), sample_cpu)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    torch.set_grad_enabled(False)

    # Warmup
    for _ in range(warmup):
        _ = model(input_tensor)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    timings: List[float] = []
    for _ in range(runs):
        start = time.perf_counter()
        _ = model(input_tensor)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        end = time.perf_counter()
        timings.append(end - start)

    peak_memory = None
    if device.type == "cuda":
        peak_memory = torch.cuda.max_memory_allocated(device)

    return {
        "model": spec.name,
        "resolution": resolution,
        "batch_size": batch_size,
        "dtype": str(dtype),
        "avg_latency_ms": sum(timings) / len(timings) * 1000,
        "throughput_fps": batch_size / (sum(timings) / len(timings)),
        "parameter_memory_mb": param_memory / (1024 ** 2),
        "activation_memory_mb": activation_memory / (1024 ** 2),
        "estimated_vram_mb": (param_memory + activation_memory) / (1024 ** 2),
        "peak_vram_mb": peak_memory / (1024 ** 2) if peak_memory else None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pose model benchmarking helper")
    parser.add_argument("--model", choices=list(MODEL_SPECS.keys()) + ["all"], default="all")
    parser.add_argument("--device", default="cuda", help="torch device string. Defaults to cuda (falls back to cpu)")
    parser.add_argument("--resolution", type=int, default=None, help="Square input resolution (pixels)")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--precision", choices=["fp32", "fp16"], default="fp16")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--output", type=str, default=None, help="Optional path to save JSON results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if args.device.startswith("cuda") and device.type != "cuda":
        print("[WARN] CUDA requested but not available. Falling back to CPU.")

    dtype = torch.float16 if args.precision == "fp16" else torch.float32
    if device.type == "cpu" and dtype == torch.float16:
        print("[WARN] FP16 is not supported on CPU. Using FP32 for execution while keeping configuration labels.")
        dtype = torch.float32
    results = []

    selected_specs: Iterable[ModelSpec]
    if args.model == "all":
        selected_specs = MODEL_SPECS.values()
    else:
        selected_specs = [MODEL_SPECS[args.model]]

    for spec in selected_specs:
        resolution = args.resolution or spec.default_resolution
        result = benchmark_model(
            spec=spec,
            device=device,
            batch_size=args.batch_size,
            resolution=resolution,
            dtype=dtype,
            warmup=args.warmup,
            runs=args.runs,
        )
        results.append(result)
        print(json.dumps(result, indent=2))

    if args.output:
        with open(args.output, "w", encoding="utf-8") as fp:
            json.dump(results, fp, indent=2)


if __name__ == "__main__":
    main()
