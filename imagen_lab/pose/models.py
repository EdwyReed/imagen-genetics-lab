"""Lightweight pose and segmentation model definitions.

The architectures mirror the small networks measured in
``benchmarks/pose_benchmark.py`` and are intentionally simple so they can be
instantiated quickly for lightweight analytics tasks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

import torch
from torch import nn


@dataclass(frozen=True)
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
        layers = []
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.backbone(x)
        return self.head(x)


class BlazePoseLite(nn.Module):
    def __init__(self, channels: int = 3):
        super().__init__()
        layers = []
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.backbone(x)
        return self.head(x)


class YOLOPoseTiny(nn.Module):
    def __init__(self, channels: int = 3):
        super().__init__()
        layers = []
        in_ch = channels
        for out_ch in [32, 64, 64, 128, 128, 256]:
            layers.append(
                nn.Conv2d(in_ch, out_ch, 3, stride=2 if out_ch != in_ch else 1, padding=1, bias=False)
            )
            layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.SiLU(inplace=True))
            in_ch = out_ch
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Conv2d(in_ch, 128, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(128, 85, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
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

__all__ = ["MODEL_SPECS", "ModelSpec"]
