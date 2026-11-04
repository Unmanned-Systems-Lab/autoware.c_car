"""Inference-only HT Scout eight-direction student network."""

from __future__ import annotations

from typing import Mapping

import torch
from torch import Tensor, nn


GRADIENT_KERNEL = torch.tensor(
    [
        [[[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 0.0]]],
        [[[0.0, 1.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 0.0]]],
        [[[0.0, 0.0, 1.0], [0.0, -1.0, 0.0], [0.0, 0.0, 0.0]]],
        [[[0.0, 0.0, 0.0], [1.0, -1.0, 0.0], [0.0, 0.0, 0.0]]],
        [[[0.0, 0.0, 0.0], [0.0, -1.0, 1.0], [0.0, 0.0, 0.0]]],
        [[[0.0, 0.0, 0.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]]],
        [[[0.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 1.0, 0.0]]],
        [[[0.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]]],
    ],
    dtype=torch.float32,
)


class HeightGradientScale(nn.Module):
    def __init__(self, std_m: float = 0.115837) -> None:
        super().__init__()
        self.std_m = float(std_m)

    def forward(self, x: Tensor) -> Tensor:
        return x / self.std_m


class BasicBlock(nn.Module):
    def __init__(self, inplanes: int, planes: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, 3, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity[:, :, 2:-2, 2:-2])


class HTStudent8Direction(nn.Module):
    """Scout student producing eight directions and two classes per cell."""

    receptive_field = 51
    direction_count = 8
    class_count = 2

    def __init__(self, gradient_std_m: float = 0.115837) -> None:
        super().__init__()
        self.conv_grad = nn.Conv2d(1, 8, kernel_size=3, bias=False)
        self.ht_grad_activate = HeightGradientScale(gradient_std_m)

        channels = [8, 16, 24, 32, 40, 48, 48]
        for index, (in_channels, out_channels) in enumerate(
            zip(channels[:-1], channels[1:]), start=1
        ):
            setattr(
                self,
                f"res_conv{index}",
                nn.Conv2d(in_channels, out_channels, 3, bias=False),
            )
            setattr(self, f"res_bn{index}", nn.BatchNorm2d(out_channels))
            setattr(self, f"res_block{index}", BasicBlock(out_channels, out_channels))

        self.hdc_conv1 = nn.Conv2d(48, 56, 3, dilation=1, bias=False)
        self.hdc_bn1 = nn.BatchNorm2d(56)
        self.hdc_conv2 = nn.Conv2d(56, 64, 3, dilation=2, bias=False)
        self.hdc_bn2 = nn.BatchNorm2d(64)
        self.hdc_conv3 = nn.Conv2d(64, 72, 3, dilation=3, bias=False)
        self.hdc_bn3 = nn.BatchNorm2d(72)
        self.conv_dim = nn.Conv2d(72, 16, kernel_size=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.set_fixed_gradient_encoder()

    @torch.no_grad()
    def set_fixed_gradient_encoder(self) -> None:
        self.conv_grad.weight.copy_(GRADIENT_KERNEL)
        self.conv_grad.weight.requires_grad_(False)

    def forward_logits(self, x: Tensor) -> Tensor:
        x = self.ht_grad_activate(self.conv_grad(x))
        for index in range(1, 7):
            x = getattr(self, f"res_conv{index}")(x)
            x = getattr(self, f"res_bn{index}")(x)
            x = self.relu(x)
            x = getattr(self, f"res_block{index}")(x)
        x = self.relu(self.hdc_bn1(self.hdc_conv1(x)))
        x = self.relu(self.hdc_bn2(self.hdc_conv2(x)))
        x = self.relu(self.hdc_bn3(self.hdc_conv3(x)))
        return self.conv_dim(x)

    def forward(self, x: Tensor) -> Tensor:
        logits = self.forward_logits(x)
        batch, _, height, width = logits.shape
        return logits.view(batch, 8, 2, height, width)


def normalize_checkpoint(checkpoint: object) -> Mapping[str, Tensor]:
    """Return a plain state dict and tolerate common training wrappers."""

    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"Checkpoint must be a mapping, got {type(checkpoint).__name__}")

    for key in ("state_dict", "model_state_dict", "model"):
        wrapped = checkpoint.get(key)
        if isinstance(wrapped, Mapping):
            checkpoint = wrapped
            break

    state = {}
    for key, value in checkpoint.items():
        if not isinstance(key, str) or not isinstance(value, Tensor):
            continue
        state[key.removeprefix("module.")] = value
    if not state:
        raise ValueError("Checkpoint does not contain tensor parameters")
    return state


def load_student_model(weight_path: str, device: torch.device) -> HTStudent8Direction:
    """Load the validated student checkpoint strictly for inference."""

    try:
        checkpoint = torch.load(weight_path, map_location="cpu", weights_only=True)
    except TypeError:  # Older Jetson PyTorch builds do not expose weights_only.
        checkpoint = torch.load(weight_path, map_location="cpu")

    model = HTStudent8Direction()
    model.load_state_dict(normalize_checkpoint(checkpoint), strict=True)
    model.to(device)
    model.eval()
    return model
