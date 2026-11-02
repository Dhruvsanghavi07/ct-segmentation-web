"""A compact 2D U-Net (Ronneberger et al., 2015) sized to train on a CPU and run in a browser.

Four resolution levels (16, 32, 64, 128 channels) and a 256-channel bottleneck, batch norm,
transposed-conv upsampling and skip connections. The parameter count and ONNX size are printed by
train.py and export.py and recorded in the run's results.
Only operators that onnxruntime-web supports are used (Conv, BatchNorm, Relu, MaxPool, ConvTranspose, Concat).
"""

from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Sequential):
    def __init__(self, cin: int, cout: int):
        super().__init__(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )


class UNet(nn.Module):
    def __init__(self, in_channels: int = 1, n_classes: int = 2, base: int = 16, depth: int = 4):
        super().__init__()
        widths = [base * 2**i for i in range(depth)]
        self.down = nn.ModuleList()
        c = in_channels
        for w in widths:
            self.down.append(ConvBlock(c, w))
            c = w
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(widths[-1], widths[-1] * 2)
        self.up, self.dec = nn.ModuleList(), nn.ModuleList()
        c = widths[-1] * 2
        for w in reversed(widths):
            self.up.append(nn.ConvTranspose2d(c, w, 2, stride=2))
            self.dec.append(ConvBlock(w * 2, w))
            c = w
        self.head = nn.Conv2d(widths[0], n_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for block in self.down:
            x = block(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        for up, dec, skip in zip(self.up, self.dec, reversed(skips), strict=True):
            x = dec(torch.cat([up(x), skip], dim=1))
        return self.head(x)  # logits, (N, n_classes, H, W)


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


class DiceCELoss(nn.Module):
    """Cross-entropy plus soft Dice on the foreground class. CE gives stable early gradients; Dice
    counters the class imbalance (the spleen is a few percent of each slice)."""

    def __init__(self, dice_weight: float = 1.0, eps: float = 1.0):
        super().__init__()
        self.ce, self.w, self.eps = nn.CrossEntropyLoss(), dice_weight, eps

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prob = torch.softmax(logits, dim=1)[:, 1]
        t = target.float()
        inter = (prob * t).sum()
        soft_dice = (2 * inter + self.eps) / (prob.sum() + t.sum() + self.eps)
        return self.ce(logits, target.long()) + self.w * (1 - soft_dice)
