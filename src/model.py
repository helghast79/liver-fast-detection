"""Tiny U-Net for fast browser-side liver segmentation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TinyUNet(nn.Module):
    """Shallow U-Net with channels [8, 16, 32, 64] for small ONNX + CPU speed."""

    def __init__(self, in_channels: int = 1, out_channels: int = 1, base: int = 8) -> None:
        super().__init__()
        c1, c2, c3, c4 = base, base * 2, base * 4, base * 8

        self.enc1 = ConvBlock(in_channels, c1)
        self.enc2 = ConvBlock(c1, c2)
        self.enc3 = ConvBlock(c2, c3)
        self.bottleneck = ConvBlock(c3, c4)

        self.up3 = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(c4, c3)
        self.up2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(c3, c2)
        self.up1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(c2, c1)

        self.head = nn.Conv2d(c1, out_channels, kernel_size=1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))

        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.head(d1)


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    targets = targets.float()
    dims = (1, 2, 3)
    intersection = (probs * targets).sum(dim=dims)
    union = probs.sum(dim=dims) + targets.sum(dim=dims)
    dice = (2 * intersection + eps) / (union + eps)
    return 1 - dice.mean()


def bce_dice_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, targets.float())
    return bce + dice_loss(logits, targets)


def tversky_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.3,
    beta: float = 0.7,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Soft Tversky loss. beta > alpha penalizes false negatives more (helps recall /
    small / boundary misses).
    """
    probs = torch.sigmoid(logits)
    targets = targets.float()
    dims = (1, 2, 3)
    tp = (probs * targets).sum(dim=dims)
    fp = (probs * (1.0 - targets)).sum(dim=dims)
    fn = ((1.0 - probs) * targets).sum(dim=dims)
    tversky = (tp + eps) / (tp + alpha * fp + beta * fn + eps)
    return 1.0 - tversky.mean()


def bce_tversky_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.3,
    beta: float = 0.7,
) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, targets.float())
    return bce + tversky_loss(logits, targets, alpha=alpha, beta=beta)


def boundary_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Light boundary-aware term: L1 on spatial gradients of probs vs masks.
    Encourages sharper edges without a full Hausdorff implementation.
    """
    probs = torch.sigmoid(logits)
    targets = targets.float()
    # Finite differences along H/W
    p_dh = probs[:, :, 1:, :] - probs[:, :, :-1, :]
    p_dw = probs[:, :, :, 1:] - probs[:, :, :, :-1]
    t_dh = targets[:, :, 1:, :] - targets[:, :, :-1, :]
    t_dw = targets[:, :, :, 1:] - targets[:, :, :, :-1]
    return (p_dh - t_dh).abs().mean() + (p_dw - t_dw).abs().mean() + eps * 0.0


def bce_tversky_boundary_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.3,
    beta: float = 0.7,
    boundary_weight: float = 0.25,
) -> torch.Tensor:
    return bce_tversky_loss(logits, targets, alpha=alpha, beta=beta) + boundary_weight * boundary_loss(
        logits, targets
    )


@torch.no_grad()
def soft_dice(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> float:
    probs = (torch.sigmoid(logits) > 0.5).float()
    targets = targets.float()
    dims = (1, 2, 3)
    intersection = (probs * targets).sum(dim=dims)
    union = probs.sum(dim=dims) + targets.sum(dim=dims)
    dice = (2 * intersection + eps) / (union + eps)
    return float(dice.mean().item())


@torch.no_grad()
def soft_iou(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> float:
    preds = (torch.sigmoid(logits) > 0.5).float()
    targets = targets.float()
    dims = (1, 2, 3)
    intersection = (preds * targets).sum(dim=dims)
    union = preds.sum(dim=dims) + targets.sum(dim=dims) - intersection
    iou = (intersection + eps) / (union + eps)
    return float(iou.mean().item())
