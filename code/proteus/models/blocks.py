"""Attention-enhanced convolutional building blocks (1-D port of the
reference architecture: residual blocks with CBAM attention and a light FPN),
following Lee et al. (Adv. Intell. Syst., 2026)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(ch: int) -> nn.GroupNorm:
    return nn.GroupNorm(min(8, ch), ch)


class CBAM1d(nn.Module):
    """Convolutional Block Attention Module (channel then spatial), 1-D."""

    def __init__(self, ch: int, reduction: int = 8, k: int = 7):
        super().__init__()
        hidden = max(1, ch // reduction)
        self.mlp = nn.Sequential(
            nn.Conv1d(ch, hidden, 1), nn.ReLU(), nn.Conv1d(hidden, ch, 1))
        self.spatial = nn.Conv1d(2, 1, kernel_size=k, padding=k // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = self.mlp(F.adaptive_avg_pool1d(x, 1))
        mx = self.mlp(F.adaptive_max_pool1d(x, 1))
        x = x * torch.sigmoid(avg + mx)
        s = torch.cat([x.mean(dim=1, keepdim=True),
                       x.max(dim=1, keepdim=True).values], dim=1)
        return x * torch.sigmoid(self.spatial(s))


class ResBlock1d(nn.Module):
    """Residual conv block with GroupNorm, GELU, CBAM and dropout."""

    def __init__(self, cin: int, cout: int, stride: int = 1,
                 dropout: float = 0.05):
        super().__init__()
        self.conv1 = nn.Conv1d(cin, cout, 3, stride=stride, padding=1)
        self.gn1 = _gn(cout)
        self.conv2 = nn.Conv1d(cout, cout, 3, padding=1)
        self.gn2 = _gn(cout)
        self.cbam = CBAM1d(cout)
        self.drop = nn.Dropout(dropout)
        self.skip = None
        if stride != 1 or cin != cout:
            self.skip = nn.Sequential(nn.Conv1d(cin, cout, 1, stride=stride),
                                      _gn(cout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        idn = self.skip(x) if self.skip is not None else x
        h = F.gelu(self.gn1(self.conv1(x)))
        h = self.gn2(self.conv2(h))
        h = self.cbam(h)
        return self.drop(F.gelu(h + idn))


class UpBlock1d(nn.Module):
    """Transposed-conv residual block with CBAM (decoder side)."""

    def __init__(self, cin: int, cout: int, dropout: float = 0.05):
        super().__init__()
        self.conv1 = nn.ConvTranspose1d(cin, cout, 4, stride=2, padding=1)
        self.gn1 = _gn(cout)
        self.conv2 = nn.Conv1d(cout, cout, 3, padding=1)
        self.gn2 = _gn(cout)
        self.cbam = CBAM1d(cout)
        self.drop = nn.Dropout(dropout)
        self.skip = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv1d(cin, cout, 1), _gn(cout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        idn = self.skip(x)
        h = F.gelu(self.gn1(self.conv1(x)))
        h = self.gn2(self.conv2(h))
        h = self.cbam(h)
        return self.drop(F.gelu(h + idn))


class FPN1d(nn.Module):
    """Light top-down feature-pyramid fusion of three encoder scales."""

    def __init__(self, chans: tuple[int, int, int], out: int):
        super().__init__()
        self.lat = nn.ModuleList([nn.Conv1d(c, out, 1) for c in chans])
        self.smooth = ResBlock1d(out, out)

    def forward(self, feats: list[torch.Tensor]) -> torch.Tensor:
        # feats ordered fine -> coarse; fuse top-down into the finest map
        f3 = self.lat[2](feats[2])
        f2 = self.lat[1](feats[1]) + F.interpolate(
            f3, size=feats[1].shape[-1], mode="nearest")
        f1 = self.lat[0](feats[0]) + F.interpolate(
            f2, size=feats[0].shape[-1], mode="nearest")
        return self.smooth(f1)


def mlp(sizes: list[int], act=nn.GELU, out_act=None,
        layer_norm: bool = True) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        last = i == len(sizes) - 2
        if not last:
            if layer_norm:
                layers.append(nn.LayerNorm(sizes[i + 1]))
            layers.append(act())
        elif out_act is not None:
            layers.append(out_act())
    return nn.Sequential(*layers)


def init_weights(m: nn.Module):
    if isinstance(m, (nn.Linear, nn.Conv1d, nn.ConvTranspose1d)):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
