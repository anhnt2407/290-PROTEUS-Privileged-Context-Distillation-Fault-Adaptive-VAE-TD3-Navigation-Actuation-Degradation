"""Embodiment-context pathway: the privileged encoder g_phi (teacher side)
and the proprioceptive+latent-flow adaptation module A_theta (student side).

Teacher:  c = g_phi(e)             with e the true normalised fault vector.
Student:  c_hat = A_theta(h_t)     with h_t a causal window of per-tick
          features [a_{t-k}, odom_{t-k}, P(d mu_{t-k})], where P is a learned
          linear projection of the frame-to-frame difference of the frozen
          VAE posterior mean (visual ego-motion evidence that survives when
          odometry itself is faulted)."""
from __future__ import annotations

import torch
import torch.nn as nn

from ..config import DistillConfig, E_DIM
from .blocks import mlp


class PrivilegedContextEncoder(nn.Module):
    """g_phi: true fault vector e -> compact bounded context c."""

    def __init__(self, context_dim: int = 8):
        super().__init__()
        self.net = mlp([E_DIM, 64, 64, context_dim], out_act=nn.Tanh)

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        return self.net(e)


class AdaptationModule(nn.Module):
    """A_theta: causal dilated TCN over the recent command-response history.

    Per-tick input feature (dim 8):
        [ a_v, a_w, odom_v, odom_w, P(d mu) (4) ]
    The latent-flow projection P is learned jointly. `use_latent_flow=False`
    (A2 ablation) zeroes that branch. `temporal="ff"` (A4) replaces the TCN
    with a feed-forward network on the newest tick only.
    """

    IN_PROPRIO = 4

    def __init__(self, cfg: DistillConfig, latent_dim: int = 32,
                 context_dim: int = 8, use_latent_flow: bool = True,
                 temporal: str = "tcn", window: int | None = None):
        super().__init__()
        self.cfg = cfg
        self.use_latent_flow = use_latent_flow
        self.temporal = temporal
        self.window = window or cfg.window
        self.flow_proj = nn.Linear(latent_dim, cfg.latent_flow_dim)
        f = self.IN_PROPRIO + cfg.latent_flow_dim
        ch = cfg.tcn_channels
        if temporal == "tcn":
            layers: list[nn.Module] = []
            cin = f
            for d in (1, 2, 4, 8):
                layers += [CausalConv1d(cin, ch, 3, dilation=d), nn.GELU()]
                cin = ch
            self.tcn = nn.Sequential(*layers)
            self.head = nn.Sequential(nn.LayerNorm(ch),
                                      nn.Linear(ch, context_dim), nn.Tanh())
        else:  # feed-forward on the newest tick only
            self.tcn = None
            self.head = mlp([f, 64, 64, context_dim], out_act=nn.Tanh)

    def forward(self, proprio: torch.Tensor,
                dmu: torch.Tensor) -> torch.Tensor:
        """proprio: (B, W, 4); dmu: (B, W, latent_dim). Returns (B, c_dim)."""
        flow = self.flow_proj(dmu)
        if not self.use_latent_flow:
            flow = torch.zeros_like(flow)
        x = torch.cat([proprio, flow], dim=-1)          # (B, W, F)
        if self.temporal == "tcn":
            h = self.tcn(x.transpose(1, 2))             # (B, C, W)
            return self.head(h[:, :, -1])
        return self.head(x[:, -1, :])


class CausalConv1d(nn.Module):
    """Left-padded conv so the output at t sees inputs up to t only."""

    def __init__(self, cin: int, cout: int, k: int, dilation: int = 1):
        super().__init__()
        self.pad = (k - 1) * dilation
        self.conv = nn.Conv1d(cin, cout, k, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = nn.functional.pad(x, (self.pad, 0))
        return self.conv(x)
