"""Actor and twin-critic networks (family-style dual-pathway critics)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import init_weights, mlp


class Actor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int = 2, hidden: int = 256):
        super().__init__()
        self.net = mlp([obs_dim, hidden, hidden, act_dim], out_act=nn.Tanh)
        self.apply(init_weights)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class QNetwork(nn.Module):
    """Dual-pathway Q: separate state and action encoders, then a joint head."""

    def __init__(self, obs_dim: int, act_dim: int = 2, hidden: int = 256):
        super().__init__()
        self.s_path = nn.Sequential(nn.Linear(obs_dim, hidden),
                                    nn.LayerNorm(hidden), nn.GELU())
        self.a_path = nn.Sequential(nn.Linear(act_dim, hidden),
                                    nn.LayerNorm(hidden), nn.GELU())
        self.head = nn.Sequential(nn.Linear(2 * hidden, hidden),
                                  nn.LayerNorm(hidden), nn.GELU(),
                                  nn.Linear(hidden, 1))
        self.apply(init_weights)

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        h = torch.cat([self.s_path(obs), self.a_path(act)], dim=-1)
        return self.head(h)


class TwinCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int = 2, hidden: int = 256):
        super().__init__()
        self.q1 = QNetwork(obs_dim, act_dim, hidden)
        self.q2 = QNetwork(obs_dim, act_dim, hidden)

    def forward(self, obs, act):
        return self.q1(obs, act), self.q2(obs, act)
