"""Replay buffers.

TD3Replay caches the FROZEN VAE feature (posterior mean mu) instead of raw
profiles, so every RL update is MLP-only — the URSA efficiency convention.
The true fault vector e is stored so the privileged context c = g_phi(e) can
be recomputed with gradients at update time.

WindowBuffer stores fixed-length causal history windows for the phase-2
distillation regression."""
from __future__ import annotations

import numpy as np
import torch


class TD3Replay:
    def __init__(self, capacity: int, feat_dim: int, e_dim: int,
                 act_dim: int = 2):
        self.capacity = capacity
        self.n = 0
        self.ptr = 0
        self.feat = np.zeros((capacity, feat_dim), dtype=np.float32)
        self.feat2 = np.zeros((capacity, feat_dim), dtype=np.float32)
        self.e = np.zeros((capacity, e_dim), dtype=np.float32)
        self.e2 = np.zeros((capacity, e_dim), dtype=np.float32)
        self.act = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rew = np.zeros((capacity, 1), dtype=np.float32)
        self.done = np.zeros((capacity, 1), dtype=np.float32)

    def add(self, feat, e, act, rew, feat2, e2, done):
        i = self.ptr
        self.feat[i] = feat
        self.e[i] = e
        self.act[i] = act
        self.rew[i] = rew
        self.feat2[i] = feat2
        self.e2[i] = e2
        self.done[i] = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.n = min(self.n + 1, self.capacity)

    def sample(self, batch: int, device) -> dict:
        idx = np.random.randint(0, self.n, size=batch)
        to = lambda x: torch.as_tensor(x[idx], device=device)  # noqa: E731
        return {"feat": to(self.feat), "e": to(self.e), "act": to(self.act),
                "rew": to(self.rew), "feat2": to(self.feat2),
                "e2": to(self.e2), "done": to(self.done)}


class WindowBuffer:
    """(proprio window, d-mu window, target context) triplets for distillation."""

    def __init__(self, capacity: int, window: int, proprio_dim: int,
                 latent_dim: int, c_dim: int):
        self.capacity = capacity
        self.n = 0
        self.ptr = 0
        self.proprio = np.zeros((capacity, window, proprio_dim),
                                dtype=np.float32)
        self.dmu = np.zeros((capacity, window, latent_dim), dtype=np.float32)
        self.target = np.zeros((capacity, c_dim), dtype=np.float32)

    def add(self, proprio, dmu, target):
        i = self.ptr
        self.proprio[i] = proprio
        self.dmu[i] = dmu
        self.target[i] = target
        self.ptr = (self.ptr + 1) % self.capacity
        self.n = min(self.n + 1, self.capacity)

    def sample(self, batch: int, device) -> dict:
        idx = np.random.randint(0, self.n, size=batch)
        to = lambda x: torch.as_tensor(x[idx], device=device)  # noqa: E731
        return {"proprio": to(self.proprio), "dmu": to(self.dmu),
                "target": to(self.target)}


class HistoryTracker:
    """Rolling causal window of per-tick student inputs during rollouts."""

    def __init__(self, window: int, proprio_dim: int, latent_dim: int):
        self.window = window
        self.proprio = np.zeros((window, proprio_dim), dtype=np.float32)
        self.dmu = np.zeros((window, latent_dim), dtype=np.float32)
        self._prev_mu: np.ndarray | None = None

    def reset(self):
        self.proprio[:] = 0.0
        self.dmu[:] = 0.0
        self._prev_mu = None

    def push(self, action: np.ndarray, odom: np.ndarray, mu: np.ndarray):
        self.proprio = np.roll(self.proprio, -1, axis=0)
        self.dmu = np.roll(self.dmu, -1, axis=0)
        self.proprio[-1] = np.concatenate([action, odom]).astype(np.float32)
        dmu = np.zeros_like(mu) if self._prev_mu is None else mu - self._prev_mu
        self.dmu[-1] = dmu.astype(np.float32)
        self._prev_mu = mu.copy()
