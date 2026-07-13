"""Shared observation featurisation: frozen-VAE embedding or pooled raw
profile, plus assembly of the context-free base observation vector."""
from __future__ import annotations

import os

import numpy as np
import torch

from .config import Config, VAE_DIR
from .models.vae import AttentionVAE1D


class Featurizer:
    """Maps env observation dicts to flat base-obs vectors (no context)."""

    def __init__(self, cfg: Config, perception: str, device: str = "cpu",
                 vae_path: str | None = None):
        self.cfg = cfg
        self.perception = perception
        self.device = device
        self.vae = None
        if perception == "vae":
            path = vae_path or os.path.join(VAE_DIR, "vae.pt")
            self.vae = AttentionVAE1D(cfg.vae, cfg.sim.n_beams).to(device)
            self.vae.load_state_dict(torch.load(path, map_location=device))
            self.vae.eval()
            for p in self.vae.parameters():
                p.requires_grad_(False)
            self.feat_dim = cfg.vae.latent_dim
        else:
            self.feat_dim = cfg.rl.raw_pool
        self.base_dim = self.feat_dim + 6

    # ------------------------------------------------------------------ #
    def visual(self, profiles: np.ndarray) -> np.ndarray:
        """profiles: (B, n_beams) -> visual features (B, feat_dim)."""
        if profiles.ndim == 1:
            profiles = profiles[None, :]
        if self.perception == "vae":
            with torch.no_grad():
                x = torch.as_tensor(profiles, dtype=torch.float32,
                                    device=self.device)
                mu = self.vae.embed(x)
            return mu.cpu().numpy()
        # raw: average-pool the profile to raw_pool bins
        B, L = profiles.shape
        k = L // self.cfg.rl.raw_pool
        return profiles.reshape(B, self.cfg.rl.raw_pool, k).mean(axis=2)

    def base_obs(self, obs: dict, feat: np.ndarray | None = None) -> np.ndarray:
        """Flat [feat | goal | odom | last_action] for a single env obs."""
        if feat is None:
            feat = self.visual(obs["profile"])[0]
        return np.concatenate([feat, obs["goal"], obs["odom"],
                               obs["last_action"]]).astype(np.float32)
