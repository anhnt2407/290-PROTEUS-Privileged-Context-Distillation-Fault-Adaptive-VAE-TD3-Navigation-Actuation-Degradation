"""Attention-enhanced 1-D variational autoencoder (inherited perception).

Direct 1-D port of the reference VAE+ (residual CBAM encoder + light FPN +
Gaussian latent) trained to reconstruct the CLEAN depth profile from the
degraded one, then frozen for control. beta-VAE objective with free-bits KL
to prevent posterior collapse."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import VAEConfig
from .blocks import FPN1d, ResBlock1d, UpBlock1d


class AttentionVAE1D(nn.Module):
    def __init__(self, cfg: VAEConfig, n_beams: int = 128):
        super().__init__()
        c = cfg.base_channels
        self.cfg = cfg
        self.n_beams = n_beams
        self.enc1 = ResBlock1d(1, c, stride=2)          # L/2
        self.enc2 = ResBlock1d(c, 2 * c, stride=2)      # L/4
        self.enc3 = ResBlock1d(2 * c, 4 * c, stride=2)  # L/8
        self.fpn = FPN1d((c, 2 * c, 4 * c), 4 * c)
        self.fc_mu = nn.Linear(4 * c, cfg.latent_dim)
        self.fc_logvar = nn.Linear(4 * c, cfg.latent_dim)

        self.dec_len = n_beams // 8
        self.dec_in = nn.Linear(cfg.latent_dim, 4 * c * self.dec_len)
        self.dec1 = UpBlock1d(4 * c, 2 * c)             # L/4
        self.dec2 = UpBlock1d(2 * c, c)                 # L/2
        self.dec3 = UpBlock1d(c, c)                     # L
        self.head = nn.Conv1d(c, 1, 3, padding=1)

    # ------------------------------------------------------------------ #
    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (B, n_beams) or (B, 1, n_beams) -> (mu, logvar)."""
        if x.dim() == 2:
            x = x.unsqueeze(1)
        f1 = self.enc1(x)
        f2 = self.enc2(f1)
        f3 = self.enc3(f2)
        fused = self.fpn([f1, f2, f3])
        h = F.adaptive_avg_pool1d(fused, 1).flatten(1)
        mu = self.fc_mu(h)
        logvar = torch.clamp(self.fc_logvar(h), -8.0, 8.0)
        return mu, logvar

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.dec_in(z).view(z.shape[0], -1, self.dec_len)
        h = self.dec3(self.dec2(self.dec1(h)))
        return torch.sigmoid(self.head(h)).squeeze(1)

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    # ------------------------------------------------------------------ #
    def loss(self, x_in: torch.Tensor, x_clean: torch.Tensor) -> dict:
        recon, mu, logvar = self.forward(x_in)
        rec = F.mse_loss(recon, x_clean, reduction="none").sum(dim=1).mean()
        kl_dim = 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar)   # (B, d)
        kl_fb = torch.clamp(kl_dim.mean(dim=0),
                            min=self.cfg.free_bits).sum()
        total = rec + self.cfg.beta * kl_fb
        return {"loss": total, "recon": rec.detach(),
                "kl": kl_dim.mean(dim=0).sum().detach()}

    @torch.no_grad()
    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Posterior mean used as the frozen visual feature."""
        mu, _ = self.encode(x)
        return mu
