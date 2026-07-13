"""TD3 agent with an optional privileged embodiment-context pathway.

Backbone: twin critics, target-policy smoothing, delayed actor updates and
Polyak-averaged targets (Fujimoto et al., 2018); smooth-L1 TD loss and AdamW
follow the family convention. `backbone="ddpg"` (A5 ablation) collapses to a
single critic without smoothing or delay.

Observation layout fed to the networks:
    [ visual feature (mu or pooled raw) | goal (2) | odom (2) | last action (2)
      | context c (context_dim, only when the method uses context) ]

The privileged encoder g_phi is owned by the agent and updated through BOTH
TD losses and the actor loss (context must be informative for value and
action alike); critics see c with gradients, targets use the frozen copy.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn.functional as F

from ..config import Config, E_DIM, MethodSpec
from ..models.actor_critic import Actor, TwinCritic
from ..models.context import PrivilegedContextEncoder
from .replay import TD3Replay


class OUNoise:
    """Ornstein-Uhlenbeck exploration noise with linearly decaying sigma."""

    def __init__(self, dim: int, theta: float, sigma0: float, sigma1: float,
                 decay_steps: int):
        self.dim = dim
        self.theta = theta
        self.sigma0 = sigma0
        self.sigma1 = sigma1
        self.decay_steps = decay_steps
        self.state = np.zeros(dim)

    def reset(self):
        self.state = np.zeros(self.dim)

    def sample(self, step: int) -> np.ndarray:
        frac = min(1.0, step / max(1, self.decay_steps))
        sigma = self.sigma0 + frac * (self.sigma1 - self.sigma0)
        self.state += (-self.theta * self.state
                       + sigma * np.random.randn(self.dim))
        return self.state


class TD3Agent:
    def __init__(self, cfg: Config, spec: MethodSpec, feat_dim: int,
                 device: str = "cpu"):
        self.cfg = cfg
        self.spec = spec
        self.device = device
        rl = cfg.rl
        self.use_context = spec.context != "none"
        self.base_dim = feat_dim + 2 + 2 + 2
        self.obs_dim = self.base_dim + (rl.context_dim if self.use_context else 0)

        self.actor = Actor(self.obs_dim, 2, rl.hidden).to(device)
        self.actor_t = copy.deepcopy(self.actor).requires_grad_(False)
        self.critic = TwinCritic(self.obs_dim, 2, rl.hidden).to(device)
        self.critic_t = copy.deepcopy(self.critic).requires_grad_(False)

        self.g_phi = None
        self.g_phi_t = None
        params_ctx = []
        if self.use_context:
            self.g_phi = PrivilegedContextEncoder(rl.context_dim).to(device)
            self.g_phi_t = copy.deepcopy(self.g_phi).requires_grad_(False)
            params_ctx = list(self.g_phi.parameters())

        self.opt_actor = torch.optim.AdamW(self.actor.parameters(), rl.actor_lr)
        self.opt_critic = torch.optim.AdamW(self.critic.parameters(),
                                            rl.critic_lr)
        self.opt_ctx = (torch.optim.AdamW(params_ctx, rl.context_lr)
                        if params_ctx else None)

        # replay rows store the full context-free base observation
        # [visual feature | goal | odom | last action]
        self.replay = TD3Replay(rl.buffer_size, self.base_dim, E_DIM)
        self.total_updates = 0
        self.is_ddpg = spec.backbone == "ddpg"
        self.noise = OUNoise(2, rl.expl_theta, rl.expl_sigma,
                             rl.expl_sigma_end, rl.expl_decay_steps)

    # ------------------------------------------------------------------ #
    def assemble_obs(self, feat, goal, odom, last_a, c=None) -> torch.Tensor:
        parts = [feat, goal, odom, last_a]
        if self.use_context:
            assert c is not None
            parts.append(c)
        return torch.cat(parts, dim=-1)

    @torch.no_grad()
    def act(self, obs_vec: torch.Tensor, explore: bool = False,
            step: int = 0) -> np.ndarray:
        a = self.actor(obs_vec.unsqueeze(0)).squeeze(0).cpu().numpy()
        if explore:
            a = a + self.noise.sample(step)
        return np.clip(a, -1.0, 1.0)

    @torch.no_grad()
    def act_batch(self, obs_mat: torch.Tensor) -> np.ndarray:
        return self.actor(obs_mat).cpu().numpy()

    def context_of(self, e: torch.Tensor, target: bool = False,
                   grad: bool = False) -> torch.Tensor:
        net = self.g_phi_t if target else self.g_phi
        if grad:
            return net(e)
        with torch.no_grad():
            return net(e)

    # ------------------------------------------------------------------ #
    def update(self) -> dict:
        rl = self.cfg.rl
        batch = self.replay.sample(rl.batch_size, self.device)
        s_base, s2_base = batch["feat"], batch["feat2"]
        act, rew, done = batch["act"], batch["rew"], batch["done"]

        if self.use_context:
            c = self.g_phi(batch["e"])                      # grads flow
            with torch.no_grad():
                c2 = self.g_phi_t(batch["e2"])
            s = torch.cat([s_base, c], dim=-1)
            s2 = torch.cat([s2_base, c2], dim=-1)
        else:
            s, s2 = s_base, s2_base

        # --- critic update ---
        with torch.no_grad():
            a2 = self.actor_t(s2)
            if not self.is_ddpg:
                noise = torch.clamp(
                    torch.randn_like(a2) * rl.policy_noise,
                    -rl.noise_clip, rl.noise_clip)
                a2 = torch.clamp(a2 + noise, -1.0, 1.0)
            q1_t, q2_t = self.critic_t(s2, a2)
            q_t = q1_t if self.is_ddpg else torch.min(q1_t, q2_t)
            y = rew + (1.0 - done) * rl.gamma * q_t
        q1, q2 = self.critic(s, act)
        loss_c = F.smooth_l1_loss(q1, y)
        if not self.is_ddpg:
            loss_c = loss_c + F.smooth_l1_loss(q2, y)
        self.opt_critic.zero_grad()
        if self.opt_ctx:
            self.opt_ctx.zero_grad()
        loss_c.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 2.0)
        self.opt_critic.step()
        if self.opt_ctx:
            torch.nn.utils.clip_grad_norm_(self.g_phi.parameters(), 2.0)
            self.opt_ctx.step()

        out = {"loss_critic": float(loss_c.detach())}

        # --- delayed actor + target updates ---
        self.total_updates += 1
        delay = 1 if self.is_ddpg else rl.policy_delay
        if self.total_updates % delay == 0:
            if self.use_context:
                c = self.g_phi(batch["e"])
                s_pi = torch.cat([s_base, c], dim=-1)
            else:
                s_pi = s_base
            a_pi = self.actor(s_pi)
            loss_a = -self.critic.q1(s_pi, a_pi).mean()
            self.opt_actor.zero_grad()
            if self.opt_ctx:
                self.opt_ctx.zero_grad()
            loss_a.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 2.0)
            self.opt_actor.step()
            if self.opt_ctx:
                torch.nn.utils.clip_grad_norm_(self.g_phi.parameters(), 2.0)
                self.opt_ctx.step()
            out["loss_actor"] = float(loss_a.detach())
            self._polyak(self.actor_t, self.actor)
            self._polyak(self.critic_t, self.critic)
            if self.use_context:
                self._polyak(self.g_phi_t, self.g_phi)
        return out

    def _polyak(self, target, source):
        tau = self.cfg.rl.tau
        with torch.no_grad():
            for pt, p in zip(target.parameters(), source.parameters()):
                pt.mul_(1.0 - tau).add_(p, alpha=tau)

    # ------------------------------------------------------------------ #
    def state_dict(self) -> dict:
        d = {"actor": self.actor.state_dict(),
             "critic": self.critic.state_dict(),
             "spec": self.spec.name}
        if self.g_phi is not None:
            d["g_phi"] = self.g_phi.state_dict()
        return d

    def load_state_dict(self, d: dict):
        self.actor.load_state_dict(d["actor"])
        self.critic.load_state_dict(d["critic"])
        if self.g_phi is not None and "g_phi" in d:
            self.g_phi.load_state_dict(d["g_phi"])
        self.actor_t = copy.deepcopy(self.actor).requires_grad_(False)
        self.critic_t = copy.deepcopy(self.critic).requires_grad_(False)
        if self.g_phi is not None:
            self.g_phi_t = copy.deepcopy(self.g_phi).requires_grad_(False)
