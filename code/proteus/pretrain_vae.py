"""Phase 0: scripted-explorer corpus collection and VAE pretraining.

A privileged potential-field explorer (mixed with random actions) drives the
robot through fault-randomised episodes, logging (degraded, clean) profile
pairs. The attention VAE is trained to reconstruct the clean profile, then
frozen for all later phases — perception is inherited, not the contribution.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import torch

from .config import DEFAULT, CORPUS_DIR, VAE_DIR, Config
from .env.faultnav import FaultNavEnv
from .env.faults import FaultSampler
from .models.vae import AttentionVAE1D


def traversable_distance(profile: np.ndarray, fov_rad: float,
                         range_max: float, footprint: float) -> np.ndarray:
    """True straight-line traversable distance per heading for a DISC robot.

    Converts the depth profile into a planar point cloud and, for every beam
    heading, computes how far a disc of radius `footprint` can advance along
    that heading before touching any sensed point (zero-width rays overstate
    freedom -- the classic beam-steering pitfall)."""
    B = profile.shape[0]
    angles = np.linspace(fov_rad / 2, -fov_rad / 2, B)
    d = (1.0 - profile) * range_max                    # distance per beam
    hit = d < range_max * 0.995                        # beams that saw a surface
    px = d * np.cos(angles)
    py = d * np.sin(angles)
    dirx, diry = np.cos(angles), np.sin(angles)        # candidate headings
    # component of each point along/perpendicular to each heading: (H, P)
    par = dirx[:, None] * px[None, :] + diry[:, None] * py[None, :]
    perp = -diry[:, None] * px[None, :] + dirx[:, None] * py[None, :]
    blocking = hit[None, :] & (np.abs(perp) < footprint) & (par > 0.0)
    t = par - np.sqrt(np.maximum(footprint ** 2 - perp ** 2, 0.0))
    t = np.where(blocking, np.maximum(t, 0.0), np.inf)
    return np.minimum(t.min(axis=1), range_max)


def scripted_action(obs: dict, rng: np.random.Generator,
                    noise: float = 1.0, state: dict | None = None
                    ) -> np.ndarray:
    """Footprint-aware most-open-heading steerer biased toward the goal.

    Used only for corpus collection and feasibility checks -- never as a
    learned baseline."""
    prof = obs["profile"]
    B = prof.shape[0]
    fov = math.radians(120.0)
    angles = np.linspace(fov / 2, -fov / 2, B)
    trav = traversable_distance(prof, fov, 3.5, footprint=0.22)
    goal_ang = float(np.clip(obs["goal"][1], -1, 1)) * math.pi
    goal_dist = float(obs["goal"][0]) * 7.07
    # a heading is as useful as the distance it allows, capped by the goal
    useful = np.minimum(trav, goal_dist + 0.3)
    score = useful - 0.9 * np.abs(angles - goal_ang)
    if state is not None and "steer" in state:      # mild hysteresis
        score += 0.05 * np.exp(-0.5 * ((angles - state["steer"]) / 0.2) ** 2)
    best = int(np.argmax(score))
    steer_to = float(angles[best])
    if abs(goal_ang) > fov / 2:                     # goal outside FOV
        steer_to = math.copysign(fov / 2, goal_ang)
    if state is not None:
        state["steer"] = steer_to
    w = float(np.clip(2.2 * steer_to, -1.0, 1.0))
    # speed: advance along the chosen arc, slowing with narrowness and turn
    mid = B // 2
    front = trav[max(0, mid - 6): mid + 7].min()    # straight-ahead corridor
    chosen = trav[max(0, best - 3): best + 4].min()
    ahead = min(front if abs(steer_to) < 0.15 else np.inf, chosen)
    # committed rotation with hysteresis: once cornered, keep turning the
    # same way until genuinely free (prevents left-right dithering)
    if state is not None and state.get("rot") is not None:
        if front > 0.45 and abs(steer_to) < 0.45:
            state["rot"] = None                     # escaped
        else:
            a = np.array([-1.0, state["rot"]])
            a += rng.normal(0.0, [0.05 * noise, 0.10 * noise])
            return np.clip(a, -1.0, 1.0)
    if ahead < 0.20:                                # cornered: latch a turn
        left, right = trav[:mid].mean(), trav[mid:].mean()
        rot = 1.0 if left >= right else -1.0
        if abs(goal_ang) < 1.4 and abs(left - right) < 0.3:
            rot = math.copysign(1.0, goal_ang)      # tie-break toward goal
        if state is not None:
            state["rot"] = rot
        v_cmd, w = 0.0, rot
    else:
        v_cmd = float(np.clip((ahead - 0.20) / 0.6, 0.05, 1.0))
        v_cmd *= float(np.clip(1.0 - 0.55 * abs(steer_to), 0.25, 1.0))
    a_v = 2.0 * v_cmd - 1.0
    a = np.array([a_v, w])
    a += rng.normal(0.0, [0.10 * noise, 0.18 * noise])
    return np.clip(a, -1.0, 1.0)


def collect_corpus(cfg: Config, seed: int, out_path: str) -> dict:
    """Mixed-driver corpus: privileged planner (goal-directed viewpoints),
    footprint-aware reactive explorer, and random actions."""
    from .env.planner import PrivilegedPlanner
    rng = np.random.default_rng(seed)
    env = FaultNavEnv(cfg, "train", seed=seed)
    planner = PrivilegedPlanner(cfg)
    sampler = FaultSampler(cfg.fault, cfg.curriculum, mode="uniform")
    degraded = np.zeros((cfg.vae.corpus_steps, cfg.sim.n_beams),
                        dtype=np.float32)
    clean = np.zeros_like(degraded)
    i = 0
    episodes = 0
    while i < cfg.vae.corpus_steps:
        obs = env.reset(spec=sampler.sample(rng))
        planner.reset()
        done = False
        u = rng.random()
        driver = "planner" if u < 0.4 else \
            ("reactive" if u < 0.4 + cfg.vae.scripted_frac * 0.5 else "random")
        state: dict = {}
        while not done and i < cfg.vae.corpus_steps:
            degraded[i] = obs["profile"]
            clean[i] = obs["clean_profile"]
            i += 1
            if driver == "planner":
                a = planner.act(env, rng, noise=0.5)
            elif driver == "reactive":
                a = scripted_action(obs, rng, noise=0.6, state=state)
            else:
                a = rng.uniform(-1.0, 1.0, size=2)
            obs, _, done, _ = env.step(a)
        episodes += 1
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(out_path, degraded=degraded, clean=clean)
    return {"steps": int(i), "episodes": episodes}


def train_vae(cfg: Config, corpus_path: str, out_dir: str, seed: int,
              device: str = "cpu") -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    data = np.load(corpus_path)
    x_in = torch.as_tensor(data["degraded"])
    x_cl = torch.as_tensor(data["clean"])
    n = x_in.shape[0]
    n_val = n // 10
    perm = torch.randperm(n)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    model = AttentionVAE1D(cfg.vae, cfg.sim.n_beams).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.vae.lr)
    bs = cfg.vae.batch_size
    history = []
    t0 = time.time()
    for epoch in range(cfg.vae.epochs):
        model.train()
        idx = tr_idx[torch.randperm(tr_idx.shape[0])]
        tr_loss = tr_rec = 0.0
        nb = 0
        for k in range(0, idx.shape[0], bs):
            b = idx[k:k + bs]
            out = model.loss(x_in[b].to(device), x_cl[b].to(device))
            opt.zero_grad()
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tr_loss += float(out["loss"])
            tr_rec += float(out["recon"])
            nb += 1
        model.eval()
        with torch.no_grad():
            vout = model.loss(x_in[val_idx].to(device), x_cl[val_idx].to(device))
        history.append({"epoch": epoch, "train_loss": tr_loss / nb,
                        "train_recon": tr_rec / nb,
                        "val_loss": float(vout["loss"]),
                        "val_recon": float(vout["recon"]),
                        "val_kl": float(vout["kl"])})
        print(f"[vae] epoch {epoch:3d} train {tr_loss / nb:9.4f} "
              f"val {float(vout['loss']):9.4f} "
              f"(rec {float(vout['recon']):8.4f} kl {float(vout['kl']):6.2f})",
              flush=True)
    os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(out_dir, "vae.pt"))
    summary = {"history": history, "wall_time_s": time.time() - t0,
               "params": sum(p.numel() for p in model.parameters())}
    with open(os.path.join(out_dir, "vae_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=DEFAULT.vae.corpus_seed)
    args = ap.parse_args()
    cfg = DEFAULT
    corpus_path = os.path.join(CORPUS_DIR, f"corpus_s{args.seed}.npz")
    if not os.path.exists(corpus_path):
        print("[vae] collecting corpus ...", flush=True)
        meta = collect_corpus(cfg, args.seed, corpus_path)
        print(f"[vae] corpus: {meta}", flush=True)
    train_vae(cfg, corpus_path, VAE_DIR, args.seed, args.device)


if __name__ == "__main__":
    main()
