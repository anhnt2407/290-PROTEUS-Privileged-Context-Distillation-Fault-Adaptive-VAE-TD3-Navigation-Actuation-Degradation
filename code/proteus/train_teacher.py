"""Phase 1: TD3 training (all non-student methods).

Covers the entire non-student ladder: RAW-DR, NOM, DR, PROTEUS-T (privileged
context) and the A1/A5 ablations, differing only through their MethodSpec.
Fault episodes are drawn by the FaultSampler in the spec's regime, with the
self-paced severity curriculum where enabled."""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from .config import DEFAULT, METHODS, RUNS_DIR, Config, run_name
from .env.faultnav import COLLISION, SUCCESS, TIMEOUT, FaultNavEnv
from .env.faults import FaultSampler
from .features import Featurizer
from .agents.td3 import TD3Agent

OUTCOME_NAME = {SUCCESS: "success", COLLISION: "collision", TIMEOUT: "timeout"}


def train(method: str, seed: int, cfg: Config = DEFAULT, device: str = "cpu",
          out_root: str = RUNS_DIR) -> dict:
    spec = METHODS[method]
    assert spec.context in ("none", "privileged"), \
        "students are produced by distill_student.py"
    torch.set_num_threads(cfg.rl.torch_threads)
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(1000 * seed + 7)

    run = run_name(method, seed)
    out_dir = os.path.join(out_root, run)
    os.makedirs(out_dir, exist_ok=True)

    env = FaultNavEnv(cfg, "train", seed=seed)
    sampler = FaultSampler(cfg.fault, cfg.curriculum, mode=spec.train_faults)
    feat = Featurizer(cfg, spec.perception, device)
    agent = TD3Agent(cfg, spec, feat.feat_dim, device)

    episode_log: list[dict] = []
    losses: list[dict] = []
    t0 = time.time()
    step = 0
    episode = 0
    while step < cfg.rl.total_steps:
        fault_spec = sampler.sample(rng)
        obs = env.reset(spec=fault_spec)
        agent.noise.reset()
        ep_ret, ep_len = 0.0, 0
        done = False
        while not done and step < cfg.rl.total_steps:
            base = feat.base_obs(obs)
            e_now = obs["e_severity"]
            if step < cfg.rl.warmup_steps:
                a = rng.uniform(-1.0, 1.0, size=2)
            else:
                t_base = torch.as_tensor(base, device=device)
                if agent.use_context:
                    c = agent.context_of(torch.as_tensor(
                        e_now, device=device).unsqueeze(0)).squeeze(0)
                    obs_vec = torch.cat([t_base, c])
                else:
                    obs_vec = t_base
                a = agent.act(obs_vec, explore=True, step=step)
            nobs, r, done, info = env.step(a)
            nbase = feat.base_obs(nobs)
            agent.replay.add(base, e_now, a, r, nbase,
                             nobs["e_severity"], done)
            obs = nobs
            ep_ret += r
            ep_len += 1
            step += 1
            if (step >= cfg.rl.warmup_steps
                    and step % cfg.rl.update_every == 0):
                out = agent.update()
                if step % 500 == 0:
                    out["step"] = step
                    losses.append(out)
        episode += 1
        success = info["outcome"] == SUCCESS
        sampler.update(success)
        episode_log.append({
            "episode": episode, "step": step, "ret": round(ep_ret, 3),
            "len": ep_len, "outcome": OUTCOME_NAME.get(info["outcome"], "?"),
            "fault": fault_spec.describe(), "s_max": round(sampler.s_max, 3),
            "ema": None if sampler._ema is None else round(sampler._ema, 3)})
        if episode % 25 == 0:
            recent = episode_log[-25:]
            sr = np.mean([e["outcome"] == "success" for e in recent])
            print(f"[{run}] ep {episode:4d} step {step:6d} "
                  f"sr25 {sr:.2f} s_max {sampler.s_max:.2f} "
                  f"ret {np.mean([e['ret'] for e in recent]):7.2f}",
                  flush=True)

    torch.save(agent.state_dict(), os.path.join(out_dir, "agent.pt"))
    summary = {
        "run": run, "method": method, "seed": seed,
        "episodes": episode, "steps": step,
        "wall_time_s": round(time.time() - t0, 1),
        "curriculum": [(e, round(s, 3), round(m, 3) if m is not None else None)
                       for e, s, m in sampler.history],
        "final_sr100": float(np.mean(
            [e["outcome"] == "success" for e in episode_log[-100:]])),
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f)
    with open(os.path.join(out_dir, "episodes.json"), "w") as f:
        json.dump(episode_log, f)
    with open(os.path.join(out_dir, "losses.json"), "w") as f:
        json.dump(losses, f)
    print(f"[{run}] done in {summary['wall_time_s']}s "
          f"final sr100={summary['final_sr100']:.2f}", flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("method", choices=list(METHODS.keys()))
    ap.add_argument("seed", type=int)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    train(args.method, args.seed, device=args.device)


if __name__ == "__main__":
    main()
