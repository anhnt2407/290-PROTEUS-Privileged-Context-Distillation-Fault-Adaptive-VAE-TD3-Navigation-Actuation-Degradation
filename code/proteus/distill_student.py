"""Phase 2: on-policy distillation of the adaptation module A_theta.

The teacher's actor and privileged encoder g_phi are FROZEN. Episodes are
rolled out with the student's own c_hat in the control loop (DAgger-style:
the student visits the states its imperfect estimates produce) while the
regression target is the privileged c = g_phi(e_t) at every tick. The A6
ablation instead rolls out with the teacher's c (offline distillation) and
suffers the resulting distribution shift at deployment."""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from .config import DEFAULT, METHODS, RUNS_DIR, Config, run_name
from .env.faultnav import SUCCESS, FaultNavEnv
from .env.faults import FaultSampler
from .features import Featurizer
from .agents.td3 import TD3Agent
from .agents.replay import HistoryTracker, WindowBuffer
from .models.context import AdaptationModule


def distill(method: str, seed: int, cfg: Config = DEFAULT,
            device: str = "cpu", out_root: str = RUNS_DIR) -> dict:
    spec = METHODS[method]
    assert spec.context == "student"
    torch.set_num_threads(cfg.rl.torch_threads)
    torch.manual_seed(seed + 500)
    np.random.seed(seed + 500)
    rng = np.random.default_rng(9000 * seed + 11)

    run = run_name(method, seed)
    out_dir = os.path.join(out_root, run)
    os.makedirs(out_dir, exist_ok=True)

    # frozen teacher
    teacher_spec = METHODS[spec.teacher]
    feat = Featurizer(cfg, teacher_spec.perception, device)
    teacher = TD3Agent(cfg, teacher_spec, feat.feat_dim, device)
    tpath = os.path.join(out_root, run_name(spec.teacher, seed), "agent.pt")
    teacher.load_state_dict(torch.load(tpath, map_location=device))
    teacher.actor.eval()
    teacher.g_phi.eval()

    dcfg = cfg.distill
    window = spec.window
    student = AdaptationModule(dcfg, cfg.vae.latent_dim, cfg.rl.context_dim,
                               use_latent_flow=spec.use_latent_flow,
                               temporal=spec.temporal,
                               window=window).to(device)
    opt = torch.optim.AdamW(student.parameters(), dcfg.lr, weight_decay=0.0)
    buffer = WindowBuffer(dcfg.buffer_size, window, 4, feat.feat_dim,
                          cfg.rl.context_dim)
    tracker = HistoryTracker(window, 4, feat.feat_dim)

    env = FaultNavEnv(cfg, "train", seed=seed + 500)
    sampler = FaultSampler(cfg.fault, cfg.curriculum, mode="uniform")

    losses: list[dict] = []
    successes: list[bool] = []
    t0 = time.time()
    step = 0
    episode = 0
    while step < dcfg.steps:
        fault_spec = sampler.sample(rng)
        obs = env.reset(spec=fault_spec)
        tracker.reset()
        prev_a = np.zeros(2, dtype=np.float32)
        done = False
        while not done and step < dcfg.steps:
            mu = feat.visual(obs["profile"])[0]
            tracker.push(prev_a, obs["odom"], mu)
            e_now = torch.as_tensor(obs["e_severity"], device=device)
            with torch.no_grad():
                c_true = teacher.g_phi(e_now.unsqueeze(0)).squeeze(0)
                if spec.distill_onpolicy:
                    c_hat = student(
                        torch.as_tensor(tracker.proprio,
                                        device=device).unsqueeze(0),
                        torch.as_tensor(tracker.dmu,
                                        device=device).unsqueeze(0)).squeeze(0)
                    c_used = c_hat
                else:
                    c_used = c_true
                base = feat.base_obs(obs, feat=mu)
                obs_vec = torch.cat([torch.as_tensor(base, device=device),
                                     c_used])
                a = teacher.act(obs_vec, explore=False)
            buffer.add(tracker.proprio.copy(), tracker.dmu.copy(),
                       c_true.cpu().numpy())
            obs, _, done, info = env.step(a)
            prev_a = a.astype(np.float32)
            step += 1
            if step > dcfg.warmup_steps:
                batch = buffer.sample(dcfg.batch_size, device)
                pred = student(batch["proprio"], batch["dmu"])
                loss = torch.nn.functional.mse_loss(pred, batch["target"])
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(), 2.0)
                opt.step()
                if step % 250 == 0:
                    losses.append({"step": step, "loss": float(loss.detach())})
        episode += 1
        successes.append(info["outcome"] == SUCCESS)
        if episode % 25 == 0:
            print(f"[{run}] ep {episode:4d} step {step:6d} "
                  f"distill-loss {losses[-1]['loss'] if losses else float('nan'):.4f} "
                  f"sr25 {np.mean(successes[-25:]):.2f}", flush=True)

    torch.save({"student": student.state_dict(), "teacher": spec.teacher,
                "window": window}, os.path.join(out_dir, "student.pt"))
    summary = {"run": run, "method": method, "seed": seed,
               "episodes": episode, "steps": step,
               "wall_time_s": round(time.time() - t0, 1),
               "final_loss": losses[-1]["loss"] if losses else None,
               "rollout_sr": float(np.mean(successes))}
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f)
    with open(os.path.join(out_dir, "losses.json"), "w") as f:
        json.dump(losses, f)
    print(f"[{run}] done in {summary['wall_time_s']}s "
          f"final-loss {summary['final_loss']:.4f}", flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("method")
    ap.add_argument("seed", type=int)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    distill(args.method, args.seed, device=args.device)


if __name__ == "__main__":
    main()
