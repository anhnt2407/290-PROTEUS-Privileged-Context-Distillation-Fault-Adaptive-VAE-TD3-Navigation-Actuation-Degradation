"""Evaluation campaign for a trained policy.

Regimes
-------
sweep         : per-fault-type severity grids on the training layout family
sweep_unseen  : reduced grids on the held-out layout family
heldout       : unseen fault type (deadzone), unseen compound (latency+slip),
                seen compound (gain_L+bias), encoder-scale (odometry lies)
onset         : sudden mid-episode fault onset with per-tick timelines
intervention  : student/teacher context clamped to the nominal embedding
                (decision-time contribution of adaptation, cf. URSA u-override)
latency       : per-decision wall-clock

Scenario seeds are a stable CRC32 hash of (regime, type, severity index,
episode index): byte-identical initial conditions across methods."""
from __future__ import annotations

import argparse
import json
import os
import time
import zlib

import numpy as np
import torch

from .config import (DEFAULT, EVAL_DIR, METHODS, RUNS_DIR, TRAIN_FAULT_TYPES,
                     Config, run_name)
from .env.faultnav import COLLISION, SUCCESS, FaultNavEnv
from .env.faults import FaultSpec, spec_from
from .features import Featurizer
from .agents.td3 import TD3Agent
from .agents.replay import HistoryTracker
from .models.context import AdaptationModule

OUT_NAME = {SUCCESS: "success", COLLISION: "collision"}


def scenario_seed(cfg: Config, *parts) -> int:
    key = "|".join(str(p) for p in parts)
    return cfg.eval.eval_seed_offset + zlib.crc32(key.encode())


# --------------------------------------------------------------------------- #
# Policy wrapper (handles every method uniformly, batched across envs)
# --------------------------------------------------------------------------- #
class Policy:
    def __init__(self, cfg: Config, method: str, seed: int,
                 device: str = "cpu", run_root: str = RUNS_DIR,
                 clamp_context: bool = False):
        self.cfg = cfg
        self.spec = METHODS[method]
        self.seed = seed
        self.device = device
        self.clamp = clamp_context
        self.is_student = self.spec.context == "student"

        base_spec = (METHODS[self.spec.teacher] if self.is_student
                     else self.spec)
        self.feat = Featurizer(cfg, base_spec.perception, device)
        self.agent = TD3Agent(cfg, base_spec, self.feat.feat_dim, device)
        load_run = (run_name(self.spec.teacher, seed) if self.is_student
                    else run_name(method, seed))
        self.agent.load_state_dict(torch.load(
            os.path.join(run_root, load_run, "agent.pt"),
            map_location=device))
        self.agent.actor.eval()

        self.student = None
        if self.is_student:
            d = torch.load(os.path.join(run_root, run_name(method, seed),
                                        "student.pt"), map_location=device)
            self.student = AdaptationModule(
                cfg.distill, cfg.vae.latent_dim, cfg.rl.context_dim,
                use_latent_flow=self.spec.use_latent_flow,
                temporal=self.spec.temporal,
                window=self.spec.window).to(device)
            self.student.load_state_dict(d["student"])
            self.student.eval()
        self.trackers: list[HistoryTracker] = []
        self._prev_a: list[np.ndarray] = []

    @property
    def uses_context(self) -> bool:
        return self.agent.use_context

    def reset_state(self, n: int):
        if self.is_student:
            self.trackers = [HistoryTracker(self.spec.window, 4,
                                            self.feat.feat_dim)
                             for _ in range(n)]
        self._prev_a = [np.zeros(2, dtype=np.float32) for _ in range(n)]

    @torch.no_grad()
    def act(self, obs_list: list[dict], active: np.ndarray) -> tuple:
        """Batched action for active envs. Returns (actions (n,2), aux dict)."""
        n = len(obs_list)
        profiles = np.stack([o["profile"] for o in obs_list])
        feats = self.feat.visual(profiles)                       # (n, F)
        base = np.stack([np.concatenate(
            [feats[i], obs_list[i]["goal"], obs_list[i]["odom"],
             obs_list[i]["last_action"]]) for i in range(n)]).astype(np.float32)
        t_base = torch.as_tensor(base, device=self.device)
        aux: dict = {}
        if not self.uses_context:
            actions = self.agent.act_batch(t_base)
            return np.clip(actions, -1, 1), aux

        e_true = torch.as_tensor(
            np.stack([o["e_severity"] for o in obs_list]),
            dtype=torch.float32, device=self.device)
        c_true = self.agent.g_phi(e_true)
        if self.is_student:
            for i in range(n):
                if active[i]:
                    self.trackers[i].push(self._prev_a[i],
                                          obs_list[i]["odom"], feats[i])
            prop = torch.as_tensor(
                np.stack([tr.proprio for tr in self.trackers]),
                device=self.device)
            dmu = torch.as_tensor(
                np.stack([tr.dmu for tr in self.trackers]),
                device=self.device)
            c_hat = self.student(prop, dmu)
            aux["c_err"] = torch.norm(c_hat - c_true, dim=-1).cpu().numpy()
            aux["c_hat"] = c_hat.cpu().numpy()
            c_used = c_hat
        else:
            c_used = c_true
        aux["c_true"] = c_true.cpu().numpy()
        if self.clamp:      # decision-time intervention: pretend nominal body
            c_used = self.agent.g_phi(torch.zeros_like(e_true))
        obs_mat = torch.cat([t_base, c_used], dim=-1)
        actions = self.agent.act_batch(obs_mat)
        actions = np.clip(actions, -1, 1)
        for i in range(n):
            if active[i]:
                self._prev_a[i] = actions[i].astype(np.float32)
        return actions, aux


# --------------------------------------------------------------------------- #
# Vectorised cell execution
# --------------------------------------------------------------------------- #
def run_cell(cfg: Config, policy: Policy, fault_spec_fn, n_eps: int,
             seeds: list[int], family: str = "train",
             record_timelines: bool = False, id_ticks: tuple = ()) -> dict:
    envs = [FaultNavEnv(cfg, family, seed=s) for s in seeds[:n_eps]]
    obs_list, specs = [], []
    for i, env in enumerate(envs):
        spec = fault_spec_fn(i)
        specs.append(spec)
        obs_list.append(env.reset(spec=spec, scenario_seed=seeds[i]))
    policy.reset_state(n_eps)
    active = np.ones(n_eps, dtype=bool)
    rows = [{"ticks": 0, "outcome": "timeout", "path_len": 0.0,
             "straight": envs[i]._straight, "min_clear": np.inf,
             "sum_speed_cmd": 0.0, "sum_speed_real": 0.0, "osc": 0.0,
             "c_err_sum": 0.0, "ret": 0.0, "sum_aw": 0.0}
            for i in range(n_eps)]
    prev_aw = np.zeros(n_eps)
    tl = {k: [[] for _ in range(n_eps)] for k in
          ("goal_dist", "head_err", "c_err", "a_v", "a_w", "v_real", "w_real")} \
        if record_timelines else None
    id_pairs = []
    tick = 0
    while active.any():
        actions, aux = policy.act(obs_list, active)
        for i, env in enumerate(envs):
            if not active[i]:
                continue
            obs, r, done, info = env.step(actions[i])
            obs_list[i] = obs
            row = rows[i]
            row["ticks"] += 1
            row["ret"] += r
            row["path_len"] = info["path_len"]
            row["min_clear"] = min(row["min_clear"], info["clearance"])
            row["sum_speed_cmd"] += info["commanded"][0]
            row["sum_speed_real"] += abs(info["realized"][0])
            row["osc"] += abs(actions[i][1] - prev_aw[i])
            row["sum_aw"] += float(actions[i][1])
            prev_aw[i] = actions[i][1]
            if "c_err" in aux:
                row["c_err_sum"] += float(aux["c_err"][i])
            if record_timelines:
                tl["goal_dist"][i].append(info["goal_dist"])
                tl["head_err"][i].append(abs(obs["goal"][1]))
                tl["c_err"][i].append(float(aux.get("c_err", np.zeros(len(envs)))[i])
                                      if "c_err" in aux else 0.0)
                tl["a_v"][i].append(float(actions[i][0]))
                tl["a_w"][i].append(float(actions[i][1]))
                tl["v_real"][i].append(float(info["realized"][0]))
                tl["w_real"][i].append(float(info["realized"][1]))
            if tick in id_ticks and "c_hat" in aux:
                id_pairs.append(np.concatenate(
                    [aux["c_hat"][i], aux["c_true"][i],
                     obs["e_severity"]]))
            if done:
                active[i] = False
                row["outcome"] = OUT_NAME.get(info["outcome"], "timeout")
        tick += 1
    # per-episode records
    out = []
    for i, row in enumerate(rows):
        succ = row["outcome"] == "success"
        spl = (row["straight"] / max(row["straight"], row["path_len"])
               if succ and row["path_len"] > 0 else 0.0)
        out.append({
            "outcome": row["outcome"], "success": int(succ),
            "collision": int(row["outcome"] == "collision"),
            "ticks": row["ticks"], "spl": round(float(spl), 4),
            "ret": round(row["ret"], 3),
            "path_len": round(row["path_len"], 3),
            "min_clear": round(float(row["min_clear"]), 4),
            "mean_speed_cmd": round(row["sum_speed_cmd"] / row["ticks"], 4),
            "mean_speed_real": round(row["sum_speed_real"] / row["ticks"], 4),
            "osc": round(row["osc"] / row["ticks"], 4),
            "mean_aw": round(row["sum_aw"] / row["ticks"], 4),
            "c_err": round(row["c_err_sum"] / row["ticks"], 4),
            "fault": specs[i].describe(),
        })
    res = {"episodes": out}
    if record_timelines:
        res["timelines"] = tl
    if id_pairs:
        res["id_pairs"] = np.stack(id_pairs)
    return res


# --------------------------------------------------------------------------- #
# Regimes
# --------------------------------------------------------------------------- #
def _sev_spec(fault_type: str, s: float, ep_idx: int,
              onset: int = 0) -> FaultSpec:
    if s <= 0:
        return FaultSpec(onset_tick=onset)
    val = s
    if fault_type == "bias":
        val = s if ep_idx % 2 == 0 else -s
    return spec_from({fault_type: val}, onset_tick=onset)


def _compound_spec(pair: tuple[str, str], s: float, ep_idx: int) -> FaultSpec:
    d = {}
    for t in pair:
        d[t] = (s if ep_idx % 2 == 0 else -s) if t == "bias" else s
    return spec_from(d) if s > 0 else FaultSpec()


def eval_sweep(cfg, policy, method, seed, family, severities, n_eps,
               regime_tag, id_dump=False):
    cells = {}
    id_all = []
    for ft in TRAIN_FAULT_TYPES:
        for si, s in enumerate(severities):
            seeds = [scenario_seed(cfg, regime_tag, ft, si, k)
                     for k in range(n_eps)]
            res = run_cell(cfg, policy,
                           lambda i, ft=ft, s=s: _sev_spec(ft, s, i),
                           n_eps, seeds, family=family,
                           id_ticks=(60, 140, 220) if id_dump else ())
            cells[f"{ft}|{si}"] = {"severity": s, "episodes": res["episodes"]}
            if "id_pairs" in res:
                id_all.append(res["id_pairs"])
    out = {"regime": regime_tag, "severities": list(severities),
           "cells": cells}
    return out, (np.concatenate(id_all) if id_all else None)


def eval_heldout(cfg, policy, n_eps):
    ev = cfg.eval
    cells = {}
    groups = [("deadzone", lambda i, s: _sev_spec("deadzone", s, i)),
              ("encoder", lambda i, s: _sev_spec("encoder", s, i))]
    for pair in ev.compound_pairs:
        tag = "+".join(pair)
        groups.append((tag,
                       lambda i, s, pair=pair: _compound_spec(pair, s, i)))
    for tag, fn in groups:
        for si, s in enumerate(ev.severities):
            seeds = [scenario_seed(cfg, "heldout", tag, si, k)
                     for k in range(n_eps)]
            res = run_cell(cfg, policy, lambda i, s=s, fn=fn: fn(i, s),
                           n_eps, seeds, family="train")
            cells[f"{tag}|{si}"] = {"severity": s,
                                    "episodes": res["episodes"]}
    return {"regime": "heldout", "severities": list(ev.severities),
            "cells": cells}


def eval_onset(cfg, policy):
    ev = cfg.eval
    cells = {}
    tls = {}
    for ft in TRAIN_FAULT_TYPES:
        seeds = [scenario_seed(cfg, "onset", ft, 0, k)
                 for k in range(ev.onset_episodes)]
        res = run_cell(cfg, policy,
                       lambda i, ft=ft: _sev_spec(ft, ev.onset_severity, i,
                                                  onset=ev.onset_tick),
                       ev.onset_episodes, seeds, family="train",
                       record_timelines=True)
        cells[ft] = {"severity": ev.onset_severity,
                     "onset_tick": ev.onset_tick,
                     "episodes": res["episodes"]}
        tls[ft] = res["timelines"]
    return {"regime": "onset", "cells": cells}, tls


def eval_latency(cfg, policy, n_calls=500):
    env = FaultNavEnv(cfg, "train", seed=123)
    obs = env.reset(spec=FaultSpec())
    policy.reset_state(1)
    active = np.ones(1, dtype=bool)
    ts = []
    for _ in range(n_calls):
        t0 = time.perf_counter()
        actions, _ = policy.act([obs], active)
        ts.append(time.perf_counter() - t0)
        obs, _, done, _ = env.step(actions[0])
        if done:
            obs = env.reset(spec=FaultSpec())
            policy.reset_state(1)
    return {"mean_ms": 1e3 * float(np.mean(ts)),
            "p95_ms": 1e3 * float(np.percentile(ts, 95))}


# --------------------------------------------------------------------------- #
def evaluate(method: str, seed: int, cfg: Config = DEFAULT,
             device: str = "cpu", regimes: list[str] | None = None,
             run_root: str = RUNS_DIR, out_root: str = EVAL_DIR) -> None:
    torch.set_num_threads(cfg.rl.torch_threads)
    run = run_name(method, seed)
    out_dir = os.path.join(out_root, run)
    os.makedirs(out_dir, exist_ok=True)
    policy = Policy(cfg, method, seed, device, run_root)
    ev = cfg.eval
    regimes = regimes or ["sweep", "sweep_unseen", "heldout", "onset",
                          "latency", "intervention"]

    if "sweep" in regimes:
        t0 = time.time()
        out, id_pairs = eval_sweep(cfg, policy, method, seed, "train",
                                   ev.severities, ev.episodes_per_cell,
                                   "sweep", id_dump=policy.is_student)
        out["wall_s"] = round(time.time() - t0, 1)
        with open(os.path.join(out_dir, "sweep.json"), "w") as f:
            json.dump(out, f)
        if id_pairs is not None:
            np.savez_compressed(os.path.join(out_dir, "id_pairs.npz"),
                                pairs=id_pairs)
        print(f"[{run}] sweep done ({out['wall_s']}s)", flush=True)

    if "sweep_unseen" in regimes:
        out, _ = eval_sweep(cfg, policy, method, seed, "unseen",
                            ev.unseen_severities, ev.unseen_episodes_per_cell,
                            "sweep_unseen")
        with open(os.path.join(out_dir, "sweep_unseen.json"), "w") as f:
            json.dump(out, f)
        print(f"[{run}] sweep_unseen done", flush=True)

    if "heldout" in regimes:
        out = eval_heldout(cfg, policy, ev.unseen_episodes_per_cell)
        with open(os.path.join(out_dir, "heldout.json"), "w") as f:
            json.dump(out, f)
        print(f"[{run}] heldout done", flush=True)

    if "onset" in regimes:
        out, tls = eval_onset(cfg, policy)
        with open(os.path.join(out_dir, "onset.json"), "w") as f:
            json.dump(out, f)
        np.savez_compressed(
            os.path.join(out_dir, "onset_timelines.npz"),
            **{f"{ft}_{k}": np.array(
                [np.pad(np.asarray(x, dtype=np.float32),
                        (0, cfg.sim.horizon - len(x)), constant_values=np.nan)
                 for x in tls[ft][k]])
               for ft in tls for k in tls[ft]})
        print(f"[{run}] onset done", flush=True)

    if "intervention" in regimes and policy.uses_context:
        clamped = Policy(cfg, method, seed, device, run_root,
                         clamp_context=True)
        out, _ = eval_sweep(cfg, clamped, method, seed, "train",
                            ev.severities, ev.unseen_episodes_per_cell,
                            "sweep")           # same scenarios as sweep
        out["regime"] = "intervention"
        with open(os.path.join(out_dir, "intervention.json"), "w") as f:
            json.dump(out, f)
        print(f"[{run}] intervention done", flush=True)

    if "latency" in regimes:
        out = eval_latency(cfg, policy)
        with open(os.path.join(out_dir, "latency.json"), "w") as f:
            json.dump(out, f)
        print(f"[{run}] latency: {out['mean_ms']:.2f} ms", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("method")
    ap.add_argument("seed", type=int)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--regimes", nargs="*", default=None)
    args = ap.parse_args()
    evaluate(args.method, args.seed, device=args.device,
             regimes=args.regimes)


if __name__ == "__main__":
    main()
