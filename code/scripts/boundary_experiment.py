#!/usr/bin/env python3
"""The observability-boundary experiment: *when* would embodiment knowledge
help learned navigation?

Two localization regimes are compared on the same trained policies and the same
seed-matched scenarios:

  oracle       goal bearing from the true pose (external localization present)
  dead_reckon  goal bearing from a pose integrated out of the faulty wheel
               odometry (a cheap robot with no external localization)

For a proprioception-*preserving* fault (bias) the dead-reckoned belief stays
accurate and feedback compensates in both regimes. For a proprioception-
*corrupting* fault (encoder scale) the belief drifts, reactive navigation
collapses, and a gap opens between what a policy achieves and what perfect
localization (the scripted oracle controller) could achieve -- the head-room a
successful embodiment identifier would need to capture. The companion
perception analysis shows that head-room is not exteroceptively recoverable on
this sensor, which is the paper's central boundary.

Outputs results/boundary/boundary.json consumed by make_figures.py.
"""
from __future__ import annotations

import dataclasses
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from proteus.config import DEFAULT, RESULTS_DIR, Config, SimConfig
from proteus.env.faultnav import FaultNavEnv, SUCCESS
from proteus.env.faults import FaultSpec, spec_from
from proteus.evaluate import Policy, scenario_seed

OUT_DIR = os.path.join(RESULTS_DIR, "boundary")
FAULTS = ["encoder", "bias"]           # corrupting vs preserving
SEVERITIES = [0.0, 1 / 3, 2 / 3, 1.0]
N_EPS = 40
POLICY_METHODS = ["nom", "dr", "proteus"]


def cfg_with(localization: str) -> Config:
    sim = dataclasses.replace(DEFAULT.sim, localization=localization)
    return dataclasses.replace(DEFAULT, sim=sim)


# --------------------------------------------------------------------------- #
# Scripted controllers: bound the achievable success with perfect vs
# dead-reckoned localization, independent of any learned policy.
# --------------------------------------------------------------------------- #
def _script_action(bearing: float, profile: np.ndarray) -> np.ndarray:
    n = len(profile)
    front = float(profile[int(n * 0.30):int(n * 0.70)].max())
    steer = 2.2 * bearing
    if front > 0.55:
        left = float(profile[:n // 2].max())
        right = float(profile[n // 2:].max())
        steer += 1.4 if right > left else -1.4
    w = float(np.clip(steer, -1.0, 1.0))
    v = float(np.clip(0.9 * (1 - 0.6 * abs(w)) * (1 - min(1.0, front)), -1, 1))
    if front > 0.8:
        v = -0.2
    return np.array([v, w])


def scripted_cell(fault: str, s: float, seeds: list[int], use_true: bool) -> float:
    """Scripted proportional controller; `use_true` selects the true bearing
    (oracle localization) or the dead-reckoned bearing (reactive)."""
    cfg = DEFAULT
    env = FaultNavEnv(cfg, "train", seed=1)
    succ = 0
    for k in range(len(seeds)):
        spec = FaultSpec() if s <= 0 else spec_from(
            {fault: (s if k % 2 == 0 else -s) if fault == "bias" else s})
        env.reset(spec=spec, scenario_seed=seeds[k])
        pos_dr = env.layout.start.copy()
        head_dr = env.layout.start_heading
        goal = env.layout.goal
        obs = env._observe()
        done = False
        while not done:
            if use_true:
                bearing = env._goal_angle()
            else:
                g = goal - pos_dr
                bearing = math.atan2(g[1], g[0]) - head_dr
                bearing = (bearing + math.pi) % (2 * math.pi) - math.pi
            obs, _, done, info = env.step(_script_action(bearing, obs["profile"]))
            v_meas, w_meas = env._odom
            head_dr += w_meas * cfg.sim.dt
            pos_dr = pos_dr + v_meas * cfg.sim.dt * np.array(
                [math.cos(head_dr), math.sin(head_dr)])
        succ += info["outcome"] == SUCCESS
    return succ / len(seeds)


# --------------------------------------------------------------------------- #
# Trained-policy cells under a chosen localization regime
# --------------------------------------------------------------------------- #
def policy_cell(cfg: Config, policy: Policy, fault: str, s: float,
                seeds: list[int]) -> float:
    envs = [FaultNavEnv(cfg, "train", seed=sd) for sd in seeds]
    obs_list = []
    for i, env in enumerate(envs):
        spec = FaultSpec() if s <= 0 else spec_from(
            {fault: (s if i % 2 == 0 else -s) if fault == "bias" else s})
        obs_list.append(env.reset(spec=spec, scenario_seed=seeds[i]))
    policy.reset_state(len(envs))
    active = np.ones(len(envs), dtype=bool)
    succ = 0
    while active.any():
        actions, _ = policy.act(obs_list, active)
        for i, env in enumerate(envs):
            if not active[i]:
                continue
            obs, _, done, info = env.step(actions[i])
            obs_list[i] = obs
            if done:
                active[i] = False
                succ += info["outcome"] == SUCCESS
    return succ / len(envs)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    cfgs = {"oracle": cfg_with("oracle"),
            "dead_reckon": cfg_with("dead_reckon")}
    out = {"faults": FAULTS, "severities": SEVERITIES, "n_eps": N_EPS,
           "policies": {}, "scripted": {}}

    # scripted head-room bound (localization is chosen inside the controller,
    # so the env stays in its default oracle mode for physics)
    for fault in FAULTS:
        out["scripted"][fault] = {"oracle": [], "reactive": []}
        for si, s in enumerate(SEVERITIES):
            seeds = [scenario_seed(DEFAULT, "boundary", fault, si, k)
                     for k in range(N_EPS)]
            out["scripted"][fault]["oracle"].append(
                scripted_cell(fault, s, seeds, use_true=True))
            out["scripted"][fault]["reactive"].append(
                scripted_cell(fault, s, seeds, use_true=False))
        print(f"[boundary] scripted {fault} done", flush=True)

    # trained policies under both localization regimes
    for method in POLICY_METHODS:
        out["policies"][method] = {}
        for loc, cfg in cfgs.items():
            pol = Policy(cfg, method, 1)
            out["policies"][method][loc] = {}
            for fault in FAULTS:
                srs = []
                for si, s in enumerate(SEVERITIES):
                    seeds = [scenario_seed(DEFAULT, "boundary", fault, si, k)
                             for k in range(N_EPS)]
                    srs.append(policy_cell(cfg, pol, fault, s, seeds))
                out["policies"][method][loc][fault] = srs
            print(f"[boundary] policy {method}/{loc} done", flush=True)

    with open(os.path.join(OUT_DIR, "boundary.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"[boundary] wrote {OUT_DIR}/boundary.json", flush=True)


if __name__ == "__main__":
    main()
