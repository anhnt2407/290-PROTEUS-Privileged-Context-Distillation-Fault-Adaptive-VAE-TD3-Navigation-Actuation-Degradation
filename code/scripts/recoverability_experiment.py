#!/usr/bin/env python3
"""Perception motion-recoverability analysis: is the ego-motion a policy needs
to correct drifting proprioception recoverable from vision?

For two visual representations of consecutive depth frames --
  vae_flow : frame-to-frame difference of the frozen VAE posterior mean, mu_t
             - mu_{t-1} (the signal PROTEUS's adaptation module consumes)
  raw_flow : the raw depth frames themselves, [x_{t-1} | x_t | x_t - x_{t-1}]
-- we probe how well true angular velocity (rotation) and true forward velocity
(translation) are linearly and non-linearly decodable (5-fold CV R^2). We then
convert the best rotation R^2 into the heading error a policy would accumulate
by integrating that estimate over an episode, and compare it against the
heading error caused by the odometry bias it is meant to cancel -- quantifying
why exteroceptive correction is not viable on a forward depth sensor.

Writes results/recoverability/recoverability.json for make_figures.py.
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from proteus.config import DEFAULT, RESULTS_DIR, VAE_DIR
from proteus.env.faultnav import FaultNavEnv
from proteus.env.faults import FaultSpec, spec_from
from proteus.models.vae import AttentionVAE1D

from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

CFG = DEFAULT
OUT_DIR = os.path.join(RESULTS_DIR, "recoverability")
REGIMES = [("nominal", 0.0), ("encoder", 1.0), ("slip", 0.8)]
N_EPS = 70
HORIZON_TICKS = 200          # representative episode length for drift accounting


def load_vae():
    vae = AttentionVAE1D(CFG.vae, CFG.sim.n_beams)
    vae.load_state_dict(torch.load(os.path.join(VAE_DIR, "vae.pt"),
                                   map_location="cpu"))
    vae.eval()
    return vae


@torch.no_grad()
def collect(vae, fault, sev):
    env = FaultNavEnv(CFG, "train", seed=4)
    vae_flow, raw_flow, vr, wr = [], [], [], []
    for k in range(N_EPS):
        spec = FaultSpec() if fault == "nominal" else spec_from({fault: sev})
        obs = env.reset(spec=spec, scenario_seed=700000 + k)
        prev_raw = obs["profile"].astype(np.float64)
        prev_mu = vae(torch.as_tensor(obs["profile"]).unsqueeze(0))[1].squeeze(0).numpy()
        for t in range(120):
            a = np.array([0.5 + 0.3 * math.sin(t / 5),
                          0.8 * math.sin(t / 6 + k * 0.3)])
            obs, _, done, info = env.step(a)
            cur_raw = obs["profile"].astype(np.float64)
            cur_mu = vae(torch.as_tensor(obs["profile"]).unsqueeze(0))[1].squeeze(0).numpy()
            vae_flow.append(cur_mu - prev_mu)
            raw_flow.append(np.concatenate([prev_raw, cur_raw, cur_raw - prev_raw]))
            v_real, w_real = info["realized"]
            vr.append(v_real)
            wr.append(w_real)
            prev_raw, prev_mu = cur_raw, cur_mu
            if done:
                break
    return (np.array(vae_flow), np.array(raw_flow),
            np.array(vr), np.array(wr))


def probe(X, y, kind):
    if np.std(y) < 1e-6:
        return None
    est = (make_pipeline(StandardScaler(), Ridge(alpha=1.0)) if kind == "ridge"
           else make_pipeline(StandardScaler(),
                              MLPRegressor(hidden_layer_sizes=(64,),
                                           max_iter=300, alpha=1e-3,
                                           random_state=0)))
    return float(np.mean(cross_val_score(est, X, y, cv=4, scoring="r2")))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    vae = load_vae()
    out = {"regimes": [f"{f}@{s}" for f, s in REGIMES], "table": {},
           "std_w": None, "std_v": None, "drift": {}}
    for fault, sev in REGIMES:
        vae_flow, raw_flow, vr, wr = collect(vae, fault, sev)
        tag = f"{fault}@{sev}"
        row = {}
        for rep_name, X in (("vae_flow", vae_flow), ("raw_flow", raw_flow)):
            row[rep_name] = {
                "w_ridge": probe(X, wr, "ridge"),
                "w_mlp": probe(X, wr, "mlp"),
                "v_ridge": probe(X, vr, "ridge"),
                "v_mlp": probe(X, vr, "mlp"),
            }
        out["table"][tag] = row
        if fault == "nominal":
            out["std_w"] = float(np.std(wr))
            out["std_v"] = float(np.std(vr))
        print(f"[recover] {tag} done", flush=True)

    # Drift accounting: an unbiased rotation estimate with residual std sigma_w
    # integrated over T ticks random-walks the heading by sigma_w*dt*sqrt(T).
    # Compare to the systematic heading error from an uncorrected encoder bias.
    sw = out["std_w"]
    best_r2 = max(v for tag in out["table"]
                  for v in [out["table"][tag]["raw_flow"]["w_mlp"]]
                  if v is not None)
    sigma_w = sw * math.sqrt(max(0.0, 1.0 - best_r2))     # residual rotation std
    dt = CFG.sim.dt
    integ_heading_rms = sigma_w * dt * math.sqrt(HORIZON_TICKS)
    # encoder bias: measured w is scaled by (1 - enc_max*s); at s=1 the belief
    # loses that fraction of every turn -> heading error grows with total turn.
    enc_scale_err = CFG.fault.encoder_scale_max                     # fraction
    typical_total_turn = sw * dt * HORIZON_TICKS                     # ~ rad turned
    bias_heading_err = enc_scale_err * typical_total_turn
    out["drift"] = {
        "best_raw_w_r2": best_r2,
        "residual_sigma_w": sigma_w,
        "integrated_heading_rms_rad": integ_heading_rms,
        "encoder_bias_heading_err_rad": bias_heading_err,
        "horizon_ticks": HORIZON_TICKS,
    }
    with open(os.path.join(OUT_DIR, "recoverability.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"[recover] wrote {OUT_DIR}/recoverability.json", flush=True)


if __name__ == "__main__":
    main()
