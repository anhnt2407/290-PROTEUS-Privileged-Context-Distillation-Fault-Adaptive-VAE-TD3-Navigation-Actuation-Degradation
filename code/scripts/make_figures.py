#!/usr/bin/env python3
"""Generate every manuscript figure and table from campaign outputs.
Safe to run with partial results: each artefact is skipped when its inputs
are missing."""
from __future__ import annotations

import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from proteus.config import (ABLATION_SEED, DEFAULT, HEADLINE_METHODS,   # noqa: E402
                            HEADLINE_SEEDS, RUNS_DIR, VAE_DIR, METHODS,
                            run_name)
from proteus.env.faultnav import FaultNavEnv
from proteus.env.faults import FaultSpec, spec_from
from proteus.analysis import aggregate as agg
from proteus.analysis import plots
from proteus.analysis import tables

CFG = DEFAULT


# --------------------------------------------------------------------------- #
def fig_testbed():
    fig = plt.figure(figsize=(7.0, 4.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.25, 1.0])
    # (a) train layout, (b) unseen layout
    for k, fam in enumerate(("train", "unseen")):
        ax = fig.add_subplot(gs[0, k])
        env = FaultNavEnv(CFG, fam, seed=11 + k)
        env.reset(spec=FaultSpec(), scenario_seed=101 + 5 * k)
        env.render_top_down(ax, title=f"{fam} layout family")
    # (c) same open-loop command script under five embodiments
    ax = fig.add_subplot(gs[0, 2])
    faults = [("nominal", FaultSpec()),
              ("gain-L 0.8", spec_from({"gain_L": 0.8})),
              ("bias +0.8", spec_from({"bias": 0.8})),
              ("latency 1.0", spec_from({"latency": 1.0})),
              ("droop 1.0", spec_from({"droop": 1.0}))]
    cmap = plt.get_cmap("tab10")
    styles = [dict(color="0.15", lw=2.2, alpha=0.65, zorder=5),
              dict(color=cmap(1), lw=1.4, zorder=3),
              dict(color=cmap(2), lw=1.4, zorder=3),
              dict(color=cmap(3), lw=1.6, ls="--", zorder=6),
              dict(color=cmap(4), lw=1.8, ls=":", zorder=6)]
    env = FaultNavEnv(CFG, "train", seed=3)
    for j, (lbl, spec) in enumerate(faults):
        env.reset(spec=spec, scenario_seed=77)
        # overwrite clutter for a clean demo: drive from fixed pose
        env.layout.centers = np.zeros((0, 2))
        env.layout.radii = np.zeros(0)
        env.pos = np.array([-1.8, -1.4])
        env.heading = 0.35
        env._prev_goal_dist = env._goal_dist()
        traj = [env.pos.copy()]
        for t in range(90):
            _, _, done, _ = env.step(np.array([0.7, 0.0]))
            traj.append(env.pos.copy())
            if done:
                break
        traj = np.array(traj)
        ax.plot(traj[:, 0], traj[:, 1], label=lbl, **styles[j])
    ax.set_xlim(-2.6, 2.6)
    ax.set_ylim(-2.6, 2.6)
    ax.set_aspect("equal")
    ax.legend(fontsize=5.5, loc="upper left", frameon=False)
    ax.set_title("one command script, five bodies")
    ax.set_xticks([])
    ax.set_yticks([])
    # (d,e) depth profiles clean vs degraded; (f) fault chain sketch
    env = FaultNavEnv(CFG, "train", seed=5)
    obs = env.reset(spec=FaultSpec(), scenario_seed=55)
    ax = fig.add_subplot(gs[1, 0])
    beams = np.arange(CFG.sim.n_beams)
    ax.plot(beams, obs["clean_profile"], color="k", lw=1.0, label="clean")
    ax.plot(beams, obs["profile"], color="tab:red", lw=0.7, alpha=0.8,
            label="observed")
    ax.set_xlabel("beam index")
    ax.set_ylabel("graded depth")
    ax.legend(frameon=False)
    ax.set_title("depth profile (120$^\\circ$, 128 beams)")
    # (e) wheel command vs realised under gain_L fault
    ax = fig.add_subplot(gs[1, 1])
    env = FaultNavEnv(CFG, "train", seed=6)
    env.reset(spec=spec_from({"gain_L": 0.8}), scenario_seed=66)
    cmds, reals = [], []
    for t in range(60):
        _, _, done, info = env.step(np.array([0.6, 0.2 * math.sin(t / 6)]))
        cmds.append(info["commanded"])
        reals.append(info["realized"])
        if done:
            break
    cmds, reals = np.array(cmds), np.array(reals)
    t = np.arange(len(cmds)) / 10
    ax.plot(t, cmds[:, 1], color="k", lw=1.0, label=r"commanded $\omega$")
    ax.plot(t, reals[:, 1], color="tab:red", lw=1.0,
            label=r"realised $\omega$")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("yaw rate [rad/s]")
    ax.legend(frameon=False)
    ax.set_title("gain-L fault: command vs realised")
    # (f) severity scaling of the six axes (schematic curves)
    ax = fig.add_subplot(gs[1, 2])
    s = np.linspace(0, 1, 50)
    fc = CFG.fault
    ax.plot(s, 1 - fc.gain_max_loss * s, label="wheel gain $g$")
    ax.plot(s, fc.bias_yaw_max * s, label=r"bias $b_\omega$ [rad/s]")
    ax.plot(s, np.round(fc.latency_max_ticks * s) / 10, label="latency [s]")
    ax.plot(s, fc.slip_sigma_max * s, label=r"slip $\sigma$")
    ax.plot(s, 1 - fc.droop_max * s, label="speed ceiling")
    ax.set_xlabel("severity $s$")
    ax.legend(fontsize=5.5, frameon=False)
    ax.set_title("fault-axis scaling")
    fig.tight_layout()
    plots._save(fig, "fig_testbed")


def fig_reconstruction():
    from proteus.models.vae import AttentionVAE1D
    path = os.path.join(VAE_DIR, "vae.pt")
    if not os.path.exists(path):
        print("[fig_reconstruction] no VAE yet")
        return
    vae = AttentionVAE1D(CFG.vae, CFG.sim.n_beams)
    vae.load_state_dict(torch.load(path, map_location="cpu"))
    vae.eval()
    fig, axes = plt.subplots(3, 1, figsize=(3.6, 3.8), sharex=True)
    env = FaultNavEnv(CFG, "train", seed=21)
    for k, ax in enumerate(axes):
        obs = env.reset(spec=FaultSpec(), scenario_seed=210 + k)
        for _ in range(15 + 12 * k):
            obs, _, done, _ = env.step(np.array([0.5, 0.15]))
            if done:
                break
        with torch.no_grad():
            recon, _, _ = vae(torch.as_tensor(obs["profile"]).unsqueeze(0))
        b = np.arange(CFG.sim.n_beams)
        ax.plot(b, obs["clean_profile"], color="k", lw=1.0, label="clean")
        ax.plot(b, obs["profile"], color="0.6", lw=0.6, label="observed")
        ax.plot(b, recon.squeeze(0).numpy(), color="tab:red", lw=1.0,
                ls="--", label="reconstruction")
        ax.set_ylabel("depth")
        if k == 0:
            ax.legend(frameon=False, fontsize=6)
    axes[-1].set_xlabel("beam index")
    fig.tight_layout()
    plots._save(fig, "fig_reconstruction")


def fig_architecture():
    """Block schematic of the PROTEUS pipeline (pure matplotlib)."""
    fig, ax = plt.subplots(figsize=(7.0, 3.3))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")

    def box(x, y, w, h, text, fc="#eaf2fb", ec="#1f4e79", fs=7, ls="-"):
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec,
                                   lw=1.1, zorder=2, linestyle=ls))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, zorder=3)

    def arrow(x0, y0, x1, y1, text="", fs=6, color="0.2"):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.1))
        if text:
            ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.12, text, fontsize=fs,
                    ha="center", color=color)

    # environment (left), perception (top), context (middle/bottom), policy (right)
    box(0.2, 1.7, 1.8, 1.6,
        "FaultNav\nenvironment\n(wheel-space\nfault channel $F_e$)",
        fc="#fdf3e7", ec="#8a5a00")
    box(2.9, 3.5, 2.0, 1.1, "attention VAE\n(CBAM+FPN, frozen)",
        fc="#eaf7ea", ec="#1e6b1e")
    box(5.6, 2.7, 2.0, 1.0, "privileged encoder\n$g_\\phi(e_t) = c_t$",
        fc="#f2eafd", ec="#4b1a8b", ls="--")
    box(2.9, 0.4, 2.0, 1.1,
        "history window $W$\n$[a, \\tilde{o}, P\\Delta\\mu]_{t-W+1:t}$")
    box(5.6, 0.4, 2.0, 1.1, "adaptation TCN\n$A_\\theta(h_t) = \\hat{c}_t$",
        fc="#fdeaea", ec="#8b1a1a")
    box(8.1, 1.7, 1.8, 1.7,
        "TD3 actor $\\pi$\ntwin critics $Q_{1,2}$\n"
        "$[\\mu_t, g_t, \\tilde{o}_t,$\n$a_{t-1}, c]$",
        fc="#eaf2fb", ec="#1f4e79")
    arrow(2.0, 3.0, 2.9, 3.8, "depth profile $x_t$")
    arrow(2.0, 1.9, 2.9, 1.1, "commands $+$ odometry")
    arrow(4.9, 4.15, 8.5, 3.4, "$\\mu_t$")
    arrow(3.9, 3.5, 3.9, 1.5, "$\\Delta\\mu$", color="#8b1a1a")
    arrow(4.9, 0.95, 5.6, 0.95)
    arrow(7.6, 1.0, 8.2, 1.75, "$\\hat{c}_t$ (deploy)")
    arrow(7.6, 3.2, 8.35, 3.0, "$c_t$ (train)")
    arrow(6.6, 2.7, 6.6, 1.55, "", color="#4b1a8b")
    ax.text(6.72, 2.05, "distill $\\|\\hat{c}-c\\|^2$", fontsize=6,
            color="#4b1a8b")
    arrow(9.9, 2.55, 9.99, 2.55)
    ax.text(9.72, 2.75, "$a_t$", fontsize=8)
    ax.text(4.6, 4.82, "true fault $e_t$ (simulation only)", fontsize=6.5,
            color="#4b1a8b")
    arrow(5.6, 4.75, 6.6, 3.75, "", color="#4b1a8b")
    fig.tight_layout()
    plots._save(fig, "fig_architecture")


def fig_trajectories():
    """Qualitative: DR vs PROTEUS under a mid-episode gain-L onset."""
    from proteus.evaluate import Policy
    ok = all(os.path.exists(os.path.join(RUNS_DIR, run_name(m, 1),
                                         "agent.pt" if m != "proteus"
                                         else "student.pt"))
             for m in ("dr", "proteus"))
    if not ok:
        print("[fig_trajectories] policies missing")
        return
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.5))
    scen = [901, 907, 911]
    for ax, sc in zip(axes, scen):
        for m, color in (("dr", "#1f77b4"), ("proteus", "#d62728")):
            pol = Policy(CFG, m, 1)
            env = FaultNavEnv(CFG, "train", seed=1)
            spec = spec_from({"gain_L": 0.85}, onset_tick=60)
            obs = env.reset(spec=spec, scenario_seed=sc)
            pol.reset_state(1)
            traj = [env.pos.copy()]
            done = False
            import numpy as _np
            active = _np.ones(1, dtype=bool)
            while not done:
                a, _ = pol.act([obs], active)
                obs, _, done, info = env.step(a[0])
                traj.append(env.pos.copy())
            traj = np.array(traj)
            ax.plot(traj[:, 0], traj[:, 1], color=color, lw=1.4,
                    label=plots.STYLE[m]["label"])
            if len(traj) > 60:
                ax.plot(*traj[60], marker="x", color=color, ms=6, mew=1.6)
            mark = "*" if info["outcome"] == 1 else "s"
            ax.plot(*traj[-1], marker=mark, color=color, ms=7)
        env.render_top_down(ax, title=f"scenario {sc}")
    axes[0].legend(fontsize=6, frameon=False, loc="lower left")
    fig.tight_layout()
    plots._save(fig, "fig_trajectories")


# --------------------------------------------------------------------------- #
def main():
    fig_testbed()
    fig_reconstruction()
    fig_architecture()

    ms = [(m, s) for m in HEADLINE_METHODS for s in HEADLINE_SEEDS]
    df = agg.load_all_sweeps(ms)
    if df.empty:
        print("no sweep results yet; stopping after static figures")
        return
    cells = agg.cell_means(df)
    plots.fig_stress(cells, HEADLINE_METHODS)
    plots.fig_stress(cells, HEADLINE_METHODS, name="fig_stress_collision",
                     metric="collision", ylabel="collision rate")
    plots.fig_learning(["raw_dr", "nom", "dr", "teacher"], HEADLINE_SEEDS)
    plots.fig_onset()
    plots.fig_identification(seed=1)
    plots.fig_compensation(cells)

    abl = [(m, ABLATION_SEED) for m in
           ("abl_unif", "abl_ddpg", "abl_noflow", "abl_w5", "abl_ff",
            "abl_offline")]
    df_all = agg.load_all_sweeps(ms + abl)
    cells_all = agg.cell_means(df_all)
    auc_s = agg.auc_table(cells_all, "success")
    auc_c = agg.auc_table(cells_all, "collision")
    auc_map, coll_map = {}, {}
    for m in cells_all.method.unique():
        a = auc_s[(auc_s.method == m) & (auc_s.fault_type == "macro")]
        c = auc_c[(auc_c.method == m) & (auc_c.fault_type == "macro")]
        auc_map[m] = (a["auc_success"].mean(),
                      a["auc_success"].std() if len(a) > 1 else 0.0)
        coll_map[m] = (c["auc_collision"].mean(),
                       c["auc_collision"].std() if len(c) > 1 else 0.0)
    plots.fig_ablation(auc_map, coll_map)

    df_unseen = agg.load_all_sweeps(ms, fname="sweep_unseen.json")
    cells_unseen = agg.cell_means(df_unseen) if not df_unseen.empty \
        else df_unseen
    df_held = agg.load_all_sweeps(ms + [("abl_noflow", ABLATION_SEED)],
                                  fname="heldout.json")
    cells_held = agg.cell_means(df_held) if not df_held.empty else df_held
    if not df_unseen.empty and not df_held.empty:
        plots.fig_generalization(cells_unseen, cells_held)
        plots.fig_encoder_probe(cells_held)
    df_int = agg.load_all_sweeps([("proteus", s) for s in HEADLINE_SEEDS],
                                 fname="intervention.json")
    if not df_int.empty:
        plots.fig_intervention(cells, agg.cell_means(df_int))

    fig_trajectories()

    tables.tab_main(cells)
    tables.tab_ablation(cells_all)
    tables.tab_compute()
    tables.numbers_digest(
        cells, cells_unseen, cells_held,
        agg.cell_means(df_int) if not df_int.empty else df_int)


if __name__ == "__main__":
    main()
