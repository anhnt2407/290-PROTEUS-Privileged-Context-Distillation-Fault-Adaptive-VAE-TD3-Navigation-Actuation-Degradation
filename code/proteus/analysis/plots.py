"""Publication figures. Every figure is written as PDF+PNG into both
code/figures and manuscript/figs."""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..config import (CODE_DIR, DEFAULT, EVAL_DIR, FIGURES_DIR, RUNS_DIR,
                      TRAIN_FAULT_TYPES, run_name)
from . import aggregate as agg

MS_FIGS = os.path.normpath(os.path.join(CODE_DIR, "..", "manuscript", "figs"))

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "figure.dpi": 200, "savefig.bbox": "tight", "axes.grid": True,
    "grid.alpha": 0.25, "grid.linewidth": 0.5, "lines.linewidth": 1.4,
})

STYLE = {
    "raw_dr":  dict(color="#9467bd", ls=":",  label="RAW-DR"),
    "nom":     dict(color="#7f7f7f", ls="-",  label="NOM"),
    "dr":      dict(color="#1f77b4", ls="-",  label="DR"),
    "teacher": dict(color="#2ca02c", ls="--", label="PROTEUS-T (oracle)"),
    "proteus": dict(color="#d62728", ls="-",  label="PROTEUS"),
}
FT_LABEL = {"gain_L": "left-gain loss", "gain_R": "right-gain loss",
            "bias": "yaw-rate bias", "latency": "command latency",
            "slip": "traction slip", "droop": "saturation droop",
            "deadzone": "deadzone (unseen)", "encoder": "encoder fault",
            "gain_L+bias": "gain+bias (seen pair)",
            "latency+slip": "latency+slip (unseen pair)"}


def _save(fig, name: str):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(MS_FIGS, exist_ok=True)
    for root in (FIGURES_DIR, MS_FIGS):
        fig.savefig(os.path.join(root, f"{name}.pdf"))
        fig.savefig(os.path.join(root, f"{name}.png"))
    plt.close(fig)
    print(f"[fig] {name}")


def _seed_band(ax, cells, method, ft, metric="success", **kw):
    g = cells[(cells.method == method) & (cells.fault_type == ft)]
    if g.empty:
        return
    p = g.pivot_table(index="severity", columns="seed", values=metric)
    x = p.index.values
    mean, std = p.mean(axis=1).values, p.std(axis=1).fillna(0).values
    st = dict(STYLE.get(method, {"color": "k", "ls": "-", "label": method}))
    st.update(kw)
    lbl = st.pop("label")
    ax.plot(x, mean, label=lbl, **st)
    ax.fill_between(x, mean - std, mean + std, color=st["color"], alpha=0.15,
                    lw=0)


# --------------------------------------------------------------------------- #
def fig_stress(cells: pd.DataFrame, methods: list[str],
               name="fig_stress", metric="success", ylabel="success rate"):
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.2), sharex=True,
                             sharey=True)
    for ax, ft in zip(axes.flat, TRAIN_FAULT_TYPES):
        for m in methods:
            _seed_band(ax, cells, m, ft, metric=metric)
        ax.set_title(FT_LABEL[ft])
        ax.set_ylim(-0.03, 1.03)
        ax.set_xlim(0, 1)
    for ax in axes[1]:
        ax.set_xlabel("fault severity $s$")
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(methods),
               bbox_to_anchor=(0.5, -0.035), frameon=False)
    fig.tight_layout()
    _save(fig, name)


def fig_learning(methods: list[str], seeds: list[int]):
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.2))
    ax0, ax1, ax2 = axes
    for m in methods:
        dfs = [agg.training_curves(m, s) for s in seeds]
        dfs = [d for d in dfs if d is not None]
        if not dfs:
            continue
        # align on episode index; moving-average return and success
        w = 40
        curves_r, curves_s = [], []
        L = min(len(d) for d in dfs)
        for d in dfs:
            r = d["ret"].rolling(w, min_periods=5).mean().values[:L]
            sr = d["success"].rolling(w, min_periods=5).mean().values[:L]
            curves_r.append(r)
            curves_s.append(sr)
        x = np.arange(L)
        st = STYLE.get(m, {"color": "k", "ls": "-", "label": m})
        for ax, curves in ((ax0, curves_r), (ax1, curves_s)):
            arr = np.stack(curves)
            ax.plot(x, np.nanmean(arr, 0), color=st["color"], ls=st["ls"],
                    label=st["label"])
            ax.fill_between(x, np.nanmean(arr, 0) - np.nanstd(arr, 0),
                            np.nanmean(arr, 0) + np.nanstd(arr, 0),
                            color=st["color"], alpha=0.15, lw=0)
    ax0.set_xlabel("episode")
    ax0.set_ylabel("return (mov. avg.)")
    ax1.set_xlabel("episode")
    ax1.set_ylabel("success rate (mov. avg.)")
    ax1.set_ylim(0, 1)
    # curriculum ceiling trace (teacher runs)
    for s in seeds:
        d = agg.load_json(os.path.join(RUNS_DIR, run_name("teacher", s),
                                       "summary.json"))
        if d is None:
            continue
        cur = np.array([(e, sm) for e, sm, _ in d["curriculum"]])
        ax2.plot(cur[:, 0], cur[:, 1], color="#2ca02c", alpha=0.7,
                 label="teacher" if s == seeds[0] else None)
        d2 = agg.load_json(os.path.join(RUNS_DIR, run_name("abl_unif", s),
                                        "summary.json"))
        if d2 is not None:
            cur2 = np.array([(e, sm) for e, sm, _ in d2["curriculum"]])
            ax2.plot(cur2[:, 0], cur2[:, 1], color="#ff7f0e", ls=":",
                     label="uniform (A1)" if s == seeds[0] else None)
    ax2.set_xlabel("episode")
    ax2.set_ylabel(r"severity ceiling $s_{\max}$")
    ax2.set_ylim(0, 1.05)
    ax2.legend(frameon=False)
    ax0.legend(frameon=False, fontsize=6)
    fig.tight_layout()
    _save(fig, "fig_learning")


def fig_onset(methods=("dr", "proteus"), onset_tick=80):
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.2))
    ax0, ax1, ax2 = axes
    # (a) goal-distance timeline, bias fault, mean over episodes/seeds
    for m in methods:
        gts = []
        for s in (1, 2, 3):
            p = os.path.join(EVAL_DIR, run_name(m, s), "onset_timelines.npz")
            if not os.path.exists(p):
                continue
            tl = np.load(p)
            gts.append(tl["bias_goal_dist"])
        if not gts:
            continue
        g = np.concatenate(gts, axis=0)
        st = STYLE[m]
        t = np.arange(g.shape[1]) / 10.0
        ax0.plot(t, np.nanmean(g, 0), color=st["color"], ls=st["ls"],
                 label=st["label"])
    ax0.axvline(onset_tick / 10.0, color="k", lw=0.8, ls="--")
    ax0.text(onset_tick / 10.0 + 0.2, ax0.get_ylim()[1] * 0.92, "onset",
             fontsize=7)
    ax0.set_xlabel("time [s]")
    ax0.set_ylabel("distance to goal [m]  (bias fault)")
    ax0.legend(frameon=False)
    # (b) context-error timeline (student), averaged over fault types
    errs = []
    for s in (1, 2, 3):
        p = os.path.join(EVAL_DIR, run_name("proteus", s),
                         "onset_timelines.npz")
        if not os.path.exists(p):
            continue
        tl = np.load(p)
        for ft in TRAIN_FAULT_TYPES:
            errs.append(tl[f"{ft}_c_err"])
    if errs:
        e = np.concatenate(errs, axis=0)
        t = np.arange(e.shape[1]) / 10.0
        ax1.plot(t, np.nanmean(e, 0), color="#d62728")
        q1, q3 = np.nanpercentile(e, [25, 75], axis=0)
        ax1.fill_between(t, q1, q3, color="#d62728", alpha=0.15, lw=0)
    ax1.axvline(onset_tick / 10.0, color="k", lw=0.8, ls="--")
    ax1.set_xlabel("time [s]")
    ax1.set_ylabel(r"context error $\|\hat{c}-c\|_2$")
    # (c) post-onset collision within 5 s, per fault type
    width = 0.38
    xs = np.arange(len(TRAIN_FAULT_TYPES))
    for k, m in enumerate(methods):
        vals = []
        for ft in TRAIN_FAULT_TYPES:
            per_seed = []
            for s in (1, 2, 3):
                st_ = agg.onset_stats(m, s)
                if st_ and ft in st_:
                    per_seed.append(st_[ft]["collision_within_5s"])
            vals.append(np.mean(per_seed) if per_seed else np.nan)
        ax2.bar(xs + (k - 0.5) * width, vals, width,
                color=STYLE[m]["color"], label=STYLE[m]["label"])
    ax2.set_xticks(xs)
    ax2.set_xticklabels([ft.replace("_", "-") for ft in TRAIN_FAULT_TYPES],
                        rotation=40, ha="right")
    ax2.set_ylabel("post-onset collisions (5 s)")
    ax2.legend(frameon=False)
    fig.tight_layout()
    _save(fig, "fig_onset")


def fig_identification(seed=1):
    from sklearn.manifold import TSNE
    from ..config import E_AXES
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.3))
    ax0, ax1, ax2 = axes
    stats = agg.identification_stats("proteus", seed)
    if stats:
        names = [n for n, v in stats["probe_r2"].items() if v is not None]
        vals = [stats["probe_r2"][n] for n in names]
        ax0.barh(np.arange(len(names)), vals, color="#1f77b4")
        ax0.set_yticks(np.arange(len(names)))
        ax0.set_yticklabels([n.replace("_", "-") for n in names])
        ax0.set_xlabel(r"linear-probe $R^2$ from $\hat{c}$")
        ax0.set_xlim(0, 1)
    p = os.path.join(EVAL_DIR, run_name("proteus", seed), "id_pairs.npz")
    if os.path.exists(p):
        pairs = np.load(p)["pairs"]
        c_hat, e = pairs[:, :8], pairs[:, 16:]
        # colour by dominant fault axis
        dom = np.argmax(np.abs(e), axis=1)
        dom[np.abs(e).max(axis=1) < 0.05] = -1
        sub = np.random.default_rng(0).choice(
            len(c_hat), size=min(1500, len(c_hat)), replace=False)
        emb = TSNE(n_components=2, random_state=0,
                   perplexity=30).fit_transform(c_hat[sub])
        cmap = plt.get_cmap("tab10")
        for j, name in enumerate(["nominal"] + E_AXES[:6]):
            mask = dom[sub] == (j - 1)
            if mask.sum() == 0:
                continue
            ax1.scatter(emb[mask, 0], emb[mask, 1], s=3,
                        color="0.7" if j == 0 else cmap(j - 1),
                        label=name.replace("_", "-"), alpha=0.7)
        ax1.legend(frameon=False, fontsize=5, markerscale=2, ncol=2)
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.set_title(r"t-SNE of $\hat{c}$")
        # severity gradient along one axis (gain_L)
        mask = (dom == 0)
        if mask.sum() > 30:
            sev = e[mask, 0]
            ax2.scatter(c_hat[mask, :2][:, 0], c_hat[mask, :2][:, 1], s=4,
                        c=sev, cmap="viridis")
            sm = plt.cm.ScalarMappable(cmap="viridis")
            sm.set_array(sev)
            fig.colorbar(sm, ax=ax2, label="left-gain severity")
            ax2.set_xlabel(r"$\hat{c}_1$")
            ax2.set_ylabel(r"$\hat{c}_2$")
            ax2.set_title("severity gradient")
    fig.tight_layout()
    _save(fig, "fig_identification")


def fig_compensation(cells: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.2))
    ax0, ax1, ax2 = axes
    # (a) counter-steer: mean commanded a_w vs signed bias severity
    for m in ("dr", "proteus", "teacher"):
        g = cells[(cells.method == m) & (cells.fault_type == "bias")]
        if g.empty or "mean_aw" not in g:
            continue
        p = g.groupby("severity")["mean_aw"].mean()
        st = STYLE[m]
        ax0.plot(p.index, p.values, marker="o", ms=3, color=st["color"],
                 ls=st["ls"], label=st["label"])
    ax0.set_xlabel("bias severity (episode-averaged $\\pm$)")
    ax0.set_ylabel(r"mean commanded $a_\omega$")
    ax0.legend(frameon=False, fontsize=6)
    # (b) counter-steer under left-gain loss
    for m in ("dr", "proteus", "teacher"):
        g = cells[(cells.method == m) & (cells.fault_type == "gain_L")]
        if g.empty:
            continue
        p = g.groupby("severity")["mean_aw"].mean()
        st = STYLE[m]
        ax1.plot(p.index, p.values, marker="o", ms=3, color=st["color"],
                 ls=st["ls"], label=st["label"])
    ax1.set_xlabel("left-gain severity")
    ax1.set_ylabel(r"mean commanded $a_\omega$")
    # (c) commanded speed vs macro severity (caution component)
    for m in ("dr", "proteus", "teacher"):
        g = cells[cells.method == m]
        if g.empty:
            continue
        p = g.groupby("severity")["mean_speed_cmd"].mean()
        st = STYLE[m]
        ax2.plot(p.index, p.values, marker="o", ms=3, color=st["color"],
                 ls=st["ls"], label=st["label"])
    ax2.set_xlabel("fault severity (macro over types)")
    ax2.set_ylabel("mean commanded speed [m/s]")
    fig.tight_layout()
    _save(fig, "fig_compensation")


def fig_ablation(auc_map: dict[str, tuple[float, float]],
                 coll_map: dict[str, tuple[float, float]]):
    order = ["dr", "abl_unif", "abl_ddpg", "teacher", "abl_offline",
             "abl_ff", "abl_w5", "abl_noflow", "proteus"]
    labels = {"dr": "DR (no ctx)", "abl_unif": "A1 uniform sev.",
              "abl_ddpg": "A5 DDPG", "teacher": "PROTEUS-T",
              "abl_offline": "A6 offline", "abl_ff": "A4 feed-fwd",
              "abl_w5": "A3 W=5", "abl_noflow": "A2 no latent flow",
              "proteus": "PROTEUS"}
    present = [m for m in order if m in auc_map]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4))
    for ax, dmap, ttl in ((axes[0], auc_map, "success AUC"),
                          (axes[1], coll_map, "collision AUC")):
        vals = [dmap[m][0] for m in present]
        errs = [dmap[m][1] for m in present]
        colors = ["#d62728" if m == "proteus" else
                  "#2ca02c" if m == "teacher" else "#1f77b4"
                  for m in present]
        ax.bar(np.arange(len(present)), vals, yerr=errs, capsize=2,
               color=colors)
        ax.set_xticks(np.arange(len(present)))
        ax.set_xticklabels([labels[m] for m in present], rotation=40,
                           ha="right")
        ax.set_ylabel(ttl)
    fig.tight_layout()
    _save(fig, "fig_ablation")


def fig_generalization(cells_unseen: pd.DataFrame, heldout: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.3))
    ax0, ax1, ax2 = axes
    # (a) unseen layout, macro success vs severity
    for m in ("dr", "teacher", "proteus"):
        g = cells_unseen[cells_unseen.method == m]
        if g.empty:
            continue
        p = g.groupby("severity")["success"].mean()
        st = STYLE[m]
        ax0.plot(p.index, p.values, marker="o", ms=3, color=st["color"],
                 ls=st["ls"], label=st["label"])
    ax0.set_ylim(0, 1)
    ax0.set_xlabel("severity (macro, unseen layouts)")
    ax0.set_ylabel("success rate")
    ax0.legend(frameon=False, fontsize=6)
    # (b) held-out fault type / unseen pair
    for ax, tag in ((ax1, "deadzone"), (ax2, "latency+slip")):
        for m in ("dr", "teacher", "proteus"):
            g = heldout[(heldout.method == m) & (heldout.fault_type == tag)]
            if g.empty:
                continue
            p = g.groupby("severity")["success"].mean()
            st = STYLE[m]
            ax.plot(p.index, p.values, marker="o", ms=3, color=st["color"],
                    ls=st["ls"], label=st["label"])
        ax.set_ylim(0, 1)
        ax.set_xlabel(f"severity ({FT_LABEL[tag]})")
        ax.set_ylabel("success rate")
    fig.tight_layout()
    _save(fig, "fig_generalization")


def fig_encoder_probe(heldout: pd.DataFrame):
    """Encoder-fault regime: odometry lies; latent flow should rescue."""
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    show = [("dr", STYLE["dr"]),
            ("proteus", STYLE["proteus"]),
            ("abl_noflow", dict(color="#ff7f0e", ls="--",
                                label="A2 proprio-only"))]
    for m, st in show:
        g = heldout[(heldout.method == m) & (heldout.fault_type == "encoder")]
        if g.empty:
            continue
        p = g.groupby("severity")["success"].mean()
        ax.plot(p.index, p.values, marker="o", ms=3, color=st["color"],
                ls=st["ls"], label=st["label"])
    ax.set_ylim(0, 1)
    ax.set_xlabel("encoder-fault severity (odometry corrupted)")
    ax.set_ylabel("success rate")
    ax.legend(frameon=False, fontsize=6)
    fig.tight_layout()
    _save(fig, "fig_encoder_probe")


def fig_intervention(cells: pd.DataFrame, inter: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.3))
    for ax, metric, ylabel in ((axes[0], "success", "success rate"),
                               (axes[1], "collision", "collision rate")):
        g = cells[cells.method == "proteus"].groupby("severity")[metric].mean()
        h = inter[inter.method == "proteus"].groupby("severity")[metric].mean()
        ax.plot(g.index, g.values, marker="o", ms=3, color="#d62728",
                label=r"PROTEUS ($\hat{c}$ live)")
        ax.plot(h.index, h.values, marker="s", ms=3, color="#8c564b",
                ls="--", label=r"clamped $\hat{c} \equiv c_{nom}$")
        ax.set_xlabel("fault severity (macro)")
        ax.set_ylabel(ylabel)
        ax.legend(frameon=False, fontsize=6)
        ax.set_ylim(-0.03, 1.03)
    fig.tight_layout()
    _save(fig, "fig_intervention")
