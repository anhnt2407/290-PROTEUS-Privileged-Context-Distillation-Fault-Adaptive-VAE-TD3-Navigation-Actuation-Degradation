"""Aggregation of raw evaluation JSONs into tidy DataFrames, robustness
AUCs, onset-transient statistics, identification quality and the paper's
headline numbers (numbers.json)."""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from ..config import EVAL_DIR, RUNS_DIR, TRAIN_FAULT_TYPES, run_name

EP_METRICS = ["success", "collision", "ticks", "spl", "ret", "path_len",
              "min_clear", "mean_speed_cmd", "mean_speed_real", "osc",
              "c_err", "mean_aw"]


def load_sweep(method: str, seed: int, fname: str = "sweep.json",
               eval_dir: str = EVAL_DIR) -> pd.DataFrame | None:
    path = os.path.join(eval_dir, run_name(method, seed), fname)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    rows = []
    for key, cell in data["cells"].items():
        ft, si = key.rsplit("|", 1)
        for ep_i, ep in enumerate(cell["episodes"]):
            row = {"method": method, "seed": seed, "fault_type": ft,
                   "sev_idx": int(si), "severity": cell["severity"],
                   "episode": ep_i}
            for m in EP_METRICS:
                if m in ep:
                    row[m] = ep[m]
            rows.append(row)
    return pd.DataFrame(rows)


def load_all_sweeps(methods_seeds: list[tuple[str, int]],
                    fname: str = "sweep.json",
                    eval_dir: str = EVAL_DIR) -> pd.DataFrame:
    dfs = [load_sweep(m, s, fname, eval_dir) for m, s in methods_seeds]
    dfs = [d for d in dfs if d is not None]
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def cell_means(df: pd.DataFrame) -> pd.DataFrame:
    """Per (method, seed, fault_type, severity) means of episode metrics."""
    keys = ["method", "seed", "fault_type", "sev_idx", "severity"]
    metrics = [c for c in EP_METRICS if c in df.columns]
    return df.groupby(keys, as_index=False)[metrics].mean()


def auc_table(cells: pd.DataFrame, metric: str = "success") -> pd.DataFrame:
    """Trapezoidal area under the metric-vs-severity curve, per fault type,
    plus the macro average over types. Severity axis normalised to [0, 1]."""
    rows = []
    for (m, s, ft), g in cells.groupby(["method", "seed", "fault_type"]):
        g = g.sort_values("severity")
        auc = np.trapezoid(g[metric].values, g["severity"].values)
        rows.append({"method": m, "seed": s, "fault_type": ft,
                     f"auc_{metric}": auc})
    df = pd.DataFrame(rows)
    macro = (df.groupby(["method", "seed"], as_index=False)
             [f"auc_{metric}"].mean())
    macro["fault_type"] = "macro"
    return pd.concat([df, macro], ignore_index=True)


def bootstrap_ci(x: np.ndarray, n_boot: int = 2000, alpha: float = 0.05,
                 rng=None) -> tuple[float, float]:
    rng = rng or np.random.default_rng(0)
    boots = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    return (float(np.quantile(boots, alpha / 2)),
            float(np.quantile(boots, 1 - alpha / 2)))


def paired_test(cells_a: pd.DataFrame, cells_b: pd.DataFrame,
                metric: str = "success") -> dict:
    """Wilcoxon signed-rank over matched (fault_type, severity) cells,
    seed-averaged."""
    from scipy.stats import wilcoxon
    ka = cells_a.groupby(["fault_type", "sev_idx"])[metric].mean()
    kb = cells_b.groupby(["fault_type", "sev_idx"])[metric].mean()
    joined = pd.concat([ka, kb], axis=1, keys=["a", "b"]).dropna()
    stat, p = wilcoxon(joined["a"], joined["b"])
    return {"n_cells": int(len(joined)), "wilcoxon_stat": float(stat),
            "p_value": float(p),
            "mean_delta": float((joined["a"] - joined["b"]).mean())}


# --------------------------------------------------------------------------- #
# Onset transients
# --------------------------------------------------------------------------- #
def onset_stats(method: str, seed: int, onset_tick: int = 80,
                horizon: int = 300, eval_dir: str = EVAL_DIR) -> dict | None:
    run = run_name(method, seed)
    jpath = os.path.join(eval_dir, run, "onset.json")
    npath = os.path.join(eval_dir, run, "onset_timelines.npz")
    if not (os.path.exists(jpath) and os.path.exists(npath)):
        return None
    with open(jpath) as f:
        data = json.load(f)
    tl = np.load(npath)
    out = {}
    for ft, cell in data["cells"].items():
        eps = cell["episodes"]
        gd = tl[f"{ft}_goal_dist"]                      # (n, horizon) padded nan
        n = gd.shape[0]
        # collision within 50 ticks of onset
        coll_soon = 0
        succ = 0
        recovery = []
        for i, ep in enumerate(eps):
            end = ep["ticks"]
            collided = ep["outcome"] == "collision"
            if collided and onset_tick <= end <= onset_tick + 50:
                coll_soon += 1
            if ep["outcome"] == "success":
                succ += 1
            # recovery: progress rate back to >= 50 % of pre-onset average
            g = gd[i, :end]
            if end > onset_tick + 12:
                pre = -np.diff(g[10:onset_tick]).mean()
                post = -np.diff(g[onset_tick:])
                if pre > 1e-5 and len(post) > 10:
                    k = np.convolve(post, np.ones(10) / 10, mode="valid")
                    idx = np.where(k >= 0.5 * pre)[0]
                    if len(idx) and ep["outcome"] == "success":
                        recovery.append(float(idx[0]))
        out[ft] = {
            "n": n, "success": succ / n, "collision_within_5s": coll_soon / n,
            "recovery_ticks_median": (float(np.median(recovery))
                                      if recovery else None),
            "c_err_mean_post": float(np.nanmean(
                tl[f"{ft}_c_err"][:, onset_tick:])),
            "c_err_mean_pre": float(np.nanmean(
                tl[f"{ft}_c_err"][:, 10:onset_tick])),
        }
    return out


# --------------------------------------------------------------------------- #
# Identification quality (students)
# --------------------------------------------------------------------------- #
def identification_stats(method: str, seed: int, c_dim: int = 8,
                         eval_dir: str = EVAL_DIR) -> dict | None:
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import cross_val_score
    path = os.path.join(eval_dir, run_name(method, seed), "id_pairs.npz")
    if not os.path.exists(path):
        return None
    pairs = np.load(path)["pairs"]
    c_hat = pairs[:, :c_dim]
    c_true = pairs[:, c_dim:2 * c_dim]
    e = pairs[:, 2 * c_dim:]
    ctx_err = float(np.linalg.norm(c_hat - c_true, axis=1).mean())
    r2 = {}
    from ..config import E_AXES
    for j, name in enumerate(E_AXES):
        y = e[:, j]
        if np.std(y) < 1e-6:
            r2[name] = None
            continue
        scores = cross_val_score(Ridge(alpha=1.0), c_hat, y, cv=5,
                                 scoring="r2")
        r2[name] = float(np.mean(scores))
    cos = float(np.mean(
        np.sum(c_hat * c_true, axis=1) /
        (np.linalg.norm(c_hat, axis=1) * np.linalg.norm(c_true, axis=1)
         + 1e-9)))
    return {"n_pairs": int(len(pairs)), "context_err": ctx_err,
            "cosine": cos, "probe_r2": r2}


def load_json(path: str) -> dict | None:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def training_curves(method: str, seed: int,
                    runs_dir: str = RUNS_DIR) -> pd.DataFrame | None:
    path = os.path.join(runs_dir, run_name(method, seed), "episodes.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        eps = json.load(f)
    df = pd.DataFrame(eps)
    df["method"] = method
    df["seed"] = seed
    df["success"] = (df["outcome"] == "success").astype(float)
    return df
