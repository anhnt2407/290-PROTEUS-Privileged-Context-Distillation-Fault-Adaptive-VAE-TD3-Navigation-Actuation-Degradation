#!/usr/bin/env python3
"""Resumable campaign orchestrator.

Stages: vae -> train -> distill -> eval -> figures. Every unit of work is
skipped when its output already exists, so the campaign can be re-launched
after interruption. Workers are separate processes, each limited to
cfg.rl.torch_threads BLAS threads."""
from __future__ import annotations

import argparse
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from proteus.config import (ABLATION_SEED, CORPUS_DIR, EVAL_DIR,   # noqa: E402
                            HEADLINE_SEEDS, METHODS, RUNS_DIR, VAE_DIR,
                            run_name)

TRAIN_METHODS = ["raw_dr", "nom", "dr", "teacher"]
TRAIN_ABLATIONS = ["abl_unif", "abl_ddpg"]
STUDENT_METHODS = ["proteus"]
STUDENT_ABLATIONS = ["abl_noflow", "abl_w5", "abl_ff", "abl_offline"]

EVAL_REGIMES = {
    "raw_dr": ["sweep", "sweep_unseen", "heldout", "onset", "latency"],
    "nom": ["sweep", "sweep_unseen", "heldout", "onset", "latency"],
    "dr": ["sweep", "sweep_unseen", "heldout", "onset", "latency"],
    "teacher": ["sweep", "sweep_unseen", "heldout", "onset", "latency",
                "intervention"],
    "proteus": ["sweep", "sweep_unseen", "heldout", "onset", "latency",
                "intervention"],
    "abl_unif": ["sweep"],
    "abl_ddpg": ["sweep"],
    "abl_noflow": ["sweep", "heldout", "latency"],
    "abl_w5": ["sweep", "latency"],
    "abl_ff": ["sweep", "latency"],
    "abl_offline": ["sweep", "latency"],
}
_REGIME_FILE = {"sweep": "sweep.json", "sweep_unseen": "sweep_unseen.json",
                "heldout": "heldout.json", "onset": "onset.json",
                "latency": "latency.json",
                "intervention": "intervention.json"}


def _train_one(method: str, seed: int):
    from proteus.train_teacher import train
    return train(method, seed)


def _distill_one(method: str, seed: int):
    from proteus.distill_student import distill
    return distill(method, seed)


def _eval_one(method: str, seed: int, regimes: list[str]):
    from proteus.evaluate import evaluate
    evaluate(method, seed, regimes=regimes)
    return f"eval {method}_s{seed} {regimes}"


def pending_train() -> list[tuple[str, int]]:
    jobs = [(m, s) for m in TRAIN_METHODS for s in HEADLINE_SEEDS]
    jobs += [(m, ABLATION_SEED) for m in TRAIN_ABLATIONS]
    return [(m, s) for m, s in jobs
            if not os.path.exists(os.path.join(RUNS_DIR, run_name(m, s),
                                               "agent.pt"))]


def pending_distill() -> list[tuple[str, int]]:
    jobs = [(m, s) for m in STUDENT_METHODS for s in HEADLINE_SEEDS]
    jobs += [(m, ABLATION_SEED) for m in STUDENT_ABLATIONS]
    out = []
    for m, s in jobs:
        done = os.path.exists(os.path.join(RUNS_DIR, run_name(m, s),
                                           "student.pt"))
        teacher_ok = os.path.exists(os.path.join(
            RUNS_DIR, run_name(METHODS[m].teacher, s), "agent.pt"))
        if not done and teacher_ok:
            out.append((m, s))
    return out


def pending_eval() -> list[tuple[str, int, list[str]]]:
    jobs = [(m, s) for m in TRAIN_METHODS + STUDENT_METHODS
            for s in HEADLINE_SEEDS]
    jobs += [(m, ABLATION_SEED) for m in TRAIN_ABLATIONS + STUDENT_ABLATIONS]
    out = []
    for m, s in jobs:
        spec = METHODS[m]
        trained = os.path.exists(os.path.join(
            RUNS_DIR, run_name(m, s),
            "student.pt" if spec.context == "student" else "agent.pt"))
        if not trained:
            continue
        missing = [r for r in EVAL_REGIMES[m]
                   if not os.path.exists(os.path.join(
                       EVAL_DIR, run_name(m, s), _REGIME_FILE[r]))]
        if missing:
            out.append((m, s, missing))
    return out


def run_pool(jobs, fn, workers: int, tag: str):
    if not jobs:
        print(f"[{tag}] nothing to do")
        return
    print(f"[{tag}] {len(jobs)} jobs on {workers} workers")
    failures = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fn, *j): j for j in jobs}
        for fut in as_completed(futs):
            j = futs[fut]
            try:
                fut.result()
                print(f"[{tag}] done {j}", flush=True)
            except Exception:
                failures.append(j)
                print(f"[{tag}] FAILED {j}\n{traceback.format_exc()}",
                      flush=True)
    if failures:
        raise RuntimeError(f"[{tag}] failures: {failures}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", nargs="*",
                    default=["vae", "train", "distill", "eval", "figures"])
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    if "vae" in args.stages:
        if not os.path.exists(os.path.join(VAE_DIR, "vae.pt")):
            from proteus.config import DEFAULT
            from proteus.pretrain_vae import collect_corpus, train_vae
            corpus = os.path.join(CORPUS_DIR, "corpus_s0.npz")
            if not os.path.exists(corpus):
                print("[vae] collecting corpus ...", flush=True)
                collect_corpus(DEFAULT, 0, corpus)
            print("[vae] training ...", flush=True)
            train_vae(DEFAULT, corpus, VAE_DIR, 0)
        else:
            print("[vae] exists, skipping")

    if "train" in args.stages:
        run_pool(pending_train(), _train_one, args.workers, "train")
    if "distill" in args.stages:
        run_pool(pending_distill(), _distill_one, args.workers, "distill")
    if "eval" in args.stages:
        run_pool(pending_eval(), _eval_one, args.workers, "eval")
    if "figures" in args.stages:
        import subprocess
        subprocess.run([sys.executable,
                        os.path.join(os.path.dirname(__file__),
                                     "make_figures.py")], check=True)


if __name__ == "__main__":
    main()
