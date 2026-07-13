#!/usr/bin/env python3
"""End-to-end smoke test on a miniature configuration.

Exercises: corpus -> VAE -> teacher (privileged TD3 + curriculum) ->
student distillation -> evaluation regimes -> aggregation. Writes to a
throwaway results tree and asserts every artefact exists."""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import proteus.config as C                                        # noqa: E402
from proteus.config import Config, METHODS                       # noqa: E402


def small_config() -> Config:
    cfg = Config()
    cfg.vae.corpus_steps = 2500
    cfg.vae.epochs = 3
    cfg.rl.total_steps = 2500
    cfg.rl.warmup_steps = 400
    cfg.rl.buffer_size = 20000
    cfg.distill.steps = 1200
    cfg.distill.warmup_steps = 100
    cfg.eval.severities = (0.0, 0.5, 1.0)
    cfg.eval.episodes_per_cell = 4
    cfg.eval.unseen_severities = (0.0, 1.0)
    cfg.eval.unseen_episodes_per_cell = 3
    cfg.eval.onset_episodes = 4
    return cfg


def main():
    t0 = time.time()
    torch.set_num_threads(4)
    tmp = tempfile.mkdtemp(prefix="proteus_smoke_")
    runs = os.path.join(tmp, "runs")
    evals = os.path.join(tmp, "eval")
    vae_dir = os.path.join(tmp, "vae")
    cfg = small_config()

    from proteus.pretrain_vae import collect_corpus, train_vae
    corpus = os.path.join(tmp, "corpus.npz")
    meta = collect_corpus(cfg, 0, corpus)
    print("corpus:", meta)
    train_vae(cfg, corpus, vae_dir, 0)
    assert os.path.exists(os.path.join(vae_dir, "vae.pt"))

    # redirect the module-level VAE path used by Featurizer
    import proteus.features as F
    F.VAE_DIR = vae_dir

    from proteus.train_teacher import train
    train("teacher", 1, cfg=cfg, out_root=runs)
    train("dr", 1, cfg=cfg, out_root=runs)
    assert os.path.exists(os.path.join(runs, "teacher_s1", "agent.pt"))

    from proteus.distill_student import distill
    distill("proteus", 1, cfg=cfg, out_root=runs)
    assert os.path.exists(os.path.join(runs, "proteus_s1", "student.pt"))

    from proteus.evaluate import evaluate
    for m in ("dr", "teacher", "proteus"):
        evaluate(m, 1, cfg=cfg, run_root=runs, out_root=evals,
                 regimes=["sweep", "sweep_unseen", "heldout", "onset",
                          "latency", "intervention"])
    for m in ("dr", "teacher", "proteus"):
        for f in ("sweep.json", "sweep_unseen.json", "heldout.json",
                  "onset.json", "latency.json"):
            assert os.path.exists(os.path.join(evals, f"{m}_s1", f)), (m, f)
    assert os.path.exists(os.path.join(evals, "proteus_s1", "id_pairs.npz"))
    assert os.path.exists(os.path.join(evals, "proteus_s1",
                                       "intervention.json"))

    from proteus.analysis import aggregate as agg
    df = agg.load_all_sweeps([("dr", 1), ("teacher", 1), ("proteus", 1)],
                             eval_dir=evals)
    cells = agg.cell_means(df)
    auc = agg.auc_table(cells)
    print(auc[auc.fault_type == "macro"])
    ostats = agg.onset_stats("proteus", 1, eval_dir=evals)
    print("onset:", json.dumps(ostats, indent=1)[:300])
    ids = agg.identification_stats("proteus", 1, eval_dir=evals)
    print("identification:", ids and {k: ids[k] for k in
                                      ("n_pairs", "context_err", "cosine")})
    shutil.rmtree(tmp)
    print(f"SMOKE TEST PASSED in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
