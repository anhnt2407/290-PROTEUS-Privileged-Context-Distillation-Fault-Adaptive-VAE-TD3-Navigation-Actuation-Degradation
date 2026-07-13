# PROTEUS — Privileged Embodiment-Context Distillation for Fault-Adaptive VAE+TD3 Navigation under Actuation Degradation

Third member of the research family derived from the reference study (Lee et al.,
*VAE+DDPG: An Attention-Enhanced Variational Autoencoder for Deep Reinforcement
Learning-Based Autonomous Navigation in Low-Light Environments*, Advanced
Intelligent Systems, 2026):

| Project | Robustness axis | Mechanism |
|---|---|---|
| Reference | illumination | attention-enhanced VAE features + DDPG |
| 506-URSA | perception integrity | VAE-posterior uncertainty → adaptive-CVaR distributional critic |
| 507-TEMPO | world dynamics (moving obstacles) | temporal-attention VAE + collision forecast + predictive shield |
| **508-PROTEUS** | **embodiment (actuation faults)** | **privileged context distillation → online embodiment identification → context-conditioned TD3** |

**Idea.** All prior members assume the commanded twist is executed faithfully.
PROTEUS studies navigation when the *body* degrades — per-wheel gain loss,
yaw-rate trim bias, command latency, traction slip, saturation droop, deadzone —
and shows that a policy conditioned on an *online-identified embodiment context*
adapts (counter-steers, re-times, slows only when physically necessary) where
domain randomization merely averages. A privileged teacher learns with the true
fault vector under a self-paced severity curriculum; a deployable student infers
the context from the recent command–response history (proprioception) fused with
frame-to-frame motion of the frozen VAE latent (visual ego-motion evidence that
survives when odometry itself lies).

## Layout

```
code/
├── proteus/
│   ├── config.py              # every dial in one place + method registry
│   ├── env/                   # FaultNav simulator (geometry, faults, env)
│   ├── models/                # attention VAE, context modules, actor-critic
│   ├── agents/                # TD3(+DDPG) with context, replay structures
│   ├── pretrain_vae.py        # phase 0: corpus + VAE (then frozen)
│   ├── train_teacher.py       # phase 1: privileged TD3 + curriculum
│   ├── distill_student.py     # phase 2: DAgger-style context distillation
│   ├── evaluate.py            # stress sweeps, onset, held-out, intervention
│   └── analysis/              # aggregation, figures, LaTeX tables
├── scripts/
│   ├── run_experiments.py     # resumable parallel campaign orchestrator
│   ├── smoke_test.py          # end-to-end miniature pipeline (~3 min)
│   └── make_figures.py        # all manuscript figures + tables
└── tests/test_proteus.py      # unit tests (geometry, faults, models, agent)
manuscript/                    # IEEEtran journal paper
```

## Reproduce

```bash
cd code
python3 -m pytest tests -q          # unit tests
python3 scripts/smoke_test.py       # end-to-end miniature pipeline
python3 scripts/run_experiments.py --workers 12   # full campaign
```

The campaign is resumable; artefacts land in `code/results/` and figures/tables
are written into `manuscript/`.

## Method ladder

M0 RAW-DR (no VAE) · M1 NOM (nominal-trained, reference-style) · M2 DR
(fault-randomized, no context) · M3 PROTEUS-T (privileged oracle upper bound) ·
M4 PROTEUS (distilled student, deployable) — plus ablations A1 (uniform
severity instead of curriculum), A2 (proprio-only student), A3 (short window),
A4 (feed-forward student), A5 (DDPG backbone), A6 (offline distillation).
