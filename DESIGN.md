# PROTEUS — Design Document

> **Outcome note (post-campaign).** This document specifies the system as designed and built. The
> full campaign then produced an honest *negative/boundary* result rather than the hypothesized win,
> and the paper was reframed accordingly (title: *"PROTEUS: When Does Identifying the Body Help?
> Embodiment-Context Adaptation and Its Limits for Actuation-Fault-Robust Visual Navigation"*).
> **Finding:** the benefit of embodiment identification is gated by two conditions and a
> quasi-static depth-guided differential-drive robot meets neither across most of the taxonomy.
> (1) *Feedback compensability* — under reliable localization a fault-naive policy matches/beats the
> fault-aware ladder (macro AUC$_S$ NOM 0.735 vs DR 0.680 vs PROTEUS 0.610), and clamping the
> inferred context to nominal changes success by <2 pts (the pathway is inert). (2) *Exteroceptive
> observability* — the one regime that would benefit (odometry-corrupting faults under dead-reckoned
> localization; oracle 0.70–0.90 vs dead-reckon 0.08–0.23 at high severity) needs an independent
> motion reference a forward depth sensor cannot supply (rotation decodable only weakly, $R^2$≈0.60;
> translation not at all). The contribution is the **compensability–observability boundary** plus the
> FaultNav benchmark and the context-collapse engineering lessons. Everything below documents the
> as-built system; see `scripts/boundary_experiment.py`, `scripts/recoverability_experiment.py`, and
> the manuscript for the boundary experiments that located the result.

**Full title:** PROTEUS: Privileged Embodiment-Context Distillation for Fault-Adaptive Deep
Reinforcement Learning in Visual Autonomous Navigation under Actuation Degradation

**Lineage:** Third sibling derived from the reference study (Lee et al., *VAE+DDPG: An
Attention-Enhanced Variational Autoencoder for Deep Reinforcement Learning-Based Autonomous
Navigation in Low-Light Environments*, Advanced Intelligent Systems, 2026). The family now
covers three orthogonal robustness axes:

| Project | Stress axis | Mechanism | Backbone |
|---|---|---|---|
| Reference (published) | illumination | attention-VAE (CBAM+FPN) features | DDPG |
| 506-URSA | perception integrity (image corruption) | VAE-posterior uncertainty → adaptive-CVaR distributional critic | DDPG + QR |
| 507-TEMPO | world dynamics (moving obstacles) | temporal-attention VAE + latent collision forecast + predictive shield | DDPG (+TD3 check) |
| **508-PROTEUS** | **embodiment / actuation faults** | **privileged context → proprioceptive+latent-flow distilled online system identification → context-conditioned policy + self-paced severity curriculum** | **TD3** |

**Slogan (dual to URSA's):** URSA estimates what it does not know about the *world* and dares
less; PROTEUS estimates what its *body* has become and steers differently. Adaptation over
conservatism: do not merely slow down — re-identify and compensate.

## 1. Problem

All prior members assume the actuation channel is intact: the commanded twist is executed up to
benign zero-mean noise. Deployed wheeled robots violate this constantly: motor-driver gain loss,
battery sag, per-wheel wear/underinflation (systematic veer), miscalibrated trim (yaw-rate bias),
control-loop latency under CPU contention, traction loss (slip bursts), deadzone from gear
backlash, and speed-limit droop. Under such faults a fixed policy silently veers, oscillates,
overshoots, and collides even with perfect perception. The stressor lives in the *action*
channel, so perception-side remedies (reference, URSA) and world-prediction remedies (TEMPO)
cannot see it — the epistemic gap is about the *self*, not the scene.

## 2. Idea

Treat the embodiment as a latent context vector `e` (fault parameters). Learn navigation that is
*context-conditioned*: a policy π(z_vis, goal, proprio, c) where c = g(e) is a compact embedding
of the embodiment. Because `e` is unobservable at deployment, distill an **adaptation module**
that infers ĉ online purely from onboard evidence: the recent window of (commanded action,
measured odometry) pairs — the command–response fingerprint of the body — optionally
cross-checked with **ego-motion evidence in the frozen VAE latent space** (Δμ between consecutive
frames), which survives when odometry itself is faulted.

Three-phase training inheriting the reference two-stage pipeline:

- **Phase 0 (perception, inherited):** collect a scripted-explorer corpus; pretrain the
  attention-enhanced 1-D VAE (Conv1d residual + CBAM1d + FPN1d, β-VAE with free-bits KL) to
  reconstruct clean depth profiles from mildly degraded ones. Freeze; cache μ.
- **Phase 1 (teacher, privileged):** TD3 with ground-truth `e` → context encoder g_φ(e) = c,
  trained under a **self-paced fault-severity curriculum** (success-gated expansion of the
  severity ceiling with hysteresis).
- **Phase 2 (student, deployable):** freeze π and g_φ; roll out with the student's ĉ in the loop
  (DAgger-style on-policy distillation) and regress ĉ → c with a causal dilated TCN over a
  W=25-tick history of [a, odom, proj(Δμ)].

## 3. Simulator — FaultNav

Custom, self-contained NumPy raycast simulator (no ROS/Gazebo), following the family convention
but with a **physically explicit differential-drive command path** so faults act where real
faults act — in wheel space:

- Arena 5×5 m; walls + 3–5 random static discs (r 0.15–0.35) + 1–2 thin legs (r 0.06) + one
  guaranteed on-path obstacle; unseen-layout family (bars/corridor fragments) for generalization.
- TurtleBot3-waffle-class disc robot: r_robot 0.16 m, wheel radius 0.033 m, half-track 0.1435 m,
  v ∈ [0, 0.26] m/s, ω ∈ [−1.82, 1.82] rad/s, 10 Hz control, Euler substeps with per-substep
  collision checks.
- Sensor: forward 120° FOV, 128-beam depth profile, cap 3.5 m, graded encoding x = 1 − r/r_max;
  fixed mild degradation (speckle, additive noise, 1% dropout, 8-bit quantization) — NOT the
  stress axis (that is URSA's).
- **Command path:** twist → inverse kinematics → wheel-rate commands → fault operator F_e →
  realized wheel rates → forward kinematics → integration. Odometry = wheel rates + encoder
  noise (and, in one held-out regime, encoder scale fault).

**Fault taxonomy** (e ∈ R^8, each axis normalized to severity [0,1]):

| # | Fault | Physical cause | Effect |
|---|---|---|---|
| 1 | left-wheel gain loss g_L | motor wear / PWM droop | veer + speed loss |
| 2 | right-wheel gain loss g_R | idem | veer (other side) |
| 3 | yaw-rate bias b_ω | trim miscalibration | constant drift |
| 4 | command latency L (0–4 ticks) | CPU contention / bus delay | phase lag, oscillation |
| 5 | slip noise σ_slip + Bernoulli slip bursts | traction loss | stochastic tracking error |
| 6 | saturation droop | battery sag | max-speed reduction |
| 7 | deadzone δ (HELD OUT of training) | gear backlash | small commands ignored |
| 8 | encoder scale error (special regime) | encoder fault | odometry lies |

Training episodes mix: 15% nominal, 55% single-fault, 20% compound (2 faults), 10% sudden onset
mid-episode. Evaluation sweeps each type over 7 severity levels with the rest nominal, plus
compound and onset regimes.

## 4. Methods ladder

| ID | Name | Perception | Faults in training | Context |
|---|---|---|---|---|
| M0 | RAW-DR | raw profile ↓32 | randomized | none |
| M1 | NOM | frozen VAE μ | none (nominal) | none |
| M2 | DR | frozen VAE μ | randomized (curriculum) | none |
| M3 | PROTEUS-T | frozen VAE μ | curriculum | privileged c=g(e) (oracle, upper bound) |
| M4 | PROTEUS | frozen VAE μ | curriculum | distilled ĉ from history (deployable) |

Ablations: A1 uniform-severity DR instead of curriculum (context kept); A2 proprio-only student
(no latent-flow input) — contrasted under the encoder-fault regime; A3 short history W=5;
A4 feed-forward student (no temporal aggregation); A5 DDPG backbone swap; A6 offline (off-policy)
distillation instead of DAgger-style.

## 5. Analyses & figures

Stress curves/AUC per fault type; sudden-onset transients (collision-within-Δ after onset,
recovery time, ĉ convergence); unseen-fault (deadzone, latency+slip compound) and unseen-layout
generalization; identification quality (linear-probe R² per fault axis, t-SNE of ĉ, tracking
after onset); behavioral compensation (commanded wheel-asymmetry vs true gain imbalance —
counter-steering evidence; speed modulation vs severity — caution vs compensation decomposition
via c-clamp counterfactual, the analog of URSA's u-override intervention); learning curves with
curriculum ceiling trace; VAE reconstructions; compute cost.

Propositions: (1) least-squares identifiability of (g_L, g_R) from a W-tick window under
persistent excitation, error O(σ/√W); (2) value-difference bound linear in E‖ĉ − c‖ (Lipschitz
policy/critic argument) — together: the context error the student can achieve bounds the return
it gives up.

## 6. Repro & compute

3 seeds for M0–M4, 1 seed for ablations; 60k RL steps/run; frozen-VAE latent caching makes RL
updates MLP-only (CPU-friendly); vectorized eval. Campaign orchestrated by a resumable runner;
deployable locally (48 cores) and/or on the vdm machine (server5, 64 cores).
