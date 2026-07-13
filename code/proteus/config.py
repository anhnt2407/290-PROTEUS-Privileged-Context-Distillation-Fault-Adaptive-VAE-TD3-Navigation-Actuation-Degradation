"""Central configuration for PROTEUS.

Every experiment is fully described by (MethodSpec, seed) on top of the frozen
defaults below, mirroring the convention of the sibling projects (506-URSA,
507-TEMPO) derived from the reference VAE+DDPG study.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(CODE_DIR, "results")
FIGURES_DIR = os.path.join(CODE_DIR, "figures")
RUNS_DIR = os.path.join(RESULTS_DIR, "runs")
EVAL_DIR = os.path.join(RESULTS_DIR, "eval")
VAE_DIR = os.path.join(RESULTS_DIR, "vae")
CORPUS_DIR = os.path.join(RESULTS_DIR, "corpus")

MASTER_SEED = 20260713


# --------------------------------------------------------------------------- #
# Simulator (FaultNav)
# --------------------------------------------------------------------------- #
@dataclass
class SimConfig:
    # Arena
    arena_half: float = 2.5          # half side of the square arena [m]
    n_obstacles_min: int = 2
    n_obstacles_max: int = 4
    obst_r_min: float = 0.15
    obst_r_max: float = 0.30
    n_thin_legs: int = 1             # thin desk-leg discs
    leg_radius: float = 0.06
    onpath_obstacle: bool = True     # guarantee one obstacle near the start-goal chord
    # Robot (TurtleBot3 waffle-class differential drive)
    robot_radius: float = 0.16
    wheel_radius: float = 0.033      # r_w [m]
    half_track: float = 0.1435       # b [m] (axle half-width)
    v_max: float = 0.26              # [m/s]
    w_max: float = 1.82              # [rad/s]
    dt: float = 0.10                 # control period (10 Hz)
    substeps: int = 4                # integration substeps with collision checks
    motor_noise: float = 0.01        # benign multiplicative wheel noise (always on)
    encoder_noise: float = 0.02      # odometry wheel-rate noise (fraction of wheel max)
    # Sensor: forward depth profile
    fov_deg: float = 120.0
    n_beams: int = 128
    range_max: float = 3.5
    # Fixed mild sensor degradation (NOT the stress axis; that is URSA's)
    speckle_sigma: float = 0.03
    additive_sigma: float = 0.02
    dropout_prob: float = 0.01
    quantize_bits: int = 8
    # Episode
    goal_dist_min: float = 1.2
    goal_dist_max: float = 2.5
    goal_radius: float = 0.30
    spawn_clearance: float = 0.16    # extra clearance beyond robot radius at spawn
    horizon: int = 300               # ticks (30 s)


# --------------------------------------------------------------------------- #
# Fault taxonomy (the PROTEUS stress axis)
# --------------------------------------------------------------------------- #
# e-vector layout (normalised severities; bias is signed):
E_AXES = ["gain_L", "gain_R", "bias", "latency", "slip", "droop", "deadzone", "encoder"]
E_DIM = len(E_AXES)

# Fault types available during TRAINING (deadzone and encoder are held out).
TRAIN_FAULT_TYPES = ["gain_L", "gain_R", "bias", "latency", "slip", "droop"]
HELDOUT_FAULT_TYPES = ["deadzone"]
# encoder-scale faults corrupt odometry: a special regime probing the
# latent-flow pathway of the adaptation module.


@dataclass
class FaultConfig:
    gain_max_loss: float = 0.65      # g = 1 - gain_max_loss * s  (worst gain 0.35)
    bias_yaw_max: float = 0.30       # |yaw-rate bias| at s=1 [rad/s]
    latency_max_ticks: int = 4       # L = round(4 s) control periods
    slip_sigma_max: float = 0.18     # multiplicative wheel noise std at s=1
    slip_burst_prob: float = 0.06    # per-tick burst probability at s=1
    slip_burst_gain: float = 0.35    # wheel gain during a slip burst
    slip_burst_mean_len: int = 3     # geometric mean burst duration [ticks]
    droop_max: float = 0.45          # wheel speed ceiling reduced by up to 45 %
    deadzone_max: float = 0.22       # commands below this fraction of wheel max -> 0
    encoder_scale_max: float = 0.45  # odometry wheel rates scaled by (1 - 0.45 s)
    # Training mixture over episodes
    p_nominal: float = 0.15
    p_single: float = 0.55
    p_compound: float = 0.20
    p_onset: float = 0.10
    single_sev_min: float = 0.15
    compound_sev_frac: float = 0.80  # compound severities ~ U(min, 0.8*s_max)
    onset_sev_min_frac: float = 0.50
    onset_tick_min: int = 50
    onset_tick_max: int = 150


@dataclass
class CurriculumConfig:
    enabled: bool = True
    s_max_init: float = 0.25
    s_max_floor: float = 0.25
    s_max_ceiling: float = 1.0
    ema_alpha: float = 0.05
    check_every: int = 20            # episodes
    grow_thresh: float = 0.60
    shrink_thresh: float = 0.30
    grow_step: float = 0.05
    shrink_step: float = 0.025


# --------------------------------------------------------------------------- #
# Perception (inherited attention-enhanced VAE, 1-D port, frozen after pretrain)
# --------------------------------------------------------------------------- #
@dataclass
class VAEConfig:
    latent_dim: int = 32
    base_channels: int = 16
    beta: float = 0.01               # KL weight
    free_bits: float = 0.05          # nats per latent dim
    lr: float = 1e-3
    batch_size: int = 256
    epochs: int = 25
    corpus_steps: int = 30000        # scripted-explorer corpus size
    corpus_seed: int = 0
    scripted_frac: float = 0.7       # potential-field explorer vs random actions


# --------------------------------------------------------------------------- #
# Control (TD3 backbone + privileged context / distilled adaptation)
# --------------------------------------------------------------------------- #
@dataclass
class RLConfig:
    total_steps: int = 90000
    warmup_steps: int = 2500
    batch_size: int = 256
    buffer_size: int = 200000
    gamma: float = 0.99
    tau: float = 0.003
    actor_lr: float = 1e-4
    critic_lr: float = 1e-3
    context_lr: float = 1e-4         # privileged encoder (updated with the actor)
    hidden: int = 256
    # exploration: Ornstein-Uhlenbeck (family convention) with linear decay
    expl_sigma: float = 0.30         # initial OU sigma
    expl_sigma_end: float = 0.10
    expl_decay_steps: int = 45000
    expl_theta: float = 0.15
    policy_noise: float = 0.2        # TD3 target smoothing
    noise_clip: float = 0.5
    policy_delay: int = 2
    update_every: int = 1
    context_dim: int = 8
    raw_pool: int = 32               # raw baseline: profile average-pooled to 32 dims
    torch_threads: int = 2
    eval_every: int = 0              # in-training probes disabled (post-hoc eval)


@dataclass
class DistillConfig:
    steps: int = 20000
    warmup_steps: int = 200          # fill the history window before training
    buffer_size: int = 50000
    batch_size: int = 256
    lr: float = 3e-4
    window: int = 25                 # history window W [ticks]
    tcn_channels: int = 32
    latent_flow_dim: int = 4         # learned projection of consecutive-frame d(mu)


# --------------------------------------------------------------------------- #
# Reward (family-"A"-style shaping, identical across every method)
# --------------------------------------------------------------------------- #
@dataclass
class RewardConfig:
    k_progress: float = 6.0          # per metre of progress toward the goal
    k_heading: float = 0.05
    k_angular: float = 0.03
    k_proximity: float = 0.40
    proximity_margin: float = 0.30   # [m] clearance below which penalty grows
    step_cost: float = 0.01
    r_goal: float = 20.0
    r_collision: float = -30.0
    r_timeout: float = -3.0


# --------------------------------------------------------------------------- #
# Evaluation protocol
# --------------------------------------------------------------------------- #
@dataclass
class EvalConfig:
    severities: tuple = (0.0, 1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6, 1.0)
    episodes_per_cell: int = 40
    unseen_severities: tuple = (0.0, 1 / 3, 2 / 3, 1.0)
    unseen_episodes_per_cell: int = 30
    onset_severity: float = 0.8
    onset_tick: int = 80
    onset_episodes: int = 40
    compound_pairs: tuple = (("gain_L", "bias"), ("latency", "slip"))
    eval_seed_offset: int = 900000


# --------------------------------------------------------------------------- #
# Method registry
# --------------------------------------------------------------------------- #
@dataclass
class MethodSpec:
    name: str
    perception: str = "vae"          # "vae" | "raw"
    train_faults: str = "curriculum"  # "none" | "curriculum" | "uniform"
    context: str = "none"            # "none" | "privileged" | "student"
    backbone: str = "td3"            # "td3" | "ddpg"
    # student options (used when context == "student")
    teacher: str = ""                # run name of the teacher to distill from
    use_latent_flow: bool = True
    window: int = 25
    temporal: str = "tcn"            # "tcn" | "ff"
    distill_onpolicy: bool = True
    label: str = ""


METHODS: dict[str, MethodSpec] = {
    # --- headline ladder ---
    "raw_dr": MethodSpec("raw_dr", perception="raw", train_faults="curriculum",
                         label="RAW-DR"),
    "nom": MethodSpec("nom", train_faults="none", label="NOM (reference-style)"),
    "dr": MethodSpec("dr", train_faults="curriculum", label="DR"),
    "teacher": MethodSpec("teacher", train_faults="curriculum", context="privileged",
                          label="PROTEUS-T (oracle)"),
    "proteus": MethodSpec("proteus", context="student", teacher="teacher",
                          label="PROTEUS"),
    # --- ablations ---
    "abl_unif": MethodSpec("abl_unif", train_faults="uniform", context="privileged",
                           label="A1 no-curriculum"),
    "abl_ddpg": MethodSpec("abl_ddpg", train_faults="curriculum", context="privileged",
                           backbone="ddpg", label="A5 DDPG backbone"),
    "abl_noflow": MethodSpec("abl_noflow", context="student", teacher="teacher",
                             use_latent_flow=False, label="A2 proprio-only"),
    "abl_w5": MethodSpec("abl_w5", context="student", teacher="teacher", window=5,
                         label="A3 short window"),
    "abl_ff": MethodSpec("abl_ff", context="student", teacher="teacher", temporal="ff",
                         label="A4 feed-forward"),
    "abl_offline": MethodSpec("abl_offline", context="student", teacher="teacher",
                              distill_onpolicy=False, label="A6 offline distillation"),
}

HEADLINE_METHODS = ["raw_dr", "nom", "dr", "teacher", "proteus"]
HEADLINE_SEEDS = [1, 2, 3]
ABLATION_SEED = 1


@dataclass
class Config:
    sim: SimConfig = field(default_factory=SimConfig)
    fault: FaultConfig = field(default_factory=FaultConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    vae: VAEConfig = field(default_factory=VAEConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    distill: DistillConfig = field(default_factory=DistillConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    def to_dict(self):
        return asdict(self)


DEFAULT = Config()


def run_name(method: str, seed: int) -> str:
    return f"{method}_s{seed}"
