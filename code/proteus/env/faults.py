"""Actuation-fault taxonomy, the wheel-space fault operator F_e, episode-level
fault sampling, and the self-paced severity curriculum.

The e-vector (config.E_AXES) holds normalised severities in [0, 1] except the
signed bias axis in [-1, 1]:

    e = [gain_L, gain_R, bias, latency, slip, droop, deadzone, encoder]

Physical action of each axis (applied to commanded wheel rates, in order):
latency (FIFO of control periods) -> deadzone -> saturation droop ->
per-wheel gain -> differential trim bias -> slip (multiplicative noise +
Bernoulli traction bursts). The encoder axis corrupts odometry only.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from ..config import E_AXES, E_DIM, TRAIN_FAULT_TYPES, FaultConfig, CurriculumConfig

_IDX = {name: i for i, name in enumerate(E_AXES)}


@dataclass
class FaultSpec:
    """A fully instantiated episode fault: severities plus onset tick."""
    e: np.ndarray = field(default_factory=lambda: np.zeros(E_DIM, dtype=np.float64))
    onset_tick: int = 0                  # 0 => active from the first tick

    def active(self, tick: int) -> np.ndarray:
        return self.e if tick >= self.onset_tick else np.zeros_like(self.e)

    def describe(self) -> str:
        parts = [f"{n}={v:+.2f}" for n, v in zip(E_AXES, self.e) if abs(v) > 1e-9]
        tag = ",".join(parts) if parts else "nominal"
        if self.onset_tick > 0:
            tag += f"@t{self.onset_tick}"
        return tag


def spec_from(types_sev: dict[str, float], onset_tick: int = 0) -> FaultSpec:
    e = np.zeros(E_DIM, dtype=np.float64)
    for t, s in types_sev.items():
        e[_IDX[t]] = s
    return FaultSpec(e=e, onset_tick=onset_tick)


class WheelFaultChannel:
    """Stateful per-episode fault operator acting on commanded wheel rates.

    Wheel rates are (omega_L, omega_R) in rad/s; `wheel_max` is the healthy
    actuator ceiling. The channel owns the latency FIFO and the slip-burst
    state, which persist across ticks within an episode.
    """

    def __init__(self, fc: FaultConfig, wheel_max: float, spec: FaultSpec,
                 rng: np.random.Generator, bias_wheel_factor: float):
        self.fc = fc
        self.wheel_max = wheel_max
        self.spec = spec
        self.rng = rng
        # kinematic factor b / r_w: wheel-rate offset producing a unit yaw bias
        self._bias_wheel_factor = bias_wheel_factor
        self._queue: deque[np.ndarray] = deque()
        self._burst_left = np.zeros(2, dtype=np.int64)
        self._last_e = None
        self._last_lat = 0

    def _rebuild_queue(self, latency_ticks: int):
        old = list(self._queue)
        self._queue = deque(old[-latency_ticks:] if latency_ticks > 0 else [],
                            maxlen=None)
        while len(self._queue) < latency_ticks:
            self._queue.appendleft(np.zeros(2, dtype=np.float64))

    def apply(self, cmd: np.ndarray, tick: int) -> np.ndarray:
        """cmd: commanded (omega_L, omega_R). Returns realised wheel rates."""
        fc = self.fc
        e = self.spec.active(tick)
        # --- latency (integer control periods) ---
        lat = int(round(fc.latency_max_ticks * e[_IDX["latency"]]))
        if self._last_e is None or lat != self._last_lat:
            self._rebuild_queue(lat)
            self._last_lat = lat
        if lat > 0:
            self._queue.append(cmd.copy())
            w = self._queue.popleft()
        else:
            w = cmd.copy()
        # --- deadzone ---
        dz = fc.deadzone_max * e[_IDX["deadzone"]] * self.wheel_max
        if dz > 0.0:
            w = np.where(np.abs(w) < dz, 0.0, w)
        # --- saturation droop ---
        ceil = self.wheel_max * (1.0 - fc.droop_max * e[_IDX["droop"]])
        w = np.clip(w, -ceil, ceil)
        # --- per-wheel gain loss ---
        g = np.array([1.0 - fc.gain_max_loss * e[_IDX["gain_L"]],
                      1.0 - fc.gain_max_loss * e[_IDX["gain_R"]]])
        w = w * g
        # --- differential trim bias (signed): yaw-rate offset ---
        # yaw = r_w (w_R - w_L) / (2 b)  =>  +/- yaw_bias * (b / r_w) per wheel
        yaw_bias = fc.bias_yaw_max * e[_IDX["bias"]]
        w = w + np.array([-1.0, 1.0]) * yaw_bias * self._bias_wheel_factor
        # --- slip: multiplicative noise + Bernoulli traction bursts ---
        s_slip = e[_IDX["slip"]]
        if s_slip > 0.0:
            sigma = fc.slip_sigma_max * s_slip
            w = w * (1.0 + self.rng.normal(0.0, sigma, size=2))
            p_burst = fc.slip_burst_prob * s_slip
            for i in range(2):
                if self._burst_left[i] > 0:
                    self._burst_left[i] -= 1
                elif self.rng.random() < p_burst:
                    self._burst_left[i] = self.rng.geometric(
                        1.0 / fc.slip_burst_mean_len)
            w = np.where(self._burst_left > 0, w * fc.slip_burst_gain, w)
        self._last_e = e
        return w

    def encoder_scale(self, tick: int) -> float:
        e = self.spec.active(tick)
        return 1.0 - self.fc.encoder_scale_max * e[_IDX["encoder"]]


# --------------------------------------------------------------------------- #
# Episode-level sampling and the self-paced curriculum
# --------------------------------------------------------------------------- #
class FaultSampler:
    """Draws per-episode FaultSpecs for training.

    Mixture: nominal / single-fault / compound / sudden-onset, with severities
    bounded by the curriculum ceiling s_max (self-paced) or 1.0 (uniform DR).
    """

    def __init__(self, fc: FaultConfig, cc: CurriculumConfig,
                 mode: str = "curriculum"):
        assert mode in ("none", "curriculum", "uniform")
        self.fc = fc
        self.cc = cc
        self.mode = mode
        self.s_max = cc.s_max_init if mode == "curriculum" else 1.0
        self._ema = None
        self._since_check = 0
        self.history: list[tuple[int, float, float]] = []   # (episode, s_max, ema)
        self._episode = 0

    def _sev(self, rng, lo, hi):
        hi = max(lo + 1e-6, hi)
        return float(rng.uniform(lo, hi))

    def _signed(self, rng, s):
        return s * (1.0 if rng.random() < 0.5 else -1.0)

    def sample(self, rng: np.random.Generator) -> FaultSpec:
        if self.mode == "none":
            return FaultSpec()
        fc = self.fc
        u = rng.random()
        if u < fc.p_nominal:
            return FaultSpec()
        u -= fc.p_nominal
        if u < fc.p_single:
            t = TRAIN_FAULT_TYPES[rng.integers(len(TRAIN_FAULT_TYPES))]
            s = self._sev(rng, fc.single_sev_min, self.s_max)
            if t == "bias":
                s = self._signed(rng, s)
            return spec_from({t: s})
        u -= fc.p_single
        if u < fc.p_compound:
            ts = rng.choice(len(TRAIN_FAULT_TYPES), size=2, replace=False)
            spec = {}
            for ti in ts:
                t = TRAIN_FAULT_TYPES[ti]
                s = self._sev(rng, fc.single_sev_min,
                              fc.compound_sev_frac * self.s_max)
                spec[t] = self._signed(rng, s) if t == "bias" else s
            return spec_from(spec)
        # sudden onset
        t = TRAIN_FAULT_TYPES[rng.integers(len(TRAIN_FAULT_TYPES))]
        s = self._sev(rng, fc.onset_sev_min_frac * self.s_max, self.s_max)
        if t == "bias":
            s = self._signed(rng, s)
        tick = int(rng.integers(fc.onset_tick_min, fc.onset_tick_max + 1))
        return spec_from({t: s}, onset_tick=tick)

    def update(self, success: bool):
        """Self-paced ceiling update with hysteresis (per finished episode)."""
        self._episode += 1
        x = 1.0 if success else 0.0
        self._ema = x if self._ema is None else \
            (1 - self.cc.ema_alpha) * self._ema + self.cc.ema_alpha * x
        if self.mode != "curriculum":
            self.history.append((self._episode, self.s_max, self._ema))
            return
        self._since_check += 1
        if self._since_check >= self.cc.check_every:
            self._since_check = 0
            if self._ema >= self.cc.grow_thresh:
                self.s_max = min(self.cc.s_max_ceiling,
                                 self.s_max + self.cc.grow_step)
            elif self._ema <= self.cc.shrink_thresh:
                self.s_max = max(self.cc.s_max_floor,
                                 self.s_max - self.cc.shrink_step)
        self.history.append((self._episode, self.s_max, self._ema))
