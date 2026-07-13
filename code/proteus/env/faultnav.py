"""FaultNav: a self-contained differential-drive navigation simulator with a
physically explicit wheel-space command path, so actuation faults act where
real faults act.

Pipeline per control tick (10 Hz):
  action a in [-1,1]^2 -> twist (v, w) -> inverse kinematics -> commanded
  wheel rates -> WheelFaultChannel F_e -> realised wheel rates -> forward
  kinematics -> Euler integration with per-substep collision checks ->
  depth-profile sensing (ray cast + fixed mild degradation) -> noisy odometry.

Outcome codes follow the family convention: SUCCESS / COLLISION / TIMEOUT.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..config import Config, E_DIM
from . import geometry as geo
from .faults import FaultSpec, WheelFaultChannel

SUCCESS, COLLISION, TIMEOUT, RUNNING = 1, 2, 3, 0


@dataclass
class Layout:
    centers: np.ndarray        # (C, 2) disc centres (obstacles + legs)
    radii: np.ndarray          # (C,)
    seg_a: np.ndarray          # (S, 2) segment starts (walls + bars)
    seg_b: np.ndarray          # (S, 2) segment ends
    start: np.ndarray          # (2,)
    start_heading: float
    goal: np.ndarray           # (2,)


def _wall_segments(half: float) -> tuple[np.ndarray, np.ndarray]:
    c = np.array([[-half, -half], [half, -half], [half, half], [-half, half]])
    return c, np.roll(c, -1, axis=0)


class FaultNavEnv:
    """Single-robot goal navigation among static clutter under actuation faults."""

    def __init__(self, cfg: Config, layout_family: str = "train",
                 seed: int = 0):
        self.cfg = cfg
        self.family = layout_family
        self.rng = np.random.default_rng(seed)
        s = cfg.sim
        self._fov = math.radians(s.fov_deg)
        self._wheel_max = (s.v_max + s.w_max * s.half_track) / s.wheel_radius
        self._bias_factor = s.half_track / s.wheel_radius
        self.layout: Layout | None = None
        self.spec = FaultSpec()
        self._channel: WheelFaultChannel | None = None
        self.tick = 0

    # ------------------------------------------------------------------ #
    # Episode generation
    # ------------------------------------------------------------------ #
    def _sample_free(self, rng, clear, centers, radii, seg_a, seg_b, margin):
        half = self.cfg.sim.arena_half
        for _ in range(400):
            p = rng.uniform(-half + margin, half - margin, size=2)
            if geo.min_clearance(p, centers, radii, seg_a, seg_b) > clear:
                return p
        raise RuntimeError("could not sample a free pose")

    def _build_layout(self, rng: np.random.Generator) -> Layout:
        s = self.cfg.sim
        half = s.arena_half
        wall_a, wall_b = _wall_segments(half)
        centers, radii = [], []
        seg_a_list, seg_b_list = [wall_a], [wall_b]

        def place_disc(r: float):
            """Place a disc keeping a navigable gap to existing discs."""
            for _ in range(80):
                c = rng.uniform(-half + r + 0.30, half - r - 0.30, size=2)
                gap_ok = all(np.linalg.norm(c - c2) > r + r2 + 0.55
                             for c2, r2 in zip(centers, radii))
                wall_gap = half - np.max(np.abs(c)) - r
                if gap_ok and wall_gap > 0.45:
                    centers.append(c)
                    radii.append(r)
                    return

        if self.family == "train":
            n = int(rng.integers(s.n_obstacles_min, s.n_obstacles_max + 1))
            for _ in range(n):
                place_disc(float(rng.uniform(s.obst_r_min, s.obst_r_max)))
            for _ in range(s.n_thin_legs):
                place_disc(s.leg_radius)
        else:  # unseen family: bars forming partial corridors + discs
            for _ in range(2):
                cx, cy = rng.uniform(-half + 0.8, half - 0.8, size=2)
                ang = float(rng.uniform(0, math.pi))
                w = float(rng.uniform(1.0, 1.8))
                a, b = geo.rect_segments(cx, cy, w, 0.12, ang)
                seg_a_list.append(a)
                seg_b_list.append(b)
            for _ in range(3):
                place_disc(float(rng.uniform(s.obst_r_min, s.obst_r_max)))
            place_disc(s.leg_radius)

        centers = np.array(centers) if centers else np.zeros((0, 2))
        radii = np.array(radii) if radii else np.zeros(0)
        seg_a = np.concatenate(seg_a_list, axis=0)
        seg_b = np.concatenate(seg_b_list, axis=0)

        # start and goal with clearance and separation
        clear = s.robot_radius + s.spawn_clearance
        for _ in range(200):
            start = self._sample_free(rng, clear, centers, radii, seg_a, seg_b,
                                      margin=0.30)
            goal = self._sample_free(rng, s.goal_radius + 0.05, centers, radii,
                                     seg_a, seg_b, margin=0.35)
            d = float(np.linalg.norm(goal - start))
            if s.goal_dist_min <= d <= s.goal_dist_max:
                break
        heading = float(rng.uniform(-math.pi, math.pi))

        # guaranteed on-path obstacle so perception stays load-bearing
        if s.onpath_obstacle:
            direction = (goal - start) / max(1e-9, np.linalg.norm(goal - start))
            normal = np.array([-direction[1], direction[0]])
            for _ in range(60):
                along = rng.uniform(0.40, 0.60)
                lateral = rng.uniform(-0.30, 0.30)
                c = start + along * (goal - start) + lateral * normal
                r = float(rng.uniform(s.obst_r_min, 0.24))
                ok = (np.linalg.norm(c - start) > clear + r + 0.25 and
                      np.linalg.norm(c - goal) > s.goal_radius + r + 0.20 and
                      np.all(np.abs(c) < half - r - 0.15))
                if ok:
                    centers = np.concatenate([centers, c[None, :]], axis=0)
                    radii = np.concatenate([radii, [r]])
                    break
        return Layout(centers, radii, seg_a, seg_b, start, heading, goal)

    def reset(self, spec: FaultSpec | None = None,
              scenario_seed: int | None = None) -> dict:
        if scenario_seed is not None:
            self.rng = np.random.default_rng(scenario_seed)
        rng = self.rng
        self.layout = self._build_layout(rng)
        self.spec = spec if spec is not None else FaultSpec()
        self._channel = WheelFaultChannel(
            self.cfg.fault, self._wheel_max, self.spec, rng, self._bias_factor)
        self.pos = self.layout.start.copy()
        self.heading = self.layout.start_heading
        self.tick = 0
        self.outcome = RUNNING
        self._prev_goal_dist = self._goal_dist()
        self._path_len = 0.0
        self._straight = self._goal_dist()
        self._last_cmd = np.zeros(2)
        self._realized = np.zeros(2)          # realised (v, w) of last tick
        self._odom = np.zeros(2)
        self._min_clear_run = np.inf
        return self._observe()

    # ------------------------------------------------------------------ #
    # Stepping
    # ------------------------------------------------------------------ #
    def _goal_dist(self) -> float:
        return float(np.linalg.norm(self.layout.goal - self.pos))

    def _goal_angle(self) -> float:
        g = self.layout.goal - self.pos
        ang = math.atan2(g[1], g[0]) - self.heading
        while ang > math.pi:
            ang -= 2 * math.pi
        while ang < -math.pi:
            ang += 2 * math.pi
        return ang

    def _twist_to_wheels(self, v: float, w: float) -> np.ndarray:
        s = self.cfg.sim
        wl = (v - w * s.half_track) / s.wheel_radius
        wr = (v + w * s.half_track) / s.wheel_radius
        # curvature-preserving clip in wheel space
        m = max(abs(wl), abs(wr))
        if m > self._wheel_max:
            scale = self._wheel_max / m
            wl *= scale
            wr *= scale
        return np.array([wl, wr])

    def _wheels_to_twist(self, w: np.ndarray) -> tuple[float, float]:
        s = self.cfg.sim
        v = s.wheel_radius * (w[1] + w[0]) / 2.0
        omega = s.wheel_radius * (w[1] - w[0]) / (2.0 * s.half_track)
        return float(v), float(omega)

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, dict]:
        s, rc = self.cfg.sim, self.cfg.reward
        lay = self.layout
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        v_cmd = (a[0] + 1.0) / 2.0 * s.v_max         # forward-only
        w_cmd = a[1] * s.w_max
        wheels_cmd = self._twist_to_wheels(v_cmd, w_cmd)

        wheels_real = self._channel.apply(wheels_cmd, self.tick)
        # benign motor noise (always on, all methods)
        wheels_real = wheels_real * (
            1.0 + self.rng.normal(0.0, s.motor_noise, size=2))
        v_real, w_real = self._wheels_to_twist(wheels_real)

        # integrate with substep collision checks (no tunnelling)
        h = s.dt / s.substeps
        collided = False
        for _ in range(s.substeps):
            self.heading += w_real * h
            self.pos = self.pos + v_real * h * np.array(
                [math.cos(self.heading), math.sin(self.heading)])
            clear = geo.min_clearance(self.pos, lay.centers, lay.radii,
                                      lay.seg_a, lay.seg_b)
            if clear < s.robot_radius:
                collided = True
                break
        self._path_len += abs(v_real) * s.dt
        self.tick += 1

        # odometry (wheel encoders with noise; encoder fault scales them)
        enc_scale = self._channel.encoder_scale(self.tick - 1)
        wheels_meas = wheels_real * enc_scale + self.rng.normal(
            0.0, s.encoder_noise * self._wheel_max, size=2)
        v_meas, w_meas = self._wheels_to_twist(wheels_meas)
        self._odom = np.array([v_meas, w_meas])
        self._realized = np.array([v_real, w_real])

        # outcome
        goal_dist = self._goal_dist()
        clear = geo.min_clearance(self.pos, lay.centers, lay.radii,
                                  lay.seg_a, lay.seg_b) - s.robot_radius
        self._min_clear_run = min(self._min_clear_run, clear)
        done = False
        if collided:
            self.outcome, done = COLLISION, True
        elif goal_dist < s.goal_radius:
            self.outcome, done = SUCCESS, True
        elif self.tick >= s.horizon:
            self.outcome, done = TIMEOUT, True

        # reward (identical across all methods)
        progress = self._prev_goal_dist - goal_dist
        self._prev_goal_dist = goal_dist
        goal_angle = self._goal_angle()
        r = rc.k_progress * progress
        r += rc.k_heading * (1.0 - 2.0 * abs(goal_angle) / math.pi)
        r -= rc.k_angular * (w_cmd / s.w_max) ** 2
        if clear < rc.proximity_margin:
            frac = (rc.proximity_margin - max(clear, 0.0)) / rc.proximity_margin
            r -= rc.k_proximity * frac ** 2
        r -= rc.step_cost
        if self.outcome == SUCCESS:
            r += rc.r_goal
        elif self.outcome == COLLISION:
            r += rc.r_collision
        elif self.outcome == TIMEOUT:
            r += rc.r_timeout

        self._last_cmd = a.copy()
        obs = self._observe()
        info = {
            "outcome": self.outcome,
            "goal_dist": goal_dist,
            "clearance": clear,
            "path_len": self._path_len,
            "straight": self._straight,
            "realized": self._realized.copy(),
            "commanded": np.array([v_cmd, w_cmd]),
            "wheels_cmd": wheels_cmd,
            "min_clear_run": self._min_clear_run,
        }
        return obs, float(r), done, info

    # ------------------------------------------------------------------ #
    # Sensing
    # ------------------------------------------------------------------ #
    def _observe(self) -> dict:
        s = self.cfg.sim
        lay = self.layout
        d = geo.cast_profile(self.pos, self.heading, self._fov, s.n_beams,
                             s.range_max, lay.centers, lay.radii,
                             lay.seg_a, lay.seg_b)
        clean = 1.0 - d / s.range_max            # graded encoding (near = high)
        x = clean.copy()
        # fixed mild degradation: speckle, additive, dropout, quantisation
        x *= 1.0 + self.rng.normal(0.0, s.speckle_sigma, size=x.shape)
        x += self.rng.normal(0.0, s.additive_sigma, size=x.shape)
        drop = self.rng.random(x.shape) < s.dropout_prob
        x[drop] = 0.0
        q = 2 ** s.quantize_bits - 1
        x = np.round(np.clip(x, 0.0, 1.0) * q) / q

        goal_dist = self._goal_dist()
        max_d = 2 * s.arena_half * math.sqrt(2.0)
        obs = {
            "profile": x.astype(np.float32),
            "clean_profile": np.clip(clean, 0.0, 1.0).astype(np.float32),
            "goal": np.array([goal_dist / max_d,
                              self._goal_angle() / math.pi], dtype=np.float32),
            "odom": np.array([self._odom[0] / s.v_max,
                              self._odom[1] / s.w_max], dtype=np.float32),
            "last_action": self._last_cmd.astype(np.float32),
            "e_severity": self.spec.active(self.tick).astype(np.float32),
        }
        return obs

    # convenience for figures
    def render_top_down(self, ax, traj: np.ndarray | None = None,
                        title: str = ""):
        import matplotlib.patches as mpatches
        s = self.cfg.sim
        lay = self.layout
        half = s.arena_half
        ax.set_xlim(-half - 0.1, half + 0.1)
        ax.set_ylim(-half - 0.1, half + 0.1)
        ax.set_aspect("equal")
        for a, b in zip(lay.seg_a, lay.seg_b):
            ax.plot([a[0], b[0]], [a[1], b[1]], color="0.25", lw=1.4)
        for c, r in zip(lay.centers, lay.radii):
            ax.add_patch(mpatches.Circle(c, r, facecolor="0.6",
                                         edgecolor="0.3", lw=0.8))
        ax.add_patch(mpatches.Circle(lay.goal, s.goal_radius, facecolor="none",
                                     edgecolor="tab:green", lw=1.6, ls="--"))
        ax.plot(*lay.start, marker="o", color="tab:blue", ms=6)
        if traj is not None and len(traj) > 1:
            ax.plot(traj[:, 0], traj[:, 1], color="tab:blue", lw=1.4)
        ax.set_title(title, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
