"""Privileged geometric expert (A* on an inflated occupancy grid + pure
pursuit). Used ONLY for corpus collection and environment feasibility
checks -- never as a learned method or baseline. It reads the true layout,
which no deployable policy can."""
from __future__ import annotations

import heapq
import math

import numpy as np

from ..config import Config
from . import geometry as geo


class PrivilegedPlanner:
    def __init__(self, cfg: Config, grid_n: int = 64, inflate: float = 0.24,
                 lookahead: float = 0.35, replan_every: int = 25):
        self.cfg = cfg
        self.N = grid_n
        self.inflate = inflate
        self.lookahead = lookahead
        self.replan_every = replan_every
        self._path: list[np.ndarray] | None = None
        self._age = 10 ** 9

    def reset(self):
        self._path = None
        self._age = 10 ** 9

    # ------------------------------------------------------------------ #
    def _plan(self, env) -> list[np.ndarray] | None:
        lay = env.layout
        half = self.cfg.sim.arena_half
        N = self.N
        xs = np.linspace(-half, half, N)
        gx, gy = np.meshgrid(xs, xs, indexing="ij")
        occ = np.zeros((N, N), dtype=bool)
        pts = np.stack([gx.ravel(), gy.ravel()], axis=1)
        # vectorised clearance of all grid points
        cl = np.full(len(pts), np.inf)
        for c, r in zip(lay.centers, lay.radii):
            cl = np.minimum(cl, np.linalg.norm(pts - c[None, :], axis=1) - r)
        for a, b in zip(lay.seg_a, lay.seg_b):
            e = b - a
            ee = max(float(e @ e), 1e-12)
            u = np.clip((pts - a[None, :]) @ e / ee, 0.0, 1.0)
            closest = a[None, :] + u[:, None] * e[None, :]
            cl = np.minimum(cl, np.linalg.norm(pts - closest, axis=1))
        occ = (cl < self.inflate).reshape(N, N)

        def idx(p):
            return (int(np.clip((p[0] + half) / (2 * half) * (N - 1), 0, N - 1)),
                    int(np.clip((p[1] + half) / (2 * half) * (N - 1), 0, N - 1)))

        s, g = idx(env.pos), idx(lay.goal)
        occ[s] = occ[g] = False
        pq = [(0.0, s)]
        came = {s: None}
        cost = {s: 0.0}
        while pq:
            _, cur = heapq.heappop(pq)
            if cur == g:
                break
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == dy == 0:
                        continue
                    nb = (cur[0] + dx, cur[1] + dy)
                    if not (0 <= nb[0] < N and 0 <= nb[1] < N) or occ[nb]:
                        continue
                    c = cost[cur] + math.hypot(dx, dy)
                    if c < cost.get(nb, 1e18):
                        cost[nb] = c
                        came[nb] = cur
                        heapq.heappush(
                            pq, (c + math.hypot(g[0] - nb[0], g[1] - nb[1]),
                                 nb))
        if g not in came:
            return None
        path, cur = [], g
        while cur is not None:
            path.append(np.array([xs[cur[0]], xs[cur[1]]]))
            cur = came[cur]
        path.reverse()
        return path

    # ------------------------------------------------------------------ #
    def act(self, env, rng: np.random.Generator | None = None,
            noise: float = 0.0) -> np.ndarray:
        if self._path is None or self._age >= self.replan_every:
            self._path = self._plan(env) or self._path
            self._age = 0
        self._age += 1
        if self._path is None:
            return np.array([-1.0, 0.0])
        d = [float(np.linalg.norm(p - env.pos)) for p in self._path]
        j = int(np.argmin(d))
        while (j < len(self._path) - 1 and
               np.linalg.norm(self._path[j] - env.pos) < self.lookahead):
            j += 1
        tgt = self._path[j]
        ang = math.atan2(*(tgt - env.pos)[::-1]) - env.heading
        while ang > math.pi:
            ang -= 2 * math.pi
        while ang < -math.pi:
            ang += 2 * math.pi
        w = float(np.clip(2.5 * ang, -1.0, 1.0))
        v = 0.9 if abs(ang) < 0.5 else (0.3 if abs(ang) < 1.0 else 0.0)
        a = np.array([2 * v - 1.0, w])
        if rng is not None and noise > 0:
            a = a + rng.normal(0.0, [0.10 * noise, 0.15 * noise])
        return np.clip(a, -1.0, 1.0)
