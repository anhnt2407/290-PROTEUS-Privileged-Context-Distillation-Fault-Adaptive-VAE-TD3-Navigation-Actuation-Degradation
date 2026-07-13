"""Vectorised 2-D ray casting and clearance queries for FaultNav.

Primitives are circles (discs) and line segments (walls, bars). All queries
are broadcast over beams x primitives in NumPy; no per-ray Python loops.
"""
from __future__ import annotations

import numpy as np

_EPS = 1e-9


def ray_circles(origin: np.ndarray, dirs: np.ndarray,
                centers: np.ndarray, radii: np.ndarray) -> np.ndarray:
    """First-hit distances of B rays against C circles.

    origin: (2,), dirs: (B, 2) unit, centers: (C, 2), radii: (C,).
    Returns (B,) distances (inf where no hit).
    """
    B = dirs.shape[0]
    if centers.shape[0] == 0:
        return np.full(B, np.inf)
    oc = origin[None, :] - centers                      # (C, 2)
    b = dirs @ oc.T                                     # (B, C)  d . (o - c)
    c0 = np.einsum("ij,ij->i", oc, oc) - radii ** 2     # (C,)
    disc = b ** 2 - c0[None, :]                         # (B, C)
    hit = disc >= 0.0
    sq = np.sqrt(np.maximum(disc, 0.0))
    t1 = -b - sq                                        # near root
    t2 = -b + sq                                        # far root (inside circle)
    t = np.where(t1 > _EPS, t1, np.where(t2 > _EPS, t2, np.inf))
    t = np.where(hit, t, np.inf)
    return t.min(axis=1)


def ray_segments(origin: np.ndarray, dirs: np.ndarray,
                 seg_a: np.ndarray, seg_b: np.ndarray) -> np.ndarray:
    """First-hit distances of B rays against S segments.

    seg_a, seg_b: (S, 2) endpoints. Returns (B,) distances (inf where no hit).
    """
    B = dirs.shape[0]
    if seg_a.shape[0] == 0:
        return np.full(B, np.inf)
    e = seg_b - seg_a                                   # (S, 2)
    ao = origin[None, :] - seg_a                        # (S, 2)
    # Solve origin + t d = a + u e:  t d - u e = a - o
    denom = dirs[:, 0, None] * e[None, :, 1] - dirs[:, 1, None] * e[None, :, 0]
    denom = np.where(np.abs(denom) < _EPS, np.nan, denom)  # parallel -> no hit
    t = (ao[None, :, 0] * e[None, :, 1] - ao[None, :, 1] * e[None, :, 0]) / -denom
    u = (ao[None, :, 0] * dirs[:, 1, None] - ao[None, :, 1] * dirs[:, 0, None]) / -denom
    valid = (t > _EPS) & (u >= 0.0) & (u <= 1.0)
    t = np.where(valid, t, np.inf)
    t = np.where(np.isnan(t), np.inf, t)
    return t.min(axis=1)


def cast_profile(pos: np.ndarray, heading: float, fov_rad: float, n_beams: int,
                 range_max: float, centers: np.ndarray, radii: np.ndarray,
                 seg_a: np.ndarray, seg_b: np.ndarray) -> np.ndarray:
    """Depth profile over the field of view, capped at range_max.

    Beam 0 is the leftmost beam (+fov/2), matching an image scanned
    left-to-right from the robot's viewpoint.
    """
    angles = heading + np.linspace(fov_rad / 2.0, -fov_rad / 2.0, n_beams)
    dirs = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    d_c = ray_circles(pos, dirs, centers, radii)
    d_s = ray_segments(pos, dirs, seg_a, seg_b)
    return np.minimum(range_max, np.minimum(d_c, d_s))


def clearance_circles(pos: np.ndarray, centers: np.ndarray,
                      radii: np.ndarray) -> float:
    if centers.shape[0] == 0:
        return np.inf
    d = np.linalg.norm(centers - pos[None, :], axis=1) - radii
    return float(d.min())


def clearance_segments(pos: np.ndarray, seg_a: np.ndarray,
                       seg_b: np.ndarray) -> float:
    if seg_a.shape[0] == 0:
        return np.inf
    e = seg_b - seg_a                                   # (S, 2)
    ap = pos[None, :] - seg_a                           # (S, 2)
    ee = np.einsum("ij,ij->i", e, e)
    ee = np.where(ee < _EPS, _EPS, ee)
    u = np.clip(np.einsum("ij,ij->i", ap, e) / ee, 0.0, 1.0)
    closest = seg_a + u[:, None] * e
    d = np.linalg.norm(pos[None, :] - closest, axis=1)
    return float(d.min())


def min_clearance(pos: np.ndarray, centers: np.ndarray, radii: np.ndarray,
                  seg_a: np.ndarray, seg_b: np.ndarray) -> float:
    """Distance from a point to the nearest obstacle surface (walls included)."""
    return min(clearance_circles(pos, centers, radii),
               clearance_segments(pos, seg_a, seg_b))


def rect_segments(cx: float, cy: float, w: float, h: float,
                  angle: float) -> tuple[np.ndarray, np.ndarray]:
    """Four segments of a rotated rectangle centred at (cx, cy)."""
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, -s], [s, c]])
    corners = np.array([[-w / 2, -h / 2], [w / 2, -h / 2],
                        [w / 2, h / 2], [-w / 2, h / 2]]) @ R.T
    corners += np.array([cx, cy])
    a = corners
    b = np.roll(corners, -1, axis=0)
    return a, b


def segment_hits_circle(p1: np.ndarray, p2: np.ndarray, c: np.ndarray,
                        r: float) -> bool:
    """Does segment p1-p2 pass within r of point c? (spawn-path checks)"""
    e = p2 - p1
    ee = float(e @ e)
    if ee < _EPS:
        return bool(np.linalg.norm(p1 - c) <= r)
    u = float(np.clip((c - p1) @ e / ee, 0.0, 1.0))
    closest = p1 + u * e
    return bool(np.linalg.norm(closest - c) <= r)
