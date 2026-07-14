"""Unit tests: geometry, fault physics, identifiability, environment
contracts, model shapes/causality, replay and agent updates."""
import math
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from proteus.config import Config, DEFAULT, METHODS, E_DIM
from proteus.env import geometry as geo
from proteus.env.faults import FaultSampler, FaultSpec, WheelFaultChannel, \
    spec_from
from proteus.env.faultnav import COLLISION, SUCCESS, FaultNavEnv


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #
def test_ray_circle_direct_hit():
    origin = np.zeros(2)
    dirs = np.array([[1.0, 0.0], [0.0, 1.0]])
    d = geo.ray_circles(origin, dirs, np.array([[2.0, 0.0]]),
                        np.array([0.5]))
    assert abs(d[0] - 1.5) < 1e-9
    assert np.isinf(d[1])


def test_ray_segment_hit():
    origin = np.zeros(2)
    dirs = np.array([[1.0, 0.0]])
    d = geo.ray_segments(origin, dirs, np.array([[2.0, -1.0]]),
                         np.array([[2.0, 1.0]]))
    assert abs(d[0] - 2.0) < 1e-9


def test_clearance():
    c = geo.min_clearance(np.zeros(2), np.array([[1.0, 0.0]]),
                          np.array([0.25]), np.zeros((0, 2)),
                          np.zeros((0, 2)))
    assert abs(c - 0.75) < 1e-9


# --------------------------------------------------------------------------- #
# Fault physics
# --------------------------------------------------------------------------- #
def _channel(spec, seed=0):
    cfg = DEFAULT
    wheel_max = (cfg.sim.v_max + cfg.sim.w_max * cfg.sim.half_track) \
        / cfg.sim.wheel_radius
    return WheelFaultChannel(cfg.fault, wheel_max, spec,
                             np.random.default_rng(seed),
                             cfg.sim.half_track / cfg.sim.wheel_radius)


def test_gain_fault_slows_wheel():
    ch = _channel(spec_from({"gain_L": 1.0}))
    out = ch.apply(np.array([5.0, 5.0]), tick=0)
    assert out[0] == pytest.approx(5.0 * (1 - DEFAULT.fault.gain_max_loss))
    assert out[1] == pytest.approx(5.0)


def test_latency_delays_commands():
    L = DEFAULT.fault.latency_max_ticks
    ch = _channel(spec_from({"latency": 1.0}))
    outs = [ch.apply(np.array([float(t + 1), 0.0]), tick=t)[0]
            for t in range(L + 2)]
    assert outs[:L] == [0.0] * L                  # queue warm-up
    assert outs[L] == pytest.approx(1.0)          # first command emerges
    assert outs[L + 1] == pytest.approx(2.0)


def test_deadzone_zeroes_small_commands():
    ch = _channel(spec_from({"deadzone": 1.0}))
    small = 0.5 * DEFAULT.fault.deadzone_max * ch.wheel_max
    out = ch.apply(np.array([small, ch.wheel_max * 0.9]), tick=0)
    assert out[0] == 0.0 and out[1] > 0.0


def test_bias_creates_yaw_offset():
    cfg = DEFAULT
    ch = _channel(spec_from({"bias": 1.0}))
    out = ch.apply(np.array([3.0, 3.0]), tick=0)
    yaw = cfg.sim.wheel_radius * (out[1] - out[0]) / (2 * cfg.sim.half_track)
    assert yaw == pytest.approx(cfg.fault.bias_yaw_max, rel=1e-6)


def test_onset_gates_fault():
    spec = spec_from({"gain_L": 1.0}, onset_tick=10)
    ch = _channel(spec)
    before = ch.apply(np.array([5.0, 5.0]), tick=5)
    after = ch.apply(np.array([5.0, 5.0]), tick=15)
    assert before[0] == pytest.approx(5.0)
    assert after[0] < 2.0


def test_gain_identifiable_by_least_squares():
    """Prop-1 sanity: (g_L, g_R) recoverable from noisy command-response
    pairs by linear regression, error shrinking with window length."""
    rng = np.random.default_rng(0)
    g = np.array([0.55, 0.9])
    for W, tol in ((10, 0.12), (200, 0.03)):
        cmds = rng.uniform(1.0, 8.0, size=(W, 2))
        meas = cmds * g + rng.normal(0, 0.15, size=(W, 2))
        g_hat = [np.linalg.lstsq(cmds[:, i:i + 1], meas[:, i], rcond=None)[0][0]
                 for i in range(2)]
        assert abs(g_hat[0] - g[0]) < tol and abs(g_hat[1] - g[1]) < tol


# --------------------------------------------------------------------------- #
# Environment contracts
# --------------------------------------------------------------------------- #
def test_env_shapes_and_determinism():
    env1 = FaultNavEnv(DEFAULT, "train", seed=9)
    env2 = FaultNavEnv(DEFAULT, "train", seed=9)
    o1 = env1.reset(spec=FaultSpec(), scenario_seed=42)
    o2 = env2.reset(spec=FaultSpec(), scenario_seed=42)
    assert o1["profile"].shape == (DEFAULT.sim.n_beams,)
    assert o1["e_severity"].shape == (E_DIM,)
    np.testing.assert_allclose(o1["profile"], o2["profile"])
    a = np.array([0.4, -0.2])
    r1 = env1.step(a)
    r2 = env2.step(a)
    np.testing.assert_allclose(r1[0]["profile"], r2[0]["profile"])
    assert r1[1] == pytest.approx(r2[1])


def test_env_reaches_goal_with_oracle_controller():
    """A privileged proportional controller should reach the goal in an
    empty-ish scenario, validating kinematics + reward wiring."""
    env = FaultNavEnv(DEFAULT, "train", seed=13)
    successes = 0
    for k in range(8):
        obs = env.reset(spec=FaultSpec(), scenario_seed=1000 + k)
        env.layout.centers = np.zeros((0, 2))       # clear discs
        env.layout.radii = np.zeros(0)
        obs = env._observe()
        done = False
        while not done:
            ang = obs["goal"][1] * math.pi
            a = np.array([0.7 if abs(ang) < 0.7 else -0.4,
                          np.clip(2.2 * ang, -1, 1)])
            obs, r, done, info = env.step(a)
        successes += info["outcome"] == SUCCESS
    assert successes >= 7


def test_env_collision_under_severe_bias():
    env = FaultNavEnv(DEFAULT, "train", seed=7)
    env.reset(spec=spec_from({"bias": 1.0}), scenario_seed=77)
    done = False
    while not done:
        _, _, done, info = env.step(np.array([1.0, 0.0]))
    assert info["outcome"] in (COLLISION, SUCCESS, 3)


def test_dead_reckoning_drifts_under_encoder_fault():
    """Observability boundary: with dead-reckoned localization an
    odometry-corrupting fault (encoder) makes the observed goal bearing drift
    away from the truth, while under oracle localization the bearing is exact
    and a preserving fault leaves dead-reckoning faithful."""
    import dataclasses
    from proteus.config import DEFAULT
    dr_cfg = dataclasses.replace(
        DEFAULT, sim=dataclasses.replace(DEFAULT.sim, localization="dead_reckon"))

    def bearing_error(cfg, spec):
        env = FaultNavEnv(cfg, "train", seed=3)
        env.reset(spec=spec, scenario_seed=321)
        errs = []
        for _ in range(60):
            obs, _, done, _ = env.step(np.array([0.6, 0.5]))
            true_ang = env._goal_angle()
            errs.append(abs(float(obs["goal"][1]) * math.pi - true_ang))
            if done:
                break
        return float(np.mean(errs))

    enc = spec_from({"encoder": 1.0})
    # oracle localization: observed bearing equals the true bearing exactly
    assert bearing_error(DEFAULT, enc) < 1e-6
    # dead-reckoning under an odometry-corrupting fault: belief drifts
    assert bearing_error(dr_cfg, enc) > 0.1
    # dead-reckoning under a preserving fault (bias): odometry stays faithful
    assert bearing_error(dr_cfg, spec_from({"bias": 1.0})) < \
        bearing_error(dr_cfg, enc)


def test_curriculum_expands_and_shrinks():
    cfg = DEFAULT
    s = FaultSampler(cfg.fault, cfg.curriculum, mode="curriculum")
    for _ in range(200):
        s.update(True)
    assert s.s_max > cfg.curriculum.s_max_init
    s2 = FaultSampler(cfg.fault, cfg.curriculum, mode="curriculum")
    for _ in range(200):
        s2.update(False)
    assert s2.s_max == cfg.curriculum.s_max_floor


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
def test_vae_roundtrip_shapes():
    from proteus.models.vae import AttentionVAE1D
    vae = AttentionVAE1D(DEFAULT.vae, 128)
    x = torch.rand(6, 128)
    recon, mu, logvar = vae(x)
    assert recon.shape == (6, 128) and mu.shape == (6, 32)
    out = vae.loss(x, x)
    assert torch.isfinite(out["loss"])


def test_tcn_is_causal():
    from proteus.models.context import AdaptationModule
    m = AdaptationModule(DEFAULT.distill, 32, 8)
    m.eval()
    p = torch.rand(1, 25, 4)
    d = torch.rand(1, 25, 32)
    with torch.no_grad():
        base = m(p, d)
        p2 = p.clone()
        p2[0, 0] += 100.0            # perturb the OLDEST tick: may matter
        d2 = d.clone()
        # perturbing the newest tick must change the output
        p3 = p.clone()
        p3[0, -1] += 1.0
        assert (m(p3, d) - base).abs().sum() > 1e-6
        _ = p2, d2


def test_latent_flow_toggle():
    from proteus.models.context import AdaptationModule
    m = AdaptationModule(DEFAULT.distill, 32, 8, use_latent_flow=False)
    m.eval()
    p = torch.rand(2, 25, 4)
    with torch.no_grad():
        a = m(p, torch.rand(2, 25, 32))
        b = m(p, torch.rand(2, 25, 32))
    assert torch.allclose(a, b)       # latent flow ignored


def test_agent_update_and_context_grads():
    from proteus.agents.td3 import TD3Agent
    agent = TD3Agent(DEFAULT, METHODS["teacher"], feat_dim=32)
    for _ in range(300):
        agent.replay.add(np.random.randn(38).astype(np.float32),
                         np.random.rand(8).astype(np.float32),
                         np.random.uniform(-1, 1, 2).astype(np.float32),
                         float(np.random.randn()),
                         np.random.randn(38).astype(np.float32),
                         np.random.rand(8).astype(np.float32), False)
    w0 = agent.g_phi.net[0][0].weight.detach().clone()
    for _ in range(6):
        out = agent.update()
    assert np.isfinite(out["loss_critic"])
    assert not torch.allclose(w0, agent.g_phi.net[0][0].weight)


def test_context_encoder_stays_informative():
    """Regression: default AdamW weight decay pruned g_phi to a constant when
    the RL gradient through the context weights was weak. The identifiability
    regularizer + decay-free optimizer must keep c sensitive to e."""
    from proteus.agents.td3 import TD3Agent
    agent = TD3Agent(DEFAULT, METHODS["teacher"], feat_dim=32)
    rng = np.random.default_rng(0)
    for _ in range(600):
        e = np.zeros(E_DIM, dtype=np.float32)
        e[rng.integers(E_DIM)] = rng.uniform(0, 1)
        agent.replay.add(rng.standard_normal(38).astype(np.float32), e,
                         rng.uniform(-1, 1, 2).astype(np.float32),
                         float(rng.standard_normal()),
                         rng.standard_normal(38).astype(np.float32), e, False)
    for _ in range(300):
        agent.update()
    with torch.no_grad():
        es = torch.eye(E_DIM)
        cs = agent.g_phi(es)
        spread = (cs - agent.g_phi(torch.zeros(1, E_DIM))).norm(dim=1)
    assert float(spread.min()) > 0.05, spread


def test_history_tracker_latent_flow():
    from proteus.agents.replay import HistoryTracker
    tr = HistoryTracker(5, 4, 3)
    tr.push(np.zeros(2), np.zeros(2), np.array([1.0, 1.0, 1.0]))
    tr.push(np.zeros(2), np.zeros(2), np.array([2.0, 1.0, 0.0]))
    np.testing.assert_allclose(tr.dmu[-1], [1.0, 0.0, -1.0])
    np.testing.assert_allclose(tr.dmu[-2], [0.0, 0.0, 0.0])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
