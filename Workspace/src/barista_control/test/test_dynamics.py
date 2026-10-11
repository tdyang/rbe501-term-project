"""Tests for dynamics.py and payload.py.  Run:  python3 -m pytest test/ -v

Each check uses an independent method (DH FK, Lagrangian formulas, finite
differences, the Jacobian), never RNE against itself.
"""
import os

import numpy as np
import pytest

from barista_control import ur5_params as P
from barista_control.dynamics import UR5Dynamics, _rot
from barista_control.kinematics import forward_kinematics, jacobian
from barista_control.payload import GRAVITY, Body, MassEstimator, Pitcher

URDF = os.path.join(os.path.dirname(__file__), "ur5_test.urdf")
RNG = np.random.default_rng(1)
PITCHER = Pitcher(Body(0.35, np.array([0, 0, 0.10]), 1e-3 * np.eye(3)),
                  bottom=np.array([0, 0, 0.04]), axis=np.array([0, 0, 1.0]), radius=0.045)


@pytest.fixture
def dyn():
    return UR5Dynamics.from_urdf(URDF)


def rq():
    return RNG.uniform(-np.pi, np.pi, 6)


def link_poses(dyn, q):
    T, out = np.eye(4), []
    for (R0, p, ax), t in zip(dyn.joints, q):
        A = np.eye(4)
        A[:3, :3], A[:3, 3] = R0 @ _rot(ax, t), p
        T = T @ A
        out.append(T)
    return out


def bodies(dyn):
    return dyn.links[:-1] + [dyn.links[-1] + dyn.payload]


# ---- model setup ------------------------------------------------------------
def test_urdf_tool_frame_matches_fk(dyn):
    for _ in range(20):
        q = rq()
        assert np.allclose(link_poses(dyn, q)[-1], forward_kinematics(q), atol=1e-6)


def test_gravity_points_down(dyn):
    assert np.allclose(dyn.g_base, GRAVITY)


# ---- rigid-body dynamics ------------------------------------------------------
def test_mass_matrix_lagrangian(dyn):
    """M = sum m Jv^T Jv + Jw^T R I R^T Jw   (Kebria et al. Eq. 8), payload included."""
    dyn.set_payload(PITCHER(0.6))
    for _ in range(20):
        q = rq()
        Ts, M = link_poses(dyn, q), np.zeros((6, 6))
        for i, b in enumerate(bodies(dyn)):
            R, c = Ts[i][:3, :3], Ts[i][:3, :3] @ b.c + Ts[i][:3, 3]
            Jv, Jw = np.zeros((3, 6)), np.zeros((3, 6))
            for j in range(i + 1):
                Jv[:, j] = np.cross(Ts[j][:3, 2], c - Ts[j][:3, 3])
                Jw[:, j] = Ts[j][:3, 2]
            M += b.m * Jv.T @ Jv + Jw.T @ R @ b.I @ R.T @ Jw
        Md = dyn.mass_matrix(q)
        assert np.allclose(Md, M, atol=1e-9) and np.all(np.linalg.eigvalsh(Md) > 0)


def test_gravity_is_potential_gradient(dyn):
    """g = dP/dq   (Kebria et al. Eq. 10)."""
    dyn.set_payload(PITCHER(0.6))

    def P_(q):
        return sum(-b.m * GRAVITY @ (T[:3, :3] @ b.c + T[:3, 3])
                   for b, T in zip(bodies(dyn), link_poses(dyn, q)))
    for _ in range(20):
        q, h = rq(), 1e-6
        g_fd = [(P_(q + h * e) - P_(q - h * e)) / (2 * h) for e in np.eye(6)]
        assert np.allclose(dyn.gravity(q), g_fd, atol=1e-6)


def test_coriolis_skew_symmetry(dyn):
    """qd^T (Mdot - 2C) qd = 0."""
    for _ in range(20):
        q, qd, h = rq(), RNG.normal(size=6), 1e-6
        Mdot = (dyn.mass_matrix(q + h * qd) - dyn.mass_matrix(q - h * qd)) / (2 * h)
        assert qd @ Mdot @ qd == pytest.approx(2 * qd @ dyn.coriolis(q, qd), abs=1e-6)


def test_forward_inverse_roundtrip(dyn):
    q, qd, qdd = rq(), RNG.normal(size=6), RNG.normal(size=6)
    assert np.allclose(dyn.forward_dynamics(q, qd, dyn.inverse_dynamics(q, qd, qdd)), qdd)


# ---- payload ----------------------------------------------------------------
def test_body_addition():
    a, b = Body(1.0, np.array([0, 0, 0.1])), Body(1.0, np.array([0, 0, -0.1]))
    s = a + b
    assert s.m == 2 and np.allclose(s.c, 0)
    assert np.allclose(s.I, np.diag([0.02, 0.02, 0.0]))       # 2 * m * d^2 about x, y


def test_pitcher_drains():
    full, empty = PITCHER(0.8), PITCHER(0.0)
    h = 0.8 / (1000 * np.pi * 0.045 ** 2)
    assert full.m == pytest.approx(1.15) and empty.m == pytest.approx(0.35)
    assert PITCHER.liquid(0.8).c[2] == pytest.approx(0.04 + h / 2)


def test_payload_torque_is_jacobian_transpose_wrench(dyn):
    """tau_p = J^T W_p, W_p = Newton-Euler wrench of the payload at tool0 (base frame)."""
    p_body = PITCHER(0.6)
    for _ in range(10):
        q, qd, qdd, dt = rq(), RNG.normal(size=6), RNG.normal(size=6), 1e-4
        T = [forward_kinematics(q + qd * t + 0.5 * qdd * t * t) for t in (-dt, 0, dt)]
        a = (T[2][:3, 3] - 2 * T[1][:3, 3] + T[0][:3, 3]) / dt ** 2
        vee = lambda S: np.array([S[2, 1], S[0, 2], S[1, 0]])
        J = jacobian(q)
        w = J[3:] @ qd
        wd = (jacobian(q + qd * dt + 0.5 * qdd * dt * dt)[3:] @ (qd + qdd * dt)
              - jacobian(q - qd * dt + 0.5 * qdd * dt * dt)[3:] @ (qd - qdd * dt)) / (2 * dt)
        R = T[1][:3, :3]
        h, I = R @ p_body.h, R @ p_body.I_o @ R.T
        W = np.r_[p_body.m * (a - GRAVITY) + np.cross(wd, h) + np.cross(w, np.cross(w, h)),
                  I @ wd + np.cross(w, I @ w) + np.cross(h, a - GRAVITY)]
        tau0 = dyn.inverse_dynamics(q, qd, qdd)
        dyn.set_payload(p_body)
        tau1 = dyn.inverse_dynamics(q, qd, qdd)
        dyn.set_payload(Body())
        assert np.allclose(tau1 - tau0, J.T @ W, atol=1e-4)


def test_mass_estimator_static_and_accelerating():
    est, R, m = MassEstimator(cutoff_hz=1.0), _rot(np.array([1.0, 0, 0]), 0.7), 0.9
    a = np.array([0.5, -0.2, 1.0])
    for k in range(2000):
        acc = a if k % 2 else np.zeros(3)
        f = R.T @ (m * (acc - GRAVITY))                        # force on payload, sensor frame
        m_hat = est.update(f + RNG.normal(0, 0.05, 3), R, dt=0.01, a_world=acc)
    assert m_hat == pytest.approx(m, abs=0.01)


def test_mass_estimator_tare():
    est = MassEstimator()
    est.tare([0.0, 0.0, 2.0])
    assert est.update(np.array([0, 0, 2.0 + 9.81]), np.eye(3), 0.01) == pytest.approx(1.0)