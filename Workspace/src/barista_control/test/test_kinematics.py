"""Tests for UR5 kinematics.  Run:  python3 -m pytest test/ -v

IK and manipulability are checked against a small reference FK/Jacobian
built straight from the DH table.
"""
import numpy as np
import pytest

from barista_control import ur5_params as P
from barista_control.inverse_kinematics import solve_ik, solve_ik_closest, wrap_to_pi
from barista_control.manipulability import manipulability, yoshikawa, singularity_report

RNG = np.random.default_rng(0)
N = 300


# ---- reference implementations (test-only) ------------------------------
def ref_fk_all(q):
    Ts = [np.eye(4)]
    for i in range(6):
        Ts.append(Ts[-1] @ P.link_transform(i, q[i]))
    return Ts


def ref_fk(q):
    return ref_fk_all(q)[-1]


def ref_jacobian(q):
    """Geometric Jacobian in base frame, rows [v; w]."""
    Ts = ref_fk_all(q)
    p = Ts[-1][:3, 3]
    J = np.zeros((6, 6))
    for i in range(6):
        z, o = Ts[i][:3, 2], Ts[i][:3, 3]
        J[:3, i] = np.cross(z, p - o)
        J[3:, i] = z
    return J


def random_q():
    return RNG.uniform(-np.pi, np.pi, 6)


# ---- parameters / conventions -------------------------------------------
def test_home_is_vertical():
    T = ref_fk(P.Q_HOME)
    height = P.D1 - P.A2 - P.A3 + P.D5          # ~1.0 m
    assert np.allclose(T[:3, 3], [0.0, -(P.D4 + P.D6), height], atol=1e-9)


def test_zero_is_horizontal():
    T = ref_fk(np.zeros(6))
    assert T[0, 3] == pytest.approx(P.A2 + P.A3, abs=1e-9)   # reaches out along -x


# ---- inverse kinematics -------------------------------------------------
def test_ik_all_solutions_reproduce_pose():
    for _ in range(N):
        T = ref_fk(random_q())
        sols = solve_ik(T)
        assert len(sols) >= 1
        for q in sols:
            assert np.allclose(ref_fk(q), T, atol=1e-8)


def test_ik_contains_original():
    for _ in range(N):
        q = random_q()
        sols = solve_ik(ref_fk(q))
        d = np.abs(wrap_to_pi(sols - q)).max(axis=1)
        assert d.min() < 1e-6


def test_ik_closest_returns_reference_config():
    for _ in range(N):
        q = random_q()
        q_near = q + RNG.normal(0, 0.05, 6)
        T = ref_fk(q)
        q_sol = solve_ik_closest(T, q_near)
        assert np.allclose(ref_fk(q_sol), T, atol=1e-8)
        assert np.sum((q_sol - q_near) ** 2) <= np.sum((q - q_near) ** 2) + 1e-9


def test_ik_unreachable():
    T = np.eye(4)
    T[:3, 3] = [3.0, 0.0, 0.0]                  # far outside 0.85 m reach
    assert len(solve_ik(T)) == 0
    assert solve_ik_closest(T, P.Q_HOME) is None


def test_ik_wrist_singular():
    q = P.Q_HOME.copy()
    q[4] = 0.0                                  # theta5 = 0  -> wrist singular
    T = ref_fk(q)
    for s in solve_ik(T):
        assert np.allclose(ref_fk(s), T, atol=1e-8)


# ---- manipulability -----------------------------------------------------
def test_manipulability_matches_jacobian():
    for _ in range(N):
        q = random_q()
        assert manipulability(q) == pytest.approx(yoshikawa(ref_jacobian(q)), abs=1e-10)


@pytest.mark.parametrize("joint,value,name", [(2, 0.0, "elbow"), (2, np.pi, "elbow"),
                                              (4, 0.0, "wrist"), (4, np.pi, "wrist")])
def test_singularities(joint, value, name):
    q = random_q()
    q[joint] = value
    assert manipulability(q) < 1e-12
    assert singularity_report(q)[name][1]