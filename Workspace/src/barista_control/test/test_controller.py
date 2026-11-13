"""Tests for trajectory.py and computed_torque.py.  Run:  python3 -m pytest test/ -v

The closed-loop tests integrate the plant with UR5Dynamics.forward_dynamics,
using a separate model instance as the "true" arm so that the controller's
payload can differ from the plant's.
"""
import os

import numpy as np
import pytest

from barista_control.model import ur5_params as P
from barista_control.control.computed_torque import ComputedTorqueController
from barista_control.model.dynamics import UR5Dynamics
from barista_control.model.payload import Body, Pitcher
from barista_control.planning.trajectory import Spline

URDF = os.path.join(os.path.dirname(__file__), "ur5_test.urdf")
RNG = np.random.default_rng(2)
PITCHER = Pitcher(Body(0.35, np.array([0, 0, 0.10]), 1e-3 * np.eye(3)),
                  bottom=np.array([0, 0, 0.04]), axis=np.array([0, 0, 1.0]), radius=0.045)
# upper arm vertical, forearm horizontal, tool pointing down: away from all singularities
Q_POUR = np.array([0.0, -np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, 0.0])


# ---- trajectory -------------------------------------------------------------
def test_spline_hits_waypoints_and_boundary_conditions():
    t = [0.0, 1.0, 2.5, 3.0]
    q = RNG.uniform(-1, 1, (4, 6))
    qd = RNG.normal(size=(4, 6))
    qdd = RNG.normal(size=(4, 6))
    s = Spline(t, q, qd, qdd)
    for k in range(4):
        if k in (0, 3):
            continue
        qk, qdk, qddk = s.sample(t[k])
        assert np.allclose(qk, q[k]) and np.allclose(qdk, qd[k]) and np.allclose(qddk, qdd[k])
    eps = 1e-9   # just inside the end points
    assert np.allclose(s.sample(t[0] + eps)[1], qd[0], atol=1e-6)
    assert np.allclose(s.sample(t[-1] - eps)[0], q[-1], atol=1e-6)


def test_spline_derivatives_match_finite_differences():
    s = Spline([0.0, 0.7, 2.0], RNG.uniform(-1, 1, (3, 6)))
    for t in np.linspace(0.05, 1.95, 15):
        h = 1e-5
        q_p, qd_p, _ = s.sample(t + h)
        q_m, qd_m, _ = s.sample(t - h)
        _, qd, qdd = s.sample(t)
        assert np.allclose((q_p - q_m) / (2 * h), qd, atol=1e-5)
        assert np.allclose((qd_p - qd_m) / (2 * h), qdd, atol=1e-4)


def test_spline_default_velocities_rest_at_ends_and_extrema():
    q = np.array([[0.0] * 6, [1.0] * 6, [2.0] * 6, [0.5] * 6])
    s = Spline([0, 1, 2, 3], q)
    assert np.allclose(s.qd[0], 0) and np.allclose(s.qd[-1], 0)
    assert np.allclose(s.qd[1], 1.0)        # monotone through waypoint 1
    assert np.allclose(s.qd[2], 0.0)        # turning point at waypoint 2
    # holds outside the time range
    assert np.allclose(s.sample(-1)[0], q[0]) and np.allclose(s.sample(10)[1], 0)


def test_spline_rejects_bad_times():
    with pytest.raises(ValueError):
        Spline([0.0, 0.0], np.zeros((2, 6)))


# ---- control law --------------------------------------------------------------
@pytest.fixture
def dyn():
    return UR5Dynamics.from_urdf(URDF)


def test_zero_error_gives_inverse_dynamics(dyn):
    ctc = ComputedTorqueController(dyn, wn=20.0, effort_limits=np.full(6, 1e6))
    dyn.set_payload(PITCHER(0.5))
    q, qd, qdd = RNG.uniform(-np.pi, np.pi, 6), RNG.normal(size=6), RNG.normal(size=6)
    assert np.allclose(ctc.torque(q, qd, q, qd, qdd), dyn.inverse_dynamics(q, qd, qdd))


def test_feedback_is_mass_matrix_times_pd(dyn):
    ctc = ComputedTorqueController(dyn, wn=10.0, zeta=0.7, effort_limits=np.full(6, 1e6))
    q, qd = RNG.uniform(-np.pi, np.pi, 6), RNG.normal(size=6)
    e, ed = RNG.normal(0, 0.05, 6), RNG.normal(0, 0.1, 6)
    z = np.zeros(6)
    tau = ctc.torque(q, qd, q + e, qd + ed, z)
    expected = dyn.mass_matrix(q) @ (100.0 * e + 14.0 * ed) + dyn.coriolis(q, qd) + dyn.gravity(q)
    assert np.allclose(tau, expected)


def test_torque_saturates(dyn):
    ctc = ComputedTorqueController(dyn, wn=200.0)
    z = np.zeros(6)
    tau = ctc.torque(Q_POUR, z, Q_POUR + 1.0, z, z)
    assert np.all(np.abs(tau) <= ctc.effort_limits + 1e-12)
    assert np.any(np.isclose(np.abs(tau), ctc.effort_limits))


# ---- closed loop ----------------------------------------------------------------
def simulate(ctrl_payload, plant_payload, spline, q0, T, dt=2e-3, wn=20.0):
    """Semi-implicit Euler on the true plant; controller and plant run at dt.
    ctrl_payload / plant_payload: functions t -> payload.Body.
    Returns the |q_d - q| history and the final q."""
    plant = UR5Dynamics.from_urdf(URDF)
    ctc = ComputedTorqueController(UR5Dynamics.from_urdf(URDF), wn=wn)
    q, qd = q0.copy(), np.zeros(6)
    errors = []
    for k in range(int(round(T / dt))):
        t = k * dt
        plant.set_payload(plant_payload(t))
        ctc.dyn.set_payload(ctrl_payload(t))
        q_d, qd_d, qdd_d = spline.sample(t)
        errors.append(np.abs(q_d - q))
        tau = ctc.torque(q, qd, q_d, qd_d, qdd_d)
        qd = qd + dt * plant.forward_dynamics(q, qd, tau)
        q = q + dt * qd
    return np.array(errors), q


def draining(t, m0=0.8, rate=0.8):
    """Pitcher losing 0.8 kg/s of liquid from t = 0."""
    return PITCHER(max(m0 - rate * t, 0.0))


def test_tracks_trajectory_with_exact_model():
    q1 = Q_POUR + np.array([0.3, 0.2, -0.3, 0.2, 0.0, 0.5])
    spline = Spline([0.0, 1.0], [Q_POUR, q1])
    err, _ = simulate(draining, draining, spline, Q_POUR, T=1.2)
    assert err.max() < 5e-4          # Euler integration error at dt = 2 ms


def test_payload_update_removes_pour_error():
    """Holding the pour pose while the pitcher drains: the fixed full-pitcher model
    drifts as the payload changes, the updated model stays on target."""
    spline = Spline([0.0, 1.0], [Q_POUR, Q_POUR])
    full = lambda t: PITCHER(0.8)
    fixed, _ = simulate(full, draining, spline, Q_POUR, T=1.0)
    updated, _ = simulate(draining, draining, spline, Q_POUR, T=1.0)
    assert updated.max() < 1e-3
    assert fixed[-1].max() > 10 * updated.max()


def test_steady_state_error_matches_prediction():
    """With the payload ignored, the arm settles where Kp e = M^-1 (g_true - g_model),
    both sides evaluated at the settled configuration q*."""
    spline = Spline([0.0, 1.0], [Q_POUR, Q_POUR])
    err, q_star = simulate(lambda t: Body(), lambda t: PITCHER(0.8), spline, Q_POUR, T=2.0)

    model, plant = UR5Dynamics.from_urdf(URDF), UR5Dynamics.from_urdf(URDF)
    plant.set_payload(PITCHER(0.8))
    dg = plant.gravity(q_star) - model.gravity(q_star)
    predicted = np.linalg.solve(model.mass_matrix(q_star), dg) / 20.0 ** 2
    assert np.allclose(Q_POUR - q_star, predicted, atol=1e-4)
    assert err[-1].max() > 0.05      # tens of mrad at the wrist: the light links feel it most


# ---- ROS message parsing -----------------------------------------------------------
def test_spline_from_msg_reorders_joints_and_blends_in():
    pytest.importorskip("trajectory_msgs")
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
    from barista_control.nodes.dynamic_controller import spline_from_msg

    goal = Q_POUR + 0.2
    pt = JointTrajectoryPoint(positions=list(goal[::-1]))
    pt.time_from_start.sec = 2
    msg = JointTrajectory(joint_names=P.JOINT_NAMES[::-1], points=[pt])
    s = spline_from_msg(msg, now=10.0, current=Spline([0.0, 1.0], [Q_POUR, Q_POUR]))
    assert np.allclose(s.sample(10.0)[0], Q_POUR) and np.allclose(s.sample(12.0)[0], goal)

    pt.positions = list(goal[::-1] + 10.0)                 # beyond joint limits
    with pytest.raises(ValueError):
        spline_from_msg(msg, 10.0, s)
