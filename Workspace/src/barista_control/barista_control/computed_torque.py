"""Joint-space computed-torque control with a time-varying payload.

    tau = M(q, pi_p) (qdd_d + Kd (qd_d - qd) + Kp (q_d - q)) + c(q, qd, pi_p) + g(q, pi_p)

With an exact model each joint's error obeys e'' + Kd e' + Kp e = 0, with
Kp = wn^2 and Kd = 2 zeta wn. The bracket is a commanded acceleration, so the
law is a single Newton-Euler pass. The payload pi_p is whatever was last passed
to dyn.set_payload(); leaving it empty makes the payload an unmodelled load.
"""
import numpy as np

# UR5 joint torque limits [N m] (ur_description/config/ur5/joint_limits.yaml)
EFFORT_LIMITS = np.array([150.0, 150.0, 150.0, 28.0, 28.0, 28.0])


class ComputedTorqueController:
    def __init__(self, dyn, wn=20.0, zeta=1.0, effort_limits=EFFORT_LIMITS):
        """dyn: dynamics.UR5Dynamics; wn [rad/s], zeta: scalars or per joint."""
        if np.any(np.asarray(wn) <= 0) or np.any(np.asarray(zeta) <= 0):
            raise ValueError("wn and zeta must be positive")
        self.dyn = dyn
        self.Kp = np.asarray(wn, float) ** 2
        self.Kd = 2 * np.asarray(zeta, float) * np.asarray(wn, float)
        self.effort_limits = np.asarray(effort_limits, float)

    def clip(self, tau):
        return np.clip(tau, -self.effort_limits, self.effort_limits)

    def torque(self, q, qd, q_d, qd_d, qdd_d):
        """Joint torques [N m], saturated at the effort limits."""
        a_cmd = qdd_d + self.Kd * (qd_d - qd) + self.Kp * (q_d - q)
        return self.clip(self.dyn.inverse_dynamics(q, qd, a_cmd))
