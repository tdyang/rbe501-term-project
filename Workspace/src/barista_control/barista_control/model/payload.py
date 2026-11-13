"""Time-varying payload: rigid bodies, a draining pitcher, and a mass estimator.

    Body                 rigid body (mass, COM, inertia); bodies combine with `+`
    Pitcher(m_liquid)    shell + cylindrical liquid column whose height tracks its mass
    MassEstimator        payload mass from the wrist F/T force, low-pass filtered

All vectors are in the tool0 (= wrist_3_link) frame unless stated otherwise.
"""
from dataclasses import dataclass, field

import numpy as np

GRAVITY = np.array([0.0, 0.0, -9.81])   # world frame [m/s^2]


def _parallel_axis(m, r):
    """m (|r|^2 E - r r^T): inertia shift from the COM to a point offset by -r."""
    return m * (r @ r * np.eye(3) - np.outer(r, r))


@dataclass(frozen=True)
class Body:
    """Rigid body: mass m [kg], COM c [m], inertia I about the COM [kg m^2]."""
    m: float = 0.0
    c: np.ndarray = field(default_factory=lambda: np.zeros(3))
    I: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))

    @property
    def h(self):
        """First moment of mass m c."""
        return self.m * np.asarray(self.c, float)

    @property
    def I_o(self):
        """Inertia about the frame origin."""
        return self.I + _parallel_axis(self.m, np.asarray(self.c, float))

    def __add__(self, other):
        m = self.m + other.m
        if m == 0:
            return Body()
        c = (self.h + other.h) / m
        return Body(m, c, self.I_o + other.I_o - _parallel_axis(m, c))


@dataclass(frozen=True)
class Pitcher:
    """Pitcher held by the gripper: rigid shell plus a cylinder of liquid.

    The liquid column has radius r and height m_l / (rho pi r^2), standing on
    `bottom` along `axis` (pitcher's up direction). Slosh and tilt-induced COM
    shift are neglected.
    """
    shell: Body
    bottom: np.ndarray            # center of the inside bottom [m]
    axis: np.ndarray              # unit vector, pitcher "up"
    radius: float                 # [m]
    rho: float = 1000.0           # liquid density [kg/m^3]

    def liquid(self, m_l):
        a = np.asarray(self.axis, float) / np.linalg.norm(self.axis)
        h = m_l / (self.rho * np.pi * self.radius ** 2)
        I_ax = 0.5 * m_l * self.radius ** 2                     # about the axis
        I_tr = m_l * (3 * self.radius ** 2 + h ** 2) / 12       # transverse
        I = I_tr * np.eye(3) + (I_ax - I_tr) * np.outer(a, a)
        return Body(m_l, np.asarray(self.bottom, float) + 0.5 * h * a, I)

    def __call__(self, m_l):
        """Payload with m_l [kg] of liquid left."""
        return self.shell + self.liquid(max(m_l, 0.0))


class MassEstimator:
    """Payload mass from the wrist F/T force.

    Newton on the payload: f = m (a - g), with f the force the wrist applies to
    the payload. Projecting onto e = R^T (a - g) gives m = f.e / |e|^2, which is
    low-pass filtered (first order, `cutoff_hz`).
    """

    def __init__(self, cutoff_hz=1.0, bias=np.zeros(3)):
        self.tau = 1.0 / (2 * np.pi * cutoff_hz)
        self.bias = np.asarray(bias, float)
        self.m = None

    def tare(self, f_empty):
        """Record the reading with nothing grasped (removes sensor/gripper offset)."""
        self.bias = np.asarray(f_empty, float)

    def update(self, f_sensor, R_ws, dt, a_world=np.zeros(3)):
        """f_sensor: force in sensor frame [N]; R_ws: sensor orientation in world;
        a_world: sensor linear acceleration (0 when quasi-static). Returns m_hat [kg]."""
        e = R_ws.T @ (np.asarray(a_world, float) - GRAVITY)
        m_raw = (np.asarray(f_sensor, float) - self.bias) @ e / (e @ e)
        alpha = dt / (dt + self.tau)
        self.m = m_raw if self.m is None else self.m + alpha * (m_raw - self.m)
        return self.m