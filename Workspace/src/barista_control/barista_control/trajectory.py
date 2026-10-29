"""Joint-space reference trajectories for the dynamic controller.

    Spline(times, q, qd=None, qdd=None)    piecewise quintic through waypoints
    spline.sample(t) -> (q_d, qd_d, qdd_d)

Each segment is the quintic matching position, velocity and acceleration at
both ends, so the reference is C2. Missing velocities come from neighbouring
slopes (zero at the ends and at turning points, so nothing overshoots);
missing accelerations are zero. Outside [t_0, t_N] the end points are held.
"""
import numpy as np

_K = np.arange(6)


def _quintic(p0, v0, a0, p1, v1, a1, T):
    """(6, n) coefficients c_k of p(s) = sum c_k s^k, s in [0, T]."""
    d = p1 - p0 - v0 * T - 0.5 * a0 * T ** 2
    dv, da = v1 - v0 - a0 * T, a1 - a0
    return np.array([p0, v0, 0.5 * a0,
                     (10 * d - 4 * dv * T + 0.5 * da * T ** 2) / T ** 3,
                     (-15 * d + 7 * dv * T - da * T ** 2) / T ** 4,
                     (6 * d - 3 * dv * T + 0.5 * da * T ** 2) / T ** 5])


def _slopes(t, q):
    m = np.diff(q, axis=0) / np.diff(t)[:, None]
    v = np.zeros_like(q)
    v[1:-1] = np.where(m[:-1] * m[1:] > 0, 0.5 * (m[:-1] + m[1:]), 0.0)
    return v


class Spline:
    def __init__(self, times, q, qd=None, qdd=None):
        self.t, self.q = np.asarray(times, float), np.atleast_2d(np.asarray(q, float))
        if len(self.t) != len(self.q) or len(self.t) < 2 or np.any(np.diff(self.t) <= 0):
            raise ValueError("need >= 2 waypoints with strictly increasing times")
        self.qd = _slopes(self.t, self.q) if qd is None else np.asarray(qd, float)
        self.qdd = np.zeros_like(self.q) if qdd is None else np.asarray(qdd, float)
        b = (self.q, self.qd, self.qdd)
        self.coeffs = [_quintic(*(x[k] for x in b), *(x[k + 1] for x in b), T)
                       for k, T in enumerate(np.diff(self.t))]

    @property
    def end_time(self):
        return self.t[-1]

    def sample(self, t):
        if t <= self.t[0]:
            return self.q[0].copy(), self.qd[0].copy(), self.qdd[0].copy()
        if t >= self.t[-1]:
            return self.q[-1].copy(), np.zeros_like(self.q[-1]), np.zeros_like(self.q[-1])
        k = int(np.searchsorted(self.t, t, side="right")) - 1
        c, s = self.coeffs[k], t - self.t[k]
        return (s ** _K @ c,
                (_K[1:] * s ** _K[:-1]) @ c[1:],
                (_K[2:] * _K[1:-1] * s ** _K[:-2]) @ c[2:])
