"""Manipulability measures for the UR5.

    manipulability(q)        closed form, no Jacobian needed (fast, exact)
    yoshikawa(J)             sqrt(det(J J^T)) for any Jacobian (e.g. jacobian.py)
    inverse_condition(J)     sigma_min / sigma_max in [0, 1]
    singularity_report(q)    which singularity factor is near zero

Closed form:
    w(q) = |a2 a3 s3 s5 (a2 c2 + a3 c23 + d5 s234)|

Singular when:
    s3 = 0   elbow (arm fully stretched / folded)
    s5 = 0   wrist (axes 4 and 6 parallel)
    a2 c2 + a3 c23 + d5 s234 = 0   shoulder (wrist center on axis 1/2 plane)
"""
import numpy as np

from barista_control import ur5_params as P


def _factors(q):
    q = np.asarray(q, dtype=float)
    _, t2, t3, t4, t5, _ = q
    elbow = np.sin(t3)
    wrist = np.sin(t5)
    shoulder = (P.A2 * np.cos(t2) + P.A3 * np.cos(t2 + t3)
                + P.D5 * np.sin(t2 + t3 + t4))
    return elbow, wrist, shoulder


def manipulability(q):
    """Yoshikawa manipulability w(q) = |det J(q)|, closed form [m^2]."""
    elbow, wrist, shoulder = _factors(q)
    return abs(P.A2 * P.A3 * elbow * wrist * shoulder)


def yoshikawa(J):
    """sqrt(det(J J^T)) for any Jacobian (square or 6xn)."""
    J = np.asarray(J, dtype=float)
    return float(np.sqrt(max(np.linalg.det(J @ J.T), 0.0)))


def inverse_condition(J):
    """sigma_min / sigma_max of J (1 = isotropic, 0 = singular)."""
    s = np.linalg.svd(np.asarray(J, dtype=float), compute_uv=False)
    return float(s[-1] / s[0]) if s[0] > 0 else 0.0


def singularity_report(q, tol=1e-2):
    """Dict of each singularity factor and whether it is below tol."""
    elbow, wrist, shoulder = _factors(q)
    return {
        "w": manipulability(q),
        "elbow": (float(elbow), abs(elbow) < tol),
        "wrist": (float(wrist), abs(wrist) < tol),
        "shoulder": (float(shoulder), abs(shoulder) < tol),
    }