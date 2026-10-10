import numpy as np

from barista_control.ur5_params import SCREW_W, SCREW_Q, M
from barista_control.common_function import screw_exp


def jacobian(q):
    """Geometric Jacobian in the base frame, rows [v_tool; omega] (6x6).

    Column i uses joint i's current axis z_i and a point o_i on it, both obtained by
    moving the zero-configuration axis (w_i, q_i) with the joints before it.
    """
    q = np.asarray(q, dtype=float)
    T = np.eye(4)
    z_axes, points = [], []
    for w, p, theta in zip(SCREW_W, SCREW_Q, q):
        z_axes.append(T[:3, :3] @ w)
        points.append(T[:3, :3] @ p + T[:3, 3])
        T = T @ screw_exp(w, p, theta)
    p_tool = (T @ M)[:3, 3]

    J = np.zeros((6, 6))
    for i, (z, o) in enumerate(zip(z_axes, points)):
        J[:3, i] = np.cross(z, p_tool - o)
        J[3:, i] = z
    return J