import numpy as np

from ur5_params import D1, A2, A3, D4, D5, D6, M
from common_function import screw_exp


def forward_kinematics(theta1, theta2, theta3, theta4, theta5, theta6):
    theta = np.asarray(
        [theta1, theta2, theta3, theta4, theta5, theta6],
        dtype=float,
    )

    w = [
        np.array([0.0, 0.0, 1.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        np.array([0.0, 1.0, 0.0]),
    ]

    q = [
        np.array([0.0, 0.0, 0.0]),
        np.array([0.0, D4, D1]),
        np.array([0.0, D4, D1 + A2]),
        np.array([0.0, 0.0, D1 + A2 + A3]),
        np.array([0.0, D4, D1 + A2 + A3]),
        np.array([0.0, D4, D1 + A2 + A3 + D5]),
    ]

    E = [screw_exp(w[i], q[i], theta[i]) for i in range(6)]

    # Ts
    T = np.eye(4)
    for Ei in E:
        T = T @ Ei
    T = T @ M

    return T
