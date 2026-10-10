import numpy as np

from ur5_params import D1, A2, A3, D4, D5, D6, M1, M2, M3, M4, M5, M
from common_function import screw_exp


def jacobian(theta1, theta2, theta3, theta4, theta5, theta6):
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

    # T0i
    T01 = E[0] @ M1
    T02 = E[0] @ E[1] @ M2
    T03 = E[0] @ E[1] @ E[2] @ M3
    T04 = E[0] @ E[1] @ E[2] @ E[3] @ M4
    T05 = E[0] @ E[1] @ E[2] @ E[3] @ E[4] @ M5
    T06 = E[0] @ E[1] @ E[2] @ E[3] @ E[4] @ E[5] @ M

    T_joint = [T01, T02, T03, T04, T05]
    z0 = np.array([0.0, 0.0, 1.0])
    o0 = np.zeros(3)

    # zi abd oi
    z_axes = [z0] + [Tj[:3, 2] for Tj in T_joint]
    origins = [o0] + [Tj[:3, 3] for Tj in T_joint]
    oe = T06[:3, 3]

    z_axes = np.column_stack(z_axes)
    origins = np.column_stack(origins)

    position_offsets = oe.reshape(3, 1) - origins
    J_linear = np.cross(z_axes.T, position_offsets.T).T
    J_angular = z_axes

    J = np.vstack((J_linear, J_angular))
    return J
