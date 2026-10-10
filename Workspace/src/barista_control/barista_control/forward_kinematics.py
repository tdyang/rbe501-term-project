import numpy as np

from barista_control.ur5_params import SCREW_W, SCREW_Q, M
from barista_control.common_function import screw_exp


def forward_kinematics(q):
    """Space-frame PoE: base -> tool0 for joint angles q (6,), /joint_states order."""
    T = np.eye(4)
    for w, p, theta in zip(SCREW_W, SCREW_Q, np.asarray(q, dtype=float)):
        T = T @ screw_exp(w, p, theta)
    return T @ M