"""UR5 kinematic parameters

Frames: DH frame 0 = `base`, DH frame 6 = `tool0`.
`base_link` is rotated 180 deg about z from `base`, so use `base` -> `tool0`.
"""
import numpy as np

# DH parameters [m, rad]
D1 = 0.089159
A2 = -0.425
A3 = -0.39225
D4 = 0.10915
D5 = 0.09465
D6 = 0.0823

# Home Configuration
M1 = np.array([
    [1,  0,  0,  0],
    [0,  0,  1,  D4],
    [0, -1,  0,  D1],
    [0,  0,  0,  1],
], dtype=float)

M2 = np.array([
    [1,  0,  0,  0],
    [0,  0,  1,  D4],
    [0, -1,  0,  D1 + A2],
    [0,  0,  0,  1],
], dtype=float)

M3 = np.array([
    [1,  0,  0,  0],
    [0,  0,  1,  0],
    [0, -1,  0,  D1 + A2 + A3],
    [0,  0,  0,  1],
], dtype=float)

M4 = np.array([
    [1,  0,  0,  0],
    [0,  1,  0,  D4],
    [0,  0,  1,  D1 + A2 + A3],
    [0,  0,  0,  1],
], dtype=float)

M5 = np.array([
    [1,  0,  0,  0],
    [0,  0,  1,  D4],
    [0, -1,  0,  D1 + A2 + A3 + D5],
    [0,  0,  0,  1],
], dtype=float)

M = np.array([
    [1,  0,  0,  0],
    [0,  1,  0,  D4 + D6],
    [0,  0,  1,  D1 + A2 + A3 + D5],
    [0,  0,  0,  1],
], dtype=float)


DH_A = np.array([0.0, A2, A3, 0.0, 0.0, 0.0])
DH_ALPHA = np.array([np.pi / 2, 0.0, 0.0, np.pi / 2, -np.pi / 2, 0.0])
DH_D = np.array([D1, 0.0, 0.0, D4, D5, D6])

N_JOINTS = 6

JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

# Joint limits [rad]
Q_MIN = np.full(N_JOINTS, -2 * np.pi)
Q_MAX = np.full(N_JOINTS, 2 * np.pi)

# ---------------------------------------------------------------------------
# Named joint configurations [rad]
# ---------------------------------------------------------------------------
Q_HOME = np.array([0.0, -np.pi / 2, 0.0, -np.pi / 2, 0.0, 0.0])     # arm vertical

NAMED_CONFIGS = {
    "home": Q_HOME,
}

def dh_transform(theta, d, a, alpha):
    """Standard DH homogeneous transform  ^{i-1}T_i  (4x4)."""
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0.0,      sa,       ca,      d],
        [0.0,     0.0,      0.0,    1.0],
    ])


def link_transform(i, theta):
    """^{i}T_{i+1} for joint index i = 0..5 (0-based) at angle theta."""
    return dh_transform(theta, DH_D[i], DH_A[i], DH_ALPHA[i])
