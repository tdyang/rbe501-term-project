import numpy as np


def screw_exp(w, q, theta):
    w = np.asarray(w, dtype=float).reshape(3)
    q = np.asarray(q, dtype=float).reshape(3)

    w_hat = np.array([
        [0.0, -w[2],  w[1]],
        [w[2],  0.0, -w[0]],
        [-w[1], w[0], 0.0],
    ])

    v = -np.cross(w, q)
    I = np.eye(3)

    R = I + np.sin(theta) * w_hat + (1.0 - np.cos(theta)) * (w_hat @ w_hat)
    p = (
        theta * I
        + (1.0 - np.cos(theta)) * w_hat
        + (theta - np.sin(theta)) * (w_hat @ w_hat)
    ) @ v

    return np.block([
        [R, p.reshape(3, 1)],
        [np.zeros((1, 3)), np.ones((1, 1))],
    ])
