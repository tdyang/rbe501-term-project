"""Closed-form inverse kinematics for the UR5

    solve_ik(T)                 -> (k, 6) array of all valid solutions
    solve_ik_closest(T, q_ref)  -> the single solution nearest q_ref (or None)
"""
import numpy as np

from barista_control import ur5_params as P

_EPS = 1e-9


def wrap_to_pi(q):
    """Wrap angles to (-pi, pi]."""
    return -((-np.asarray(q) + np.pi) % (2 * np.pi) - np.pi)


def solve_ik(T, q6_if_singular=0.0):
    """All IK solutions for a desired end-effector pose.

    Args:
        T: 4x4 homogeneous transform ^0T_6 (base -> tool0).
        q6_if_singular: theta6 to use when the wrist is singular (s5 ~ 0),
            where theta6 is not determined by the pose.

    Returns:
        (k, 6) numpy array, k in [0, 8], angles wrapped to (-pi, pi].
        Empty array (shape (0, 6)) if the pose is unreachable.
    """
    T = np.asarray(T, dtype=float)
    R, p = T[:3, :3], T[:3, 3]
    x_hat, y_hat, z_hat = R[:, 0], R[:, 1], R[:, 2]

    solutions = []

    # ---- theta1: shoulder left/right ------------------------------------
    p5 = p - P.D6 * z_hat                       # origin of frame 5
    r = np.hypot(p5[0], p5[1])
    if r < abs(P.D4) - _EPS:
        return np.empty((0, 6))                 # wrist center too close to axis 1
    phi = np.arctan2(p5[1], p5[0])
    psi = np.arccos(np.clip(P.D4 / r, -1.0, 1.0))

    for t1 in (phi + psi + np.pi / 2, phi - psi + np.pi / 2):
        s1, c1 = np.sin(t1), np.cos(t1)

        # ---- theta5: wrist up/down --------------------------------------
        c5 = (p[0] * s1 - p[1] * c1 - P.D4) / P.D6
        if abs(c5) > 1 + 1e-9:
            continue
        acos5 = np.arccos(np.clip(c5, -1.0, 1.0))

        for t5 in (acos5, -acos5):
            s5 = np.sin(t5)

            # ---- theta6 -------------------------------------------------
            if abs(s5) < 1e-8:
                t6 = q6_if_singular             # wrist singular: theta6 free
            else:
                t6 = np.arctan2((-y_hat[0] * s1 + y_hat[1] * c1) / s5,
                                ( x_hat[0] * s1 - x_hat[1] * c1) / s5)

            # ---- theta2, theta3, theta4: planar 3R arm ------------------
            T01 = P.link_transform(0, t1)
            T45 = P.link_transform(4, t5)
            T56 = P.link_transform(5, t6)
            T14 = np.linalg.inv(T01) @ T @ np.linalg.inv(T45 @ T56)
            x, y = T14[0, 3], T14[1, 3]

            c3 = (x * x + y * y - P.A2 ** 2 - P.A3 ** 2) / (2 * P.A2 * P.A3)
            if abs(c3) > 1 + 1e-9:
                continue                        # out of reach for this branch
            acos3 = np.arccos(np.clip(c3, -1.0, 1.0))

            for t3 in (acos3, -acos3):          # elbow up/down
                s3 = np.sin(t3)
                t2 = np.arctan2(y, x) - np.arctan2(P.A3 * s3, P.A2 + P.A3 * np.cos(t3))
                t234 = np.arctan2(T14[1, 0], T14[0, 0])
                t4 = t234 - t2 - t3
                solutions.append([t1, t2, t3, t4, t5, t6])

    if not solutions:
        return np.empty((0, 6))
    return wrap_to_pi(np.array(solutions))


def solve_ik_closest(T, q_ref, q_min=P.Q_MIN, q_max=P.Q_MAX):
    """IK solution closest to q_ref (e.g. the current joint state).

    Each joint is shifted by multiples of 2*pi to sit as close as possible to
    q_ref within the joint limits to avoid 360 deg wrist spins.

    Returns:
        (6,) array, or None if no solution exists.
    """
    q_ref = np.asarray(q_ref, dtype=float)
    sols = solve_ik(T, q6_if_singular=q_ref[5])
    if len(sols) == 0:
        return None

    best, best_cost = None, np.inf
    for q in sols:
        # unwrap each joint toward q_ref
        q = q_ref + wrap_to_pi(q - q_ref)
        q = np.where(q > q_max, q - 2 * np.pi, q)
        q = np.where(q < q_min, q + 2 * np.pi, q)
        if np.any(q < q_min) or np.any(q > q_max):
            continue
        cost = np.sum((q - q_ref) ** 2)
        if cost < best_cost:
            best, best_cost = q, cost
    return best