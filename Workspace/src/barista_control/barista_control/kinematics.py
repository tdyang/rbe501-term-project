"""UR5 kinematics: base -> tool0, joint angles in /joint_states order.

    forward_kinematics(q)        4x4 pose via space-frame PoE
    jacobian(q)                  6x6 geometric Jacobian, rows [v_tool; omega]
    solve_ik(T)                  (k, 6) all closed-form IK solutions
    solve_ik_closest(T, q_ref)   single solution nearest q_ref (or None)
    manipulability(q)            closed-form |det J(q)|
    yoshikawa(J)                 sqrt(det(J J^T)) for any Jacobian
    inverse_condition(J)         sigma_min / sigma_max in [0, 1]
    singularities(q)             names of the singularity factors near zero

Closed-form manipulability:
    w(q) = |a2 a3 s3 s5 (a2 c2 + a3 c23 + d5 s234)|
singular when
    s3 = 0                          elbow (arm fully stretched / folded)
    s5 = 0                          wrist (axes 4 and 6 parallel)
    a2 c2 + a3 c23 + d5 s234 = 0    shoulder (wrist center on axis 1/2 plane)
"""
import numpy as np

from barista_control import ur5_params as P


# ---- helpers ----------------------------------------------------------------
def skew(w):
    return np.array([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]])


def screw_exp(w, q, theta):
    """e^{[S] theta} for a revolute joint with unit axis w through point q."""
    W = skew(w)
    W2 = W @ W
    s, c = np.sin(theta), np.cos(theta)
    T = np.eye(4)
    T[:3, :3] = np.eye(3) + s * W + (1 - c) * W2
    T[:3, 3] = (theta * np.eye(3) + (1 - c) * W + (theta - s) * W2) @ -np.cross(w, q)
    return T


def wrap_to_pi(q):
    """Wrap angles to (-pi, pi]."""
    return -((-np.asarray(q) + np.pi) % (2 * np.pi) - np.pi)


def _poe_chain(q):
    """Partial PoE products [I, e^{S1 q1}, ..., e^{S1 q1}...e^{S6 q6}]."""
    Ts = [np.eye(4)]
    for w, p, theta in zip(P.SCREW_W, P.SCREW_Q, np.asarray(q, dtype=float)):
        Ts.append(Ts[-1] @ screw_exp(w, p, theta))
    return Ts


# ---- forward kinematics / Jacobian --------------------------------------------
def forward_kinematics(q):
    return _poe_chain(q)[-1] @ P.M


def jacobian(q):
    """Column i: current axis z_i and a point o_i on it (zero-config axis moved by
    the joints before it), giving [z_i x (p_tool - o_i); z_i]."""
    Ts = _poe_chain(q)
    p_tool = (Ts[-1] @ P.M)[:3, 3]
    z = np.array([T[:3, :3] @ w for T, w in zip(Ts, P.SCREW_W)])
    o = np.array([T[:3, :3] @ p + T[:3, 3] for T, p in zip(Ts, P.SCREW_Q)])
    return np.vstack([np.cross(z, p_tool - o).T, z.T])


# ---- inverse kinematics -------------------------------------------------------
def _pm_acos(c, tol=1e-9):
    """(+acos c, -acos c), or () if |c| > 1 (unreachable branch)."""
    if abs(c) > 1 + tol:
        return ()
    a = np.arccos(np.clip(c, -1.0, 1.0))
    return a, -a


def solve_ik(T, q6_if_singular=0.0):
    """All IK solutions for the pose T (base -> tool0).

    q6_if_singular is used for theta6 when the wrist is singular (s5 ~ 0) and
    theta6 is not determined by the pose. Returns a (k, 6) array, k in [0, 8],
    angles wrapped to (-pi, pi]; shape (0, 6) if unreachable.
    """
    T = np.asarray(T, dtype=float)
    R, p = T[:3, :3], T[:3, 3]
    x_hat, y_hat = R[:, 0], R[:, 1]

    # theta1 (shoulder left/right) from the frame-5 origin
    p5 = p - P.D6 * R[:, 2]
    r = np.hypot(p5[0], p5[1])
    if r < abs(P.D4) - 1e-9:
        return np.empty((0, 6))                 # wrist center too close to axis 1
    phi = np.arctan2(p5[1], p5[0]) + np.pi / 2
    psi = np.arccos(np.clip(P.D4 / r, -1.0, 1.0))

    solutions = []
    for t1 in (phi + psi, phi - psi):
        s1, c1 = np.sin(t1), np.cos(t1)
        T01_inv = np.linalg.inv(P.link_transform(0, t1))

        for t5 in _pm_acos((p[0] * s1 - p[1] * c1 - P.D4) / P.D6):    # wrist up/down
            s5 = np.sin(t5)
            if abs(s5) < 1e-8:
                t6 = q6_if_singular
            else:
                t6 = np.arctan2((-y_hat[0] * s1 + y_hat[1] * c1) / s5,
                                (x_hat[0] * s1 - x_hat[1] * c1) / s5)

            # theta2..4: planar 3R arm in frame 1
            T14 = T01_inv @ T @ np.linalg.inv(P.link_transform(4, t5) @ P.link_transform(5, t6))
            x, y = T14[0, 3], T14[1, 3]
            t234 = np.arctan2(T14[1, 0], T14[0, 0])
            c3 = (x * x + y * y - P.A2 ** 2 - P.A3 ** 2) / (2 * P.A2 * P.A3)

            for t3 in _pm_acos(c3):                                    # elbow up/down
                t2 = np.arctan2(y, x) - np.arctan2(P.A3 * np.sin(t3), P.A2 + P.A3 * np.cos(t3))
                solutions.append([t1, t2, t3, t234 - t2 - t3, t5, t6])

    return wrap_to_pi(np.array(solutions)) if solutions else np.empty((0, 6))


def solve_ik_closest(T, q_ref, q_min=P.Q_MIN, q_max=P.Q_MAX):
    """IK solution closest to q_ref (e.g. the current joint state), or None.

    Each joint is shifted by multiples of 2*pi to sit as close as possible to
    q_ref within the joint limits, avoiding 360 deg wrist spins.
    """
    q_ref = np.asarray(q_ref, dtype=float)
    sols = q_ref + wrap_to_pi(solve_ik(T, q6_if_singular=q_ref[5]) - q_ref)
    sols = np.where(sols > q_max, sols - 2 * np.pi, sols)
    sols = np.where(sols < q_min, sols + 2 * np.pi, sols)
    sols = sols[np.all((sols >= q_min) & (sols <= q_max), axis=1)]
    if len(sols) == 0:
        return None
    return sols[np.argmin(np.sum((sols - q_ref) ** 2, axis=1))]


# ---- manipulability -----------------------------------------------------------
def _singularity_factors(q):
    _, t2, t3, t4, t5, _ = np.asarray(q, dtype=float)
    return {
        "elbow": np.sin(t3),
        "wrist": np.sin(t5),
        "shoulder": P.A2 * np.cos(t2) + P.A3 * np.cos(t2 + t3) + P.D5 * np.sin(t2 + t3 + t4),
    }


def manipulability(q):
    """Yoshikawa manipulability w(q) = |det J(q)|, closed form [m^2]."""
    return float(abs(P.A2 * P.A3 * np.prod(list(_singularity_factors(q).values()))))


def yoshikawa(J):
    """sqrt(det(J J^T)) for any Jacobian (square or 6xn)."""
    J = np.asarray(J, dtype=float)
    return float(np.sqrt(max(np.linalg.det(J @ J.T), 0.0)))


def inverse_condition(J):
    """sigma_min / sigma_max of J (1 = isotropic, 0 = singular)."""
    s = np.linalg.svd(np.asarray(J, dtype=float), compute_uv=False)
    return float(s[-1] / s[0]) if s[0] > 0 else 0.0


def singularities(q, tol=1e-2):
    """Names of the singularity factors ('elbow', 'wrist', 'shoulder') below tol."""
    return [name for name, f in _singularity_factors(q).items() if abs(f) < tol]
