"""UR5 rigid-body dynamics with a time-varying payload.

    M(q) qdd + c(q, qd) + g(q) = tau,      c(q, qd) = C(q, qd) qd

Usage:
    dyn = UR5Dynamics.from_urdf_file("ur5.urdf")      # or from_urdf_string(...)
    dyn.set_payload(mass=0.8, com=[0, 0, 0.12])        # pitcher, in tool0 frame
    tau = dyn.inverse_dynamics(q, qd, qdd)
    M, c, g = dyn.mass_matrix(q), dyn.coriolis(q, qd), dyn.gravity(q)

Getting the URDF:
    offline:  ros2 run xacro xacro barista.urdf.xacro > ur5.urdf
    in ROS:   subscribe to /robot_description (std_msgs/String, transient-local QoS)
"""
import xml.etree.ElementTree as ET

import numpy as np

from barista_control import ur5_params as P

GRAVITY = 9.81  # [m/s^2], acts along -z of the URDF root (world) frame


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _floats(text, default):
    return np.array([float(v) for v in text.split()]) if text else np.array(default, float)


def _rpy_to_R(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp,     cp * sr,                cp * cr],
    ])


def _origin(elem):
    """(R, p) of a URDF <origin> element (identity if missing)."""
    if elem is None:
        return np.eye(3), np.zeros(3)
    return _rpy_to_R(*_floats(elem.get("rpy"), [0, 0, 0])), _floats(elem.get("xyz"), [0, 0, 0])


def _cross(a, b):
    """3-vector cross product (np.cross is ~10x slower on tiny arrays)."""
    return np.array([a[1] * b[2] - a[2] * b[1],
                     a[2] * b[0] - a[0] * b[2],
                     a[0] * b[1] - a[1] * b[0]])


def _axis_rot(axis, angle):
    """Rodrigues rotation about a unit axis."""
    k = axis / np.linalg.norm(axis)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------
class UR5Dynamics:
    """Serial-chain dynamics for the six UR5 joints plus a payload on wrist_3_link."""

    def __init__(self, joints, links, g_root):
        # joints: list of dicts {R0, p0, axis}: fixed origin (parent->joint frame) and axis
        # links:  list of dicts {m, c, I}: mass, COM in link frame, inertia about COM in link frame
        self._joints = joints
        self._links = links
        self._g_root = g_root               # gravity vector expressed in DH frame 0
        self.set_payload(0.0)

    # ---- construction -----------------------------------------------------
    @classmethod
    def from_urdf_file(cls, path, joint_names=P.JOINT_NAMES):
        return cls._from_root(ET.parse(path).getroot(), joint_names)

    @classmethod
    def from_urdf_string(cls, xml, joint_names=P.JOINT_NAMES):
        return cls._from_root(ET.fromstring(xml), joint_names)

    @classmethod
    def _from_root(cls, root, joint_names):
        by_name = {j.get("name"): j for j in root.findall("joint")}
        by_child = {j.find("child").get("link"): j for j in root.findall("joint")}
        link_elems = {l.get("name"): l for l in root.findall("link")}

        joints, links = [], []
        for name in joint_names:
            if name not in by_name:
                raise ValueError(f"joint '{name}' not found in URDF (tf_prefix mismatch?)")
            j = by_name[name]
            R0, p0 = _origin(j.find("origin"))
            axis = _floats(j.find("axis").get("xyz") if j.find("axis") is not None else None, [1, 0, 0])
            joints.append({"R0": R0, "p0": p0, "axis": axis / np.linalg.norm(axis)})

            child = link_elems[j.find("child").get("link")]
            inert = child.find("inertial")
            if inert is None:
                links.append({"m": 0.0, "c": np.zeros(3), "I": np.zeros((3, 3))})
                continue
            Ri, ci = _origin(inert.find("origin"))
            t = inert.find("inertia").attrib
            I_local = np.array([
                [float(t["ixx"]), float(t["ixy"]), float(t["ixz"])],
                [float(t["ixy"]), float(t["iyy"]), float(t["iyz"])],
                [float(t["ixz"]), float(t["iyz"]), float(t["izz"])],
            ])
            links.append({"m": float(inert.find("mass").get("value")),
                          "c": ci, "I": Ri @ I_local @ Ri.T})

        # gravity in the chain's root frame: walk fixed joints up to the URDF root
        root_link = by_name[joint_names[0]].find("parent").get("link")
        R = np.eye(3)
        while root_link in by_child:
            j = by_child[root_link]
            if j.get("type") not in ("fixed", None):
                raise ValueError("chain root is not fixed to the world")
            R = _origin(j.find("origin"))[0] @ R
            root_link = j.find("parent").get("link")
        g_root = R.T @ np.array([0.0, 0.0, -GRAVITY])
        return cls(joints, links, g_root)

    # ---- payload ----------------------------------------------------------
    def set_payload(self, mass, com=(0.0, 0.0, 0.0), inertia=None):
        """Set the grasped object's inertial parameters.

        Args:
            mass:    [kg] current estimate (e.g. from the wrist F/T sensor).
            com:     [m] payload center of mass in the tool0 / wrist_3_link frame.
            inertia: 3x3 [kg m^2] about the payload COM, tool0 axes.
                     None = point mass (fine for a small pitcher).

        Call this every control step to make the model time-varying.
        """
        m_l = self._links[-1]
        m_p = float(mass)
        c_p = np.asarray(com, float)
        I_p = np.zeros((3, 3)) if inertia is None else np.asarray(inertia, float)
        self.payload_mass, self.payload_com = m_p, c_p

        # combine wrist_3_link + payload into one rigid body (parallel-axis theorem)
        m = m_l["m"] + m_p
        if m <= 0:
            self._tip = dict(m_l)
            return
        c = (m_l["m"] * m_l["c"] + m_p * c_p) / m

        def shift(I, mass_, r):
            return I + mass_ * (r @ r * np.eye(3) - np.outer(r, r))

        I = shift(m_l["I"], m_l["m"], m_l["c"] - c) + shift(I_p, m_p, c_p - c)
        self._tip = {"m": m, "c": c, "I": I}

    # ---- core algorithm ---------------------------------------------------
    def _link(self, i):
        return self._tip if i == len(self._links) - 1 else self._links[i]

    def _rotations(self, q):
        return [jt["R0"] @ _axis_rot(jt["axis"], qi) for jt, qi in zip(self._joints, q)]

    def _rne(self, q, qd, qdd, gravity=True, Rs=None):
        """Recursive Newton-Euler. Returns joint torques (6,).
        Rs: precomputed joint rotations for this q (saves time in mass_matrix)."""
        n = len(self._joints)
        qd, qdd = np.asarray(qd, float), np.asarray(qdd, float)
        if Rs is None:
            Rs = self._rotations(np.asarray(q, float))

        ps = []
        w = np.zeros(3)
        wd = np.zeros(3)
        vd = -self._g_root if gravity else np.zeros(3)   # gravity as base acceleration
        F, N = [], []

        # forward pass: velocities and accelerations, link frames
        for i, jt in enumerate(self._joints):
            R = Rs[i]                                        # parent <- child rotation
            p = jt["p0"]                                     # child origin in parent
            ps.append(p)
            a = jt["axis"]

            vd = R.T @ (vd + _cross(wd, p) + _cross(w, _cross(w, p)))
            w_parent = R.T @ w
            wd = R.T @ wd + _cross(w_parent, a * qd[i]) + a * qdd[i]
            w = w_parent + a * qd[i]

            L = self._link(i)
            vdc = vd + _cross(wd, L["c"]) + _cross(w, _cross(w, L["c"]))
            F.append(L["m"] * vdc)
            N.append(L["I"] @ wd + _cross(w, L["I"] @ w))

        # backward pass: forces and torques
        tau = np.zeros(n)
        f = np.zeros(3)
        nt = np.zeros(3)
        for i in reversed(range(n)):
            L = self._link(i)
            if i < n - 1:
                f_next = Rs[i + 1] @ f
                n_next = Rs[i + 1] @ nt + _cross(ps[i + 1], f_next)
            else:
                f_next = n_next = np.zeros(3)
            f = F[i] + f_next
            nt = N[i] + n_next + _cross(L["c"], F[i])
            tau[i] = nt @ self._joints[i]["axis"]
        return tau

    # ---- public API -------------------------------------------------------
    def inverse_dynamics(self, q, qd, qdd):
        """tau = M qdd + c + g."""
        return self._rne(q, qd, qdd)

    def gravity(self, q):
        """g(q): joint torques that hold the arm still against gravity."""
        z = np.zeros(len(self._joints))
        return self._rne(q, z, z)

    def coriolis(self, q, qd):
        """c(q, qd) = C(q, qd) qd: Coriolis and centrifugal torques."""
        return self._rne(q, qd, np.zeros(len(self._joints)), gravity=False)

    def mass_matrix(self, q):
        """M(q): 6x6 symmetric positive-definite inertia matrix."""
        n = len(self._joints)
        z = np.zeros(n)
        Rs = self._rotations(np.asarray(q, float))
        M = np.column_stack([self._rne(q, z, e, gravity=False, Rs=Rs) for e in np.eye(n)])
        return 0.5 * (M + M.T)

    def forward_dynamics(self, q, qd, tau):
        """qdd = M^-1 (tau - c - g)."""
        return np.linalg.solve(self.mass_matrix(q), tau - self.coriolis(q, qd) - self.gravity(q))

    def link_poses(self, q):
        """4x4 pose of each moving link frame in DH frame 0 (for testing / plotting)."""
        T, out = np.eye(4), []
        for i, jt in enumerate(self._joints):
            A = np.eye(4)
            A[:3, :3] = jt["R0"] @ _axis_rot(jt["axis"], q[i])
            A[:3, 3] = jt["p0"]
            T = T @ A
            out.append(T.copy())
        return out