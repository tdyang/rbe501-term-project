"""ROS 2 node exposing UR5 IK and manipulability.

Subscribes
    /joint_states              sensor_msgs/JointState     current robot state
    ~/target_pose              geometry_msgs/PoseStamped  desired tool0 pose in `base`

Publishes
    ~/tool_pose                geometry_msgs/PoseStamped  FK: tool0 pose in `base`
    ~/manipulability           std_msgs/Float64           w(q) = |det J(q)|
    ~/inverse_condition        std_msgs/Float64           sigma_min/sigma_max of J (0 = singular)
    ~/ik_solution              sensor_msgs/JointState     IK solution closest to current q

Parameters
    singularity_threshold (float, default 1e-3)  warn when w drops below this

Run:
    ros2 run barista_control ur5_kinematics_node
Test:
    ros2 topic pub --once /ur5_kinematics/target_pose geometry_msgs/PoseStamped \
      "{header: {frame_id: base}, pose: {position: {x: 0.3, y: -0.2, z: 0.4},
        orientation: {x: 1.0, y: 0.0, z: 0.0, w: 0.0}}}"
"""
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64

from barista_control import ur5_params as P
from barista_control.kinematics import (
    forward_kinematics, jacobian, solve_ik_closest,
    manipulability, yoshikawa, inverse_condition, singularities)


def quat_to_rot(x, y, z, w):
    """Unit quaternion -> 3x3 rotation matrix (no tf dependency)."""
    n = np.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ])


def rot_to_quat(R):
    """3x3 rotation matrix -> unit quaternion (x, y, z, w)."""
    tr = np.trace(R)
    if tr > 0:
        s = 2.0 * np.sqrt(tr + 1.0)
        return ((R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s,
                (R[1, 0] - R[0, 1]) / s, 0.25 * s)
    i = int(np.argmax(np.diag(R)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = 2.0 * np.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k])
    q = np.zeros(4)
    q[i] = 0.25 * s
    q[j] = (R[j, i] + R[i, j]) / s
    q[k] = (R[k, i] + R[i, k]) / s
    q[3] = (R[k, j] - R[j, k]) / s
    return tuple(q)


def matrix_to_pose(T):
    pose = PoseStamped().pose
    pose.position.x, pose.position.y, pose.position.z = (float(v) for v in T[:3, 3])
    (pose.orientation.x, pose.orientation.y,
     pose.orientation.z, pose.orientation.w) = (float(v) for v in rot_to_quat(T[:3, :3]))
    return pose


def pose_to_matrix(pose):
    T = np.eye(4)
    o, p = pose.orientation, pose.position
    T[:3, :3] = quat_to_rot(o.x, o.y, o.z, o.w)
    T[:3, 3] = [p.x, p.y, p.z]
    return T


class UR5KinematicsNode(Node):
    def __init__(self):
        super().__init__("ur5_kinematics")
        self.declare_parameter("singularity_threshold", 1e-3)

        self.q = None  # current joints, kinematic order

        self.create_subscription(JointState, "/joint_states", self.on_joint_state, 10)
        self.create_subscription(PoseStamped, "~/target_pose", self.on_target_pose, 10)
        self.pub_pose = self.create_publisher(PoseStamped, "~/tool_pose", 10)
        self.pub_w = self.create_publisher(Float64, "~/manipulability", 10)
        self.pub_cond = self.create_publisher(Float64, "~/inverse_condition", 10)
        self.pub_ik = self.create_publisher(JointState, "~/ik_solution", 10)

        self.get_logger().info("UR5 kinematics node ready.")

    def on_joint_state(self, msg):
        idx = {name: i for i, name in enumerate(msg.name)}
        if not all(n in idx for n in P.JOINT_NAMES):
            return
        self.q = np.array([msg.position[idx[n]] for n in P.JOINT_NAMES])

        # forward kinematics: current tool0 pose
        out = PoseStamped()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = "base"
        out.pose = matrix_to_pose(forward_kinematics(self.q))
        self.pub_pose.publish(out)

        # manipulability from the Jacobian
        J = jacobian(self.q)
        w = yoshikawa(J)
        self.pub_w.publish(Float64(data=w))
        self.pub_cond.publish(Float64(data=inverse_condition(J)))

        thresh = self.get_parameter("singularity_threshold").value
        if w < thresh:
            self.get_logger().warn(
                f"Near singularity: w={w:.2e} ({', '.join(singularities(self.q)) or 'combined'})",
                throttle_duration_sec=1.0)

    def on_target_pose(self, msg):
        if msg.header.frame_id not in ("", "base"):
            self.get_logger().warn(
                f"target_pose frame is '{msg.header.frame_id}', expected 'base'. "
                "Transform it first (base_link is rotated 180 deg about z).")
        T_target = pose_to_matrix(msg.pose)
        q_sol = solve_ik_closest(T_target, self.q if self.q is not None else P.Q_HOME)
        if q_sol is None:
            self.get_logger().error("No IK solution: target pose unreachable.")
            return

        # check: the solution must reproduce the target through FK
        err = np.linalg.norm(forward_kinematics(q_sol)[:3, 3] - T_target[:3, 3])
        if err > 1e-4:
            self.get_logger().error(f"IK/FK mismatch of {err * 1000:.2f} mm, not publishing.")
            return

        out = JointState()
        out.header.stamp = self.get_clock().now().to_msg()
        out.name = list(P.JOINT_NAMES)
        out.position = [float(v) for v in q_sol]
        self.pub_ik.publish(out)
        self.get_logger().info(
            "IK [deg]: " + np.array2string(np.degrees(q_sol), precision=1)
            + f"   w={manipulability(q_sol):.4f}")


def main(args=None):
    rclpy.init(args=args)
    node = UR5KinematicsNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()