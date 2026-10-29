"""ROS 2 node: computed-torque joint control with a time-varying payload model.

Subscribes
    /robot_description   std_msgs/String                  URDF for the dynamic model
    /joint_states        sensor_msgs/JointState           q, qd
    ~/joint_trajectory   trajectory_msgs/JointTrajectory  reference; blends in from the
                                                          current one, stamp 0 = now
    ~/payload            geometry_msgs/Inertia            grasped object in tool0
                                                          (m, com, inertia about the COM)
Publishes
    /forward_effort_controller/commands  std_msgs/Float64MultiArray  joint torques
    ~/reference          sensor_msgs/JointState           q_d, qd_d, commanded tau
    ~/tracking_error     std_msgs/Float64MultiArray       q_d - q [rad]
Parameters
    control_rate (500.0 Hz), natural_frequency (20.0 rad/s), damping_ratio (1.0),
    use_payload (true; false = fixed-payload baseline),
    joint_state_timeout (0.1 s; stale states -> gravity compensation only),
    go_home_duration (0.0 s; > 0 moves to Q_HOME on start instead of holding,
                      recovering the sag before the node takes over)

Test:
    ros2 topic pub --once /dynamic_controller/joint_trajectory trajectory_msgs/JointTrajectory \
      "{joint_names: [shoulder_pan_joint, shoulder_lift_joint, elbow_joint, wrist_1_joint,
        wrist_2_joint, wrist_3_joint],
        points: [{positions: [0.5, -1.2, 1.0, -1.4, -1.57, 0.0], time_from_start: {sec: 4}}]}"
    ros2 topic pub --once /dynamic_controller/payload geometry_msgs/Inertia "{m: 1.0}"
"""
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from geometry_msgs.msg import Inertia
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, String
from trajectory_msgs.msg import JointTrajectory

from barista_control import ur5_params as P
from barista_control.computed_torque import ComputedTorqueController
from barista_control.dynamics import UR5Dynamics
from barista_control.payload import Body
from barista_control.trajectory import Spline


def spline_from_msg(msg, now, current):
    """JointTrajectory -> Spline in node time, starting from `current` if the first
    point is in the future. Raises ValueError on a malformed or out-of-limit message."""
    if not msg.points:
        raise ValueError('empty trajectory')
    try:
        idx = [list(msg.joint_names).index(n) for n in P.JOINT_NAMES]
    except ValueError:
        raise ValueError('trajectory must name all six UR5 joints') from None

    def column(field):
        rows = [getattr(p, field) for p in msg.points]
        if not all(len(r) == len(msg.joint_names) for r in rows):
            return None
        return np.array([[r[i] for i in idx] for r in rows])

    q, qd, qdd = column('positions'), column('velocities'), column('accelerations')
    if q is None or not np.all(np.isfinite(q)):
        raise ValueError('every point needs six finite positions')
    if np.any(q < P.Q_MIN) or np.any(q > P.Q_MAX):
        raise ValueError('trajectory exceeds joint limits')
    stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
    t = (stamp or now) + np.array([p.time_from_start.sec + p.time_from_start.nanosec * 1e-9
                                   for p in msg.points])
    if t[0] > now:      # blend in from the current reference
        start = current.sample(now)
        t, q = np.r_[now, t], np.vstack([start[0], q])
        qd = None if qd is None else np.vstack([start[1], qd])
        qdd = None if qdd is None else np.vstack([start[2], qdd])
    return Spline(t, q, qd, qdd)


def _array(x):
    return [float(v) for v in x]


class DynamicController(Node):
    def __init__(self):
        super().__init__('dynamic_controller')
        for name, default in [('control_rate', 500.0), ('natural_frequency', 20.0),
                              ('damping_ratio', 1.0), ('use_payload', True),
                              ('joint_state_timeout', 0.1), ('go_home_duration', 0.0)]:
            self.declare_parameter(name, default)
        self.ctc = self.q = self.qd = self.spline = None
        self.state_time, self.done = 0.0, True

        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(String, '/robot_description', self.on_description, latched)
        self.create_subscription(JointState, '/joint_states', self.on_joint_state, 10)
        self.create_subscription(JointTrajectory, '~/joint_trajectory', self.on_trajectory, 10)
        self.create_subscription(Inertia, '~/payload', self.on_payload, 10)
        self.tau_pub = self.create_publisher(
            Float64MultiArray, '/forward_effort_controller/commands', 10)
        self.ref_pub = self.create_publisher(JointState, '~/reference', 10)
        self.err_pub = self.create_publisher(Float64MultiArray, '~/tracking_error', 10)
        self.create_timer(1.0 / self.param('control_rate'), self.control_step)

    def param(self, name):
        return self.get_parameter(name).value

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def on_description(self, msg):
        if self.ctc is None:
            self.ctc = ComputedTorqueController(UR5Dynamics.from_urdf_string(msg.data),
                                                self.param('natural_frequency'),
                                                self.param('damping_ratio'))
            self.get_logger().info('Dynamic model loaded.')

    def on_joint_state(self, msg):
        idx = {n: i for i, n in enumerate(msg.name)}
        if not all(n in idx for n in P.JOINT_NAMES) or len(msg.velocity) < len(msg.name):
            return
        self.q = np.array([msg.position[idx[n]] for n in P.JOINT_NAMES])
        self.qd = np.array([msg.velocity[idx[n]] for n in P.JOINT_NAMES])
        self.state_time = self.now()
        if self.spline is None:
            T, now = self.param('go_home_duration'), self.now()
            goal = P.Q_HOME if T > 0 else self.q
            self.spline = Spline([now, now + max(T, 1.0)], [self.q, goal])
            self.get_logger().info('Moving to home.' if T > 0 else 'Holding current pose.')

    def on_trajectory(self, msg):
        if self.spline is None:
            return self.get_logger().error('No joint state yet; trajectory ignored.')
        try:
            self.spline = spline_from_msg(msg, self.now(), self.spline)
        except ValueError as exc:
            return self.get_logger().error(f'Trajectory ignored: {exc}')
        self.done = False
        self.get_logger().info(f'New trajectory, ends in {self.spline.end_time - self.now():.2f} s.')

    def on_payload(self, msg):
        if self.ctc is None or not self.param('use_payload'):
            return
        I = np.array([[msg.ixx, msg.ixy, msg.ixz], [msg.ixy, msg.iyy, msg.iyz],
                      [msg.ixz, msg.iyz, msg.izz]])
        c = np.array([msg.com.x, msg.com.y, msg.com.z])
        if msg.m >= 0 and np.all(np.isfinite(np.r_[msg.m, c, I.ravel()])):
            self.ctc.dyn.set_payload(Body(msg.m, c, I))

    def control_step(self):
        if self.ctc is None or self.q is None:
            return
        now = self.now()
        if now - self.state_time > self.param('joint_state_timeout'):
            self.get_logger().warn('Stale joint states: gravity compensation only.',
                                   throttle_duration_sec=2.0)
            return self.tau_pub.publish(
                Float64MultiArray(data=_array(self.ctc.clip(self.ctc.dyn.gravity(self.q)))))

        q_d, qd_d, qdd_d = self.spline.sample(now)
        tau = self.ctc.torque(self.q, self.qd, q_d, qd_d, qdd_d)
        self.tau_pub.publish(Float64MultiArray(data=_array(tau)))
        ref = JointState(name=P.JOINT_NAMES, position=_array(q_d),
                         velocity=_array(qd_d), effort=_array(tau))
        ref.header.stamp = self.get_clock().now().to_msg()
        self.ref_pub.publish(ref)
        self.err_pub.publish(Float64MultiArray(data=_array(q_d - self.q)))
        if not self.done and now >= self.spline.end_time:
            self.done = True
            self.get_logger().info(
                f'Trajectory done, max joint error {np.max(np.abs(q_d - self.q)):.4f} rad.')


def main(args=None):
    rclpy.init(args=args)
    node = DynamicController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
