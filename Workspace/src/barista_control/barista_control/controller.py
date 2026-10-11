import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from barista_control import ur5_params as P
from barista_control.kinematics import forward_kinematics, jacobian


class CartesianController(Node):
    def __init__(self):
        super().__init__('cartesian_controller')
        self.declare_parameter('kp', 1.0)
        self.declare_parameter('damping', 0.05)
        self.declare_parameter('control_rate', 50.0)
        self.declare_parameter('max_joint_speed', 0.3)
        self.declare_parameter('max_cartesian_speed', 0.1)
        self.declare_parameter('position_tolerance', 0.01)
        self.declare_parameter('joint_limit_margin', 0.1)
        self.declare_parameter('joint_state_timeout', 0.5)
        self.declare_parameter('target_timeout', 30.0)

        self.q = None
        self.joint_state_time = None
        self.target = None
        self.target_time = None
        self.reached = False
        self.last_warning = None

        self.create_subscription(JointState, '/joint_states', self.on_joint_state, 10)
        self.create_subscription(PoseStamped, '/cartesian_target', self.on_target, 10)
        self.command_pub = self.create_publisher(
            Float64MultiArray, '/forward_velocity_controller/commands', 10
        )
        rate = float(self.get_parameter('control_rate').value)
        if rate <= 0:
            raise ValueError('control_rate must be positive')
        self.create_timer(1.0 / rate, self.control_step)
        self.get_logger().info('Cartesian position controller ready (simulation only).')

    def now_seconds(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def warn_throttled(self, message):
        now = self.now_seconds()
        if self.last_warning is None or now - self.last_warning >= 2.0:
            self.get_logger().warn(message)
            self.last_warning = now

    def on_joint_state(self, msg):
        indices = {name: i for i, name in enumerate(msg.name)}
        if not all(name in indices for name in P.JOINT_NAMES):
            return
        if len(msg.position) < len(msg.name):
            return
        q = np.array([msg.position[indices[name]] for name in P.JOINT_NAMES], dtype=float)
        if not np.all(np.isfinite(q)):
            return
        self.q = q
        self.joint_state_time = self.now_seconds()

    def on_target(self, msg):
        if msg.header.frame_id != 'base':
            self.get_logger().error("Target frame must be 'base'; target ignored.")
            return
        p = msg.pose.position
        target = np.array([p.x, p.y, p.z], dtype=float)
        if not np.all(np.isfinite(target)):
            self.get_logger().error('Non-finite target; ignored.')
            return
        self.target = target
        self.target_time = self.now_seconds()
        self.reached = False
        self.get_logger().info(f'New target [m]: {target.tolist()}')

    def send_velocity(self, velocity):
        msg = Float64MultiArray()
        msg.data = [float(x) for x in velocity]
        self.command_pub.publish(msg)

    def stop(self):
        self.send_velocity(np.zeros(P.N_JOINTS))

    def control_step(self):
        now = self.now_seconds()
        if self.q is None or self.joint_state_time is None:
            self.stop()
            return
        if now - self.joint_state_time > float(self.get_parameter('joint_state_timeout').value):
            self.stop()
            self.warn_throttled('Joint states are stale; sending zero velocities.')
            return
        if self.target is None or self.target_time is None:
            self.stop()
            return
        if now - self.target_time > float(self.get_parameter('target_timeout').value):
            self.stop()
            self.warn_throttled('Target expired; sending zero velocities.')
            return

        q = self.q.copy()
        margin = float(self.get_parameter('joint_limit_margin').value)
        q_min = np.asarray(P.Q_MIN, dtype=float) + margin
        q_max = np.asarray(P.Q_MAX, dtype=float) - margin
        if np.any(q < q_min) or np.any(q > q_max):
            self.stop()
            self.warn_throttled('Joint outside configured safety margin; stopped.')
            return

        p_current = forward_kinematics(q)[:3, 3]
        error = self.target - p_current
        distance = float(np.linalg.norm(error))
        if not np.isfinite(distance):
            self.stop()
            return
        if distance <= float(self.get_parameter('position_tolerance').value):
            self.stop()
            if not self.reached:
                self.get_logger().info(f'Target reached (position error {distance:.4f} m).')
                self.reached = True
            return

        self.reached = False
        v_des = float(self.get_parameter('kp').value) * error
        max_cart = float(self.get_parameter('max_cartesian_speed').value)
        if max_cart <= 0:
            self.stop()
            return
        v_norm = float(np.linalg.norm(v_des))
        if v_norm > max_cart:
            v_des *= max_cart / v_norm

        Jv = jacobian(q)[:3, :]
        damping = float(self.get_parameter('damping').value)
        if damping <= 0:
            self.stop()
            self.warn_throttled('Damping must be positive.')
            return
        A = Jv @ Jv.T + damping ** 2 * np.eye(3)
        try:
            q_dot = Jv.T @ np.linalg.solve(A, v_des)
        except np.linalg.LinAlgError:
            self.stop()
            self.warn_throttled('Jacobian solve failed.')
            return

        max_joint = float(self.get_parameter('max_joint_speed').value)
        if max_joint <= 0 or not np.all(np.isfinite(q_dot)):
            self.stop()
            return
        q_dot = np.clip(q_dot, -max_joint, max_joint)
        q_dot[(q >= q_max) & (q_dot > 0)] = 0.0
        q_dot[(q <= q_min) & (q_dot < 0)] = 0.0
        self.send_velocity(q_dot)


def main(args=None):
    rclpy.init(args=args)
    node = CartesianController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
