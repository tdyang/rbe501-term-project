#!/usr/bin/env python3
"""Drive dynamic_controller to a pre-grasp pose above the pitcher and record the run.

For each move duration: start a rosbag, go home -> pitcher -> home, stop the bag.
One bag per duration, holding tracking error, reference (q_d, qd_d, tau) and joint states.

Run with the sim up in dynamic mode (ros2 launch barista_gazebo barista_sim.launch.xml
dynamic:=true) and the workspace sourced:
    python3 scripts/collect_pitcher.py                    # 4 s and 1.5 s moves
    python3 scripts/collect_pitcher.py -d 4 2 1.5 --hover 0.25 -o bags
"""
import argparse
import os
import signal
import subprocess
import sys
import time

import numpy as np
import rclpy
from builtin_interfaces.msg import Duration
from rclpy.node import Node
from rclpy.parameter import Parameter
from tf2_msgs.msg import TFMessage
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from barista_control.model import ur5_params as P
from barista_control.model.kinematics import forward_kinematics, solve_ik_closest

TOPICS = ['/dynamic_controller/tracking_error', '/dynamic_controller/reference',
          '/joint_states']
TABLE_HEIGHT = 0.75                   # robot base sits on the table top [m]
TOOL_DOWN = np.diag([1.0, -1.0, -1.0])  # tool0 z axis pointing down, in `base`


def world_to_base(p_world):
    """World point -> UR `base` frame (base on the table, rotated 180 deg about z)."""
    x, y, z = p_world
    return np.array([-x, -y, z - TABLE_HEIGHT])


class Collector(Node):
    def __init__(self):
        super().__init__('data_collector',
                         parameter_overrides=[Parameter('use_sim_time', value=True)])
        self.pub = self.create_publisher(
            JointTrajectory, '/dynamic_controller/joint_trajectory', 10)

    def sim_now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def wait_for_clock(self, timeout=10.0):
        end = time.time() + timeout
        while self.sim_now() == 0.0:
            if time.time() > end:
                raise RuntimeError('no /clock: is the simulation running?')
            rclpy.spin_once(self, timeout_sec=0.1)

    def sleep_sim(self, seconds):
        """Wait in sim time, so a real-time factor below 1 doesn't cut runs short."""
        end = self.sim_now() + seconds
        while self.sim_now() < end:
            rclpy.spin_once(self, timeout_sec=0.05)

    def wait_for(self, msg_type, topic, timeout=5.0):
        box = []
        sub = self.create_subscription(msg_type, topic, box.append, 10)
        end = time.time() + timeout
        while not box and time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.1)
        self.destroy_subscription(sub)
        return box[0] if box else None

    def send(self, q, seconds):
        """Publish a single-point trajectory once the controller is subscribed."""
        end = time.time() + 10.0
        while self.pub.get_subscription_count() == 0:
            if time.time() > end:
                raise RuntimeError('dynamic_controller is not subscribed to its trajectory topic')
            rclpy.spin_once(self, timeout_sec=0.1)
        sec = int(seconds)
        point = JointTrajectoryPoint(
            positions=[float(x) for x in q],
            time_from_start=Duration(sec=sec, nanosec=int(round((seconds - sec) * 1e9))))
        self.pub.publish(JointTrajectory(joint_names=list(P.JOINT_NAMES), points=[point]))

    def move(self, q, seconds, settle):
        self.send(q, seconds)
        self.sleep_sim(seconds + settle)


def pitcher_position(node):
    msg = node.wait_for(TFMessage, '/model/pitcher/pose')
    if msg is None or not msg.transforms:
        raise RuntimeError('no message on /model/pitcher/pose')
    tf = next((t for t in msg.transforms if 'pitcher' in t.child_frame_id), msg.transforms[0])
    p = tf.transform.translation
    return np.array([p.x, p.y, p.z])


def start_bag(path):
    proc = subprocess.Popen(['ros2', 'bag', 'record', '--use-sim-time', '-o', path] + TOPICS,
                            stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    time.sleep(2.0)                   # let the recorder discover the topics
    if proc.poll() is not None:
        raise RuntimeError(f'ros2 bag record exited early (does {path} already exist?)')
    return proc


def stop_bag(proc):
    proc.send_signal(signal.SIGINT)   # SIGINT, not kill: the bag must be closed cleanly
    try:
        proc.wait(timeout=10.0)
    except subprocess.TimeoutExpired:
        proc.kill()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('-d', '--durations', type=float, nargs='+', default=[4.0, 1.5],
                    help='move durations to record [s]')
    ap.add_argument('--hover', type=float, default=0.20,
                    help='tool0 height above the pitcher origin [m]')
    ap.add_argument('--settle', type=float, default=2.0,
                    help='time to hold after each move [s]')
    ap.add_argument('-o', '--out-dir', default='bags', help='directory for the bags')
    args = ap.parse_args()

    rclpy.init()
    node = Collector()
    bag = None
    try:
        node.wait_for_clock()

        p_world = pitcher_position(node)
        p_base = world_to_base(p_world) + [0.0, 0.0, args.hover]
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = TOOL_DOWN, p_base
        q_pitcher = solve_ik_closest(T, P.Q_HOME)
        if q_pitcher is None:
            raise RuntimeError(f'pitcher target {p_base.round(3).tolist()} (base frame) '
                               'is unreachable; check the pose or change --hover')
        err = np.linalg.norm(forward_kinematics(q_pitcher)[:3, 3] - p_base)
        print(f'pitcher (world):  {p_world.round(3).tolist()}')
        print(f'target  (base):   {p_base.round(3).tolist()}  (FK check {err:.1e} m)')
        print(f'q_pitcher [rad]:  {q_pitcher.round(4).tolist()}')

        print('moving home (not recorded) ...')
        node.move(P.Q_HOME, 4.0, args.settle)

        os.makedirs(args.out_dir, exist_ok=True)
        stamp = time.strftime('%Y%m%d_%H%M%S')
        for seconds in args.durations:
            path = os.path.join(args.out_dir, f'pitcher_{seconds:g}s_{stamp}'.replace('.', 'p'))
            print(f'[{seconds:g} s] recording to {path}')
            bag = start_bag(path)
            node.sleep_sim(1.0)       # some data at rest before the move
            node.move(q_pitcher, seconds, args.settle)
            node.move(P.Q_HOME, seconds, args.settle)
            stop_bag(bag)
            bag = None
        print('done.')
    except (RuntimeError, KeyboardInterrupt) as exc:
        print(f'aborted: {exc}', file=sys.stderr)
        return 1
    finally:
        if bag is not None:
            stop_bag(bag)
        node.destroy_node()
        rclpy.try_shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())
