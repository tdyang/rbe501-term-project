# RBE 501 Term Project – Barista Robot

UR5e-based barista robot, simulated in ROS 2 Jazzy + Gazebo.

## Packages

| Package | What it is |
|---|---|
| `barista_description` | URDF/xacro for the robot, RViz display launch |
| `barista_gazebo` | Gazebo world + sim launch (wraps `ur_simulation_gz`) |
| `barista_control` | Python control nodes? |

## Prerequisites

- Ubuntu 24.04 with [ROS 2 Jazzy](https://docs.ros.org/en/jazzy/Installation.html)
- `rosdep` and `colcon`:
  ```bash
  sudo apt install python3-rosdep python3-colcon-common-extensions
  ```

## Build

```bash
cd rbe501-term-project/Workspace

source /opt/ros/jazzy/setup.bash
sudo rosdep init   # first time only, skip if already done
rosdep update
rosdep install --from-paths src --ignore-src -y

colcon build --symlink-install
source install/setup.bash
```

## Run

View the robot in RViz:
```bash
ros2 launch barista_description display.launch.xml
```

Launch the Gazebo sim:
```bash
ros2 launch barista_gazebo barista_sim.launch.xml
```

Launch with torque control (computed-torque `dynamic_controller` with the payload model):
```bash
ros2 launch barista_gazebo barista_sim.launch.xml initial_joint_controller:=forward_effort_controller
```
Send it a `trajectory_msgs/JointTrajectory` on `/dynamic_controller/joint_trajectory` and the
grasped object's `geometry_msgs/Inertia` (tool0 frame) on `/dynamic_controller/payload`.
See the docstring of `barista_control/nodes/dynamic_controller.py` for an example.

Simulated sensors (bridged by `config/sensors_bridge.yaml`):

| Topic | Type | Sensor |
|---|---|---|
| `/wrist_ft` | `geometry_msgs/WrenchStamped` | F/T in `wrist_3_joint`, tool0 frame; includes wrist_3_link (0.1879 kg) |
| `/scale/wrench` | `geometry_msgs/WrenchStamped` | scale load cell; `force.z / 9.8 - 0.2` = load [kg] |
| `/camera/{image,depth_image,points,camera_info}` | `sensor_msgs/...` | overhead RGB-D, frame `camera_optical_frame` (TF from `world`) |
