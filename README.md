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
