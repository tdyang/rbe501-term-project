"""Task-level motion planning for dynamic_controller.

Each task function decides the waypoints and timing for one motion, solves IK
and checks the path against the table, and returns (times, angles) ready for a
JointTrajectory message (positions only, the start not included).

Positions are in the base frame, z = 0 is the table top. 
The tool always faces the object from the side, so turning wrist 3 tilts the held object. 

Raises ValueError if a motion is unreachable or would hit the table.
"""
import numpy as np
from numpy.typing import NDArray

from barista_control.model import ur5_params as P
from barista_control.model.kinematics import _poe_chain, forward_kinematics, solve_ik_closest
from barista_control.planning.trajectory import Spline

Position = NDArray[np.float64]       # (3,)
Pose = NDArray[np.float64]           # (4, 4) homogeneous transform
JointAngles = NDArray[np.float64]    # (6,)
Times = NDArray[np.float64]          # (N,)
Path = tuple[Times, NDArray[np.float64]]   # (times, angles (N, 6)) without the start

# all lengths in m, speed in rad/s
GRASP_HEIGHT: float = 0.09    # tool height when grasping
PRE_GRASP: float = 0.10       # how far back from the object the approach starts
LIFT_HEIGHT: float = 0.12     # how far a lift raises the object
CARRY_HEIGHT: float = 0.25    # tool height when carrying across the table
SPEED: float = 0.05           # straight-line 
STEP: float = 0.02            # straight lines are split into steps
TILT_SPEED: float = 0.5       # wrist 3 speed when pouring (rad/s)
MOVE_TIME: float = 3.0        # duration of a free joint move
MIN_HEIGHT: float = 0.075     # lowest the arm may go, link radius + margin


# ----------- helpers

# turns point to tool pose at side of object, ready for grasp
def tool_side(position: Position, standoff: float = 0.0) -> Pose:
    facing: Position = np.r_[position[:2], 0.0] / np.hypot(position[0], position[1])
    up: Position = np.array([0.0, 0.0, 1.0])
    pose: Pose = np.eye(4)
    pose[:3, :3] = np.column_stack([up, np.cross(facing, up), facing])
    pose[:3, 3] = position - standoff * facing
    return pose

# Tool poses every step along a straight line and their durations
def line(start: Position, end: Position) -> tuple[list[Pose], list[float]]:
    distance: float = float(np.linalg.norm(end - start))
    n: int = max(1, int(np.ceil(distance / STEP)))
    poses: list[Pose] = [tool_side(start + (end - start) * k / n) for k in range(1, n + 1)]
    return poses, [distance / n / SPEED] * n


def tool_position(angles: JointAngles) -> Position:
    return forward_kinematics(angles)[:3, 3]


# (lowest point of the arm, tool0) heights above the table
# checks arm joints to make sure it stays about table
def heights(angles: JointAngles) -> tuple[float, float]:
    frames: list[NDArray[np.float64]] = _poe_chain(angles)
    tool: float = float((frames[-1] @ P.M)[2, 3])
    joints: list[float] = [float((frames[i][:3, :3] @ P.SCREW_Q[i] + frames[i][:3, 3])[2])
                           for i in (2, 3, 4, 5)]
    return min(joints + [tool]), tool


# Joint angles for each waypoint checked along the path the controller will follow
# arm stays above min_height and a held object hanging 
def plan(waypoints: list[Pose | JointAngles], durations: list[float], 
         start: JointAngles, hang: float = 0.0) -> Path:
    angles: list[JointAngles] = [np.asarray(start, float)]
    for i, waypoint in enumerate(waypoints):
        if waypoint.shape == (4, 4):
            waypoint = solve_ik_closest(waypoint, angles[-1])   # closest to the previous one
            if waypoint is None:
                raise ValueError(f'waypoint {i} is unreachable')
        angles.append(np.asarray(waypoint, float))

    times: Times = np.cumsum(np.r_[0.0, durations])
    path: Spline = Spline(times, angles)            # the path the controller will follow
    for t in np.linspace(0.0, times[-1], 500):
        arm, tool = heights(path.sample(t)[0])
        if arm < MIN_HEIGHT or tool - hang < -0.001:
            raise ValueError(f'path hits the table at t = {t:.2f} s')
    return times[1:], np.array(angles[1:])


# -------------- task motions

# move to pre_gasp back from the object, then straight in to touch its side
def approach(obj: Position, radius: float, start: JointAngles) -> Path:
    at_height: Position = obj + [0.0, 0.0, GRASP_HEIGHT]
    touch: Position = tool_side(at_height, radius)[:3, 3]
    before: Position = tool_side(at_height, radius + PRE_GRASP)[:3, 3]
    poses, durations = line(before, touch)
    return plan([tool_side(before)] + poses, [MOVE_TIME] + durations, start)

# raise the grasped object straight up by lift_height 
def lift(start: JointAngles) -> Path:
    here: Position = tool_position(start)
    poses, durations = line(here, here + [0.0, 0.0, LIFT_HEIGHT])
    return plan(poses, durations, start, hang=GRASP_HEIGHT)

# carry the held object over target at carry_height then lower it onto it
def lower(target: Position, radius: float, start: JointAngles) -> Path:
    above: Position = tool_side(np.r_[target[:2], CARRY_HEIGHT], radius)[:3, 3]
    down: Position = np.r_[above[:2], target[2] + GRASP_HEIGHT]
    poses, durations = line(above, down)
    return plan([tool_side(above)] + poses, [MOVE_TIME] + durations, start, hang=GRASP_HEIGHT)

# back the tool straight away from the released object by pre_gasp
def retreat(start: JointAngles) -> Path:
    here: Position = tool_position(start)
    poses, durations = line(here, tool_side(here, PRE_GRASP)[:3, 3])
    return plan(poses, durations, start)

# turn wrist 3 by angle [rad], the tool stays put and the held object tilts
def tilt(angle: float, start: JointAngles) -> Path:
    target: JointAngles = np.array(start, float)
    target[5] += angle
    return plan([target], [max(abs(angle) / TILT_SPEED, 1.0)], start, hang=GRASP_HEIGHT)

# move home
def home(start: JointAngles) -> Path:
    return plan([P.Q_HOME], [MOVE_TIME], start)

# (approach, lift) state machine attaches the object between the two
def pick_motion(obj: Position, radius: float, start: JointAngles) -> tuple[Path, Path]:
    approach_path: Path = approach(obj, radius, start)
    lift_path: Path = lift(approach_path[1][-1]) # starts where the approach ends
    return approach_path, lift_path

# (lower, retreat), state machine detaches the object between the two
def place_motion(target: Position, radius: float, start: JointAngles) -> tuple[Path, Path]:
    lower_path: Path = lower(target, radius, start)
    retreat_path: Path = retreat(lower_path[1][-1]) # starts where the lowering ends
    return lower_path, retreat_path
