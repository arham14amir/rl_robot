# rl_robot

A 4-wheel **skid-steer** robot with a top-mounted **2D lidar**, built as a
reinforcement-learning platform for **obstacle avoidance / goal reaching** on
**ROS 2 Humble + Gazebo Classic 11**.

The robot, sim, sensors and drive are done. The RL layer (Gym env, reward,
training) is intentionally left for you to add — this package gives you the
clean ROS interface an RL env plugs into.

## Robot at a glance

| Part        | Detail                                                        |
|-------------|---------------------------------------------------------------|
| Chassis     | 0.35 × 0.22 × 0.10 m box, 4 kg, low centre of mass            |
| Wheels      | 4 × Ø0.12 m, skid-steer (left pair + right pair)             |
| Lidar       | 360-beam 2D scan, 0.12–12 m, 10 Hz, on a mast on top          |
| Frames      | `base_footprint` → `base_link` → wheels / `lidar_mast` / `lidar_link` |

Symmetric, low, and light on purpose: stable to train and closer to sim-to-real.

## RL interface (what your env talks to)

| Role         | Topic       | Type                        |
|--------------|-------------|-----------------------------|
| **Action**   | `/cmd_vel`  | `geometry_msgs/Twist` — use `linear.x`, `angular.z` |
| **Observation** | `/scan`  | `sensor_msgs/LaserScan` — 360 beams |
| **Observation** | `/odom`  | `nav_msgs/Odometry` — pose + twist (wheel-encoder based) |
| TF           | `odom → base_footprint` | published by the drive plugin |

Tips for the env you'll write:
- Down-sample `/scan` to ~24–72 beams for the observation vector.
- Collision = min lidar range below a threshold (e.g. < 0.20 m).
- Reset each episode by teleporting the robot: Gazebo service
  `/set_entity_state` (`gazebo_msgs`), entity name `rl_robot`.
- Run headless with `gui:=false` for fast training (see below).

## Build

```bash
cd ~/ros2_git_ws
colcon build --packages-select rl_robot
source install/setup.bash
```

## Run

```bash
# Full sim: Gazebo (obstacle arena) + robot + RViz
ros2 launch rl_robot sim.launch.py

# Headless (for RL training) + no RViz
ros2 launch rl_robot sim.launch.py gui:=false rviz:=false

# Empty world, custom spawn pose
ros2 launch rl_robot sim.launch.py world:=empty x:=1.0 y:=0.5 yaw:=1.57
```

Drive it by hand to sanity-check:
```bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard
```

## Files

```
urdf/
  rl_robot.urdf.xacro   top-level, includes the rest
  robot_core.xacro      chassis, 4 wheels, lidar mast + link, dimensions
  inertial_macros.xacro physically correct inertia tensors
  materials.xacro       colors + wheel friction (skid-steer tuned)
  gazebo_control.xacro  skid-steer drive plugin  -> /cmd_vel, /odom
  lidar.xacro           2D ray sensor            -> /scan
launch/
  sim.launch.py         gazebo + rsp + spawn + rviz  (args: world, gui, rviz, x, y, z, yaw)
  rsp.launch.py         robot_state_publisher only
worlds/
  obstacles.world       10×10 m walled arena with boxes + cylinders
  empty.world           bare ground plane
rviz/view.rviz          RobotModel + LaserScan + TF
```

## Skid-steer note

4-wheel skid-steering in Gazebo Classic only turns if the wheels can slip
sideways. The wheel friction in `materials.xacro` uses `fdir1 = 1 0 0`
(rolling axis) with `mu1 = 1.0` (grip) and `mu2 = 0.0` (free lateral skid).
Raising `mu2` will make the robot refuse to turn in place.
