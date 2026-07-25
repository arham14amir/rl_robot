# rl_robot

A 4-wheel **skid-steer** robot with a top-mounted **2D lidar**, built as a
reinforcement-learning platform for **obstacle avoidance / goal reaching** on
**ROS 2 Humble + Gazebo Classic 11**.

The robot, sim, sensors, drive **and the RL layer** are done: a Gymnasium
environment, a configurable reward, and Stable-Baselines3 training/evaluation
scripts. Task: reach a randomly placed goal in the arena without hitting
anything, using only lidar + wheel odometry.

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

## The RL layer

### MDP definition

| | |
|---|---|
| **Action** | `Box(2,)` in `[-1, 1]` → `linear.x ∈ [0, 0.6] m/s`, `angular.z ∈ [-1.5, 1.5] rad/s` |
| **Observation** | `Box(29,)` = 24 lidar sectors (min-pooled from 360 beams, ÷ max range) + goal distance + `sin`/`cos` of heading error + previous action |
| **Reward** | `+200` goal, `-200` collision, `30·Δdistance` progress, heading / proximity / time / spin penalties |
| **Terminated** | within 0.35 m of the goal, or min lidar range < 0.25 m |
| **Truncated** | 500 steps (100 s of sim time at 5 Hz) |

Two design points worth knowing:

- **Sectors use the minimum, not a sample.** Sub-sampling 360 → 24 beams can
  step straight over a chair leg. The min is the pessimistic summary.
- **Heading is fed as `sin`/`cos`, not as an angle.** A raw angle jumps from
  `+π` to `-π` for an arbitrarily small rotation; the network would have to
  learn that discontinuity for nothing.

### Every knob is in one file

[config/rl_params.yaml](config/rl_params.yaml) holds the topics, action
limits, observation layout, episode limits, reward weights, reset sampling
and the algorithm hyper-parameters. The Python contains no magic numbers —
retune the task without touching code.

### Install the Python stack

Not available through rosdep:

```bash
python3 -m pip install --user torch --index-url https://download.pytorch.org/whl/cpu
python3 -m pip install --user -r requirements.txt
```

CPU-only torch is deliberate: the policy is two 256-unit MLPs, and wall-clock
time is dominated by Gazebo stepping, not by gradients. `numpy` is pinned to
`<2` because ROS 2 Humble's binaries are built against the 1.x ABI.

> **Gotcha:** installing torch upgrades `setuptools` to ≥77, which needs
> `packaging` ≥24. Ubuntu 22.04 ships 21.3, and the mismatch makes `colcon
> build` fail with `canonicalize_version() got an unexpected keyword argument
> 'strip_trailing_zero'` — *after* the install, so it looks unrelated.
> `requirements.txt` pins `packaging>=24` to prevent it.

### Simulator speed

Training at 1× real time would take ~18 hours for 300k steps. The env speeds
the simulator up on startup via `gz physics -u`, set by
`simulation.max_update_rate` (default `5000` ≈ 4×).

This is a real trade-off, not free money: physics keeps running during the
`/pause_physics` round-trip, so each step overruns by (round-trip wall time ×
speed factor) of sim time. Measured, 25 steps per rate:

| rate | speed | mean step | std | error | RL steps/s |
|---|---|---|---|---|---|
| 1000 | 0.9× | 0.200 s | 0.000 | −0.0% | 4.6 |
| 3000 | 2.6× | 0.200 s | 0.000 | +0.0% | 13.0 |
| **5000** | **4.0×** | **0.200 s** | **0.000** | **+0.0%** | **19.9** |
| 0 (uncapped) | 6.5× | 0.308 s | 0.130 | +54.1% | 21.1 |

Uncapping buys ~6% throughput and wrecks step determinism. 5000 is the knee.
Re-measure if you change `control_period` or the arena.

### Train

```bash
# terminal 1 — simulator, headless and as fast as the CPU allows
ros2 launch rl_robot sim.launch.py gui:=false rviz:=false

# terminal 2 — the learner
ros2 run rl_robot train.py --timesteps 300000
```

Or both at once: `ros2 launch rl_robot train.launch.py algo:=SAC timesteps:=300000`

Models, checkpoints and tensorboard logs go to `~/rl_robot_runs/<run_name>/`.
Watch progress with `tensorboard --logdir ~/rl_robot_runs`; the number to
follow is `rollout/ep_rew_mean`.

### Evaluate

```bash
ros2 launch rl_robot sim.launch.py                       # with the GUI
ros2 run rl_robot play.py --model ~/rl_robot_runs/sac_.../final_model.zip
ros2 run rl_robot play.py --random                       # plumbing sanity check
```

`play.py` prints a per-episode outcome (GOAL / CRASH / TIMEOUT) and a success
rate. The current goal shows up in RViz as a green disc on `/goal_marker`.

### How an episode is stepped

1. Publish the scaled action on `/cmd_vel`.
2. `/unpause_physics`, then let the sim advance `control_period` of **simulated**
   time — measured on `/clock`, so results are identical at 1× or 20× real time.
3. `/pause_physics`, then read the newest `/scan` and `/odom` and compute
   observation and reward.

Pausing between steps is what makes runs reproducible: the policy's thinking
time never leaks into the physics.

Reset teleports the robot with `/set_entity_state` (provided by
`libgazebo_ros_state.so`, loaded as a world plugin in `worlds/*.world`), then
snapshots `/odom` and treats that reading as the new start pose. The env
therefore never reads a ground-truth pose service — the policy only consumes
information a real robot would also have.

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
  train.launch.py       headless sim + learner in one shot
worlds/
  obstacles.world       10×10 m walled arena with boxes + cylinders
  empty.world           bare ground plane
rviz/view.rviz          RobotModel + LaserScan + TF + goal marker

rl_robot/               the RL layer (importable python package)
  gazebo_bridge.py      ROS/Gazebo plumbing only: topics, services, SE(2) math
  robot_env.py          the Gymnasium env: spaces, reset, step, reward
config/
  rl_params.yaml        every tunable — task, reward, reset, hyper-parameters
scripts/
  train.py              Stable-Baselines3 training (SAC / PPO / TD3)
  play.py               run a trained policy, report success rate
requirements.txt        pip deps not covered by rosdep
```

The split between the two Python files is deliberate: `gazebo_bridge.py`
knows about Gazebo but nothing about RL, and `robot_env.py` knows about RL
but talks to the simulator only through the bridge. Swapping Gazebo Classic
for Ignition or Isaac means rewriting one file.

## Skid-steer note

4-wheel skid-steering in Gazebo Classic only turns if the wheels can slip
sideways. The wheel friction in `materials.xacro` uses `fdir1 = 1 0 0`
(rolling axis) with `mu1 = 1.0` (grip) and `mu2 = 0.0` (free lateral skid).
Raising `mu2` will make the robot refuse to turn in place.
