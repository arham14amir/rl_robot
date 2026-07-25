"""Gymnasium environment wrapping the rl_robot Gazebo simulation.

Task: drive to a randomly placed goal inside the walled arena without
touching anything, using only a 2D lidar and wheel odometry.

    action       Box(2,)  in [-1, 1]  -> (linear.x, angular.z) on /cmd_vel
    observation  Box(n+5,)            -> lidar sectors + goal polar + last action
    reward       dense shaping + terminal bonus/penalty  (see _compute_reward)
    terminated   goal reached, or collision
    truncated    episode step limit reached

Every number comes from config/rl_params.yaml.
"""
from __future__ import annotations

import math
import os
from typing import Optional, Tuple

import numpy as np
import yaml

import gymnasium as gym
from gymnasium import spaces

import rclpy

from rl_robot.gazebo_bridge import (
    GazeboBridge,
    Pose2D,
    se2_compose,
    se2_inverse,
    wrap_angle,
)


def default_config_path() -> str:
    """config/rl_params.yaml from the installed share directory."""
    from ament_index_python.packages import get_package_share_directory

    return os.path.join(get_package_share_directory("rl_robot"), "config", "rl_params.yaml")


def load_config(path: Optional[str] = None) -> dict:
    path = os.path.expanduser(path) if path else default_config_path()
    with open(path, "r") as fh:
        return yaml.safe_load(fh)


class RLRobotEnv(gym.Env):
    """Goal-reaching + obstacle-avoidance env on top of Gazebo Classic."""

    metadata = {"render_modes": []}

    def __init__(self, config: Optional[dict] = None, config_path: Optional[str] = None,
                 node_name: str = "rl_robot_env"):
        super().__init__()
        self.cfg = config if config is not None else load_config(config_path)

        act = self.cfg["action"]
        obs = self.cfg["observation"]
        ep = self.cfg["episode"]

        self.v_max = float(act["max_linear_speed"])
        self.w_max = float(act["max_angular_speed"])
        self.forward_only = bool(act["forward_only"])

        self.n_beams = int(obs["n_lidar_beams"])
        self.range_max = float(obs["lidar_max_range"])
        self.range_min = float(obs["lidar_min_range"])
        self.max_goal_distance = float(obs["max_goal_distance"])
        self.use_prev_action = bool(obs["include_previous_action"])

        self.control_period = float(ep["control_period"])
        self.max_steps = int(ep["max_steps"])
        self.step_timeout = float(ep["step_timeout"])
        self.goal_tolerance = float(ep["goal_tolerance"])
        self.collision_distance = float(ep["collision_distance"])
        self.pause_between_steps = bool(ep["pause_between_steps"])

        # ---- spaces -------------------------------------------------------
        # Actions are always [-1, 1]: the algorithms assume unit scale, and we
        # do the conversion to m/s and rad/s ourselves in _scale_action().
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        obs_dim = self.n_beams + 3 + (2 if self.use_prev_action else 0)
        low = np.concatenate([
            np.zeros(self.n_beams, dtype=np.float32),          # lidar in [0,1]
            np.array([0.0, -1.0, -1.0], dtype=np.float32),     # dist, sin, cos
            *( [np.array([-1.0, -1.0], dtype=np.float32)] if self.use_prev_action else [] ),
        ])
        high = np.ones(obs_dim, dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # ---- ROS ----------------------------------------------------------
        self._owns_rclpy = not rclpy.ok()
        if self._owns_rclpy:
            rclpy.init()
        self.bridge = GazeboBridge(self.cfg, node_name=node_name)
        self.bridge.wait_for_simulator()

        sim_cfg = self.cfg.get("simulation", {})
        if sim_cfg.get("set_physics_on_start", False):
            self.bridge.set_max_update_rate(float(sim_cfg["max_update_rate"]))

        # ---- episode state ------------------------------------------------
        self.goal: Pose2D = (0.0, 0.0, 0.0)
        self.start: Pose2D = (0.0, 0.0, 0.0)
        self._T_world_odom: Pose2D = (0.0, 0.0, 0.0)
        self._prev_action = np.zeros(2, dtype=np.float32)
        self._prev_distance = 0.0
        self._steps = 0

    # ==================================================================
    #  Gymnasium API
    # ==================================================================
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        rs = self.cfg["reset"]

        self.bridge.stop_robot()
        self.bridge.unpause_physics()          # teleport + settling need physics
        self.bridge.reset_world()              # poses back to spawn, odom zeroed
        self.bridge.spin(0.1)

        # ---- choose start and goal ---------------------------------------
        if rs["randomize_start"]:
            sx, sy = self._sample_free_point()
            syaw = float(self.np_random.uniform(-math.pi, math.pi))
            self.start = (sx, sy, syaw)
        else:
            self.start = tuple(float(v) for v in rs["fixed_start"])

        if rs["randomize_goal"]:
            gx, gy = self._sample_free_point(
                away_from=self.start[:2],
                min_distance=float(rs["min_start_goal_distance"]),
            )
        else:
            gx, gy = (float(v) for v in rs["fixed_goal"])
        self.goal = (gx, gy, 0.0)

        self.bridge.teleport(self.start, z=float(rs["spawn_z"]))
        # Let the robot settle onto its wheels and a fresh scan/odom arrive.
        self.bridge.advance(0.3, wall_timeout=self.step_timeout)

        # ---- pin down the odom -> world transform -------------------------
        # The drive plugin integrates wheel encoders from wherever it last
        # zeroed itself, which after a teleport no longer matches the world.
        # We snapshot odom right after the teleport and treat that reading as
        # "the robot is at self.start". Every later world pose is then
        #     T_world_robot = T_world_odom (+) odom_reading
        # This needs no ground-truth pose service, so the policy only ever
        # consumes information a real robot would also have.
        odom_baseline = self.bridge.odom_pose()
        self._T_world_odom = se2_compose(self.start, se2_inverse(odom_baseline))

        if self.pause_between_steps:
            self.bridge.pause_physics()

        self._steps = 0
        self._prev_action[:] = 0.0
        self._prev_distance = self._distance_to_goal()
        self._publish_marker()

        return self._build_observation(), {"goal": (gx, gy), "start": self.start}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        linear, angular = self._scale_action(action)

        # 1. command  2. let physics run  3. freeze and look at the result.
        started_at = self.bridge.sim_time
        self.bridge.publish_cmd(linear, angular)

        if self.pause_between_steps:
            self.bridge.unpause_physics()
            self.bridge.advance_to(started_at + self.control_period,
                                   wall_timeout=self.step_timeout)
            # The simulator keeps running while the pause request makes its
            # round trip, so a step always overruns slightly. The overrun is
            # (round-trip wall time x speed factor) of sim time: negligible at
            # 1x, but a large fraction of a step at 10x. That is the whole
            # reason simulation.max_update_rate is a fidelity/speed trade-off
            # rather than free money - see the note in rl_params.yaml.
            self.bridge.pause_physics()
        else:
            self.bridge.advance_to(started_at + self.control_period,
                                   wall_timeout=self.step_timeout)

        step_duration = self.bridge.sim_time - started_at

        self._steps += 1
        self._prev_action[:] = action

        distance = self._distance_to_goal()
        min_range = self._min_scan_range()
        reached = distance <= self.goal_tolerance
        collided = min_range <= self.collision_distance
        truncated = (not reached and not collided) and self._steps >= self.max_steps

        reward = self._compute_reward(action, distance, min_range, reached, collided, truncated)
        self._prev_distance = distance

        terminated = bool(reached or collided)
        if terminated or truncated:
            self.bridge.stop_robot()

        info = {
            "is_success": bool(reached),
            "collision": bool(collided),
            "distance_to_goal": float(distance),
            "min_scan_range": float(min_range),
            "steps": self._steps,
            "step_sim_duration": float(step_duration),
        }
        self._publish_marker()
        return self._build_observation(), float(reward), terminated, bool(truncated), info

    def close(self):
        try:
            self.bridge.stop_robot()
            self.bridge.unpause_physics()   # never leave Gazebo frozen
            self.bridge.destroy_node()
        finally:
            if self._owns_rclpy and rclpy.ok():
                rclpy.shutdown()

    # ==================================================================
    #  Action / observation
    # ==================================================================
    def _scale_action(self, action: np.ndarray) -> Tuple[float, float]:
        if self.forward_only:
            linear = float((action[0] + 1.0) * 0.5 * self.v_max)   # [-1,1] -> [0, v_max]
        else:
            linear = float(action[0] * self.v_max)
        angular = float(action[1] * self.w_max)
        return linear, angular

    def _clean_ranges(self) -> np.ndarray:
        """Raw scan as a finite array, with no-returns pushed to max range."""
        ranges = np.asarray(self.bridge.scan.ranges, dtype=np.float32)
        ranges = np.nan_to_num(ranges, nan=self.range_max,
                               posinf=self.range_max, neginf=self.range_max)
        return np.clip(ranges, self.range_min, self.range_max)

    def _downsample_scan(self, ranges: np.ndarray) -> np.ndarray:
        """360 beams -> n_beams sectors, taking the MINIMUM of each sector.

        Min-pooling rather than plain sub-sampling: sub-sampling can step
        straight over a thin obstacle (a chair leg, a table post) and the
        policy would never see it. The minimum is the pessimistic, safe
        summary of a sector.
        """
        sectors = np.array_split(ranges, self.n_beams)
        return np.array([s.min() for s in sectors], dtype=np.float32)

    def _min_scan_range(self) -> float:
        return float(self._clean_ranges().min())

    def _build_observation(self) -> np.ndarray:
        ranges = self._clean_ranges()
        lidar = self._downsample_scan(ranges) / self.range_max

        distance, heading_error = self._goal_polar()
        parts = [
            lidar,
            np.array([
                min(distance / self.max_goal_distance, 1.0),
                math.sin(heading_error),
                math.cos(heading_error),
            ], dtype=np.float32),
        ]
        if self.use_prev_action:
            parts.append(self._prev_action.copy())

        obs = np.concatenate(parts).astype(np.float32)
        return np.clip(obs, self.observation_space.low, self.observation_space.high)

    # ==================================================================
    #  Geometry
    # ==================================================================
    def world_pose(self) -> Pose2D:
        return se2_compose(self._T_world_odom, self.bridge.odom_pose())

    def _goal_polar(self) -> Tuple[float, float]:
        """(distance, heading error) of the goal, in the robot's own frame."""
        x, y, yaw = self.world_pose()
        dx, dy = self.goal[0] - x, self.goal[1] - y
        distance = math.hypot(dx, dy)
        heading_error = wrap_angle(math.atan2(dy, dx) - yaw)
        return distance, heading_error

    def _distance_to_goal(self) -> float:
        return self._goal_polar()[0]

    def _sample_free_point(self, away_from=None, min_distance: float = 0.0):
        """Rejection-sample a point inside the arena and clear of obstacles."""
        rs = self.cfg["reset"]
        obstacles = rs["obstacles"]
        clearance = float(rs["obstacle_clearance"])

        for _ in range(int(rs["max_sampling_attempts"])):
            x = float(self.np_random.uniform(rs["arena_min_x"], rs["arena_max_x"]))
            y = float(self.np_random.uniform(rs["arena_min_y"], rs["arena_max_y"]))
            if any(math.hypot(x - o["x"], y - o["y"]) < o["radius"] + clearance for o in obstacles):
                continue
            if away_from is not None and math.hypot(x - away_from[0], y - away_from[1]) < min_distance:
                continue
            return x, y

        # Falling back is better than looping forever; warn so a badly
        # configured arena is visible rather than silently degrading training.
        self.bridge.get_logger().warn(
            "Free-space sampling failed; falling back to the fixed pose. "
            "Check reset.obstacles / arena bounds in rl_params.yaml."
        )
        fallback = rs["fixed_goal"] if away_from is not None else rs["fixed_start"]
        return float(fallback[0]), float(fallback[1])

    def _publish_marker(self) -> None:
        goal_in_odom = se2_compose(se2_inverse(self._T_world_odom), self.goal)
        self.bridge.publish_goal_marker(goal_in_odom, self.goal_tolerance)

    # ==================================================================
    #  Reward - this is the task definition
    # ==================================================================
    def _compute_reward(self, action, distance, min_range, reached, collided, truncated) -> float:
        rw = self.cfg["reward"]

        if reached:
            return float(rw["goal_reward"])
        if collided:
            return float(rw["collision_penalty"])

        reward = 0.0

        # 1. Progress: the workhorse. Rewards *closing distance*, not being
        #    close, so it cannot be farmed by parking near the goal.
        reward += float(rw["progress_weight"]) * (self._prev_distance - distance)

        # 2. Heading: gently encourage pointing at the goal. Without this the
        #    agent wanders early on, when progress signal is still noise.
        _, heading_error = self._goal_polar()
        reward -= float(rw["heading_weight"]) * abs(heading_error)

        # 3. Proximity: a soft wall in front of the hard collision penalty.
        #    Ramps linearly from 0 at safe_distance to -obstacle_weight at the
        #    collision threshold, so the agent learns to back off *before* the
        #    episode-ending event rather than only from it.
        safe = float(rw["safe_distance"])
        if min_range < safe:
            span = max(safe - self.collision_distance, 1e-6)
            severity = np.clip((safe - min_range) / span, 0.0, 1.0)
            reward -= float(rw["obstacle_weight"]) * float(severity)

        # 4. Time cost: prefer shorter paths.
        reward += float(rw["step_penalty"])

        # 5. Spin cost: a skid-steer robot can rotate in place forever and
        #    collect no penalty otherwise; this makes the motion smoother and
        #    more transferable to hardware.
        reward -= float(rw["angular_penalty"]) * abs(float(action[1]))

        if truncated:
            reward += float(rw["timeout_penalty"])

        return reward
