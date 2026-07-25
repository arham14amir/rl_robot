"""Thin ROS 2 / Gazebo Classic plumbing used by the RL environment.

This module deliberately contains *no* RL concepts - no reward, no
observation, no episode. It only knows how to:

  * read the newest /scan, /odom and /clock,
  * publish a /cmd_vel,
  * pause, unpause and reset the simulator,
  * teleport the robot,
  * advance the simulator by a fixed amount of *simulation* time.

Keeping this separate from the Gym env is what makes the env testable and
what lets you swap Gazebo Classic for Ignition/Isaac later by rewriting only
this file.
"""
from __future__ import annotations

import math
import subprocess
import time
from typing import Optional, Tuple

from gazebo_msgs.msg import EntityState
from gazebo_msgs.srv import SetEntityState
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import LaserScan
from std_srvs.srv import Empty
from visualization_msgs.msg import Marker

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy

# A BEST_EFFORT subscriber is compatible with BEST_EFFORT *and* RELIABLE
# publishers, and VOLATILE with VOLATILE and TRANSIENT_LOCAL. Subscribing this
# way means we never hit a silent QoS mismatch regardless of how the Gazebo
# plugins were compiled.
PERMISSIVE_QOS = QoSProfile(
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    durability=QoSDurabilityPolicy.VOLATILE,
)


# ---------------------------------------------------------------------------
#  Minimal SE(2) helpers.  A pose is the tuple (x, y, yaw).
#
#  We need these because the robot's /odom is measured from wherever the
#  drive plugin last zeroed itself, while goals are defined in world
#  coordinates. Composing/inverting 2D transforms converts between the two.
# ---------------------------------------------------------------------------
Pose2D = Tuple[float, float, float]


def wrap_angle(a: float) -> float:
    """Wrap an angle to (-pi, pi]."""
    return math.atan2(math.sin(a), math.cos(a))


def se2_compose(a: Pose2D, b: Pose2D) -> Pose2D:
    """Return a (+) b, i.e. b expressed in a's parent frame."""
    ca, sa = math.cos(a[2]), math.sin(a[2])
    return (
        a[0] + ca * b[0] - sa * b[1],
        a[1] + sa * b[0] + ca * b[1],
        wrap_angle(a[2] + b[2]),
    )


def se2_inverse(a: Pose2D) -> Pose2D:
    ca, sa = math.cos(a[2]), math.sin(a[2])
    return (-ca * a[0] - sa * a[1], sa * a[0] - ca * a[1], wrap_angle(-a[2]))


def yaw_from_quaternion(q) -> float:
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny, cosy)


def quaternion_from_yaw(yaw: float):
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


class GazeboBridge(Node):
    """One rclpy node owning every topic and service the env needs."""

    def __init__(self, cfg: dict, node_name: str = "rl_robot_env"):
        super().__init__(node_name)
        ros = cfg["ros"]
        self._entity_name = ros["entity_name"]
        self._service_timeout = 10.0

        # ---- state filled in by callbacks --------------------------------
        self.scan: Optional[LaserScan] = None
        self.odom: Optional[Odometry] = None
        self.sim_time: float = 0.0
        self.scan_count: int = 0

        # ---- pub / sub ----------------------------------------------------
        self._cmd_pub = self.create_publisher(Twist, ros["cmd_vel_topic"], 1)
        self._marker_pub = self.create_publisher(Marker, ros["marker_topic"], 1)
        self.create_subscription(LaserScan, ros["scan_topic"], self._on_scan, PERMISSIVE_QOS)
        self.create_subscription(Odometry, ros["odom_topic"], self._on_odom, PERMISSIVE_QOS)
        self.create_subscription(Clock, ros["clock_topic"], self._on_clock, PERMISSIVE_QOS)

        # ---- services -----------------------------------------------------
        # NOTE: this node keeps *wall* time (use_sim_time stays false) on
        # purpose. If it used sim time, every service timeout would freeze the
        # moment we pause physics and a failed call would hang forever.
        self._pause = self.create_client(Empty, ros["pause_service"])
        self._unpause = self.create_client(Empty, ros["unpause_service"])
        self._reset = self.create_client(Empty, ros["reset_service"])
        self._set_state = self.create_client(SetEntityState, ros["set_state_service"])

    # ------------------------------------------------------------------
    #  Callbacks
    # ------------------------------------------------------------------
    def _on_scan(self, msg: LaserScan) -> None:
        self.scan = msg
        self.scan_count += 1
        self._update_sim_time(msg.header.stamp)

    def _on_odom(self, msg: Odometry) -> None:
        self.odom = msg
        self._update_sim_time(msg.header.stamp)

    def _on_clock(self, msg: Clock) -> None:
        self._update_sim_time(msg.clock)

    def _update_sim_time(self, stamp) -> None:
        """Track simulation time from whichever source ticks fastest.

        /clock is only published at 10 Hz by libgazebo_ros_init, which is a
        0.1 s quantum - half of a 0.2 s control step. Measuring a step against
        it overshoots by up to 50%. Message header stamps carry sim time too,
        and /odom runs at 50 Hz, so folding those in gives a 0.02 s quantum.
        We take the max so time can never appear to run backwards.
        """
        t = stamp.sec + stamp.nanosec * 1e-9
        if t > self.sim_time:
            self.sim_time = t

    # ------------------------------------------------------------------
    #  Waiting / spinning
    # ------------------------------------------------------------------
    def spin(self, wall_seconds: float) -> None:
        """Process callbacks for `wall_seconds` of real time."""
        end = time.time() + wall_seconds
        while time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.005)

    def wait_for_simulator(self, timeout: float = 60.0) -> None:
        """Block until Gazebo is up and the first scan/odom/clock arrived."""
        for client, name in (
            (self._pause, "pause_physics"),
            (self._unpause, "unpause_physics"),
            (self._reset, "reset_world"),
            (self._set_state, "set_entity_state"),
        ):
            if not client.wait_for_service(timeout_sec=timeout):
                raise RuntimeError(
                    f"Gazebo service '{name}' never appeared. Is the simulation "
                    f"running, and does the world load libgazebo_ros_state.so?"
                )

        deadline = time.time() + timeout
        while time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            if self.scan is not None and self.odom is not None and self.sim_time > 0.0:
                return
        missing = [
            n
            for n, v in (("/scan", self.scan), ("/odom", self.odom), ("/clock", self.sim_time))
            if not v
        ]
        raise RuntimeError(f"Timed out waiting for {', '.join(missing)}.")

    def advance_to(self, target_sim_time: float, wall_timeout: float,
                   require_new_scan: bool = True) -> float:
        """Run the simulator until sim time reaches `target_sim_time`.

        Physics must already be unpaused. Progress is measured in *simulated*
        time, not wall time, so a step is identical whether Gazebo runs at
        0.5x or 20x real time - that is what makes training reproducible and
        lets you go headless as fast as the CPU allows.

        Returns the sim time actually reached, which is >= the target by one
        message quantum. Callers should feed that back into their next target
        so the small overshoot does not accumulate.
        """
        scans0 = self.scan_count
        wall_deadline = time.time() + wall_timeout
        while True:
            rclpy.spin_once(self, timeout_sec=0.002)
            reached = self.sim_time >= target_sim_time
            fresh = (self.scan_count > scans0) or not require_new_scan
            if reached and fresh:
                return self.sim_time
            if time.time() > wall_deadline:
                self.get_logger().warn(
                    f"advance_to() hit its {wall_timeout:.1f}s wall-clock timeout "
                    f"(sim time {self.sim_time:.3f} < target {target_sim_time:.3f})."
                )
                return self.sim_time

    def advance(self, sim_seconds: float, wall_timeout: float,
                require_new_scan: bool = True) -> float:
        """Run for `sim_seconds` more of simulated time."""
        return self.advance_to(self.sim_time + sim_seconds, wall_timeout, require_new_scan)

    # ------------------------------------------------------------------
    #  Actuation
    # ------------------------------------------------------------------
    def publish_cmd(self, linear: float, angular: float) -> None:
        msg = Twist()
        msg.linear.x = float(linear)
        msg.angular.z = float(angular)
        self._cmd_pub.publish(msg)

    def stop_robot(self) -> None:
        self.publish_cmd(0.0, 0.0)

    # ------------------------------------------------------------------
    #  Simulator control
    # ------------------------------------------------------------------
    def _call(self, client, request):
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=self._service_timeout)
        if not future.done():
            self.get_logger().warn(f"Service call on {client.srv_name} timed out.")
            return None
        return future.result()

    def pause_physics(self) -> None:
        self._call(self._pause, Empty.Request())

    def unpause_physics(self) -> None:
        self._call(self._unpause, Empty.Request())

    def reset_world(self) -> None:
        """Reset model poses (and the drive plugin's odometry).

        Uses /reset_world rather than /reset_simulation: the latter also
        rewinds /clock to zero, which makes every sim-time measurement in
        advance() jump backwards.
        """
        self._call(self._reset, Empty.Request())

    def set_max_update_rate(self, rate: float, timeout: float = 10.0) -> bool:
        """Throttle (or un-throttle) the physics engine.

        The worlds ship with real_time_update_rate = 1000 and a 0.001 s step,
        i.e. exactly 1x real time - comfortable to watch, but it means 300k
        training steps take ~17 hours of wall clock. Rate 0 removes the cap
        and Gazebo runs physics as fast as the CPU manages (~10x headless on
        this arena).

        This goes through Gazebo's own `gz` CLI rather than a ROS service:
        gazebo_ros in Humble ships gazebo_msgs/SetPhysicsProperties but no
        plugin that actually advertises it (that was a ROS 1 feature of
        gazebo_ros_api_plugin). `gz physics` talks to the same gzserver over
        Gazebo transport and does work.

        Changing this affects only how fast the simulation runs, never what it
        simulates: every measurement in the env is taken in sim time, so a
        policy trained uncapped is identical to one trained at 1x.
        """
        try:
            proc = subprocess.run(
                ["gz", "physics", "-u", str(int(rate))],
                capture_output=True, text=True, timeout=timeout,
            )
        except FileNotFoundError:
            self.get_logger().warn(
                "`gz` CLI not found - leaving the physics rate as the world file "
                "set it (1x real time). Training will be slow but correct."
            )
            return False
        except subprocess.TimeoutExpired:
            self.get_logger().warn("`gz physics` timed out; physics rate unchanged.")
            return False

        if proc.returncode != 0:
            self.get_logger().warn(
                f"`gz physics -u {int(rate)}` failed: {proc.stderr.strip()}"
            )
            return False

        self.get_logger().info(
            f"Physics update rate set to {int(rate)} "
            f"({'uncapped - as fast as the CPU allows' if rate == 0 else f'{rate / 1000.0:.1f}x real time'})."
        )
        return True

    def teleport(self, pose: Pose2D, z: float = 0.1) -> bool:
        """Move the robot to `pose` and zero its velocity."""
        state = EntityState()
        state.name = self._entity_name
        state.pose.position.x = float(pose[0])
        state.pose.position.y = float(pose[1])
        state.pose.position.z = float(z)
        qx, qy, qz, qw = quaternion_from_yaw(pose[2])
        state.pose.orientation.x = qx
        state.pose.orientation.y = qy
        state.pose.orientation.z = qz
        state.pose.orientation.w = qw
        state.reference_frame = "world"

        req = SetEntityState.Request()
        req.state = state
        result = self._call(self._set_state, req)
        if result is None or not result.success:
            self.get_logger().warn(f"Failed to teleport '{self._entity_name}'.")
            return False
        return True

    # ------------------------------------------------------------------
    #  Odometry
    # ------------------------------------------------------------------
    def odom_pose(self) -> Pose2D:
        """Current pose in the *odom* frame, from the wheel-encoder plugin."""
        p = self.odom.pose.pose
        return (p.position.x, p.position.y, yaw_from_quaternion(p.orientation))

    # ------------------------------------------------------------------
    #  Visualisation
    # ------------------------------------------------------------------
    def publish_goal_marker(self, goal_in_odom, radius: float) -> None:
        m = Marker()
        m.header.frame_id = "odom"
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = "rl_goal"
        m.id = 0
        m.type = Marker.CYLINDER
        m.action = Marker.ADD
        m.pose.position.x = float(goal_in_odom[0])
        m.pose.position.y = float(goal_in_odom[1])
        m.pose.position.z = 0.05
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = float(2.0 * radius)
        m.scale.z = 0.1
        m.color.r, m.color.g, m.color.b, m.color.a = 0.1, 0.9, 0.2, 0.6
        self._marker_pub.publish(m)
