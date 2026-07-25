"""rl_robot - reinforcement-learning layer for the skid-steer lidar robot.

The heavy import (gymnasium) lives in :mod:`rl_robot.robot_env`, so importing
this package alone is cheap and safe from plain ROS nodes.
"""

__all__ = ["RLRobotEnv", "load_config"]


def __getattr__(name):
    if name in __all__:
        from rl_robot.robot_env import RLRobotEnv, load_config

        return {"RLRobotEnv": RLRobotEnv, "load_config": load_config}[name]
    raise AttributeError(name)
