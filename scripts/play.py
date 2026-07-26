#!/usr/bin/env python3
"""Run a trained policy (or a random/scripted one) and report how it does.

    # terminal 1 - simulator with the GUI so you can watch
    ros2 launch rl_robot sim.launch.py

    # terminal 2
    ros2 run rl_robot play.py --model ~/rl_robot_runs/sac_.../final_model.zip
    ros2 run rl_robot play.py --random          # sanity-check the plumbing
"""
import argparse
import os
import sys

import numpy as np

from rl_robot.robot_env import RLRobotEnv, load_config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=None, help="path to a saved .zip")
    parser.add_argument("--algo", default=None, help="SAC | PPO | TD3 (defaults to config)")
    parser.add_argument("--config", default=None)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--random", action="store_true", help="ignore the model, act randomly")
    parser.add_argument("--stochastic", action="store_true",
                        help="sample from the policy instead of taking its mean action")
    parser.add_argument("--speed", type=float, default=1000.0,
                        help="physics update rate: 1000 = 1x real time (default, "
                             "so you can actually watch it), 0 = uncapped")
    # Strip the "--ros-args ..." that ros2 launch appends; see train.py.
    if argv is None:
        from rclpy.utilities import remove_ros_args
        argv = remove_ros_args(sys.argv)[1:]
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    # Training uncaps physics; for watching we want real time back.
    cfg.setdefault("simulation", {})["set_physics_on_start"] = True
    cfg["simulation"]["max_update_rate"] = args.speed
    algo = (args.algo or cfg["training"]["algo"]).upper()

    model = None
    if not args.random:
        if not args.model:
            parser.error("give --model PATH, or --random to drive with random actions")
        from stable_baselines3 import PPO, SAC, TD3
        cls = {"SAC": SAC, "PPO": PPO, "TD3": TD3}[algo]
        model = cls.load(os.path.expanduser(args.model))
        print(f"[play] loaded {algo} policy from {args.model}")

    env = RLRobotEnv(config=cfg)
    successes = collisions = timeouts = 0
    returns, lengths = [], []

    try:
        for ep in range(args.episodes):
            obs, _ = env.reset(seed=ep)
            total, steps, done = 0.0, 0, False
            info = {}
            while not done:
                if model is None:
                    action = env.action_space.sample()
                else:
                    action, _ = model.predict(obs, deterministic=not args.stochastic)
                obs, reward, terminated, truncated, info = env.step(action)
                total += reward
                steps += 1
                done = terminated or truncated

            returns.append(total)
            lengths.append(steps)
            if info.get("is_success"):
                successes += 1
                outcome = "GOAL"
            elif info.get("collision"):
                collisions += 1
                outcome = "CRASH"
            else:
                timeouts += 1
                outcome = "TIMEOUT"
            print(f"[play] episode {ep + 1:3d}/{args.episodes}  {outcome:8s} "
                  f"return={total:8.2f}  steps={steps:4d}  "
                  f"final_dist={info.get('distance_to_goal', float('nan')):.2f} m")
    except KeyboardInterrupt:
        print("\n[play] interrupted.")
    finally:
        env.close()

    if returns:
        n = len(returns)
        print("\n[play] ---------------- summary ----------------")
        print(f"[play] episodes     : {n}")
        print(f"[play] success rate : {successes / n:.0%}  ({successes} goal / "
              f"{collisions} crash / {timeouts} timeout)")
        print(f"[play] mean return  : {np.mean(returns):.2f} +/- {np.std(returns):.2f}")
        print(f"[play] mean length  : {np.mean(lengths):.1f} steps")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
