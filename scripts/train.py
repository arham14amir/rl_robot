#!/usr/bin/env python3
"""Train a policy on RLRobotEnv with Stable-Baselines3.

    # terminal 1 - the simulator, headless and as fast as the CPU allows
    ros2 launch rl_robot sim.launch.py gui:=false rviz:=false

    # terminal 2 - the learner
    ros2 run rl_robot train.py --timesteps 300000

Everything not passed on the command line comes from config/rl_params.yaml.
"""
import argparse
import os
import signal
import sys
from datetime import datetime

from rl_robot.robot_env import RLRobotEnv, load_config


def build_model(algo: str, env, cfg: dict, log_dir: str, seed: int):
    """Instantiate the SB3 algorithm named in the config."""
    from stable_baselines3 import PPO, SAC, TD3

    algo = algo.upper()
    tcfg = cfg["training"]

    if algo == "SAC":
        h = tcfg["sac"]
        return SAC(
            "MlpPolicy", env,
            learning_rate=h["learning_rate"],
            buffer_size=h["buffer_size"],
            batch_size=h["batch_size"],
            tau=h["tau"],
            gamma=h["gamma"],
            learning_starts=h["learning_starts"],
            train_freq=h["train_freq"],
            gradient_steps=h["gradient_steps"],
            policy_kwargs=dict(net_arch=list(h["net_arch"])),
            tensorboard_log=log_dir, verbose=1, seed=seed,
        )

    if algo == "PPO":
        h = tcfg["ppo"]
        return PPO(
            "MlpPolicy", env,
            learning_rate=h["learning_rate"],
            n_steps=h["n_steps"],
            batch_size=h["batch_size"],
            n_epochs=h["n_epochs"],
            gamma=h["gamma"],
            gae_lambda=h["gae_lambda"],
            clip_range=h["clip_range"],
            ent_coef=h["ent_coef"],
            policy_kwargs=dict(net_arch=list(h["net_arch"])),
            tensorboard_log=log_dir, verbose=1, seed=seed,
        )

    if algo == "TD3":
        import numpy as np
        from stable_baselines3.common.noise import NormalActionNoise

        h = tcfg["td3"]
        n_actions = env.action_space.shape[0]
        noise = NormalActionNoise(
            mean=np.zeros(n_actions),
            sigma=h["action_noise_sigma"] * np.ones(n_actions),
        )
        return TD3(
            "MlpPolicy", env,
            learning_rate=h["learning_rate"],
            buffer_size=h["buffer_size"],
            batch_size=h["batch_size"],
            tau=h["tau"],
            gamma=h["gamma"],
            learning_starts=h["learning_starts"],
            train_freq=h["train_freq"],
            gradient_steps=h["gradient_steps"],
            action_noise=noise,
            policy_kwargs=dict(net_arch=list(h["net_arch"])),
            tensorboard_log=log_dir, verbose=1, seed=seed,
        )

    raise ValueError(f"Unknown algo '{algo}' (expected SAC, PPO or TD3).")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, help="path to rl_params.yaml")
    parser.add_argument("--algo", default=None, help="SAC | PPO | TD3 (overrides config)")
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--run-name", default=None, help="subfolder under log_dir")
    parser.add_argument("--resume", default=None, help="path to a .zip to continue from")
    parser.add_argument("--check-env", action="store_true",
                        help="run SB3's API conformance check and exit")
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    cfg = load_config(args.config)
    tcfg = cfg["training"]
    algo = (args.algo or tcfg["algo"]).upper()
    timesteps = args.timesteps or int(tcfg["total_timesteps"])
    seed = tcfg["seed"] if args.seed is None else args.seed

    run_name = args.run_name or f"{algo.lower()}_{datetime.now():%Y%m%d_%H%M%S}"
    log_dir = os.path.join(os.path.expanduser(tcfg["log_dir"]), run_name)
    os.makedirs(log_dir, exist_ok=True)

    print(f"[train] algo={algo}  timesteps={timesteps}  seed={seed}")
    print(f"[train] logs and checkpoints -> {log_dir}")
    print("[train] connecting to Gazebo (start sim.launch.py first if this hangs)...")

    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.monitor import Monitor

    env = RLRobotEnv(config=cfg)

    if args.check_env:
        from stable_baselines3.common.env_checker import check_env
        check_env(env, warn=True)
        print("[train] environment passed check_env().")
        env.close()
        return 0

    # Monitor records episode return/length so they show up in the SB3 logs
    # and in tensorboard as rollout/ep_rew_mean.
    env = Monitor(env, filename=os.path.join(log_dir, "monitor.csv"),
                  info_keywords=("is_success", "collision"))

    if args.resume:
        from stable_baselines3 import PPO, SAC, TD3
        cls = {"SAC": SAC, "PPO": PPO, "TD3": TD3}[algo]
        print(f"[train] resuming from {args.resume}")
        model = cls.load(os.path.expanduser(args.resume), env=env,
                         tensorboard_log=log_dir)
    else:
        model = build_model(algo, env, cfg, log_dir, seed)

    checkpoint = CheckpointCallback(
        save_freq=int(tcfg["save_freq"]),
        save_path=os.path.join(log_dir, "checkpoints"),
        name_prefix=algo.lower(),
    )

    final_path = os.path.join(log_dir, "final_model")

    def save_and_exit(signum, frame):
        print("\n[train] interrupted - saving model before exit.")
        model.save(final_path)
        env.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, save_and_exit)

    try:
        model.learn(total_timesteps=timesteps, callback=checkpoint,
                    reset_num_timesteps=not bool(args.resume))
    finally:
        model.save(final_path)
        print(f"[train] saved {final_path}.zip")
        env.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
