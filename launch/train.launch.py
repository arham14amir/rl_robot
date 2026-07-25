"""One-shot training launch: headless Gazebo + the SB3 learner.

    ros2 launch rl_robot train.launch.py
    ros2 launch rl_robot train.launch.py algo:=PPO timesteps:=1000000
    ros2 launch rl_robot train.launch.py gui:=true          # watch it learn

Running the two halves in separate terminals (sim.launch.py in one,
`ros2 run rl_robot train.py` in the other) is usually nicer while you are
still debugging - you get clean, separated logs and can restart the learner
without restarting Gazebo. This file is for when the setup is stable.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg = get_package_share_directory('rl_robot')

    algo = LaunchConfiguration('algo')
    timesteps = LaunchConfiguration('timesteps')
    run_name = LaunchConfiguration('run_name')
    resume = LaunchConfiguration('resume')
    gui = LaunchConfiguration('gui')

    declare_args = [
        DeclareLaunchArgument('algo', default_value='SAC',
                              description='SAC | PPO | TD3'),
        DeclareLaunchArgument('timesteps', default_value='300000'),
        DeclareLaunchArgument('run_name', default_value='',
                              description='subfolder under training.log_dir'),
        DeclareLaunchArgument('resume', default_value='',
                              description='path to a checkpoint .zip to continue from'),
        DeclareLaunchArgument('gui', default_value='false',
                              description='headless by default - much faster'),
    ]

    sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pkg, 'launch', 'sim.launch.py')),
        launch_arguments={'gui': gui, 'rviz': 'false'}.items(),
    )

    trainer = Node(
        package='rl_robot',
        executable='train.py',
        name='rl_trainer',
        output='screen',
        emulate_tty=True,
        arguments=[
            '--algo', algo,
            '--timesteps', timesteps,
            '--run-name', run_name,
            '--resume', resume,
        ],
    )

    # Give gzserver time to advertise its services before the learner starts
    # polling for them (the env would wait anyway, this just keeps logs clean).
    return LaunchDescription(declare_args + [sim, TimerAction(period=8.0, actions=[trainer])])
