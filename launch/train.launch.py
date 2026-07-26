"""One-shot training launch: Gazebo + RViz + the SB3 learner, all at once.

    ros2 launch rl_robot train.launch.py                    # watch it learn
    ros2 launch rl_robot train.launch.py speed:=1000        # ...at 1x real time
    ros2 launch rl_robot train.launch.py gui:=false rviz:=false   # fastest
    ros2 launch rl_robot train.launch.py algo:=PPO timesteps:=1000000

The GUI is on by default so you can see what is happening. It costs CPU that
would otherwise go into physics, so once you trust the setup and just want
throughput, turn both viewers off.

Two things look wrong in the GUI but are not:
  * the motion stutters - the env pauses physics between steps on purpose,
    which is what keeps every step exactly control_period long;
  * everything runs ~4-5x too fast - that is simulation.max_update_rate.
    Pass speed:=1000 for real time, or run `gz physics -u 1000` live.

Running the two halves in separate terminals (sim.launch.py in one,
`ros2 run rl_robot train.py` in the other) is still nicer while debugging:
separate logs, and you can restart the learner without restarting Gazebo.
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
    rviz = LaunchConfiguration('rviz')
    speed = LaunchConfiguration('speed')

    declare_args = [
        DeclareLaunchArgument('algo', default_value='SAC',
                              description='SAC | PPO | TD3'),
        DeclareLaunchArgument('timesteps', default_value='300000'),
        DeclareLaunchArgument('run_name', default_value='',
                              description='subfolder under training.log_dir'),
        DeclareLaunchArgument('resume', default_value='',
                              description='path to a checkpoint .zip to continue from'),
        DeclareLaunchArgument('gui', default_value='true',
                              description='Gazebo GUI. false is noticeably faster.'),
        DeclareLaunchArgument('rviz', default_value='true',
                              description='RViz - the only place the goal marker '
                                          'is visible. false is faster.'),
        DeclareLaunchArgument('speed', default_value='',
                              description='physics update rate: 1000 = 1x real '
                                          'time, 5000 = ~4x (config default), '
                                          '0 = uncapped. Empty = use rl_params.yaml.'),
    ]

    sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pkg, 'launch', 'sim.launch.py')),
        launch_arguments={'gui': gui, 'rviz': rviz}.items(),
    )

    trainer = Node(
        package='rl_robot',
        executable='train.py',
        name='rl_trainer',
        output='screen',
        emulate_tty=True,
        # NOTE: `--flag=value` as a single token, not `'--flag', value`.
        # ros2 launch silently DROPS empty-string arguments, so with the
        # two-token form an unset `run_name`/`resume`/`speed` vanishes and the
        # preceding flag swallows the next flag as its value - argparse then
        # exits 2 and the trainer dies seconds after Gazebo comes up.
        arguments=[
            ['--algo=', algo],
            ['--timesteps=', timesteps],
            ['--run-name=', run_name],
            ['--resume=', resume],
            ['--speed=', speed],
        ],
    )

    # Give gzserver time to advertise its services before the learner starts
    # polling for them (the env would wait anyway, this just keeps logs clean).
    # Longer with the GUI up, since gzclient and RViz also compete for startup.
    return LaunchDescription(declare_args + [sim, TimerAction(period=12.0, actions=[trainer])])
