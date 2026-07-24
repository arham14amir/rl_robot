"""Bring up the full simulation:

    Gazebo (obstacles world) + robot_state_publisher + spawn robot + RViz

Usage:
    ros2 launch rl_robot sim.launch.py
    ros2 launch rl_robot sim.launch.py world:=empty rviz:=false
    ros2 launch rl_robot sim.launch.py x:=1.0 y:=0.5 yaw:=1.57
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node


def generate_launch_description():
    pkg = get_package_share_directory('rl_robot')
    gazebo_ros = get_package_share_directory('gazebo_ros')

    # ---- launch args ----------------------------------------------------
    world = LaunchConfiguration('world')
    use_rviz = LaunchConfiguration('rviz')
    gui = LaunchConfiguration('gui')
    x = LaunchConfiguration('x')
    y = LaunchConfiguration('y')
    z = LaunchConfiguration('z')
    yaw = LaunchConfiguration('yaw')

    declare_args = [
        DeclareLaunchArgument(
            'world', default_value='obstacles',
            description="World name in rl_robot/worlds (without .world), "
                        "e.g. 'obstacles' or 'empty'"),
        DeclareLaunchArgument('rviz', default_value='true'),
        DeclareLaunchArgument(
            'gui', default_value='true',
            description='Run the Gazebo client GUI. Set false for headless '
                        'RL training.'),
        DeclareLaunchArgument('x', default_value='0.0'),
        DeclareLaunchArgument('y', default_value='0.0'),
        DeclareLaunchArgument('z', default_value='0.1'),
        DeclareLaunchArgument('yaw', default_value='0.0'),
    ]

    world_path = PathJoinSubstitution([pkg, 'worlds', [world, '.world']])

    # ---- Gazebo (server + client) ---------------------------------------
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gazebo_ros, 'launch', 'gazebo.launch.py')),
        launch_arguments={'world': world_path, 'verbose': 'true',
                          'gui': gui}.items(),
    )

    # ---- robot_state_publisher ------------------------------------------
    rsp = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg, 'launch', 'rsp.launch.py')),
        launch_arguments={'use_sim_time': 'true'}.items(),
    )

    # ---- spawn the robot from /robot_description ------------------------
    spawn = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        output='screen',
        arguments=[
            '-topic', 'robot_description',
            '-entity', 'rl_robot',
            '-x', x, '-y', y, '-z', z, '-Y', yaw,
        ],
    )

    # ---- RViz -----------------------------------------------------------
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        condition=IfCondition(use_rviz),
        arguments=['-d', os.path.join(pkg, 'rviz', 'view.rviz')],
        parameters=[{'use_sim_time': True}],
    )

    return LaunchDescription(declare_args + [gazebo, rsp, spawn, rviz])
