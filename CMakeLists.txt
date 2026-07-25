cmake_minimum_required(VERSION 3.8)
project(rl_robot)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

# Install resource directories to share/rl_robot/...
install(
  DIRECTORY urdf launch worlds rviz config
  DESTINATION share/${PROJECT_NAME}
)

# Python module: rl_robot/{robot_env,gazebo_bridge}.py  ->  importable as
#   from rl_robot.robot_env import RLRobotEnv
ament_python_install_package(${PROJECT_NAME})

# Executable scripts: run with `ros2 run rl_robot train.py`
install(
  PROGRAMS
    scripts/train.py
    scripts/play.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
