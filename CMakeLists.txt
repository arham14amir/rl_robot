cmake_minimum_required(VERSION 3.8)
project(rl_robot)

find_package(ament_cmake REQUIRED)

# Install resource directories to share/rl_robot/...
install(
  DIRECTORY urdf launch worlds rviz
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
