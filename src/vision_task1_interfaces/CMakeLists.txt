cmake_minimum_required(VERSION 3.8)
project(vision_task1_interfaces)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)

# 声明要生成的消息文件（可以有多个，用空格隔开）
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/BlockInfo.msg"
  DEPENDENCIES std_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
