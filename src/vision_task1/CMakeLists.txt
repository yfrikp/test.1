cmake_minimum_required(VERSION 3.8)
project(vision_task1)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)
find_package(vision_task1_interfaces REQUIRED)

add_executable(color_detector src/color_detector_node.cpp)

ament_target_dependencies(color_detector
  rclcpp
  sensor_msgs
  cv_bridge
  vision_task1_interfaces
)

target_link_libraries(color_detector ${OpenCV_LIBS})
target_include_directories(color_detector PRIVATE ${OpenCV_INCLUDE_DIRS})

install(TARGETS
  color_detector
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
  OPTIONAL
)

ament_package()
