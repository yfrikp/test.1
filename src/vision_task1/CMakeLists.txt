cmake_minimum_required(VERSION 3.8)
project(vision_task1)

# ---------------------------------------------------------------------------
#  C++ 标准：OpenCV 4.x 与 ROS2 Humble 都推荐 C++17
# ---------------------------------------------------------------------------
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ---------------------------------------------------------------------------
#  依赖
#  ament_cmake      : ROS2 的 CMake 增强
#  rclcpp           : ROS2 C++ 客户端库
#  sensor_msgs      : Image / CameraInfo 消息
#  cv_bridge        : ROS Image <-> OpenCV Mat 互转（关键！）
#  OpenCV           : 视觉算法本体
#  vision_task1_interfaces : 我们自己的 BlockInfo 消息
# ---------------------------------------------------------------------------
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)
find_package(vision_task1_interfaces REQUIRED)

# ---------------------------------------------------------------------------
#  可执行文件：颜色物块检测节点
# ---------------------------------------------------------------------------
add_executable(color_detector src/color_detector_node.cpp)

# ament_target_dependencies 负责把 ROS2 包的 include 路径、链接库、编译定义都带上
ament_target_dependencies(color_detector
  rclcpp
  sensor_msgs
  cv_bridge
  vision_task1_interfaces
)

# OpenCV 不是 ament 包，所以要单独链接
target_link_libraries(color_detector ${OpenCV_LIBS})
target_include_directories(color_detector PRIVATE ${OpenCV_INCLUDE_DIRS})

# ---------------------------------------------------------------------------
#  安装：可执行文件放 lib/<包名>/，配置文件放 share/<包名>/
#  ROS2 的查找约定：ros2 run <包名> <可执行名> 会去 lib/<包名>/ 找
# ---------------------------------------------------------------------------
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
