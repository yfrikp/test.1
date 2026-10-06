cmake_minimum_required(VERSION 3.8)
project(vision_task1_sim)

find_package(ament_cmake REQUIRED)

install(DIRECTORY worlds
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY models
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/verify_sim.py
  scripts/save_frame.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
