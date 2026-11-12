cmake_minimum_required(VERSION 3.5)
project(autoware_launch)

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

install(PROGRAMS scripts/planning_fallback_publishers.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME planning_fallback_publishers)
install(PROGRAMS scripts/freespace_bypass_coordinator.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME freespace_bypass_coordinator)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_auto_package(
  INSTALL_TO_SHARE
  config
  launch
  rviz
)
