cmake_minimum_required(VERSION 3.8)
project(lidar_only_cpp)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# -------------------------------
# ROS 2 dependencies
# -------------------------------
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)

# -------------------------------
# System libraries
# -------------------------------
# PCL
find_package(PCL 1.10 REQUIRED)

# Open3D
find_package(Open3D REQUIRED)

# ONNX Runtime (manual install)
include_directories($ENV{HOME}/onnxruntime-linux-x64-1.17.1/include)

find_library(ONNXRUNTIME_LIB
  NAMES onnxruntime
  PATHS $ENV{HOME}/onnxruntime-linux-x64-1.17.1/lib
)

if(NOT ONNXRUNTIME_LIB)
  message(FATAL_ERROR "onnxruntime library not found in ~/onnxruntime-linux-x64-1.17.1/lib")
endif()

# -------------------------------
# Executable
# -------------------------------
add_executable(data_extractor_node src/data_extractor_node.cpp)

ament_target_dependencies(data_extractor_node
  rclcpp
  sensor_msgs
  visualization_msgs
  pcl_conversions
)

target_include_directories(data_extractor_node PUBLIC
  ${PCL_INCLUDE_DIRS}
  ${Open3D_INCLUDE_DIRS}
)

target_link_libraries(data_extractor_node
  ${PCL_LIBRARIES}
  Open3D::Open3D
  ${ONNXRUNTIME_LIB}   # ✅ absolute path to libonnxruntime.so
)

# -------------------------------
# Install
# -------------------------------
install(TARGETS
  data_extractor_node
  DESTINATION lib/${PROJECT_NAME}
)

# -------------------------------
# Testing
# -------------------------------
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
