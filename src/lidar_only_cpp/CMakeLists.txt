cmake_minimum_required(VERSION 3.8)
project(lidar_only_cpp)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -O3 -DNDEBUG)
endif()

# Find ROS 2 and external dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(Open3D REQUIRED)
find_package(onnxruntime REQUIRED)

# Add your executable
add_executable(data_extractor_node src/lidar_only_cpp/src/data_extractor_node.cpp)

# Link ROS 2 libraries
ament_target_dependencies(data_extractor_node
  rclcpp
  sensor_msgs
  visualization_msgs
  pcl_conversions
  ament_index_cpp
)

# Link external libraries
target_link_libraries(data_extractor_node
  PRIVATE
  Open3D::Open3D
  onnxruntime::onnxruntime
)

# Install the executable
install(TARGETS
  data_extractor_node
  DESTINATION lib/${PROJECT_NAME}
)

# Install the model files
install(DIRECTORY
  src/lidar_only_cpp/models
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()