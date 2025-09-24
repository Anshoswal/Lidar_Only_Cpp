cmake_minimum_required(VERSION 3.8)
project(lidar_only_cpp)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find necessary ROS 2, PCL, and Open3D packages
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(PCL 1.10 REQUIRED)
find_package(Open3D REQUIRED) # <-- You correctly added this

# Add PCL include directories (Open3D's are handled by the target)
include_directories(${PCL_INCLUDE_DIRS})

# Define the executable to be built from your C++ source file
add_executable(data_extractor_node src/data_extractor_node.cpp)

# Link the ROS 2 libraries your node depends on
ament_target_dependencies(data_extractor_node
  rclcpp
  sensor_msgs
  visualization_msgs
  pcl_conversions
)

# =================================================================
# === THE FIX IS HERE ===
# =================================================================
# Link the PCL and Open3D libraries your node depends on
target_link_libraries(data_extractor_node
  ${PCL_LIBRARIES}
  Open3D::Open3D  # <-- CRUCIAL FIX: This links against the Open3D library
)
# =================================================================


# Install the executable so ros2 run can find it
install(TARGETS
  data_extractor_node
  DESTINATION lib/${PROJECT_NAME}
)

# This section is for automated testing and linting
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()