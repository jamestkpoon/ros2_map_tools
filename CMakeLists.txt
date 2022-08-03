cmake_minimum_required(VERSION 3.5)
project(ros2_map_tools)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)

find_package( PCL REQUIRED )
include_directories( ${PCL_INCLUDE_DIRS} )
add_definitions( ${PCL_DEFINITIONS} )
link_directories( ${PCL_LIBRARY_DIRS} )

find_package (Eigen3 REQUIRED)

find_package(OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )

find_package(octomap REQUIRED)
include_directories(${OCTOMAP_INCLUDE_DIRS})

include_directories(include)

add_executable(pcl_to_map src/pcl_to_map.cpp)
target_link_libraries(pcl_to_map ${PCL_LIBRARIES} Eigen3::Eigen ${OpenCV_LIBS} ${OCTOMAP_LIBRARIES})
ament_target_dependencies(pcl_to_map rclcpp geometry_msgs sensor_msgs nav_msgs std_srvs)

add_executable(dense_map_builder src/dense_map_builder.cpp)
target_link_libraries(dense_map_builder ${PCL_LIBRARIES} ${OCTOMAP_LIBRARIES})
ament_target_dependencies(dense_map_builder rclcpp tf2_ros geometry_msgs sensor_msgs nav_msgs std_msgs std_srvs)

install(TARGETS
  pcl_to_map dense_map_builder
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY
  config
  DESTINATION share/${PROJECT_NAME}
)

# Install Python modules
ament_python_install_package(${PROJECT_NAME})
# Install Python executables
install(PROGRAMS
  ros2_map_tools/map_aligner_ui.py
  DESTINATION lib/${PROJECT_NAME}
)


if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # uncomment the line when a copyright and license is not present in all source files
  #set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # uncomment the line when this package is not in a git repo
  #set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
