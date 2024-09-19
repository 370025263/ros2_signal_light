cmake_minimum_required(VERSION 3.8)
project(traffic_light_classifier)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)
find_package(traffic_light_msg REQUIRED)
find_package(autoware_perception_msgs REQUIRED)

# Include ONNX Runtime
find_package(onnxruntime REQUIRED)

include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
  ${rclcpp_INCLUDE_DIRS}
  ${rclcpp_components_INCLUDE_DIRS}
  ${ONNXRUNTIME_INCLUDE_DIRS}  # Include ONNX Runtime headers
)

add_executable(cam_node src/cam_node.cpp)
ament_target_dependencies(cam_node
  rclcpp
  sensor_msgs
  cv_bridge
  traffic_light_msg
  rclcpp_components
)
target_link_libraries(cam_node 
  ${OpenCV_LIBRARIES}
  ${rclcpp_LIBRARIES}
  ${rclcpp_components_LIBRARIES})

add_executable(classification_node src/classification_node.cpp)
ament_target_dependencies(classification_node
  rclcpp
  rclcpp_components
  sensor_msgs
  std_msgs
  cv_bridge
  traffic_light_msg
  autoware_perception_msgs
)
target_link_libraries(classification_node 
  ${OpenCV_LIBRARIES} 
  ${rclcpp_LIBRARIES} 
  ${rclcpp_components_LIBRARIES}
  ${ONNXRUNTIME_LIBRARIES}  # Link against ONNX Runtime
)

install(TARGETS
  cam_node
  classification_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/${PROJECT_NAME}
  DESTINATION include
)

install(DIRECTORY
  launch
  models
  DESTINATION share/${PROJECT_NAME}
)

ament_export_include_directories(include)
ament_export_dependencies(rclcpp sensor_msgs std_msgs cv_bridge OpenCV traffic_light_msg autoware_perception_msgs onnxruntime)

ament_package()
