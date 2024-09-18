cmake_minimum_required(VERSION 3.8)
project(traffic_light_classifier)

# 编译器选项
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# 查找依赖包
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)
# 设置libtorch路径
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "/usr/local/libtorch")
set(TORCH_INCLUDE_DIRS "/usr/local/libtorch/include" "/usr/local/libtorch/include/torch/csrc/api/include")
set(TORCH_LIBRARY_DIRS "/usr/local/libtorch/lib")

find_package(Torch REQUIRED)


find_package(traffic_light_msg REQUIRED)

# 包含目录
include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
  ${TORCH_INCLUDE_DIRS}
  ${rclcpp_INCLUDE_DIRS}
  ${rclcpp_components_INCLUDE_DIRS}
)

# 添加可执行文件
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
)
target_link_libraries(classification_node ${OpenCV_LIBRARIES} ${TORCH_LIBRARIES} ${rclcpp_LIBRARIES} ${rclcpp_components_LIBRARIES})

# 安装目标
install(TARGETS
  cam_node
  classification_node
  DESTINATION lib/${PROJECT_NAME}
)

# 安装头文件
install(DIRECTORY include/${PROJECT_NAME}
  DESTINATION include
)

# 安装其他文件
install(DIRECTORY
  launch
  models
  DESTINATION share/${PROJECT_NAME}
)

# 导出依赖
ament_export_include_directories(include)
ament_export_dependencies(rclcpp sensor_msgs std_msgs cv_bridge OpenCV Torch traffic_light_msg)

ament_package()
