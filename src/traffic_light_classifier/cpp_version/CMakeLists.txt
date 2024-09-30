cmake_minimum_required(VERSION 3.8)
project(traffic_light_classifier)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
# IDE
set(CMAKE_BUILD_TYPE "Debug")


# 编译选项
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# 查找所需的包
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)
find_package(traffic_light_msg REQUIRED)
find_package(autoware_perception_msgs REQUIRED)

# ONNX Runtime 设置
set(ONNXRUNTIME_ROOT_DIR "/home/casia/Downloads/onnxruntime-linux-x64-gpu-1.19.2")
set(ONNXRUNTIME_INCLUDE_DIRS "${ONNXRUNTIME_ROOT_DIR}/include")
set(ONNXRUNTIME_LIB_DIR "${ONNXRUNTIME_ROOT_DIR}/lib")

# 查找 ONNX Runtime 库
find_library(ONNXRUNTIME_LIBRARY
  NAMES onnxruntime
  PATHS ${ONNXRUNTIME_LIB_DIR}
  NO_DEFAULT_PATH
  REQUIRED
)

# 包含目录
include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
  ${rclcpp_INCLUDE_DIRS}
  ${rclcpp_components_INCLUDE_DIRS}
  ${ONNXRUNTIME_INCLUDE_DIRS}
  ${autoware_perception_msgs_INCLUDE_DIRS}
)

# 添加可执行文件
add_executable(cam_node src/cam_node.cpp
        )
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
  ${rclcpp_components_LIBRARIES}
)

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
  ${ONNXRUNTIME_LIBRARY}
  ${autoware_perception_msgs_LIBRARIES}  # 添加这一行
)

# 安装
install(TARGETS
  cam_node
  classification_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/${PROJECT_NAME}
  DESTINATION include
)

install(DIRECTORY
  #launch
  models
  DESTINATION share/${PROJECT_NAME}
)

# 导出依赖
ament_export_include_directories(include)
ament_export_dependencies(
  rclcpp 
  sensor_msgs 
  std_msgs 
  cv_bridge 
  OpenCV 
  traffic_light_msg 
  autoware_perception_msgs
        
)

ament_package()
