cmake_minimum_required(VERSION 3.5)
project(traffic_light_classifier)

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# Generate interfaces
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/TrafficLightMsg.msg"
  DEPENDENCIES sensor_msgs
)

ament_package()
