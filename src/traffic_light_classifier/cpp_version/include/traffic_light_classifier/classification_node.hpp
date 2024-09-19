#ifndef TRAFFIC_LIGHT_CLASSIFIER__CLASSIFICATION_NODE_HPP_
#define TRAFFIC_LIGHT_CLASSIFIER__CLASSIFICATION_NODE_HPP_

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/string.hpp>
#include <cv_bridge/cv_bridge.h>
#include "traffic_light_msg/msg/traffic_light_msg.hpp"
#include <autoware_perception_msgs/msg/traffic_light_group_array.hpp>

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

#include <map>

namespace traffic_light_classifier
{

class ClassificationNode : public rclcpp::Node
{
public:
  explicit ClassificationNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());

private:
  void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg);
  void debugCallback(const traffic_light_msg::msg::TrafficLightMsg::SharedPtr msg);
  void processImage(const cv::Mat& cv_image, const std::string& true_label = "");
  cv::Mat drawResult(const cv::Mat& image, const std::vector<cv::Rect>& boxes, const std::vector<std::string>& labels);

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;
  rclcpp::Subscription<traffic_light_msg::msg::TrafficLightMsg>::SharedPtr debug_subscription_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr result_publisher_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_result_publisher_;
  rclcpp::Publisher<autoware_perception_msgs::msg::TrafficLightGroupArray>::SharedPtr traffic_signals_publisher_;

  // 使用 OpenCV DNN 模块加载 ONNX 模型
  cv::dnn::Net detection_net_;
  cv::dnn::Net classification_net_;

  std::map<std::string, int> class_to_idx_;
  std::map<int, std::string> idx_to_class_;
  
  int total_predictions_;
  int correct_predictions_;
  double total_processing_time_;
  int total_processed_images_;

  // 假设的相机内参矩阵，用于将像素坐标转换为世界坐标（根据您的相机参数进行设置）
  cv::Mat camera_intrinsic_matrix_;
};

}  // namespace traffic_light_classifier

#endif  // TRAFFIC_LIGHT_CLASSIFIER__CLASSIFICATION_NODE_HPP_
