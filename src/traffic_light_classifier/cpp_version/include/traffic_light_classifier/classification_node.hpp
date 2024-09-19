#ifndef TRAFFIC_LIGHT_CLASSIFIER__CLASSIFICATION_NODE_HPP_
#define TRAFFIC_LIGHT_CLASSIFIER__CLASSIFICATION_NODE_HPP_

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/string.hpp>
#include <cv_bridge/cv_bridge.h>
#include "traffic_light_msg/msg/traffic_light_msg.hpp"
#include <autoware_perception_msgs/msg/traffic_light_group_array.hpp>

#include <opencv2/opencv.hpp>

#include <map>

// 包含 ONNX Runtime 头文件
#include <onnxruntime_cxx_api.h>

namespace traffic_light_classifier
{

class ClassificationNode : public rclcpp::Node
{
public:
  explicit ClassificationNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
  ~ClassificationNode(); // 添加析构函数声明

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

  // ONNX Runtime 会话指针
  Ort::Env env_;
  Ort::Session* detection_session_;
  Ort::Session* classification_session_;

  Ort::SessionOptions session_options_;

  std::map<std::string, int> class_to_idx_;
  std::map<int, std::string> idx_to_class_;
  
  int total_predictions_;
  int correct_predictions_;
  double total_processing_time_;
  int total_processed_images_;

  // 相机内参矩阵的占位符（如果需要）
  cv::Mat camera_intrinsic_matrix_;
};

}  // namespace traffic_light_classifier

#endif  // TRAFFIC_LIGHT_CLASSIFIER__CLASSIFICATION_NODE_HPP_
