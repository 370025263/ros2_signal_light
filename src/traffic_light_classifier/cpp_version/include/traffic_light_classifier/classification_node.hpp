#ifndef TRAFFIC_LIGHT_CLASSIFIER__CLASSIFICATION_NODE_HPP_
#define TRAFFIC_LIGHT_CLASSIFIER__CLASSIFICATION_NODE_HPP_

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/string.hpp>
#include <cv_bridge/cv_bridge.h>
#include "traffic_light_msg/msg/traffic_light_msg.hpp"

#include <torch/script.h>
#include <opencv2/opencv.hpp>
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
  cv::Mat drawResult(const cv::Mat& image, const std::string& label);

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;
  rclcpp::Subscription<traffic_light_msg::msg::TrafficLightMsg>::SharedPtr debug_subscription_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr result_publisher_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_result_publisher_;

  torch::jit::script::Module model_;
  std::map<std::string, int> class_to_idx_;
  std::map<int, std::string> idx_to_class_;
  
  int total_predictions_;
  int correct_predictions_;
  double total_processing_time_;
  int total_processed_images_;
};

}  // namespace traffic_light_classifier

#endif  // TRAFFIC_LIGHT_CLASSIFIER__CLASSIFICATION_NODE_HPP_
