#ifndef TRAFFIC_LIGHT_CLASSIFIER__CAM_NODE_HPP_
#define TRAFFIC_LIGHT_CLASSIFIER__CAM_NODE_HPP_

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include "traffic_light_msg/msg/traffic_light_msg.hpp"

#include <opencv2/opencv.hpp>
#include <random>
#include <vector>
#include <string>
#include <map>

namespace traffic_light_classifier
{

struct Annotation {
  std::string img_path;
  std::string label;
  std::vector<int> bbox;
};

class CamNode : public rclcpp::Node
{
public:
  explicit CamNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());

private:
  void timerCallback();
  void loadDataset();
  cv::Mat cropImage(const cv::Mat& image, const std::vector<int>& bbox);
  std::string getClassesString() const;

  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr publisher_;
  rclcpp::Publisher<traffic_light_msg::msg::TrafficLightMsg>::SharedPtr debug_publisher_;
  rclcpp::TimerBase::SharedPtr timer_;
  
  std::string annotations_dir_;
  std::string data_dir_;
  std::vector<Annotation> annotations_;
  std::map<std::string, int> class_to_idx_;
  std::random_device rd_;
  std::mt19937 gen_;
  std::uniform_int_distribution<> dis_;

  // 新增成员变量
  bool publish_cropped_images_;
};

}  // namespace traffic_light_classifier

#endif  // TRAFFIC_LIGHT_CLASSIFIER__CAM_NODE_HPP_
