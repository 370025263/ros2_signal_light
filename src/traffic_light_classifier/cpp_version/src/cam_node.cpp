#include "traffic_light_classifier/cam_node.hpp"

#include <filesystem>
#include <fstream>
#include <sstream>

namespace traffic_light_classifier
{

CamNode::CamNode(const rclcpp::NodeOptions & options)
: Node("cam_node", options), gen_(rd_())
{
  publisher_ = this->create_publisher<sensor_msgs::msg::Image>(
    "/perception/traffic_light_recognition/traffic_light_image", 10);
  debug_publisher_ = this->create_publisher<traffic_light_msg::msg::TrafficLightMsg>(
    "/perception/traffic_light_recognition/traffic_light_data", 10);
  timer_ = this->create_wall_timer(
    std::chrono::seconds(1), std::bind(&CamNode::timerCallback, this));

  this->declare_parameter("annotations_dir", "/home/casia/jupyter_notebooks/Annotations/Annotations/dayTrain");
  this->declare_parameter("data_dir", "/home/casia/jupyter_notebooks/dayTrain/dayTrain");
  this->declare_parameter("debug_mode", false);

  // 新增参数
  this->declare_parameter("publish_cropped_images", false);

  annotations_dir_ = this->get_parameter("annotations_dir").as_string();
  data_dir_ = this->get_parameter("data_dir").as_string();
  
  // 获取参数值
  publish_cropped_images_ = this->get_parameter("publish_cropped_images").as_bool();

  RCLCPP_INFO(this->get_logger(), "Annotations directory: %s", annotations_dir_.c_str());
  RCLCPP_INFO(this->get_logger(), "Data directory: %s", data_dir_.c_str());
  RCLCPP_INFO(this->get_logger(), "Publish cropped images: %s", publish_cropped_images_ ? "true" : "false");

  loadDataset();

  RCLCPP_INFO(this->get_logger(), "Dataset size: %zu", annotations_.size());
  RCLCPP_INFO(this->get_logger(), "Classes: %s", getClassesString().c_str());
}

void CamNode::timerCallback()
{
  if (annotations_.empty()) {
    RCLCPP_ERROR(this->get_logger(), "No images loaded. Cannot publish.");
    return;
  }

  int idx = dis_(gen_);
  const auto& ann = annotations_[idx];

  cv::Mat image = cv::imread(ann.img_path);
  if (image.empty()) {
    RCLCPP_ERROR(this->get_logger(), "Failed to read image: %s", ann.img_path.c_str());
    return;
  }

  cv::Mat image_to_publish;

  if (publish_cropped_images_) {
    cv::Mat cropped_img = cropImage(image, ann.bbox);
    image_to_publish = cropped_img;
  } else {
    image_to_publish = image;
  }

  sensor_msgs::msg::Image::SharedPtr img_msg = 
    cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", image_to_publish).toImageMsg();
  
  publisher_->publish(*img_msg);
  
  if (this->get_parameter("debug_mode").as_bool()) {
    auto debug_msg = std::make_unique<traffic_light_msg::msg::TrafficLightMsg>();
    debug_msg->image = *img_msg;
    debug_msg->label = ann.label;
    debug_publisher_->publish(std::move(debug_msg));
    RCLCPP_INFO(this->get_logger(), "Published debug image with label: %s", ann.label.c_str());
  } else {
    RCLCPP_INFO(this->get_logger(), "Published image");
  }
}

// 其余代码保持不变

}  // namespace traffic_light_classifier

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(traffic_light_classifier::CamNode)
