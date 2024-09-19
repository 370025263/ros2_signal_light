#include "traffic_light_classifier/classification_node.hpp"

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <chrono>

namespace traffic_light_classifier
{

ClassificationNode::ClassificationNode(const rclcpp::NodeOptions & options)
: Node("classification_node", options),
  total_predictions_(0),
  correct_predictions_(0),
  total_processing_time_(0.0),
  total_processed_images_(0)
{
  subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
    "/perception/traffic_light_recognition/traffic_light_image", 10,
    std::bind(&ClassificationNode::imageCallback, this, std::placeholders::_1));
  
  debug_subscription_ = this->create_subscription<traffic_light_msg::msg::TrafficLightMsg>(
    "/perception/traffic_light_recognition/traffic_light_data", 10,
    std::bind(&ClassificationNode::debugCallback, this, std::placeholders::_1));
  
  result_publisher_ = this->create_publisher<std_msgs::msg::String>(
    "/perception/traffic_light_recognition/traffic_light_class", 10);
  
  image_result_publisher_ = this->create_publisher<sensor_msgs::msg::Image>(
    "traffic_light_result_image", 10);

  traffic_signals_publisher_ = this->create_publisher<autoware_perception_msgs::msg::TrafficLightGroupArray>(
    "/perception/traffic_light_recognition/traffic_signals", 10);

  // 初始化模型和其他成员变量
  std::string package_share_dir = ament_index_cpp::get_package_share_directory("traffic_light_classifier");
  std::string detection_model_path = package_share_dir + "/models/fasterrcnn_resnet50_fpn.onnx";
  std::string classification_model_path = package_share_dir + "/models/best_model.onnx";

  // 加载 ONNX 模型
  detection_net_ = cv::dnn::readNetFromONNX(detection_model_path);
  classification_net_ = cv::dnn::readNetFromONNX(classification_model_path);

  if (detection_net_.empty() || classification_net_.empty()) {
    RCLCPP_ERROR(this->get_logger(), "Failed to load the models");
    rclcpp::shutdown();
  }

  // 初始化 class_to_idx_ 和 idx_to_class_
  class_to_idx_ = {{"warning", 0}, {"stop", 1}, {"stopLeft", 2}, {"go", 3}, {"goLeft", 4}, {"warningLeft", 5}};
  for (const auto& pair : class_to_idx_) {
    idx_to_class_[pair.second] = pair.first;
  }

  this->declare_parameter("visualization_enabled", true);

  // 初始化相机内参矩阵（根据实际相机参数设置）
  // 假设焦距为 fx=fy=1000，主点在图像中心 (cx, cy)
  int image_width = 800;  // 根据实际情况设置
  int image_height = 600; // 根据实际情况设置
  double fx = 1000.0;
  double fy = 1000.0;
  double cx = image_width / 2.0;
  double cy = image_height / 2.0;
  camera_intrinsic_matrix_ = (cv::Mat_<double>(3,3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
}

void ClassificationNode::processImage(const cv::Mat& cv_image, const std::string& true_label)
{
  auto start_time = std::chrono::high_resolution_clock::now();

  // 预处理图像以供检测模型使用
  cv::Mat blob;
  cv::dnn::blobFromImage(cv_image, blob, 1.0 / 255, cv::Size(800, 800), cv::Scalar(0, 0, 0), true, false);

  // 设置输入并执行检测
  detection_net_.setInput(blob);
  std::vector<cv::Mat> detection_outputs;
  std::vector<std::string> detection_output_names = detection_net_.getUnconnectedOutLayersNames();
  detection_net_.forward(detection_outputs, detection_output_names);

  // 解析检测结果
  std::vector<cv::Rect> detected_boxes;
  std::vector<std::string> detected_labels;
  std::vector<float> detected_scores;

  // 填充 autoware_perception_msgs::msg::TrafficLightGroupArray 消息
  autoware_perception_msgs::msg::TrafficLightGroupArray traffic_light_group_array_msg;
  traffic_light_group_array_msg.stamp = this->now();

  // 创建一个 TrafficLightGroup
  autoware_perception_msgs::msg::TrafficLightGroup traffic_light_group;
  traffic_light_group.traffic_light_group_id = 0;  // 根据实际情况设置

  // 处理检测结果
  float score_threshold = 0.5;

  // 假设检测模型的输出为 [num_detections, 7]，每一行格式为 [batch_id, class_id, score, x1, y1, x2, y2]
  cv::Mat detections = detection_outputs[0];
  for (int i = 0; i < detections.rows; ++i) {
    float score = detections.at<float>(i, 2);
    if (score > score_threshold) {
      int class_id = static_cast<int>(detections.at<float>(i, 1));
      int x1 = static_cast<int>(detections.at<float>(i, 3) * cv_image.cols);
      int y1 = static_cast<int>(detections.at<float>(i, 4) * cv_image.rows);
      int x2 = static_cast<int>(detections.at<float>(i, 5) * cv_image.cols);
      int y2 = static_cast<int>(detections.at<float>(i, 6) * cv_image.rows);

      cv::Rect box(x1, y1, x2 - x1, y2 - y1);
      box &= cv::Rect(0, 0, cv_image.cols, cv_image.rows);

      detected_boxes.push_back(box);
      detected_scores.push_back(score);

      // 对每个检测到的区域进行分类
      cv::Mat cropped_img = cv_image(box);

      cv::Mat classification_blob;
      cv::dnn::blobFromImage(cropped_img, classification_blob, 1.0 / 255, cv::Size(224, 224), cv::Scalar(0, 0, 0), true, false);

      classification_net_.setInput(classification_blob);
      cv::Mat classifier_output = classification_net_.forward();

      cv::Point classIdPoint;
      double confidence;
      minMaxLoc(classifier_output, nullptr, &confidence, nullptr, &classIdPoint);
      int predicted_class_id = classIdPoint.x;

      std::string predicted_label = idx_to_class_[predicted_class_id];
      detected_labels.push_back(predicted_label);

      total_processed_images_++;

      // 构建 TrafficLightElement
      autoware_perception_msgs::msg::TrafficLightElement traffic_light_element;

      // 设置颜色
      if (predicted_label == "stop" || predicted_label == "stopLeft") {
        traffic_light_element.color = autoware_perception_msgs::msg::TrafficLightElement::RED;
      } else if (predicted_label == "warning" || predicted_label == "warningLeft") {
        traffic_light_element.color = autoware_perception_msgs::msg::TrafficLightElement::AMBER;
      } else if (predicted_label == "go" || predicted_label == "goLeft") {
        traffic_light_element.color = autoware_perception_msgs::msg::TrafficLightElement::GREEN;
      } else {
        traffic_light_element.color = autoware_perception_msgs::msg::TrafficLightElement::UNKNOWN;
      }

      // 设置形状
      if (predicted_label.find("Left") != std::string::npos) {
        traffic_light_element.shape = autoware_perception_msgs::msg::TrafficLightElement::LEFT_ARROW;
      } else {
        traffic_light_element.shape = autoware_perception_msgs::msg::TrafficLightElement::CIRCLE;
      }

      // 设置状态，假设为 SOLID_ON
      traffic_light_element.status = autoware_perception_msgs::msg::TrafficLightElement::SOLID_ON;

      // 设置置信度
      traffic_light_element.confidence = static_cast<float>(confidence);

      // 将 TrafficLightElement 添加到 TrafficLightGroup 中
      traffic_light_group.elements.push_back(traffic_light_element);

      // 交通信号灯组的 ID 可以根据实际需求设置
      traffic_light_group.traffic_light_group_id = 0;  // 示例中设为0
    }
  }

  // 将 TrafficLightGroup 添加到 TrafficLightGroupArray 中
  traffic_light_group_array_msg.traffic_light_groups.push_back(traffic_light_group);

  traffic_signals_publisher_->publish(traffic_light_group_array_msg);

  // 发布分类结果
  auto result_msg = std::make_unique<std_msgs::msg::String>();
  std::string concatenated_labels;
  for (const auto& label : detected_labels) {
    concatenated_labels += label + " ";
  }
  result_msg->data = "Predicted labels: " + concatenated_labels;
  result_publisher_->publish(std::move(result_msg));

  // 可视化
  if (this->get_parameter("visualization_enabled").as_bool()) {
    cv::Mat result_image = drawResult(cv_image, detected_boxes, detected_labels);
    sensor_msgs::msg::Image::SharedPtr result_img_msg = 
      cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", result_image).toImageMsg();
    image_result_publisher_->publish(*result_img_msg);
  }

  // 计算处理时间
  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
  total_processing_time_ += duration.count() / 1000.0;
  double avg_processing_time = total_processing_time_ / total_processed_images_;

  RCLCPP_INFO(this->get_logger(), "Processing time: %.4f seconds", duration.count() / 1000.0);
  RCLCPP_INFO(this->get_logger(), "Average processing time: %.4f seconds", avg_processing_time);
  RCLCPP_INFO(this->get_logger(), "Total processed images: %d", total_processed_images_);
}

cv::Mat ClassificationNode::drawResult(const cv::Mat& image, const std::vector<cv::Rect>& boxes, const std::vector<std::string>& labels)
{
  cv::Mat result_image = image.clone();

  for (size_t i = 0; i < boxes.size(); ++i) {
    cv::rectangle(result_image, boxes[i], cv::Scalar(0, 255, 0), 2);
    cv::putText(result_image, labels[i], cv::Point(boxes[i].x, boxes[i].y - 10),
                cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(0, 255, 0), 2);
  }

  return result_image;
}

void ClassificationNode::imageCallback(const sensor_msgs::msg::Image::SharedPtr msg)
{
  cv_bridge::CvImagePtr cv_ptr;
  try {
    cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
  } catch (cv_bridge::Exception& e) {
    RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
    return;
  }
  
  processImage(cv_ptr->image);
}

void ClassificationNode::debugCallback(const traffic_light_msg::msg::TrafficLightMsg::SharedPtr msg)
{
  cv_bridge::CvImagePtr cv_ptr;
  try {
    cv_ptr = cv_bridge::toCvCopy(msg->image, sensor_msgs::image_encodings::BGR8);
  } catch (cv_bridge::Exception& e) {
    RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
    return;
  }
  
  processImage(cv_ptr->image, msg->label);
}

}  // namespace traffic_light_classifier

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(traffic_light_classifier::ClassificationNode)
