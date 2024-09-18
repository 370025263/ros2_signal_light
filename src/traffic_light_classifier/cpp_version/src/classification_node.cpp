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

  // 初始化模型和其他成员变量
  std::string package_share_dir = ament_index_cpp::get_package_share_directory("traffic_light_classifier");
  std::string model_path = package_share_dir + "/models/best_model.pt";
  
  try {
    model_ = torch::jit::load(model_path);
    model_.eval();
  } catch (const c10::Error& e) {
    RCLCPP_ERROR(this->get_logger(), "Error loading the model: %s", e.what());
    rclcpp::shutdown();
  }

  // 初始化 class_to_idx_ 和 idx_to_class_
  class_to_idx_ = {{"warning", 0}, {"stop", 1}, {"stopLeft", 2}, {"go", 3}, {"goLeft", 4}, {"warningLeft", 5}};
  for (const auto& pair : class_to_idx_) {
    idx_to_class_[pair.second] = pair.first;
  }

  this->declare_parameter("plot_result", true);
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

void ClassificationNode::processImage(const cv::Mat& cv_image, const std::string& true_label)
{
  auto start_time = std::chrono::high_resolution_clock::now();

  // 将OpenCV图像转换为Torch张量
  cv::Mat resized_image;
  cv::resize(cv_image, resized_image, cv::Size(224, 224));
  cv::Mat float_image;
  resized_image.convertTo(float_image, CV_32F, 1.0 / 255);
  
  auto input_tensor = torch::from_blob(float_image.data, {1, 224, 224, 3});
  input_tensor = input_tensor.permute({0, 3, 1, 2});
  
  // 标准化
  input_tensor[0][0] = input_tensor[0][0].sub_(0.485).div_(0.229);
  input_tensor[0][1] = input_tensor[0][1].sub_(0.456).div_(0.224);
  input_tensor[0][2] = input_tensor[0][2].sub_(0.406).div_(0.225);
  
  // 进行预测
  torch::NoGradGuard no_grad;
  auto output = model_.forward({input_tensor}).toTensor();
  auto predicted = output.argmax(1);
  
  std::string predicted_label = idx_to_class_[predicted.item<int>()];
  
  // 更新统计信息
  total_processed_images_++;
  if (!true_label.empty()) {
    total_predictions_++;
    if (predicted_label == true_label) {
      correct_predictions_++;
    }
    double accuracy = static_cast<double>(correct_predictions_) / total_predictions_;
    RCLCPP_INFO(this->get_logger(), "Predicted: %s, True: %s, Accuracy: %.2f", 
                predicted_label.c_str(), true_label.c_str(), accuracy);
  } else {
    RCLCPP_INFO(this->get_logger(), "Predicted: %s", predicted_label.c_str());
  }

  // 发布分类结果
  auto result_msg = std::make_unique<std_msgs::msg::String>();
  result_msg->data = "Predicted: " + predicted_label;
  if (!true_label.empty()) {
    result_msg->data += ", True: " + true_label;
  }
  result_publisher_->publish(std::move(result_msg));

  // 如果启用了绘图，则绘制结果并发布
  if (this->get_parameter("plot_result").as_bool()) {
    cv::Mat result_image = drawResult(cv_image, predicted_label);
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

cv::Mat ClassificationNode::drawResult(const cv::Mat& image, const std::string& label)
{
  cv::Mat result_image = image.clone();
  int label_height = 40;
  int margin = 10;
  
  // 创建一个新的图像，包括原始图像和标签区域
  cv::Mat output_image(result_image.rows + label_height + margin, result_image.cols, result_image.type(), cv::Scalar(255, 255, 255));
  
  // 将原始图像复制到新图像中
  result_image.copyTo(output_image(cv::Rect(0, label_height + margin, result_image.cols, result_image.rows)));
  
  // 在标签区域绘制文本
  cv::putText(output_image, "Pred: " + label, cv::Point(10, 30), 
              cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 2);
  
  return output_image;
}

}  // namespace traffic_light_classifier

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(traffic_light_classifier::ClassificationNode)
int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<traffic_light_classifier::ClassificationNode>());
  rclcpp::shutdown();
  return 0;
}
