#include "traffic_light_classifier/classification_node.hpp"

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <chrono>
#include <geometry_msgs/msg/point.hpp>
#include <cv_bridge/cv_bridge.h>

#include <onnxruntime_cxx_api.h>

namespace traffic_light_classifier
{

ClassificationNode::ClassificationNode(const rclcpp::NodeOptions &options)
: rclcpp::Node("classification_node", options),
  env_(ORT_LOGGING_LEVEL_WARNING, "TrafficLightClassifier"),
  detection_session_(nullptr),
  classification_session_(nullptr),
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

  // 初始化模型路径和其他成员变量
  std::string package_share_dir = ament_index_cpp::get_package_share_directory("traffic_light_classifier");
  std::string detection_model_path = package_share_dir + "/models/fasterrcnn_resnet50_fpn.onnx";
  std::string classification_model_path = package_share_dir + "/models/traffic_light_classifier.onnx";

  // 设置会话选项
  Ort::SessionOptions session_options;
  session_options.SetIntraOpNumThreads(1);
  session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

  // 设置CUDA执行提供程序
  OrtCUDAProviderOptions cuda_options;
  session_options.AppendExecutionProvider_CUDA(cuda_options);

  // 加载检测模型
  detection_session_ = new Ort::Session(env_, detection_model_path.c_str(), session_options);

  // 加载分类模型
  classification_session_ = new Ort::Session(env_, classification_model_path.c_str(), session_options);

  // 初始化类别映射
  class_to_idx_ = {{"warning", 0}, {"stop", 1}, {"stopLeft", 2}, {"go", 3}, {"goLeft", 4}, {"warningLeft", 5}};
  for (const auto& pair : class_to_idx_) {
    idx_to_class_[pair.second] = pair.first;
  }

  this->declare_parameter("visualization_enabled", true);
}
cv::Mat HWC2CHW(const cv::Mat& image) {
    std::vector<cv::Mat> channels(3);
    cv::split(image, channels);

    cv::Mat result(image.rows * 3, image.cols, CV_32F);

    for (int i = 0; i < 3; ++i) {
        channels[i].copyTo(result(cv::Rect(0, image.rows * i, image.cols, image.rows)));
    }

    return result;
}
void ClassificationNode::processImage(const cv::Mat& cv_image, const std::string& true_label)
{
  (void)true_label; // 抑制未使用参数警告

  auto start_time = std::chrono::high_resolution_clock::now();

  // 记录图像处理的各个步骤的时间
  std::vector<std::pair<std::string, double>> time_measurements;

  auto measure_time = [&](const std::string& step_name, auto start_time) {
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    time_measurements.push_back({step_name, duration.count() / 1000000.0});
    return end_time;
  };

  auto step_start_time = start_time;

  RCLCPP_INFO(this->get_logger(), "图像大小: %dx%d", cv_image.cols, cv_image.rows);

  // 预处理图像用于检测
  cv::Mat resized_image;
  cv::resize(cv_image, resized_image, cv::Size(224, 224));
  RCLCPP_INFO(this->get_logger(), "调整大小后的图像: %dx%d", resized_image.cols, resized_image.rows);

  cv::Mat rgb_image;
  cv::cvtColor(resized_image, rgb_image, cv::COLOR_BGR2RGB);
  rgb_image.convertTo(rgb_image, CV_32F, 1.0 / 255); // 归一化到 [0, 1]

  cv::Mat chw_image = HWC2CHW(rgb_image);
  step_start_time = measure_time("图像预处理", step_start_time);

  // 准备输入张量
  std::vector<int64_t> input_dims = {1, 3, 224, 224};
  size_t input_tensor_size = chw_image.total();
  std::vector<float> input_tensor_values(input_tensor_size);

  if (chw_image.isContinuous()) {
    std::memcpy(input_tensor_values.data(), chw_image.data, input_tensor_size * sizeof(float));
  } else {
    size_t idx = 0;
    for (int i = 0; i < chw_image.rows; ++i) {
      const float* row_ptr = chw_image.ptr<float>(i);
      std::memcpy(input_tensor_values.data() + idx, row_ptr, chw_image.cols * sizeof(float));
      idx += chw_image.cols;
    }
  }

  Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemTypeDefault);
  Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
      memory_info, input_tensor_values.data(), input_tensor_values.size(),
      input_dims.data(), input_dims.size());
  step_start_time = measure_time("准备输入张量", step_start_time);

  // 运行检测模型
  Ort::AllocatorWithDefaultOptions allocator;
  auto input_name = detection_session_->GetInputNameAllocated(0, allocator);

  std::vector<Ort::AllocatedStringPtr> output_names;
  output_names.push_back(detection_session_->GetOutputNameAllocated(0, allocator));
  output_names.push_back(detection_session_->GetOutputNameAllocated(1, allocator));
  output_names.push_back(detection_session_->GetOutputNameAllocated(2, allocator));

  std::vector<const char*> input_names = {input_name.get()};
  std::vector<const char*> output_names_char;
  for (const auto& name : output_names) {
    output_names_char.push_back(name.get());
  }

  auto output_tensors = detection_session_->Run(Ort::RunOptions{nullptr}, input_names.data(), &input_tensor, 1, output_names_char.data(), output_names_char.size());
  step_start_time = measure_time("检测模型推理", step_start_time);

  // 提取输出
  auto boxes_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
  auto labels_shape = output_tensors[1].GetTensorTypeAndShapeInfo().GetShape();
  auto scores_shape = output_tensors[2].GetTensorTypeAndShapeInfo().GetShape();

  float* boxes_data = output_tensors[0].GetTensorMutableData<float>();
  int64_t* labels_data = output_tensors[1].GetTensorMutableData<int64_t>();
  float* scores_data = output_tensors[2].GetTensorMutableData<float>();

  size_t num_detections = boxes_shape[1];  // 假设形状是 [1, num_detections, 4]
  RCLCPP_INFO(this->get_logger(), "检测到的对象数量: %zu", num_detections);

  // 处理检测结果
  std::vector<cv::Rect> detected_boxes;
  std::vector<std::string> detected_labels;
  std::vector<float> detected_scores;

  autoware_perception_msgs::msg::TrafficLightGroupArray traffic_light_group_array_msg;
  traffic_light_group_array_msg.stamp = this->now();

  autoware_perception_msgs::msg::TrafficLightGroup traffic_light_group;
  traffic_light_group.traffic_light_group_id = 0;  // 根据需要修改

  float score_threshold = 0.95;
  int traffic_light_class_id = 10;  // COCO数据集中交通灯的类别ID

  int original_width = cv_image.cols;
  int original_height = cv_image.rows;

  for (size_t i = 0; i < num_detections; ++i) {
    float score = scores_data[i];
    int label = static_cast<int>(labels_data[i]);
    if (score > score_threshold && label == traffic_light_class_id) {
      // 将坐标从模型输出缩放到原始图像尺寸
      float x1 = (boxes_data[i * 4] / 224.0f) * original_width;
      float y1 = (boxes_data[i * 4 + 1] / 224.0f) * original_height;
      float x2 = (boxes_data[i * 4 + 2] / 224.0f) * original_width;
      float y2 = (boxes_data[i * 4 + 3] / 224.0f) * original_height;

      // 确保坐标在图像范围内
      x1 = std::max(0.0f, std::min(x1, static_cast<float>(original_width - 1)));
      y1 = std::max(0.0f, std::min(y1, static_cast<float>(original_height - 1)));
      x2 = std::max(0.0f, std::min(x2, static_cast<float>(original_width - 1)));
      y2 = std::max(0.0f, std::min(y2, static_cast<float>(original_height - 1)));

      RCLCPP_INFO(this->get_logger(), "缩放后的坐标: x1=%.2f, y1=%.2f, x2=%.2f, y2=%.2f", x1, y1, x2, y2);

      cv::Rect box(cv::Point(static_cast<int>(x1), static_cast<int>(y1)),
                   cv::Point(static_cast<int>(x2), static_cast<int>(y2)));

      detected_boxes.push_back(box);
      detected_scores.push_back(score);

      // 裁剪检测到的区域并进行分类处理
      cv::Mat cropped_img = cv_image(box);
      cv::Mat class_resized_img;
      cv::resize(cropped_img, class_resized_img, cv::Size(224, 224));
      cv::cvtColor(class_resized_img, class_resized_img, cv::COLOR_BGR2RGB);
      class_resized_img.convertTo(class_resized_img, CV_32F, 1.0 / 255);
      cv::Mat chw_img = HWC2CHW(class_resized_img);

      // 创建分类模型的输入张量
      std::vector<int64_t> class_input_dims = {1, 3, 224, 224};
      size_t class_input_tensor_size = chw_img.total();
      std::vector<float> class_input_tensor_values(class_input_tensor_size);

      if (chw_img.isContinuous()) {
        std::memcpy(class_input_tensor_values.data(), chw_img.data, class_input_tensor_size * sizeof(float));
      } else {
        size_t idx = 0;
        for (int i = 0; i < chw_img.rows; ++i) {
          const float* row_ptr = chw_img.ptr<float>(i);
          std::memcpy(class_input_tensor_values.data() + idx, row_ptr, chw_img.cols * sizeof(float));
          idx += chw_img.cols;
        }
      }

      Ort::Value class_input_tensor = Ort::Value::CreateTensor<float>(
          memory_info, class_input_tensor_values.data(), class_input_tensor_values.size(),
          class_input_dims.data(), class_input_dims.size());

      // 运行分类模型
      auto class_input_name = classification_session_->GetInputNameAllocated(0, allocator);
      auto class_output_name = classification_session_->GetOutputNameAllocated(0, allocator);
      std::vector<const char*> class_input_names = {class_input_name.get()};
      std::vector<const char*> class_output_names = {class_output_name.get()};

      auto class_output_tensors = classification_session_->Run(Ort::RunOptions{nullptr}, class_input_names.data(), &class_input_tensor, 1, class_output_names.data(), class_output_names.size());

      // 提取分类结果
      float* class_scores = class_output_tensors[0].GetTensorMutableData<float>();
      size_t class_num = class_output_tensors[0].GetTensorTypeAndShapeInfo().GetShape()[1];

      auto max_elem = std::max_element(class_scores, class_scores + class_num);
      int predicted_class_id = static_cast<int>(std::distance(class_scores, max_elem));
      double confidence = *max_elem;

      std::string predicted_label = idx_to_class_[predicted_class_id];
      detected_labels.push_back(predicted_label);

      // 构建TrafficLightElement
      autoware_perception_msgs::msg::TrafficLightElement traffic_light_element;
      if (predicted_label == "stop" || predicted_label == "stopLeft") {
        traffic_light_element.color = autoware_perception_msgs::msg::TrafficLightElement::RED;
      } else if (predicted_label == "warning" || predicted_label == "warningLeft") {
        traffic_light_element.color = autoware_perception_msgs::msg::TrafficLightElement::AMBER;
      } else if (predicted_label == "go" || predicted_label == "goLeft") {
        traffic_light_element.color = autoware_perception_msgs::msg::TrafficLightElement::GREEN;
      } else {
        traffic_light_element.color = autoware_perception_msgs::msg::TrafficLightElement::UNKNOWN;
      }

      traffic_light_element.shape = (predicted_label.find("Left") != std::string::npos) ?
        autoware_perception_msgs::msg::TrafficLightElement::LEFT_ARROW :
        autoware_perception_msgs::msg::TrafficLightElement::CIRCLE;

      traffic_light_element.status = autoware_perception_msgs::msg::TrafficLightElement::SOLID_ON;
      traffic_light_element.confidence = static_cast<float>(confidence);

      traffic_light_group.elements.push_back(traffic_light_element);
    }
  }
  step_start_time = measure_time("处理检测和分类结果", step_start_time);

  traffic_light_group_array_msg.traffic_light_groups.push_back(traffic_light_group);
  traffic_signals_publisher_->publish(traffic_light_group_array_msg);

  // 发布分类结果
  auto result_msg = std::make_unique<std_msgs::msg::String>();
  std::string concatenated_labels;
  for (const auto& label : detected_labels) {
    concatenated_labels += label + " ";
  }
  result_msg->data = "预测标签: " + concatenated_labels;
  result_publisher_->publish(std::move(result_msg));

  // 可视化
  if (this->get_parameter("visualization_enabled").as_bool()) {
    cv::Mat result_image = cv_image.clone();
    result_image = drawResult(result_image, detected_boxes, detected_labels);
    sensor_msgs::msg::Image::SharedPtr result_img_msg =
      cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", result_image).toImageMsg();
    image_result_publisher_->publish(*result_img_msg);
  }
  step_start_time = measure_time("结果发布和可视化", step_start_time);

  // 计算和记录处理时间
  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
  total_processing_time_ += duration.count() / 1000.0;
  total_processed_images_++;
  double avg_processing_time = total_processing_time_ / total_processed_images_;

  // 输出所有时间测量
  RCLCPP_INFO(this->get_logger(), "==== 时间测量开始 ====");
  for (const auto& measurement : time_measurements) {
    RCLCPP_INFO(this->get_logger(), "步骤时间: %s: %.6f 秒", measurement.first.c_str(), measurement.second);
  }
  RCLCPP_INFO(this->get_logger(), "总处理时间: %.4f 秒", duration.count() / 1000.0);
  RCLCPP_INFO(this->get_logger(), "平均处理时间: %.4f 秒", avg_processing_time);
  RCLCPP_INFO(this->get_logger(), "总处理图像数: %d", total_processed_images_);
  RCLCPP_INFO(this->get_logger(), "==== 时间测量结束 ====");
}

cv::Mat ClassificationNode::drawResult(const cv::Mat& image, const std::vector<cv::Rect>& boxes, const std::vector<std::string>& labels)
{
  cv::Mat result_image = image.clone();

  for (size_t i = 0; i < boxes.size(); ++i) {
    // 在原始图像上绘制边界框
    cv::rectangle(result_image, boxes[i], cv::Scalar(0, 255, 0), 2);

    // 确保文本位置在图像内
    int text_y = std::max(boxes[i].y - 10, 0);
    cv::putText(result_image, labels[i], cv::Point(boxes[i].x, text_y),
                cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(0, 255, 0), 2);
  }

  return result_image;
}

void ClassificationNode::imageCallback(const sensor_msgs::msg::Image::SharedPtr msg)
{
  cv_bridge::CvImagePtr cv_ptr;
  try {
    cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    RCLCPP_INFO(this->get_logger(), "接收到图像大小: %dx%d", cv_ptr->image.cols, cv_ptr->image.rows);
  } catch (cv_bridge::Exception& e) {
    RCLCPP_ERROR(this->get_logger(), "cv_bridge 异常: %s", e.what());
    return;
  }

  processImage(cv_ptr->image);
}

void ClassificationNode::debugCallback(const traffic_light_msg::msg::TrafficLightMsg::SharedPtr msg)
{
  cv_bridge::CvImagePtr cv_ptr;
  try {
    cv_ptr = cv_bridge::toCvCopy(msg->image, sensor_msgs::image_encodings::BGR8);
    RCLCPP_INFO(this->get_logger(), "接收到调试图像大小: %dx%d", cv_ptr->image.cols, cv_ptr->image.rows);
  } catch (cv_bridge::Exception& e) {
    RCLCPP_ERROR(this->get_logger(), "cv_bridge 异常: %s", e.what());
    return;
  }

  processImage(cv_ptr->image, msg->label);
}

ClassificationNode::~ClassificationNode()
{
  // 清理会话
  delete detection_session_;
  delete classification_session_;
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