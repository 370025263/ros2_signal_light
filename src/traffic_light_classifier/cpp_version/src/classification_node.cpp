#include "traffic_light_classifier/classification_node.hpp"

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <chrono>
#include <geometry_msgs/msg/point.hpp>
#include <cv_bridge/cv_bridge.h>

// For ONNX Runtime
#include <onnxruntime_cxx_api.h>

namespace traffic_light_classifier
{

ClassificationNode::ClassificationNode(const rclcpp::NodeOptions & options)
: Node("classification_node", options),
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

  // Initialize model paths and other member variables
  std::string package_share_dir = ament_index_cpp::get_package_share_directory("traffic_light_classifier");
  std::string detection_model_path = package_share_dir + "/models/fasterrcnn_resnet50_fpn.onnx";
  std::string classification_model_path = package_share_dir + "/models/traffic_light_classifier.onnx";

  // Set session options (optional)
  session_options_.SetIntraOpNumThreads(1);
  session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

  // Load detection model
  detection_session_ = new Ort::Session(env_, detection_model_path.c_str(), session_options_);

  // Load classification model
  classification_session_ = new Ort::Session(env_, classification_model_path.c_str(), session_options_);

  // Initialize class mappings
  class_to_idx_ = {{"warning", 0}, {"stop", 1}, {"stopLeft", 2}, {"go", 3}, {"goLeft", 4}, {"warningLeft", 5}};
  for (const auto& pair : class_to_idx_) {
    idx_to_class_[pair.second] = pair.first;
  }

  this->declare_parameter("visualization_enabled", true);

  // Initialize camera intrinsic matrix here if needed
}

void ClassificationNode::processImage(const cv::Mat& cv_image, const std::string& true_label)
{
  (void)true_label; // Suppress unused parameter warning

  auto start_time = std::chrono::high_resolution_clock::now();

  // Preprocess image for detection
  cv::Mat resized_image;
  cv::resize(cv_image, resized_image, cv::Size(800, 800));
  cv::Mat rgb_image;
  cv::cvtColor(resized_image, rgb_image, cv::COLOR_BGR2RGB);
  rgb_image.convertTo(rgb_image, CV_32F, 1.0 / 255);

  // Create input tensor
  std::vector<int64_t> input_dims = {1, rgb_image.channels(), rgb_image.rows, rgb_image.cols};
  std::vector<float> input_tensor_values(rgb_image.begin<float>(), rgb_image.end<float>());

  // Prepare input tensor
  Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemTypeDefault);
  Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_values.size(), input_dims.data(), input_dims.size());

  // Run detection model
  Ort::AllocatorWithDefaultOptions allocator;

  // Get input and output names
  Ort::AllocatedStringPtr input_name_ptr = detection_session_->GetInputNameAllocated(0, allocator);
  std::vector<const char*> input_names = {input_name_ptr.get()};

  std::vector<Ort::AllocatedStringPtr> output_name_ptrs;
  output_name_ptrs.reserve(3);

  std::vector<const char*> output_names;
  for (size_t i = 0; i < 3; ++i) {
    output_name_ptrs.push_back(detection_session_->GetOutputNameAllocated(i, allocator));
    output_names.push_back(output_name_ptrs.back().get());
  }

  auto output_tensors = detection_session_->Run(Ort::RunOptions{nullptr}, input_names.data(), &input_tensor, 1, output_names.data(), output_names.size());

  // Extract outputs
  float* boxes_data = output_tensors[0].GetTensorMutableData<float>();
  int64_t* labels_data = output_tensors[1].GetTensorMutableData<int64_t>();
  float* scores_data = output_tensors[2].GetTensorMutableData<float>();

  std::vector<int64_t> boxes_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
  std::vector<int64_t> labels_shape = output_tensors[1].GetTensorTypeAndShapeInfo().GetShape();
  std::vector<int64_t> scores_shape = output_tensors[2].GetTensorTypeAndShapeInfo().GetShape();

  // Convert raw pointers to vectors
  size_t num_boxes = boxes_shape[1];
  std::vector<float> boxes(boxes_data, boxes_data + num_boxes * 4);
  std::vector<int64_t> labels(labels_data, labels_data + labels_shape[1]);
  std::vector<float> scores(scores_data, scores_data + scores_shape[1]);

  // Process detection results
  std::vector<cv::Rect> detected_boxes;
  std::vector<std::string> detected_labels;
  std::vector<float> detected_scores;

  autoware_perception_msgs::msg::TrafficLightGroupArray traffic_light_group_array_msg;
  traffic_light_group_array_msg.stamp = this->now();

  autoware_perception_msgs::msg::TrafficLightGroup traffic_light_group;
  traffic_light_group.traffic_light_group_id = 0;  // Modify as needed

  float score_threshold = 0.5;

  size_t num_detections = scores.size();

  for (size_t i = 0; i < num_detections; ++i) {
    float score = scores[i];
    if (score > score_threshold) {
      // int label = static_cast<int>(labels[i]);
      float x1 = boxes[i * 4];
      float y1 = boxes[i * 4 + 1];
      float x2 = boxes[i * 4 + 2];
      float y2 = boxes[i * 4 + 3];

      // Scale bounding box back to original image size
      x1 *= static_cast<float>(cv_image.cols) / 800.0f;
      y1 *= static_cast<float>(cv_image.rows) / 800.0f;
      x2 *= static_cast<float>(cv_image.cols) / 800.0f;
      y2 *= static_cast<float>(cv_image.rows) / 800.0f;

      cv::Rect box(cv::Point(static_cast<int>(x1), static_cast<int>(y1)), cv::Point(static_cast<int>(x2), static_cast<int>(y2)));
      box &= cv::Rect(0, 0, cv_image.cols, cv_image.rows);

      detected_boxes.push_back(box);
      detected_scores.push_back(score);

      // Crop detected region
      cv::Mat cropped_img = cv_image(box);

      // Preprocess for classification
      cv::Mat class_resized_img;
      cv::resize(cropped_img, class_resized_img, cv::Size(224, 224));
      cv::cvtColor(class_resized_img, class_resized_img, cv::COLOR_BGR2RGB);
      class_resized_img.convertTo(class_resized_img, CV_32F, 1.0 / 255);

      // Create input tensor for classification
      std::vector<int64_t> class_input_dims = {1, class_resized_img.channels(), class_resized_img.rows, class_resized_img.cols};
      std::vector<float> class_input_tensor_values(class_resized_img.begin<float>(), class_resized_img.end<float>());

      Ort::Value class_input_tensor = Ort::Value::CreateTensor<float>(memory_info, class_input_tensor_values.data(), class_input_tensor_values.size(), class_input_dims.data(), class_input_dims.size());

      // Run classification model
      Ort::AllocatedStringPtr class_input_name_ptr = classification_session_->GetInputNameAllocated(0, allocator);
      std::vector<const char*> class_input_names = {class_input_name_ptr.get()};

      Ort::AllocatedStringPtr class_output_name_ptr = classification_session_->GetOutputNameAllocated(0, allocator);
      std::vector<const char*> class_output_names = {class_output_name_ptr.get()};

      auto class_output_tensors = classification_session_->Run(Ort::RunOptions{nullptr}, class_input_names.data(), &class_input_tensor, 1, class_output_names.data(), class_output_names.size());

      // Extract classification results
      float* class_scores = class_output_tensors[0].GetTensorMutableData<float>();
      size_t class_num = class_output_tensors[0].GetTensorTypeAndShapeInfo().GetShape()[1];

      // Get the class with the highest score
      auto max_elem = std::max_element(class_scores, class_scores + class_num);
      int predicted_class_id = static_cast<int>(std::distance(class_scores, max_elem));
      double confidence = *max_elem;

      std::string predicted_label = idx_to_class_[predicted_class_id];
      detected_labels.push_back(predicted_label);

      total_processed_images_++;

      // Build TrafficLightElement
      autoware_perception_msgs::msg::TrafficLightElement traffic_light_element;

      // Set color
      if (predicted_label == "stop" || predicted_label == "stopLeft") {
        traffic_light_element.color = autoware_perception_msgs::msg::TrafficLightElement::RED;
      } else if (predicted_label == "warning" || predicted_label == "warningLeft") {
        traffic_light_element.color = autoware_perception_msgs::msg::TrafficLightElement::AMBER;
      } else if (predicted_label == "go" || predicted_label == "goLeft") {
        traffic_light_element.color = autoware_perception_msgs::msg::TrafficLightElement::GREEN;
      } else {
        traffic_light_element.color = autoware_perception_msgs::msg::TrafficLightElement::UNKNOWN;
      }

      // Set shape
      if (predicted_label.find("Left") != std::string::npos) {
        traffic_light_element.shape = autoware_perception_msgs::msg::TrafficLightElement::LEFT_ARROW;
      } else {
        traffic_light_element.shape = autoware_perception_msgs::msg::TrafficLightElement::CIRCLE;
      }

      // Set status, assuming SOLID_ON
      traffic_light_element.status = autoware_perception_msgs::msg::TrafficLightElement::SOLID_ON;

      // Set confidence
      traffic_light_element.confidence = static_cast<float>(confidence);

      // Add element to group
      traffic_light_group.elements.push_back(traffic_light_element);
    }
  }

  // Add group to array
  traffic_light_group_array_msg.traffic_light_groups.push_back(traffic_light_group);

  traffic_signals_publisher_->publish(traffic_light_group_array_msg);

  // Publish classification results
  auto result_msg = std::make_unique<std_msgs::msg::String>();
  std::string concatenated_labels;
  for (const auto& label : detected_labels) {
    concatenated_labels += label + " ";
  }
  result_msg->data = "Predicted labels: " + concatenated_labels;
  result_publisher_->publish(std::move(result_msg));

  // Visualization
  if (this->get_parameter("visualization_enabled").as_bool()) {
    cv::Mat result_image = drawResult(cv_image, detected_boxes, detected_labels);
    sensor_msgs::msg::Image::SharedPtr result_img_msg = 
      cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", result_image).toImageMsg();
    image_result_publisher_->publish(*result_img_msg);
  }

  // Compute processing time
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

ClassificationNode::~ClassificationNode()
{
  // Clean up sessions
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
