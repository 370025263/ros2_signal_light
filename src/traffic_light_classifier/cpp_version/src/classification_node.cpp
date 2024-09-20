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
cv::Mat HWC2CHW(const cv::Mat& image) {
    std::vector<cv::Mat> channels(3);
    cv::split(image, channels);

    // 创建一个大的单通道图像来存储所有通道
    cv::Mat result(image.rows * 3, image.cols, CV_32F);

    for (int i = 0; i < 3; ++i) {
        channels[i].copyTo(result(cv::Rect(0, image.rows * i, image.cols, image.rows)));
    }

    return result;
}
void ClassificationNode::processImage(const cv::Mat& cv_image, const std::string& true_label)
{
  (void)true_label; // Suppress unused parameter warning

  auto start_time = std::chrono::high_resolution_clock::now();

  // print image size
  RCLCPP_INFO(this->get_logger(), "Image size: %dx%d", cv_image.cols, cv_image.rows);

  // Preprocess image for detection
  cv::Mat resized_image;
  cv::resize(cv_image, resized_image, cv::Size(224, 224));

  // size after resize
    RCLCPP_INFO(this->get_logger(), "Resized image size: %dx%d", resized_image.cols, resized_image.rows);
  cv::Mat rgb_image;
  cv::cvtColor(resized_image, rgb_image, cv::COLOR_BGR2RGB);
  rgb_image.convertTo(rgb_image, CV_32F, 1.0 / 255); // Normalize to [0, 1]

// 执行HWC到CHW的转换
cv::Mat chw_image = HWC2CHW(rgb_image);

std::vector<int64_t> input_dims = {1, 3, static_cast<int64_t>(rgb_image.rows), static_cast<int64_t>(rgb_image.cols)};

size_t input_tensor_size = chw_image.total();
std::vector<float> input_tensor_values(input_tensor_size);

  // Flatten the image data and copy to input tensor

  // Copy image data to input tensor
  if (rgb_image.isContinuous()) {
    // Copy all at once
      std::memcpy(input_tensor_values.data(), chw_image.data, input_tensor_size * sizeof(float));
  } else {
    // Copy row by row
      size_t idx = 0;
      for (int i = 0; i < chw_image.rows; ++i) {
          const float* row_ptr = chw_image.ptr<float>(i);
          std::memcpy(input_tensor_values.data() + idx, row_ptr, chw_image.cols * chw_image.channels() * sizeof(float));
          idx += chw_image.cols * chw_image.channels();
      }
  }

  // Prepare input tensor
  Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemTypeDefault);
  Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
      memory_info,
      input_tensor_values.data(),
      input_tensor_values.size(),
      input_dims.data(),
      input_dims.size());

  // Run detection model
  // Get input and output names
  Ort::AllocatorWithDefaultOptions allocator;
  Ort::AllocatedStringPtr input_name_ptr = detection_session_->GetInputNameAllocated(0, allocator);
  std::vector<const char*> input_names = {input_name_ptr.get()};

  std::vector<Ort::AllocatedStringPtr> output_name_ptrs;
  output_name_ptrs.reserve(3);

  std::vector<const char*> output_names;
  for (size_t i = 0; i < 3; ++i) {
    output_name_ptrs.push_back(detection_session_->GetOutputNameAllocated(i, allocator));
    output_names.push_back(output_name_ptrs.back().get());
  }
// Print input shape
Ort::TypeInfo type_info = detection_session_->GetInputTypeInfo(0);
auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
std::vector<int64_t> input_shape = tensor_info.GetShape();
RCLCPP_INFO(this->get_logger(), "DETECTION MODEL, Input shape: %ld %ld %ld %ld", input_shape[0], input_shape[1], input_shape[2], input_shape[3]);
	// Ensure input shape is as expected
if (input_shape.size() != 4 || input_shape[1] != 3 || input_shape[2] != 224 || input_shape[3] != 224) {
  RCLCPP_ERROR(this->get_logger(), "Unexpected input shape");
  return;
}

  auto output_tensors = detection_session_->Run(Ort::RunOptions{nullptr}, input_names.data(), &input_tensor, 1, output_names.data(), output_names.size());
  // output shape print, boxes, scores, labels
  /**
* 边界框：形状通常为 [1, 100, 4]
1 表示批次
100 表示每张图像最多检测100个对象
4 表示每个边界框的坐标 (x1, y1, x2, y2)
标签：形状通常为 [1, 100]
每个检测对象的类别标签
分数：形状通常为 [1, 100]
每个检测对象的置信度分数
*/
   // print model out shape range of boxes, labels, scores
    // boxes 边界框：形状通常为 [1, 100, 4]
    Ort::Value boxes_tensor =  std::move(output_tensors[0]);
    Ort::TensorTypeAndShapeInfo boxes_info= boxes_tensor.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> boxes_shape = boxes_info.GetShape();
    RCLCPP_INFO(this->get_logger(), "DETECTION MODEL Output shape of boxes: NUM is %ld, boxDim is%ld", boxes_shape[0], boxes_shape[1]);
    int detection_num = boxes_shape[0];
    RCLCPP_INFO(this->get_logger(), "DETECT NUM: %d", detection_num);

    // labels 标签：形状通常为 [1, 100]
    Ort::Value labels_tensor =  std::move(output_tensors[1]);
    Ort::TensorTypeAndShapeInfo labels_info= labels_tensor.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> labels_shape = labels_info.GetShape();
    RCLCPP_INFO(this->get_logger(), "DETECTION MODEL Output shape of labels: (%ld,)", labels_shape[0]);

    // scores 分数：形状通常为 [1, 100]
    Ort::Value scores_tensor =  std::move(output_tensors[2]);
    Ort::TensorTypeAndShapeInfo scores_info= scores_tensor.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> scores_shape = scores_info.GetShape();
    RCLCPP_INFO(this->get_logger(), "DETECTION MODEL Output shape of scores: (%ld,)", scores_shape[0]);

  // Extract outputs
  float* boxes_data = boxes_tensor.GetTensorMutableData<float>();
  int64_t* labels_data = labels_tensor.GetTensorMutableData<int64_t>();
  float* scores_data = scores_tensor.GetTensorMutableData<float>();



  if (boxes_shape.size() < 2 || boxes_shape[0] == 0) {
  RCLCPP_ERROR(this->get_logger(), "No detections found");
  return;
    }


  // Convert raw pointers to vectors
  size_t num_boxes = boxes_shape[0];
  size_t num_detections = scores_shape[0];

  std::vector<float> boxes(boxes_data, boxes_data + num_boxes * 4);
  std::vector<int64_t> labels(labels_data, labels_data + num_detections);
  std::vector<float> scores(scores_data, scores_data + num_detections);

  // Process detection results
  std::vector<cv::Rect> detected_boxes;
  std::vector<std::string> detected_labels;
  std::vector<float> detected_scores;

  autoware_perception_msgs::msg::TrafficLightGroupArray traffic_light_group_array_msg;
  traffic_light_group_array_msg.stamp = this->now();

  autoware_perception_msgs::msg::TrafficLightGroup traffic_light_group;
  traffic_light_group.traffic_light_group_id = 0;  // Modify as needed

  float score_threshold = 0.5;
  int traffic_light_class_id = 10;  // COCO数据集中交通灯的类别ID
  // 获取原始图像的尺寸
  int original_width = cv_image.cols;
  int original_height = cv_image.rows;
  // 计算缩放因子
  float scale_x = static_cast<float>(original_width) / 224.0f;
  float scale_y = static_cast<float>(original_height) / 224.0f;

  for (size_t i = 0; i < num_detections; ++i) {
    float score = scores[i];
    int label = static_cast<int>(labels[i]);
    if (score > score_threshold && label == traffic_light_class_id) {

      // 首先将坐标除以224（模型输入大小），然后缩放到原始图像大小
      float x1 = (boxes[i * 4] / 224.0f) * original_width;
      float y1 = (boxes[i * 4 + 1] / 224.0f) * original_height;
      float x2 = (boxes[i * 4 + 2] / 224.0f) * original_width;
      float y2 = (boxes[i * 4 + 3] / 224.0f) * original_height;

      // 确保坐标在图像范围内
      x1 = std::max(0.0f, std::min(x1, static_cast<float>(original_width - 1)));
      y1 = std::max(0.0f, std::min(y1, static_cast<float>(original_height - 1)));
      x2 = std::max(0.0f, std::min(x2, static_cast<float>(original_width - 1)));
      y2 = std::max(0.0f, std::min(y2, static_cast<float>(original_height - 1)));



      // 打印缩放后的坐标以进行调试
      RCLCPP_INFO(this->get_logger(), "Scaled coordinates: x1=%.2f, y1=%.2f, x2=%.2f, y2=%.2f", x1, y1, x2, y2);


      cv::Rect box(cv::Point(static_cast<int>(x1), static_cast<int>(y1)),
                   cv::Point(static_cast<int>(x2), static_cast<int>(y2)));

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
      
      size_t class_input_tensor_size = class_resized_img.rows * class_resized_img.cols * class_resized_img.channels();
      std::vector<float> class_input_tensor_values(class_input_tensor_size);

      if (class_resized_img.isContinuous()) {
          std::memcpy(class_input_tensor_values.data(), class_resized_img.data, class_input_tensor_size * sizeof(float));
      } else {
          size_t idx = 0;
          for (int i = 0; i < class_resized_img.rows; ++i) {
              const float* row_ptr = class_resized_img.ptr<float>(i);
              std::memcpy(class_input_tensor_values.data() + idx, row_ptr, class_resized_img.cols * class_resized_img.channels() * sizeof(float));
              idx += class_resized_img.cols * class_resized_img.channels();
          }
      }

      Ort::Value class_input_tensor = Ort::Value::CreateTensor<float>(
          memory_info,
          class_input_tensor_values.data(),
          class_input_tensor_values.size(),
          class_input_dims.data(),
          class_input_dims.size());

      // Run classification model
      Ort::AllocatedStringPtr class_input_name_ptr = classification_session_->GetInputNameAllocated(0, allocator);
      std::vector<const char*> class_input_names = {class_input_name_ptr.get()};

      Ort::AllocatedStringPtr class_output_name_ptr = classification_session_->GetOutputNameAllocated(0, allocator);
      std::vector<const char*> class_output_names = {class_output_name_ptr.get()};

      // log the model input shape
      Ort::TypeInfo class_type_info = classification_session_->GetInputTypeInfo(0);
        auto class_tensor_info = class_type_info.GetTensorTypeAndShapeInfo();
        std::vector<int64_t> class_input_shape = class_tensor_info.GetShape();
        RCLCPP_INFO(this->get_logger(), "classification MODEL , Class input shape: %ld %ld %ld %ld", class_input_shape[0], class_input_shape[1], class_input_shape[2], class_input_shape[3]);

      auto class_output_tensors = classification_session_->Run(Ort::RunOptions{nullptr}, class_input_names.data(), &class_input_tensor, 1, class_output_names.data(), class_output_names.size());
		// print the output shape of classification model  from its out tensor
        // like class_output_tensors.shape?
        Ort::TypeInfo class_type_info_out = classification_session_->GetOutputTypeInfo(0);
        auto class_tensor_info_out = class_type_info_out.GetTensorTypeAndShapeInfo();
        std::vector<int64_t> class_output_shape = class_tensor_info_out.GetShape();
        RCLCPP_INFO(this->get_logger(), "classification MODEL , Class output shape: %ld %ld", class_output_shape[0], class_output_shape[1]);


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
    cv::Mat result_image = cv_image.clone(); // Use original image for drawing
    result_image = drawResult(result_image, detected_boxes, detected_labels); // Draw detections
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
    // 边界框已经是原始图像大小，不需要再次缩放
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
