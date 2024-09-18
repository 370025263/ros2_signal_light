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

  annotations_dir_ = this->get_parameter("annotations_dir").as_string();
  data_dir_ = this->get_parameter("data_dir").as_string();
  
  RCLCPP_INFO(this->get_logger(), "Annotations directory: %s", annotations_dir_.c_str());
  RCLCPP_INFO(this->get_logger(), "Data directory: %s", data_dir_.c_str());

  loadDataset();

  RCLCPP_INFO(this->get_logger(), "Dataset size: %zu", annotations_.size());
  RCLCPP_INFO(this->get_logger(), "Classes: %s", getClassesString().c_str());
}

void CamNode::loadDataset()
{
  namespace fs = std::filesystem;
  
  RCLCPP_INFO(this->get_logger(), "Starting to load dataset...");

  for (const auto & entry : fs::recursive_directory_iterator(annotations_dir_)) {
    if (fs::is_directory(entry)) {
      std::string csv_path = entry.path().string() + "/frameAnnotationsBOX.csv";
      if (fs::exists(csv_path)) {
        std::ifstream file(csv_path);
        std::string line;
        
        // Skip header
        std::getline(file, line);
        
        while (std::getline(file, line)) {
          std::istringstream iss(line);
          std::string token;
          std::vector<std::string> tokens;
          
          while (std::getline(iss, token, ';')) {
            tokens.push_back(token);
          }
          
          if (tokens.size() >= 7) {
            std::string filename = tokens[0]; // 'Filename' column
            std::string origin_file = tokens[1]; // 'Origin file' column
            std::string annotation_tag = tokens[2]; // 'Annotation tag' column
            int ulx = std::stoi(tokens[3]); // 'Upper left corner X'
            int uly = std::stoi(tokens[4]); // 'Upper left corner Y'
            int lrx = std::stoi(tokens[5]); // 'Lower right corner X'
            int lry = std::stoi(tokens[6]); // 'Lower right corner Y'

            std::string img_name = fs::path(filename).filename().string();
            std::string clip_name = fs::path(origin_file).parent_path().filename().string();

            std::string img_path = data_dir_ + "/" + clip_name + "/frames/" + img_name;

            if (fs::exists(img_path)) {
              Annotation ann;
              ann.img_path = img_path;
              ann.label = annotation_tag;
              ann.bbox = { ulx, uly, lrx, lry };
              annotations_.push_back(ann);
              if (class_to_idx_.find(ann.label) == class_to_idx_.end()) {
                class_to_idx_[ann.label] = class_to_idx_.size();
              }
            } else {
              RCLCPP_WARN(this->get_logger(), "Image file does not exist: %s", img_path.c_str());
            }
          }
        }
      }
    }
  }
  
  if (!annotations_.empty()) {
    dis_ = std::uniform_int_distribution<>(0, annotations_.size() - 1);
  } else {
    RCLCPP_ERROR(this->get_logger(), "No annotations loaded!");
  }
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

  cv::Mat cropped_img = cropImage(image, ann.bbox);
  
  sensor_msgs::msg::Image::SharedPtr img_msg = 
    cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", cropped_img).toImageMsg();
  
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

cv::Mat CamNode::cropImage(const cv::Mat& image, const std::vector<int>& bbox)
{
  int x1 = bbox[0], y1 = bbox[1], x2 = bbox[2], y2 = bbox[3];
  int h = y2 - y1, w = x2 - x1;
  int center_x = (x1 + x2) / 2, center_y = (y1 + y2) / 2;

  double expand_ratio = 1.5;
  int new_h = static_cast<int>(h * expand_ratio);
  int new_w = static_cast<int>(w * expand_ratio);
  int new_x1 = std::max(0, center_x - new_w / 2);
  int new_y1 = std::max(0, center_y - new_h / 2);
  int new_x2 = std::min(image.cols, center_x + new_w / 2);
  int new_y2 = std::min(image.rows, center_y + new_h / 2);

  return image(cv::Rect(new_x1, new_y1, new_x2 - new_x1, new_y2 - new_y1));
}

std::string CamNode::getClassesString() const
{
  std::string result = "[";
  for (const auto& pair : class_to_idx_) {
    result += pair.first + ", ";
  }
  if (!result.empty()) {
    result.pop_back();
    result.pop_back();
  }
  result += "]";
  return result;
}

}  // namespace traffic_light_classifier

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(traffic_light_classifier::CamNode)

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<traffic_light_classifier::CamNode>());
  rclcpp::shutdown();
  return 0;
}
