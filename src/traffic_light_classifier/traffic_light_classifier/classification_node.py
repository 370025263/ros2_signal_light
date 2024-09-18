import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
import cv2
import os
import numpy as np
from traffic_light_msg.msg import TrafficLightMsg
from ament_index_python.packages import get_package_share_directory
import time

class TrafficLightClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

class ClassificationNode(Node):
    def __init__(self):
        super().__init__('classification_node')
        self.subscription = self.create_subscription(
            Image,
            '/perception/traffic_light_recognition/traffic_light_image',
            self.image_callback,
            10)
        self.debug_subscription = self.create_subscription(
            TrafficLightMsg,
            '/perception/traffic_light_recognition/traffic_light_data',
            self.debug_callback,
            10)
        self.cv_bridge = CvBridge()

        # 初始化模型
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_to_idx = {"warning": 0, "stop": 1, "stopLeft": 2, "go": 3, "goLeft": 4, "warningLeft": 5}
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        num_classes = len(self.class_to_idx)
        self.model = TrafficLightClassifier(num_classes).to(self.device)
        package_share_dir = get_package_share_directory('traffic_light_classifier')
        model_path = os.path.join(package_share_dir, 'models', 'best_model.pth')
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.total_predictions = 0
        self.correct_predictions = 0
        self.total_processing_time = 0
        self.total_processed_images = 0  # New counter for timing

        # 发布分类结果
        self.result_publisher = self.create_publisher(
            String, '/perception/traffic_light_recognition/traffic_light_class', 10)
        self.image_result_publisher = self.create_publisher(
            Image, 'traffic_light_result_image', 10)

        # 添加参数来控制是否绘制结果
        self.declare_parameter('plot_result', True)

    def image_callback(self, msg):
        start_time = time.time()
        cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        self.process_image(cv_image, start_time=start_time)

    def debug_callback(self, msg):
        start_time = time.time()
        cv_image = self.cv_bridge.imgmsg_to_cv2(msg.image, desired_encoding="rgb8")
        true_label = msg.label
        self.process_image(cv_image, true_label, start_time=start_time)

    def process_image(self, cv_image, true_label=None, start_time=None):
        self.get_logger().info(f'Input image shape: {cv_image.shape}')

        # 预处理图像，保持原始尺寸
        image_tensor = self.transform(cv_image).unsqueeze(0).to(self.device)
        self.get_logger().info(f'Transformed tensor shape: {image_tensor.shape}')

        # 进行预测
        with torch.no_grad():
            outputs = self.model(image_tensor)
            _, predicted = torch.max(outputs, 1)

        predicted_label = self.idx_to_class[predicted.item()]

        # 更新统计信息（如果有真实标签）
        if true_label is not None:
            self.total_predictions += 1
            if predicted_label == true_label:
                self.correct_predictions += 1
            accuracy = self.correct_predictions / self.total_predictions
            self.get_logger().info(
                f'Predicted: {predicted_label}, True: {true_label}, Accuracy: {accuracy:.2f}')
        else:
            self.get_logger().info(f'Predicted: {predicted_label}')

        # 发布分类结果
        result_msg = String()
        result_msg.data = f'Predicted: {predicted_label}'
        if true_label:
            result_msg.data += f', True: {true_label}'
        self.result_publisher.publish(result_msg)

        # 如果启用了绘图，则绘制结果并发布
        if self.get_parameter('plot_result').value:
            result_image = self.draw_result(cv_image, predicted_label)
            result_img_msg = self.cv_bridge.cv2_to_imgmsg(result_image, encoding="rgb8")
            self.image_result_publisher.publish(result_img_msg)

        # 计算并记录处理时间
        if start_time:
            end_time = time.time()
            processing_time = end_time - start_time
            self.total_processing_time += processing_time
            self.total_processed_images += 1  # Increment processed images count
            avg_processing_time = self.total_processing_time / self.total_processed_images
            self.get_logger().info(f'Processing time: {processing_time:.4f} seconds')
            self.get_logger().info(f'Average processing time: {avg_processing_time:.4f} seconds')
            self.get_logger().info(f'Total processed images: {self.total_processed_images}')

    def draw_result(self, image, label):
        # 设置标签区域的高度和边距
        label_height = 40
        margin = 10

        # 获取输入图像的尺寸
        height, width = image.shape[:2]

        # 创建结果图像，保持原始图像尺寸，只在顶部添加标签区域
        result_height = height + label_height + margin
        result_image = np.zeros((result_height, width, 3), dtype=np.uint8)

        # 在顶部绘制白色背景用于标签
        result_image[:label_height + margin, :] = (255, 255, 255)

        # 将原始图像复制到结果图像中
        result_image[label_height + margin:, :] = image

        # 绘制预测结果
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5  # 减小字体大小以适应可能的窄图像
        font_thickness = 1
        text = f'Pred: {label}'
        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]

        # 如果文本宽度超过图像宽度，进一步缩小字体
        if text_size[0] > width:
            font_scale *= width / text_size[0]
            text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]

        text_x = (width - text_size[0]) // 2
        text_y = (label_height + margin) // 2 + text_size[1] // 2
        cv2.putText(result_image, text, (text_x, text_y), font,
                    font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)

        self.get_logger().info(
            f'Original image shape: {image.shape}, Result image shape: {result_image.shape}')
        return result_image

def main(args=None):
    rclpy.init(args=args)
    classification_node = ClassificationNode()
    rclpy.spin(classification_node)
    classification_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
