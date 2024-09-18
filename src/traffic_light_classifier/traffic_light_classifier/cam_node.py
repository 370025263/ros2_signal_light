import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import random
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image as PILImage
from traffic_light_msg.msg import TrafficLightMsg

class TrafficLightDataset(Dataset):
    def __init__(self, annotations_dir, data_dir, transform=None):
        self.annotations = []
        self.transform = transform
        self.data_dir = data_dir
        
        for clip_folder in os.listdir(annotations_dir):
            csv_path = os.path.join(annotations_dir, clip_folder, 'frameAnnotationsBOX.csv')
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path, sep=';')
                for _, row in df.iterrows():
                    img_name = row['Filename'].split('/')[-1]
                    clip_name = row['Origin file'].split('/')[-2]
                    img_path = os.path.join(data_dir, clip_name, 'frames', img_name)
                    if os.path.exists(img_path):
                        self.annotations.append({
                            'img_path': img_path,
                            'label': row['Annotation tag'],
                            'bbox': [
                                row['Upper left corner X'],
                                row['Upper left corner Y'],
                                row['Lower right corner X'],
                                row['Lower right corner Y']
                            ]
                        })
        
        self.class_to_idx = {cls: idx for idx, cls in enumerate(set([ann['label'] for ann in self.annotations]))}
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        ann = self.annotations[idx]
        image = PILImage.open(ann['img_path']).convert('RGB')
        
        bbox = ann['bbox']
        label = ann['label']
        
        x1, y1, x2, y2 = bbox
        h, w = y2 - y1, x2 - x1
        center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
        
        expand_ratio = 1.5
        new_h, new_w = int(h * expand_ratio), int(w * expand_ratio)
        new_x1 = max(0, int(center_x - new_w / 2))
        new_y1 = max(0, int(center_y - new_h / 2))
        new_x2 = min(image.width, int(center_x + new_w / 2))
        new_y2 = min(image.height, int(center_y + new_h / 2))
        
        cropped_img = image.crop((new_x1, new_y1, new_x2, new_y2))
        
        if self.transform:
            cropped_img = self.transform(cropped_img)
        
        return cropped_img, self.class_to_idx[label], torch.tensor(bbox), label


class CamNode(Node):
    def __init__(self):
        super().__init__('cam_node')
        self.publisher_ = self.create_publisher(Image, '/perception/traffic_light_recognition/traffic_light_image', 10)
        self.debug_publisher_ = self.create_publisher(TrafficLightMsg, '/perception/traffic_light_recognition/traffic_light_data', 10)
        self.timer = self.create_timer(1.0, self.timer_callback)
        self.cv_bridge = CvBridge()
        
        # 初始化数据集
        annotations_dir = '/home/casia/jupyter_notebooks/Annotations/Annotations/dayTrain'
        data_dir = '/home/casia/jupyter_notebooks/dayTrain/dayTrain'
        self.dataset = TrafficLightDataset(annotations_dir, data_dir, transform=transforms.ToTensor())
        
        self.get_logger().info(f"数据集大小: {len(self.dataset)}")
        self.get_logger().info(f"类别: {list(self.dataset.class_to_idx.keys())}")
        
        # 添加参数来控制是否使用调试模式
        self.declare_parameter('debug_mode', False)
        
    def timer_callback(self):
        # 随机选择一个样本
        idx = random.randint(0, len(self.dataset) - 1)
        image, class_idx, bbox, label = self.dataset[idx]
        
        # 将PyTorch张量转换为OpenCV图像
        image_np = (image.permute(1, 2, 0).numpy() * 255).astype('uint8')
        
        # 创建Image消息
        img_msg = self.cv_bridge.cv2_to_imgmsg(image_np, encoding="rgb8")
        
        # 发布Image消息
        self.publisher_.publish(img_msg)
        
        # 如果在调试模式下，发布TrafficLightMsg
        if self.get_parameter('debug_mode').value:
            debug_msg = TrafficLightMsg()
            debug_msg.image = img_msg
            debug_msg.label = label
            self.debug_publisher_.publish(debug_msg)
            self.get_logger().info(f'Published debug image with label: {label}')
        else:
            self.get_logger().info('Published image')

def main(args=None):
    rclpy.init(args=args)
    cam_node = CamNode()
    rclpy.spin(cam_node)
    cam_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
