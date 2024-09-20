import torch
import torch.nn as nn
from torchvision import models
import onnx
import onnxruntime

class TrafficLightClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
    
    def forward(self, x):
        return self.model(x)

def convert_to_onnx(model_path, onnx_path, num_classes):
    # 加载PyTorch模型
    model = TrafficLightClassifier(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    # 创建一个示例输入
    dummy_input = torch.randn(1, 3, 224, 224)

    # 导出ONNX模型
    torch.onnx.export(model, 
                      dummy_input, 
                      onnx_path,
                      export_params=True,
                      opset_version=11,
                      do_constant_folding=True,
                      input_names=['input'],
                      output_names=['output'],
                      dynamic_axes={'input': {0: 'batch_size'},
                                    'output': {0: 'batch_size'}})

    print(f"Model has been converted to ONNX and saved at {onnx_path}")

    # 验证ONNX模型
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model is valid")

    # 使用ONNX Runtime进行推理测试
    ort_session = onnxruntime.InferenceSession(onnx_path)
    ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
    ort_outputs = ort_session.run(None, ort_inputs)
    print("ONNX Runtime inference test passed")

if __name__ == "__main__":
    model_path = "best_model.pth"  # 您的PyTorch模型路径
    onnx_path = "traffic_light_classifier.onnx"  # 输出的ONNX模型路径
    num_classes = 6  # 根据您的模型调整这个值

    convert_to_onnx(model_path, onnx_path, num_classes)
