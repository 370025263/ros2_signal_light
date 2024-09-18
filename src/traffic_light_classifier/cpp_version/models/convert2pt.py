import torch
import torchvision.models as models

# 假设您的模型是基于 ResNet18 的。如果不是，请相应地修改这部分。
class TrafficLightClassifier(torch.nn.Module):
    def __init__(self, num_classes):
        super(TrafficLightClassifier, self).__init__()
        self.model = models.resnet18(pretrained=False)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型
num_classes = 6  # 根据您的实际类别数进行修改
model = TrafficLightClassifier(num_classes)
model.load_state_dict(torch.load('best_model.pth', map_location=device))
model.eval()

# 创建一个示例输入
example_input = torch.rand(1, 3, 224, 224)

# 转换为 TorchScript 模型
traced_script_module = torch.jit.trace(model, example_input)

# 保存 TorchScript 模型
traced_script_module.save("best_model.pt")

print("模型已成功转换为 TorchScript 格式并保存为 best_model.pt")
