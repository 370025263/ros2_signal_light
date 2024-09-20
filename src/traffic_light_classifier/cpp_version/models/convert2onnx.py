import torch
import torchvision
from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn
import onnx
import onnxruntime

def convert_fasterrcnn_to_onnx(input_path, output_path):
    # 加载预训练模型
    model = fasterrcnn_resnet50_fpn(pretrained=False)
    model.load_state_dict(torch.load(input_path, map_location='cpu'))
    model.eval()

    # 创建虚拟输入
    dummy_input = torch.randn(1, 3, 224, 224)

    # 导出 ONNX 模型
    torch.onnx.export(model, dummy_input, output_path,
                      opset_version=11,
                      do_constant_folding=True,
                      input_names=['input'],
                      output_names=['boxes', 'labels', 'scores'],
                      dynamic_axes={'input': {0: 'batch_size'},
                                    'boxes': {0: 'batch_size'},
                                    'labels': {0: 'batch_size'},
                                    'scores': {0: 'batch_size'}})

    print(f"模型已成功转换为 ONNX 格式并保存为 {output_path}")

    # 验证 ONNX 模型
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX 模型检查通过")

    # 使用 ONNX Runtime 进行推理测试
    ort_session = onnxruntime.InferenceSession(output_path)

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    # 计算 ONNX Runtime 输出
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(dummy_input)}
    ort_outputs = ort_session.run(None, ort_inputs)

    print("ONNX Runtime 推理测试通过")

if __name__ == "__main__":
    input_path = "fasterrcnn_resnet50_fpn_coco-258fb6c6.pth"
    output_path = "fasterrcnn_resnet50_fpn.onnx"
    
    convert_fasterrcnn_to_onnx(input_path, output_path)
