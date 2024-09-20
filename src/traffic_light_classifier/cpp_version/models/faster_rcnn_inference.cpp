#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

int main() {
    // 初始化ONNX Runtime
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "faster-rcnn-inference");
    Ort::SessionOptions session_options;
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    // 加载模型
    const char* model_path = "./fasterrcnn_resnet50_fpn.onnx";
    Ort::Session session(env, model_path, session_options);

    // 读取并预处理图像
    cv::Mat image = cv::imread("./d.jpg");
    cv::Mat resized_image;
    cv::resize(image, resized_image, cv::Size(224, 224)); // 调整大小为800x800
    cv::Mat float_image;
    resized_image.convertTo(float_image, CV_32F, 1.0 / 255.0);
    cv::cvtColor(float_image, float_image, cv::COLOR_BGR2RGB);

    // 准备输入tensor
    std::vector<float> input_tensor_values(float_image.total() * float_image.channels());
    memcpy(input_tensor_values.data(), float_image.data, input_tensor_values.size() * sizeof(float));
    std::vector<int64_t> input_shape = {1, 3, 224, 224};

    // 设置输入输出节点名
    const char* input_names[] = {"input"};
    const char* output_names[] = {"boxes", "labels", "scores"};

    // 创建输入tensor
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_values.size(), input_shape.data(), input_shape.size());

    // 运行推理
    auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 3);

    // 处理输出
    float* boxes = output_tensors[0].GetTensorMutableData<float>();
    int64_t* labels = output_tensors[1].GetTensorMutableData<int64_t>();
    float* scores = output_tensors[2].GetTensorMutableData<float>();

    int num_detections = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape()[1];

    // 计算缩放因子
    float scale_x = static_cast<float>(image.cols) / 224.0f;
    float scale_y = static_cast<float>(image.rows) / 224.0f;


// 检查输出张量的形状
for (size_t i = 0; i < output_tensors.size(); i++) {
    auto shape = output_tensors[i].GetTensorTypeAndShapeInfo().GetShape();
    std::cout << "Output " << i << " shape: ";
    for (auto dim : shape) {
        std::cout << dim << " ";
    }
    std::cout << std::endl;
}


std::cout << "Number of detections: " << num_detections << std::endl;

if (scores == nullptr) {
    std::cout << "Scores array is null" << std::endl;
    return 1;
}

// 打印检测结果
std::cout << "检测到的对象:" << std::endl;
for (int i = 0; i < std::min(num_detections, 10); i++) {
    if (scores[i] > 0.5) { // 只打印置信度大于0.5的结果
        std::cout << "对象 " << i + 1 << ":" << std::endl;
        std::cout << "  标签: " << labels[i] << std::endl;
        std::cout << "  置信度: " << scores[i] << std::endl;
        std::cout << "  边界框: [" 
                  << boxes[i*4] * scale_x << ", " 
                  << boxes[i*4+1] * scale_y << ", " 
                  << boxes[i*4+2] * scale_x << ", " 
                  << boxes[i*4+3] * scale_y << "]" << std::endl;
    }
}
    return 0;
}
