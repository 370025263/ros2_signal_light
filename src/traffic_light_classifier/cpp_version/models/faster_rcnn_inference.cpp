#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

cv::Mat preprocess_image(const cv::Mat& image, const cv::Size& target_size = cv::Size(800, 800)) {
    // 调整图像大小
    cv::Mat resized_image;
    cv::resize(image, resized_image, target_size);

    // 转换颜色空间从BGR到RGB
    cv::Mat rgb_image;
    cv::cvtColor(resized_image, rgb_image, cv::COLOR_BGR2RGB);

    // 归一化
    cv::Mat float_image;
    rgb_image.convertTo(float_image, CV_32F, 1.0 / 255.0);

    // 转置图像以匹配 CHW 格式
    std::vector<cv::Mat> channels(3);
    cv::split(float_image, channels);

    cv::Mat preprocessed_image(1, channels[0].total() * 3, CV_32F);
    float* preprocessed_data = preprocessed_image.ptr<float>();

    for (int c = 0; c < 3; ++c) {
        memcpy(preprocessed_data + c * channels[c].total(), channels[c].data, channels[c].total() * sizeof(float));
    }

    return preprocessed_image;
}

void printNodeInfo(Ort::Session& session, bool isInput) {
    Ort::AllocatorWithDefaultOptions allocator;
    size_t num_nodes = isInput ? session.GetInputCount() : session.GetOutputCount();
    std::vector<int64_t> node_dims;

    std::cout << "Number of " << (isInput ? "inputs" : "outputs") << " = " << num_nodes << std::endl;

    for (size_t i = 0; i < num_nodes; i++) {
        auto name = isInput ? session.GetInputNameAllocated(i, allocator) : session.GetOutputNameAllocated(i, allocator);
        std::cout << (isInput ? "Input " : "Output ") << i << " : name=" << name.get() << std::endl;

        auto type_info = isInput ? session.GetInputTypeInfo(i) : session.GetOutputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

        node_dims = tensor_info.GetShape();
        std::cout << (isInput ? "Input " : "Output ") << i << " : num_dims=" << node_dims.size() << std::endl;
        for (size_t j = 0; j < node_dims.size(); j++)
            std::cout << (isInput ? "Input " : "Output ") << i << " : dim " << j << "=" << node_dims[j] << std::endl;
    }
}

int main() {
    // 初始化ONNX Runtime
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "faster-rcnn-inference");
    Ort::SessionOptions session_options;
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    // 加载模型
    const char* model_path = "./fasterrcnn_resnet50_fpn.onnx";
    Ort::Session session(env, model_path, session_options);

    // 打印输入信息
    std::cout << "=== Input Information ===" << std::endl;
    printNodeInfo(session, true);

    // 打印输出信息
    std::cout << "\n=== Output Information ===" << std::endl;
    printNodeInfo(session, false);

    // 读取并预处理图像
    cv::Mat image = cv::imread("./d.jpg");
    if (image.empty()) {
        std::cout << "Failed to read image" << std::endl;
        return 1;
    }

    cv::Mat preprocessed_image = preprocess_image(image, cv::Size(224, 224));

    // 准备输入tensor
    std::vector<float> input_tensor_values(preprocessed_image.total());
    memcpy(input_tensor_values.data(), preprocessed_image.ptr<float>(), preprocessed_image.total() * sizeof(float));

    std::vector<int64_t> input_shape = {1, 3, 224, 224};

    // 设置输入输出节点名
    const char* input_names[] = {"input"};
    const char* output_names[] = {"boxes", "labels", "scores"};

    // 创建输入tensor
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_values.size(), input_shape.data(), input_shape.size());

    // 运行推理
    auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 3);

    // 打印输出大小
    std::cout << "Output size: " << output_tensors.size() << std::endl;

    // 检查输出张量的形状
    for (size_t i = 0; i < output_tensors.size(); i++) {
        auto shape = output_tensors[i].GetTensorTypeAndShapeInfo().GetShape();
        std::cout << "Output " << i << " shape: ";
        for (auto dim : shape) {
            std::cout << dim << " ";
        }
        std::cout << std::endl;
    }

    // 处理输出
    if (!output_tensors.empty() && output_tensors[0].GetTensorTypeAndShapeInfo().GetShape()[0] > 0) {
        float* boxes = output_tensors[0].GetTensorMutableData<float>();
        int64_t* labels = output_tensors[1].GetTensorMutableData<int64_t>();
        float* scores = output_tensors[2].GetTensorMutableData<float>();

        int num_detections = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape()[0];
        std::cout << "Number of detections: " << num_detections << std::endl;

        // 计算缩放因子
        float scale_x = static_cast<float>(image.cols) / 224.0f;
        float scale_y = static_cast<float>(image.rows) / 224.0f;

        // 打印检测结果
        std::cout << "检测到的对象:" << std::endl;
        for (int i = 0; i < num_detections; i++) {
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
    } else {
        std::cout << "No detections found" << std::endl;
    }

    return 0;
}
