#ifndef TRT_INFER_H
#define TRT_INFER_H

#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <memory>
#include <numeric>

class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kINFO)
            std::cout << "[TensorRT] " << msg << std::endl;
    }
};

struct LetterBoxInfo {
    float dw;
    float dh;
    float ratio;
    int new_unpad_h;
    int new_unpad_w;
    int stride;
};

class TRTInference {
public:
    TRTInference(const std::string& engine_path);
    ~TRTInference();

    // 预处理函数
    void preprocess(const cv::Mat& img,
                   std::vector<float>& input,
                   LetterBoxInfo& letterbox_info);

    // 推理函数
    std::vector<float> inference(const cv::Mat& img,
                               LetterBoxInfo& letterbox_info);

    // 后处理函数
    std::vector<std::vector<float>> process_output(
        const std::vector<float>& output,
        float conf_thres,
        float iou_thres,
        int num_classes,
        const LetterBoxInfo& letterbox_info);

private:
    // TensorRT相关
    Logger logger_;
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;
    cudaStream_t stream_;

    // 输入输出buffer
    std::vector<void*> input_buffers_;
    std::vector<void*> output_buffers_;
    std::vector<nvinfer1::Dims> input_dims_;
    std::vector<nvinfer1::Dims> output_dims_;

    // 图像尺寸
    int input_h_ = 640;  // 最大输入高度
    int input_w_ = 640;  // 最大输入宽度
    int stride_ = 32;    // 模型步长

    // 辅助函数
    void cleanup();
    std::vector<int> nms(const std::vector<std::vector<float>>& boxes,
                        const std::vector<float>& scores,
                        float iou_threshold);
    
    // 关键点处理函数
    std::vector<std::vector<float>> process_keypoints(
        const std::vector<float>& output,
        const std::vector<std::vector<float>>& boxes,
        const LetterBoxInfo& letterbox_info);

    // 坐标还原函数
    void rescale_coords(std::vector<float>& coords, 
                       const LetterBoxInfo& letterbox_info,
                       bool is_keypoint = false);
};

#endif // TRT_INFER_H 