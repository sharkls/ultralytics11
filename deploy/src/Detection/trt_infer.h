#ifndef TRT_INFER_H
#define TRT_INFER_H

#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <memory>
#include <numeric>  // for std::iota

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
};

class TRTInference {
public:
    TRTInference(const std::string& engine_path);
    ~TRTInference();

    // 预处理函数
    void preprocess(const cv::Mat& rgb_img, const cv::Mat& ir_img, 
                   const std::vector<float>& homography,
                   std::vector<float>& rgb_input,
                   std::vector<float>& ir_input,
                   std::vector<float>& h_input,
                   LetterBoxInfo& letterbox_info);

    // 推理函数
    std::vector<float> inference(const cv::Mat& rgb_img, 
                               const cv::Mat& ir_img,
                               const std::vector<float>& homography,
                               LetterBoxInfo& letterbox_info);

    // 后处理函数
    std::vector<std::vector<float>> process_output(const std::vector<float>& output,
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
    int input_h_;
    int input_w_;
    int stride_;

    // 辅助函数
    void cleanup();
    std::vector<int> nms(const std::vector<std::vector<float>>& boxes,
                        const std::vector<float>& scores,
                        float iou_threshold);
};

#endif // TRT_INFER_H