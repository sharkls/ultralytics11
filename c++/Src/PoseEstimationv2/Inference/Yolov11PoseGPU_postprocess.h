#pragma once

#include <cuda_runtime.h>
#include <vector>

// 检测结果结构体
struct DetectionResult {
    float x1, y1, x2, y2;  // 边界框坐标
    float confidence;      // 置信度
    int class_id;          // 类别ID
    float keypoints[51];   // 17个关键点 * 3 (x, y, conf)
};

// GPU端后处理函数声明
extern "C" int processOutputGPU(
    const float* gpu_output,
    DetectionResult* gpu_detections,
    int* gpu_valid_count,
    int batch_size,
    int feature_dim,
    int num_anchors,
    int num_classes,
    int num_keys,
    float conf_threshold,
    float iou_threshold,
    float* gpu_preprocess_params,
    int max_detections,
    cudaStream_t stream
);

// GPU后处理管理类
class GPUPostProcessor {
public:
    GPUPostProcessor();
    ~GPUPostProcessor();
    
    // 初始化GPU后处理
    bool initialize(int max_batch_size, int max_detections);
    
    // 执行GPU后处理
    std::vector<std::vector<float>> processOutput(
        const float* gpu_output,
        int batch_size,
        int feature_dim,
        int num_anchors,
        int num_classes,
        int num_keys,
        float conf_threshold,
        float iou_threshold,
        const std::vector<float>& preprocess_params,
        cudaStream_t stream
    );
    
    // 清理资源
    void cleanup();

private:
    DetectionResult* gpu_detections_;
    int* gpu_valid_count_;
    float* gpu_preprocess_params_;
    int max_detections_;
    int max_batch_size_;
    bool initialized_;
}; 