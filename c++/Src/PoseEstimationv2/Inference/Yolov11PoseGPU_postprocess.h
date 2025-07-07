#pragma once

#include <vector>
#include <memory>
#include "log.h"
#include <cuda_runtime.h>

// 检测结果结构体 - 针对A6000优化内存对齐
struct __align__(32) DetectionResult {
    float x1, y1, x2, y2;  // 边界框坐标
    float confidence;      // 置信度
    int class_id;          // 类别ID
    float keypoints[51];   // 17个关键点 * 3 (x, y, conf)
    // 添加填充以确保32字节对齐
    float padding[3];
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

class GPUPostProcessor {
public:
    GPUPostProcessor();
    ~GPUPostProcessor();
    
    bool initialize(int max_batch_size, int max_detections);
    void cleanup();
    
    std::vector<std::vector<std::vector<float>>> processOutput(
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

private:
    bool initialized_;
    int max_batch_size_;
    int max_detections_;
    
    // GPU内存指针
    DetectionResult* gpu_detections_;
    int* gpu_valid_count_;
    
    // 针对A6000的GPU错误恢复机制
    bool gpu_error_recovery_;
    int consecutive_gpu_errors_;
    static const int MAX_CONSECUTIVE_ERRORS = 3;
    
    // GPU错误恢复函数
    bool attemptGPURecovery();
    void resetGPUState();
    
public:
    // CPU后处理辅助函数
    std::vector<std::vector<std::vector<float>>> processOutputCPU(
        const float* cpu_output,
        int batch_size,
        int feature_dim,
        int num_anchors,
        int num_classes,
        int num_keys,
        float conf_threshold,
        float iou_threshold,
        const std::vector<float>& preprocess_params
    );

private:
}; 