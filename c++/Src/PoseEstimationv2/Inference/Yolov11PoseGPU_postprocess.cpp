#include "Yolov11PoseGPU_postprocess.h"
#include "log.h"
#include <cuda_runtime.h>
#include <algorithm>

GPUPostProcessor::GPUPostProcessor() 
    : gpu_detections_(nullptr)
    , gpu_valid_count_(nullptr)
    , gpu_preprocess_params_(nullptr)
    , max_detections_(0)
    , max_batch_size_(0)
    , initialized_(false) {
}

GPUPostProcessor::~GPUPostProcessor() {
    cleanup();
}

bool GPUPostProcessor::initialize(int max_batch_size, int max_detections) {
    if (initialized_) {
        LOG(WARNING) << "GPUPostProcessor already initialized";
        return true;
    }
    
    max_batch_size_ = max_batch_size;
    max_detections_ = max_detections;
    
    // 分配GPU内存
    cudaError_t status;
    
    // 分配检测结果内存
    status = cudaMalloc(&gpu_detections_, max_detections * sizeof(DetectionResult));
    if (status != cudaSuccess) {
        LOG(ERROR) << "Failed to allocate GPU detections memory: " << cudaGetErrorString(status);
        return false;
    }
    
    // 分配有效检测计数内存
    status = cudaMalloc(&gpu_valid_count_, sizeof(int));
    if (status != cudaSuccess) {
        LOG(ERROR) << "Failed to allocate GPU valid count memory: " << cudaGetErrorString(status);
        cudaFree(gpu_detections_);
        gpu_detections_ = nullptr;
        return false;
    }
    
    // 分配预处理参数内存
    status = cudaMalloc(&gpu_preprocess_params_, max_batch_size * 5 * sizeof(float));
    if (status != cudaSuccess) {
        LOG(ERROR) << "Failed to allocate GPU preprocess params memory: " << cudaGetErrorString(status);
        cudaFree(gpu_detections_);
        cudaFree(gpu_valid_count_);
        gpu_detections_ = nullptr;
        gpu_valid_count_ = nullptr;
        return false;
    }
    
    initialized_ = true;
    LOG(INFO) << "GPUPostProcessor initialized successfully, max_batch_size=" << max_batch_size 
              << ", max_detections=" << max_detections;
    return true;
}

std::vector<std::vector<float>> GPUPostProcessor::processOutput(
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
) {
    if (!initialized_) {
        LOG(ERROR) << "GPUPostProcessor not initialized";
        return std::vector<std::vector<float>>();
    }
    
    if (batch_size > max_batch_size_) {
        LOG(ERROR) << "Batch size " << batch_size << " exceeds maximum " << max_batch_size_;
        return std::vector<std::vector<float>>();
    }
    
    // 上传预处理参数到GPU
    cudaError_t status = cudaMemcpyAsync(gpu_preprocess_params_, preprocess_params.data(),
                                        preprocess_params.size() * sizeof(float),
                                        cudaMemcpyHostToDevice, stream);
    if (status != cudaSuccess) {
        LOG(ERROR) << "Failed to copy preprocess params to GPU: " << cudaGetErrorString(status);
        return std::vector<std::vector<float>>();
    }
    
    // 执行GPU后处理
    int valid_count = processOutputGPU(
        gpu_output,
        gpu_detections_,
        gpu_valid_count_,
        batch_size,
        feature_dim,
        num_anchors,
        num_classes,
        num_keys,
        conf_threshold,
        iou_threshold,
        gpu_preprocess_params_,
        max_detections_,
        stream
    );
    
    if (valid_count == 0) {
        LOG(INFO) << "No valid detections found";
        return std::vector<std::vector<float>>();
    }
    
    // 下载检测结果到CPU
    std::vector<DetectionResult> cpu_detections(valid_count);
    status = cudaMemcpyAsync(cpu_detections.data(), gpu_detections_,
                            valid_count * sizeof(DetectionResult),
                            cudaMemcpyDeviceToHost, stream);
    if (status != cudaSuccess) {
        LOG(ERROR) << "Failed to copy detections from GPU: " << cudaGetErrorString(status);
        return std::vector<std::vector<float>>();
    }
    
    // 同步流
    cudaStreamSynchronize(stream);
    
    // 转换为标准格式
    std::vector<std::vector<float>> results;
    results.reserve(valid_count);
    
    for (int i = 0; i < valid_count; ++i) {
        const DetectionResult& detection = cpu_detections[i];
        
        std::vector<float> result;
        result.reserve(4 + 1 + 1 + num_keys * 3);  // bbox + confidence + class_id + keypoints
        
        // 添加边界框坐标
        result.push_back(detection.x1);
        result.push_back(detection.y1);
        result.push_back(detection.x2);
        result.push_back(detection.y2);
        
        // 添加置信度
        result.push_back(detection.confidence);
        
        // 添加类别ID
        result.push_back(static_cast<float>(detection.class_id));
        
        // 添加关键点
        for (int k = 0; k < num_keys * 3; ++k) {
            result.push_back(detection.keypoints[k]);
        }
        
        results.push_back(std::move(result));
    }
    
    LOG(INFO) << "GPU post-processing completed, found " << valid_count << " detections";
    return results;
}

void GPUPostProcessor::cleanup() {
    if (gpu_detections_) {
        cudaFree(gpu_detections_);
        gpu_detections_ = nullptr;
    }
    
    if (gpu_valid_count_) {
        cudaFree(gpu_valid_count_);
        gpu_valid_count_ = nullptr;
    }
    
    if (gpu_preprocess_params_) {
        cudaFree(gpu_preprocess_params_);
        gpu_preprocess_params_ = nullptr;
    }
    
    initialized_ = false;
    LOG(INFO) << "GPUPostProcessor cleanup completed";
} 