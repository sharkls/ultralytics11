#include "Yolov11PoseGPU_postprocess.h"
#include <cuda_runtime.h>
#include "log.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>

GPUPostProcessor::GPUPostProcessor()
    : initialized_(false)
    , max_batch_size_(0)
    , max_detections_(0)
    , gpu_detections_(nullptr)
    , gpu_valid_count_(nullptr)
    , gpu_error_recovery_(false)
    , consecutive_gpu_errors_(0)
{
}

GPUPostProcessor::~GPUPostProcessor() {
    if (initialized_) {
        LOG(INFO) << "GPUPostProcessor cleanup";
        
        if (gpu_detections_) {
            cudaFree(gpu_detections_);
            gpu_detections_ = nullptr;
        }
        
        if (gpu_valid_count_) {
            cudaFree(gpu_valid_count_);
            gpu_valid_count_ = nullptr;
        }
        
        initialized_ = false;
    }
}

bool GPUPostProcessor::initialize(int max_batch_size, int max_detections) {
    if (initialized_) {
        LOG(WARNING) << "GPUPostProcessor already initialized";
        return true;
    }
    
    // 检查CUDA设备
    int device_count;
    cudaError_t status = cudaGetDeviceCount(&device_count);
    if (status != cudaSuccess) {
        LOG(ERROR) << "Failed to get CUDA device count: " << cudaGetErrorString(status);
        return false;
    }
    
    if (device_count == 0) {
        LOG(ERROR) << "No CUDA devices available";
        return false;
    }
    
    // 获取当前设备信息
    cudaDeviceProp prop;
    status = cudaGetDeviceProperties(&prop, 0);
    if (status != cudaSuccess) {
        LOG(ERROR) << "Failed to get device properties: " << cudaGetErrorString(status);
        return false;
    }
    
    LOG(INFO) << "Using CUDA device: " << prop.name;
    LOG(INFO) << "Compute capability: " << prop.major << "." << prop.minor;
    LOG(INFO) << "Total memory: " << (prop.totalGlobalMem / (1024 * 1024)) << " MB";
    
    // 设置参数
    max_batch_size_ = max_batch_size;
    max_detections_ = max_detections;
    
    // 尝试分配GPU内存 - 针对A6000优化
    cudaError_t alloc_status = cudaMalloc(&gpu_detections_, max_detections * sizeof(DetectionResult));
    if (alloc_status != cudaSuccess) {
        LOG(WARNING) << "Failed to allocate GPU memory for detections: " << cudaGetErrorString(alloc_status);
        LOG(WARNING) << "GPU post-processing will use CPU fallback";
        gpu_detections_ = nullptr;
    } else {
        LOG(INFO) << "Successfully allocated GPU memory for detections";
        
        // 针对A6000：初始化内存为0
        cudaMemset(gpu_detections_, 0, max_detections * sizeof(DetectionResult));
    }
    
    alloc_status = cudaMalloc(&gpu_valid_count_, sizeof(int));
    if (alloc_status != cudaSuccess) {
        LOG(WARNING) << "Failed to allocate GPU memory for valid count: " << cudaGetErrorString(alloc_status);
        if (gpu_detections_) {
            cudaFree(gpu_detections_);
            gpu_detections_ = nullptr;
        }
        gpu_valid_count_ = nullptr;
    } else {
        LOG(INFO) << "Successfully allocated GPU memory for valid count";
        
        // 针对A6000：初始化计数器为0
        cudaMemset(gpu_valid_count_, 0, sizeof(int));
    }
    
    initialized_ = true;
    LOG(INFO) << "GPUPostProcessor initialized successfully, max_batch_size=" << max_batch_size 
              << ", max_detections=" << max_detections;
    
    return true;
}

std::vector<std::vector<std::vector<float>>> GPUPostProcessor::processOutput(
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
        return std::vector<std::vector<std::vector<float>>>();
    }
    
    if (batch_size > max_batch_size_) {
        LOG(ERROR) << "Batch size " << batch_size << " exceeds maximum " << max_batch_size_;
        return std::vector<std::vector<std::vector<float>>>();
    }
    
    // 检查预处理参数数量
    if (preprocess_params.size() != batch_size * 5) {
        LOG(ERROR) << "Invalid preprocess params size: " << preprocess_params.size() 
                   << ", expected: " << (batch_size * 5);
        return std::vector<std::vector<std::vector<float>>>();
    }
    
    // 检查GPU状态
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        LOG(WARNING) << "GPU had previous error, clearing: " << cudaGetErrorString(status);
    }
    
    // 检查GPU设备状态
    int device_id;
    status = cudaGetDevice(&device_id);
    if (status != cudaSuccess) {
        LOG(ERROR) << "Failed to get current device: " << cudaGetErrorString(status);
        LOG(INFO) << "Using CPU fallback due to GPU device error";
        
        // 尝试重置GPU状态
        cudaError_t reset_status = cudaDeviceReset();
        if (reset_status != cudaSuccess) {
            LOG(ERROR) << "Failed to reset GPU device: " << cudaGetErrorString(reset_status);
        }
        
        // 使用CPU后处理
        int total_output_size = batch_size * feature_dim * num_anchors;
        std::vector<float> cpu_output(total_output_size);
        
        // 尝试从GPU拷贝数据，如果失败则返回空结果
        status = cudaMemcpy(cpu_output.data(), gpu_output, 
                           total_output_size * sizeof(float),
                           cudaMemcpyDeviceToHost);
        if (status != cudaSuccess) {
            LOG(ERROR) << "Failed to copy GPU output to CPU: " << cudaGetErrorString(status);
            return std::vector<std::vector<std::vector<float>>>();
        }
        
        return processOutputCPU(cpu_output.data(), batch_size, feature_dim, num_anchors, 
                               num_classes, num_keys, conf_threshold, iou_threshold, 
                               preprocess_params);
    }
    
    // 检查GPU内存是否可用
    if (!gpu_detections_ || !gpu_valid_count_) {
        LOG(WARNING) << "GPU memory not available, using CPU fallback";
        
        // 下载GPU输出到CPU
        int total_output_size = batch_size * feature_dim * num_anchors;
        std::vector<float> cpu_output(total_output_size);
        
        status = cudaMemcpy(cpu_output.data(), gpu_output, 
                           total_output_size * sizeof(float),
                           cudaMemcpyDeviceToHost);
        if (status != cudaSuccess) {
            LOG(ERROR) << "Failed to copy GPU output to CPU: " << cudaGetErrorString(status);
            return std::vector<std::vector<std::vector<float>>>();
        }
        
        // 使用CPU后处理
        return processOutputCPU(cpu_output.data(), batch_size, feature_dim, num_anchors, 
                               num_classes, num_keys, conf_threshold, iou_threshold, 
                               preprocess_params);
    }
    
    // 针对A6000：优先使用CPU后处理以确保稳定性
    LOG(INFO) << "Using CPU post-processing for A6000 compatibility";
    
    // 下载GPU输出到CPU
    int total_output_size = batch_size * feature_dim * num_anchors;
    std::vector<float> cpu_output(total_output_size);
    
    status = cudaMemcpy(cpu_output.data(), gpu_output, 
                       total_output_size * sizeof(float),
                       cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) {
        LOG(ERROR) << "Failed to copy GPU output to CPU: " << cudaGetErrorString(status);
        return std::vector<std::vector<std::vector<float>>>();
    }
    
    // 使用CPU后处理
    return processOutputCPU(cpu_output.data(), batch_size, feature_dim, num_anchors, 
                           num_classes, num_keys, conf_threshold, iou_threshold, 
                           preprocess_params);
}

void GPUPostProcessor::cleanup() {
    if (initialized_) {
        LOG(INFO) << "GPUPostProcessor cleanup";
        
        if (gpu_detections_) {
            cudaFree(gpu_detections_);
            gpu_detections_ = nullptr;
        }
        
        if (gpu_valid_count_) {
            cudaFree(gpu_valid_count_);
            gpu_valid_count_ = nullptr;
        }
        
        initialized_ = false;
    }
}

// CPU后处理辅助函数
std::vector<std::vector<std::vector<float>>> GPUPostProcessor::processOutputCPU(
    const float* cpu_output,
    int batch_size,
    int feature_dim,
    int num_anchors,
    int num_classes,
    int num_keys,
    float conf_threshold,
    float iou_threshold,
    const std::vector<float>& preprocess_params
) {
    // 为每个batch单独处理
    std::vector<std::vector<std::vector<float>>> batch_results;
    batch_results.reserve(batch_size);
    
    for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        // 计算当前batch的输出偏移
        int batch_output_offset = batch_idx * feature_dim * num_anchors;
        const float* batch_output = cpu_output + batch_output_offset;
        
        // 计算当前batch的预处理参数偏移
        int param_offset = batch_idx * 5;
        float ratio = preprocess_params[param_offset + 0];
        float padTop = preprocess_params[param_offset + 1];
        float padLeft = preprocess_params[param_offset + 2];
        float originalWidth = preprocess_params[param_offset + 3];
        float originalHeight = preprocess_params[param_offset + 4];
        
        // CPU端后处理
        std::vector<std::vector<float>> batch_result;
        
        // 遍历所有anchor
        for (int anchor_idx = 0; anchor_idx < num_anchors; ++anchor_idx) {
            // 边界检查
            if (anchor_idx >= num_anchors) {
                LOG(ERROR) << "Anchor index out of bounds: " << anchor_idx << " >= " << num_anchors;
                break;
            }
            
            // 获取边界框坐标
            int bbox_offset = 0 * num_anchors + anchor_idx;
            if (bbox_offset >= feature_dim * num_anchors) {
                LOG(ERROR) << "Bbox offset out of bounds: " << bbox_offset;
                continue;
            }
            
            float x = batch_output[0 * num_anchors + anchor_idx];
            float y = batch_output[1 * num_anchors + anchor_idx];
            float w = batch_output[2 * num_anchors + anchor_idx];
            float h = batch_output[3 * num_anchors + anchor_idx];
            
            // 获取类别置信度
            float max_conf = 0.0f;
            int max_class = 0;
            for (int c = 0; c < num_classes; ++c) {
                int conf_offset = (4 + c) * num_anchors + anchor_idx;
                if (conf_offset >= feature_dim * num_anchors) {
                    LOG(ERROR) << "Confidence offset out of bounds: " << conf_offset;
                    continue;
                }
                float conf = batch_output[conf_offset];
                if (conf > max_conf) {
                    max_conf = conf;
                    max_class = c;
                }
            }
            
            // 置信度过滤
            if (max_conf < conf_threshold) continue;
            
            // 检查边界框坐标的有效性
            if (w <= 0 || h <= 0) continue;
            
            // 坐标转换 (xywh -> xyxy)
            float x1 = ((x - w / 2) - padLeft) / ratio;
            float y1 = ((y - h / 2) - padTop) / ratio;
            float x2 = ((x + w / 2) - padLeft) / ratio;
            float y2 = ((y + h / 2) - padTop) / ratio;
            
            // 坐标裁剪
            x1 = std::max(0.0f, std::min(x1, originalWidth));
            y1 = std::max(0.0f, std::min(y1, originalHeight));
            x2 = std::max(0.0f, std::min(x2, originalWidth));
            y2 = std::max(0.0f, std::min(y2, originalHeight));
            
            // 检查边界框有效性
            float box_width = x2 - x1;
            float box_height = y2 - y1;
            if (box_width < 1 || box_height < 1 || 
                box_width > originalWidth || box_height > originalHeight) {
                continue;
            }
            
            // 创建检测结果
            std::vector<float> result;
            result.reserve(4 + 1 + 1 + num_keys * 3);  // bbox + confidence + class_id + keypoints
            
            // 添加边界框坐标
            result.push_back(x1);
            result.push_back(y1);
            result.push_back(x2);
            result.push_back(y2);
            
            // 添加置信度
            result.push_back(max_conf);
            
            // 添加类别ID
            result.push_back(static_cast<float>(max_class));
            
            // 处理关键点
            for (int k = 0; k < num_keys; ++k) {
                int kpt_x_offset = (4 + num_classes + k * 3) * num_anchors + anchor_idx;
                int kpt_y_offset = (4 + num_classes + k * 3 + 1) * num_anchors + anchor_idx;
                int kpt_conf_offset = (4 + num_classes + k * 3 + 2) * num_anchors + anchor_idx;
                
                // 边界检查
                if (kpt_x_offset >= feature_dim * num_anchors || 
                    kpt_y_offset >= feature_dim * num_anchors || 
                    kpt_conf_offset >= feature_dim * num_anchors) {
                    LOG(ERROR) << "Keypoint offset out of bounds for k=" << k;
                    result.push_back(0.0f);  // x
                    result.push_back(0.0f);  // y
                    result.push_back(0.0f);  // conf
                    continue;
                }
                
                float kpt_x = batch_output[kpt_x_offset];
                float kpt_y = batch_output[kpt_y_offset];
                float kpt_conf = batch_output[kpt_conf_offset];
                
                // 坐标转换
                kpt_x = (kpt_x - padLeft) / ratio;
                kpt_y = (kpt_y - padTop) / ratio;
                
                // 关键点置信度过滤
                if (kpt_conf < 0.1f) {
                    result.push_back(0.0f);  // x
                    result.push_back(0.0f);  // y
                    result.push_back(0.0f);  // conf
                } else {
                    result.push_back(kpt_x);
                    result.push_back(kpt_y);
                    result.push_back(kpt_conf);
                }
            }
            
            batch_result.push_back(std::move(result));
        }
        
        // 按置信度排序
        std::sort(batch_result.begin(), batch_result.end(),
                  [](const std::vector<float>& a, const std::vector<float>& b) {
                      return a[4] > b[4];  // 按置信度排序
                  });
        
        // 执行NMS
        std::vector<std::vector<float>> nms_result;
        std::vector<bool> suppressed(batch_result.size(), false);
        
        for (size_t i = 0; i < batch_result.size(); ++i) {
            if (suppressed[i]) continue;
            
            nms_result.push_back(batch_result[i]);
            
            // 计算与其他检测框的IOU
            for (size_t j = i + 1; j < batch_result.size(); ++j) {
                if (suppressed[j]) continue;
                
                const std::vector<float>& box1 = batch_result[i];
                const std::vector<float>& box2 = batch_result[j];
                
                // 计算IOU
                float x1 = std::max(box1[0], box2[0]);
                float y1 = std::max(box1[1], box2[1]);
                float x2 = std::min(box1[2], box2[2]);
                float y2 = std::min(box1[3], box2[3]);
                
                if (x2 <= x1 || y2 <= y1) continue;  // 无重叠
                
                float intersection = (x2 - x1) * (y2 - y1);
                float area1 = (box1[2] - box1[0]) * (box1[3] - box1[1]);
                float area2 = (box2[2] - box2[0]) * (box2[3] - box2[1]);
                float union_area = area1 + area2 - intersection;
                
                float iou = intersection / union_area;
                
                if (iou > iou_threshold) {
                    suppressed[j] = true;
                }
            }
        }
        
        // 限制检测数量
        if (nms_result.size() > max_detections_) {
            nms_result.resize(max_detections_);
        }
        
        batch_results.push_back(std::move(nms_result));
        LOG(INFO) << "Batch " << batch_idx << ": found " << batch_results.back().size() << " detections after NMS (CPU)";
    }
    
    LOG(INFO) << "CPU post-processing completed for " << batch_size << " batches";
    return batch_results;
}

// 针对A6000的GPU错误恢复函数
bool GPUPostProcessor::attemptGPURecovery() {
    LOG(WARNING) << "Attempting GPU recovery for A6000 compatibility";
    
    consecutive_gpu_errors_++;
    
    if (consecutive_gpu_errors_ >= MAX_CONSECUTIVE_ERRORS) {
        LOG(ERROR) << "Too many consecutive GPU errors, switching to CPU fallback";
        gpu_error_recovery_ = true;
        return false;
    }
    
    // 尝试重置GPU状态
    resetGPUState();
    
    // 重新初始化GPU内存
    if (gpu_detections_) {
        cudaFree(gpu_detections_);
        gpu_detections_ = nullptr;
    }
    
    if (gpu_valid_count_) {
        cudaFree(gpu_valid_count_);
        gpu_valid_count_ = nullptr;
    }
    
    // 重新分配内存
    cudaError_t status = cudaMalloc(&gpu_detections_, max_detections_ * sizeof(DetectionResult));
    if (status != cudaSuccess) {
        LOG(ERROR) << "Failed to reallocate GPU memory for detections: " << cudaGetErrorString(status);
        return false;
    }
    
    status = cudaMalloc(&gpu_valid_count_, sizeof(int));
    if (status != cudaSuccess) {
        LOG(ERROR) << "Failed to reallocate GPU memory for valid count: " << cudaGetErrorString(status);
        cudaFree(gpu_detections_);
        gpu_detections_ = nullptr;
        return false;
    }
    
    // 初始化内存
    cudaMemset(gpu_detections_, 0, max_detections_ * sizeof(DetectionResult));
    cudaMemset(gpu_valid_count_, 0, sizeof(int));
    
    LOG(INFO) << "GPU recovery successful";
    consecutive_gpu_errors_ = 0;
    return true;
}

void GPUPostProcessor::resetGPUState() {
    LOG(INFO) << "Resetting GPU state for A6000 compatibility";
    
    // 重置CUDA设备
    cudaError_t status = cudaDeviceReset();
    if (status != cudaSuccess) {
        LOG(ERROR) << "Failed to reset GPU device: " << cudaGetErrorString(status);
    } else {
        LOG(INFO) << "GPU device reset successful";
    }
    
    // 同步设备
    status = cudaDeviceSynchronize();
    if (status != cudaSuccess) {
        LOG(ERROR) << "Failed to synchronize GPU device: " << cudaGetErrorString(status);
    }
    
    // 清除错误状态
    cudaGetLastError();
}