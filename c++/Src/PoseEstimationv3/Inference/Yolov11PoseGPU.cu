#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/remove.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <vector>
#include <algorithm>
#include <thrust/device_ptr.h>
#include <cub/cub.cuh>
#include <cmath>

// 检测结果结构体 - 针对A6000优化内存对齐
struct __align__(32) DetectionResult {
    float x1, y1, x2, y2;  // 边界框坐标
    float confidence;      // 置信度
    int class_id;          // 类别ID
    float keypoints[51];   // 17个关键点 * 3 (x, y, conf)
    // 添加填充以确保32字节对齐
    float padding[3];
};

// 简单的比较函数，用于Thrust排序
struct DetectionResultCompare {
    __host__ __device__ bool operator()(const DetectionResult& a, const DetectionResult& b) const {
        return a.confidence > b.confidence;
    }
};

// GPU端置信度过滤和坐标转换内核 - 针对A6000优化
__global__ void filterDetectionsKernel(
    const float* output,           // TensorRT输出数据
    DetectionResult* detections,   // 过滤后的检测结果
    int* valid_count,              // 有效检测数量
    int batch_size,                // 批次大小
    int feature_dim,               // 特征维度
    int num_anchors,               // anchor数量
    int num_classes,               // 类别数量
    int num_keys,                  // 关键点数量
    float conf_threshold,          // 置信度阈值
    float* preprocess_params,      // 预处理参数 [ratio, padTop, padLeft, originalWidth, originalHeight] * batch_size
    int max_detections             // 最大检测数量
) {
    // 计算线程索引
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = batch_size * num_anchors;
    
    // 严格的边界检查
    if (idx >= total_threads) return;
    
    // 计算batch和anchor索引
    int batch_idx = idx / num_anchors;
    int anchor_idx = idx % num_anchors;
    
    // 额外的边界检查
    if (batch_idx >= batch_size || anchor_idx >= num_anchors) return;
    
    // 检查输入指针有效性
    if (output == nullptr || detections == nullptr || valid_count == nullptr || preprocess_params == nullptr) {
        return;
    }
    
    // 获取预处理参数 - 针对A6000优化内存访问
    int param_offset = batch_idx * 5;
    if (param_offset + 4 >= batch_size * 5) return;
    
    // 使用共享内存缓存预处理参数（如果可能）
    float ratio, padTop, padLeft, originalWidth, originalHeight;
    
    // 安全的参数读取
    ratio = preprocess_params[param_offset + 0];
    padTop = preprocess_params[param_offset + 1];
    padLeft = preprocess_params[param_offset + 2];
    originalWidth = preprocess_params[param_offset + 3];
    originalHeight = preprocess_params[param_offset + 4];
    
    // 检查参数有效性
    if (ratio <= 0.0f || originalWidth <= 0.0f || originalHeight <= 0.0f) return;
    
    // 计算数据起始位置
    int batch_start = batch_idx * feature_dim * num_anchors;
    
    // 严格的边界检查
    int total_elements = batch_size * feature_dim * num_anchors;
    if (batch_start + 4 * num_anchors + anchor_idx >= total_elements) return;
    
    // 安全的边界框坐标读取
    float x, y, w, h;
    x = output[batch_start + 0 * num_anchors + anchor_idx];
    y = output[batch_start + 1 * num_anchors + anchor_idx];
    w = output[batch_start + 2 * num_anchors + anchor_idx];
    h = output[batch_start + 3 * num_anchors + anchor_idx];
    
    // 严格的数值检查
    if (isnan(x) || isnan(y) || isnan(w) || isnan(h) ||
        isinf(x) || isinf(y) || isinf(w) || isinf(h)) {
        return;
    }
    
    // 获取类别置信度
    float max_conf = 0.0f;
    int max_class = 0;
    
    for (int c = 0; c < num_classes; ++c) {
        int conf_idx = batch_start + (4 + c) * num_anchors + anchor_idx;
        if (conf_idx >= total_elements) break;
        
        float conf = output[conf_idx];
        if (isnan(conf) || isinf(conf)) continue;
        
        if (conf > max_conf) {
            max_conf = conf;
            max_class = c;
        }
    }
    
    // 置信度过滤
    if (max_conf < conf_threshold) return;
    
    // 检查边界框坐标的有效性
    if (w <= 0.0f || h <= 0.0f) return;
    
    // 坐标转换 (xywh -> xyxy) - 针对A6000优化计算
    float x1 = ((x - w * 0.5f) - padLeft) / ratio;
    float y1 = ((y - h * 0.5f) - padTop) / ratio;
    float x2 = ((x + w * 0.5f) - padLeft) / ratio;
    float y2 = ((y + h * 0.5f) - padTop) / ratio;
    
    // 坐标裁剪
    x1 = fmaxf(0.0f, fminf(x1, originalWidth));
    y1 = fmaxf(0.0f, fminf(y1, originalHeight));
    x2 = fmaxf(0.0f, fminf(x2, originalWidth));
    y2 = fmaxf(0.0f, fminf(y2, originalHeight));
    
    // 检查边界框有效性
    float box_width = x2 - x1;
    float box_height = y2 - y1;
    if (box_width < 1.0f || box_height < 1.0f || 
        box_width > originalWidth || box_height > originalHeight) {
        return;
    }
    
    // 原子操作获取有效检测索引 - 针对A6000优化
    int detection_idx = atomicAdd(valid_count, 1);
    if (detection_idx >= max_detections) return;
    
    // 安全的检测结果保存
    DetectionResult& detection = detections[detection_idx];
    detection.x1 = x1;
    detection.y1 = y1;
    detection.x2 = x2;
    detection.y2 = y2;
    detection.confidence = max_conf;
    detection.class_id = max_class;
    
    // 初始化关键点为0
    for (int k = 0; k < 51; ++k) {
        detection.keypoints[k] = 0.0f;
    }
    
    // 处理关键点 - 针对A6000优化
    for (int k = 0; k < num_keys && k < 17; ++k) {  // 限制最大关键点数量
        int kpt_x_idx = batch_start + (4 + num_classes + k * 3) * num_anchors + anchor_idx;
        int kpt_y_idx = batch_start + (4 + num_classes + k * 3 + 1) * num_anchors + anchor_idx;
        int kpt_conf_idx = batch_start + (4 + num_classes + k * 3 + 2) * num_anchors + anchor_idx;
        
        // 边界检查
        if (kpt_conf_idx >= total_elements) break;
        
        float kpt_x = output[kpt_x_idx];
        float kpt_y = output[kpt_y_idx];
        float kpt_conf = output[kpt_conf_idx];
        
        // 检查关键点坐标有效性
        if (isnan(kpt_x) || isnan(kpt_y) || isnan(kpt_conf) ||
            isinf(kpt_x) || isinf(kpt_y) || isinf(kpt_conf)) {
            detection.keypoints[k * 3 + 0] = 0.0f;
            detection.keypoints[k * 3 + 1] = 0.0f;
            detection.keypoints[k * 3 + 2] = 0.0f;
            continue;
        }
        
        // 坐标转换
        kpt_x = (kpt_x - padLeft) / ratio;
        kpt_y = (kpt_y - padTop) / ratio;
        
        // 关键点置信度过滤
        if (kpt_conf < 0.1f) {
            detection.keypoints[k * 3 + 0] = 0.0f;
            detection.keypoints[k * 3 + 1] = 0.0f;
            detection.keypoints[k * 3 + 2] = 0.0f;
        } else {
            detection.keypoints[k * 3 + 0] = kpt_x;
            detection.keypoints[k * 3 + 1] = kpt_y;
            detection.keypoints[k * 3 + 2] = kpt_conf;
        }
    }
    
    // 初始化填充字段
    detection.padding[0] = 0.0f;
    detection.padding[1] = 0.0f;
    detection.padding[2] = 0.0f;
}

// GPU端NMS内核
__global__ void nmsKernel(
    const DetectionResult* detections,
    int* keep_indices,
    int* keep_count,
    int num_detections,
    float iou_threshold
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= num_detections) return;
    
    // 标记是否保留当前检测
    bool keep = true;
    
    // 检查当前检测框是否与之前保留的检测框重叠
    for (int i = 0; i < idx; ++i) {
        // 检查索引i是否已被保留
        bool i_kept = false;
        for (int j = 0; j < *keep_count; ++j) {
            if (keep_indices[j] == i) {
                i_kept = true;
                break;
            }
        }
        
        if (!i_kept) continue;  // 索引i未被保留，跳过
        
        const DetectionResult& current = detections[idx];
        const DetectionResult& previous = detections[i];
        
        // 计算IOU
        float x1 = max(current.x1, previous.x1);
        float y1 = max(current.y1, previous.y1);
        float x2 = min(current.x2, previous.x2);
        float y2 = min(current.y2, previous.y2);
        
        if (x2 <= x1 || y2 <= y1) continue;  // 无重叠
        
        float intersection = (x2 - x1) * (y2 - y1);
        float area1 = (current.x2 - current.x1) * (current.y2 - current.y1);
        float area2 = (previous.x2 - previous.x1) * (previous.y2 - previous.y1);
        float union_area = area1 + area2 - intersection;
        
        if (union_area <= 0) continue;  // 避免除零错误
        
        float iou = intersection / union_area;
        
        if (iou > iou_threshold) {
            keep = false;
            break;
        }
    }
    
    if (keep) {
        int keep_idx = atomicAdd(keep_count, 1);
        keep_indices[keep_idx] = idx;
    }
}

// GPU端后处理主函数
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
) {
    // 检查输入参数的有效性
    if (!gpu_output || !gpu_detections || !gpu_valid_count || !gpu_preprocess_params) {
        printf("Invalid GPU pointers in processOutputGPU\n");
        return 0;
    }
    
    // 检查参数合理性
    if (batch_size <= 0 || feature_dim <= 0 || num_anchors <= 0 || 
        num_classes <= 0 || num_keys <= 0 || max_detections <= 0) {
        printf("Invalid parameters in processOutputGPU\n");
        return 0;
    }
    
    // 检查CUDA错误状态
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        printf("Previous CUDA error detected: %s\n", cudaGetErrorString(status));
    }
    
    // 重置有效检测计数 - 使用同步版本避免流同步问题
    status = cudaMemset(gpu_valid_count, 0, sizeof(int));
    if (status != cudaSuccess) {
        printf("Failed to reset valid count: %s\n", cudaGetErrorString(status));
        return 0;
    }
    
    // 重置检测结果内存 - 使用同步版本
    status = cudaMemset(gpu_detections, 0, max_detections * sizeof(DetectionResult));
    if (status != cudaSuccess) {
        printf("Failed to reset detections memory: %s\n", cudaGetErrorString(status));
        return 0;
    }
    
    // 计算线程块配置 - 针对A6000优化
    int total_threads = batch_size * num_anchors;
    
    // A6000优化：使用更小的block_size以提高稳定性
    int block_size = 128;  // 从256减少到128
    int grid_size = (total_threads + block_size - 1) / block_size;
    
    // 限制grid大小，避免过度并行
    if (grid_size > 65535) {
        grid_size = 65535;
        block_size = (total_threads + grid_size - 1) / grid_size;
        // 确保block_size不超过A6000的限制
        if (block_size > 1024) {
            block_size = 1024;
            grid_size = (total_threads + block_size - 1) / block_size;
        }
    }
    
    // 针对A6000的额外限制
    if (block_size > 1024) block_size = 1024;
    if (grid_size > 65535) grid_size = 65535;
    
    printf("GPU processing: batch_size=%d, feature_dim=%d, num_anchors=%d, grid_size=%d, block_size=%d\n",
           batch_size, feature_dim, num_anchors, grid_size, block_size);
    
    // 执行置信度过滤和坐标转换 - 针对A6000优化
    filterDetectionsKernel<<<grid_size, block_size, 0, stream>>>(
        gpu_output,
        gpu_detections,
        gpu_valid_count,
        batch_size,
        feature_dim,
        num_anchors,
        num_classes,
        num_keys,
        conf_threshold,
        gpu_preprocess_params,
        max_detections
    );
    
    // 检查CUDA错误 - 针对A6000增强错误处理
    status = cudaGetLastError();
    if (status != cudaSuccess) {
        printf("CUDA error in filterDetectionsKernel: %s\n", cudaGetErrorString(status));
        // 尝试重置GPU状态
        cudaDeviceReset();
        return 0;
    }
    
    // 同步流，确保内核执行完成 - 针对A6000优化
    status = cudaStreamSynchronize(stream);
    if (status != cudaSuccess) {
        printf("Failed to synchronize stream: %s\n", cudaGetErrorString(status));
        // 尝试重置GPU状态
        cudaDeviceReset();
        return 0;
    }
    
    // 获取有效检测数量
    int valid_count;
    status = cudaMemcpy(&valid_count, gpu_valid_count, sizeof(int), cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) {
        printf("Failed to copy valid count: %s\n", cudaGetErrorString(status));
        return 0;
    }
    
    if (valid_count == 0) {
        printf("No valid detections found after confidence filtering\n");
        return 0;
    }
    
    printf("Found %d valid detections after confidence filtering\n", valid_count);
    
    // 限制检测数量
    if (valid_count > max_detections) {
        valid_count = max_detections;
        printf("Limited detections to %d (max_detections)\n", valid_count);
    }
    
    // 验证检测结果的有效性
    std::vector<DetectionResult> temp_detections(valid_count);
    status = cudaMemcpy(temp_detections.data(), gpu_detections,
                       valid_count * sizeof(DetectionResult),
                       cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) {
        printf("Failed to copy detections for validation: %s\n", cudaGetErrorString(status));
        return 0;
    }
    
    // 验证检测结果
    bool valid_detections = true;
    for (int i = 0; i < valid_count; ++i) {
        const DetectionResult& det = temp_detections[i];
        if (std::isnan(det.confidence) || std::isinf(det.confidence) ||
            std::isnan(det.x1) || std::isnan(det.y1) || std::isnan(det.x2) || std::isnan(det.y2)) {
            printf("Invalid detection result at index %d\n", i);
            valid_detections = false;
            break;
        }
    }
    
    if (!valid_detections) {
        printf("Invalid detection results found, skipping sort and NMS\n");
        return 0;
    }
    
    // 按置信度排序 - 使用简单的比较函数而不是lambda
    try {
        thrust::device_ptr<DetectionResult> detections_ptr(gpu_detections);
        thrust::sort(thrust::cuda::par.on(stream), detections_ptr, detections_ptr + valid_count, DetectionResultCompare());
        
        // 同步流，确保排序完成
        status = cudaStreamSynchronize(stream);
        if (status != cudaSuccess) {
            printf("Failed to synchronize stream after sort: %s\n", cudaGetErrorString(status));
            return valid_count;  // 返回排序前的结果，不进行NMS
        }
    } catch (const thrust::system_error& e) {
        printf("Thrust sort error: %s\n", e.what());
        return valid_count;  // 返回排序前的结果，不进行NMS
    }
    
    // 检查排序后的CUDA错误
    status = cudaGetLastError();
    if (status != cudaSuccess) {
        printf("CUDA error in thrust::sort: %s\n", cudaGetErrorString(status));
        return valid_count;  // 返回排序后的结果，不进行NMS
    }
    
    printf("Sorting completed successfully\n");
    
    // 下载检测结果到CPU进行NMS
    std::vector<DetectionResult> cpu_detections(valid_count);
    status = cudaMemcpy(cpu_detections.data(), gpu_detections,
                       valid_count * sizeof(DetectionResult),
                       cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) {
        printf("Failed to copy detections for NMS: %s\n", cudaGetErrorString(status));
        return valid_count;  // 返回排序后的结果，不进行NMS
    }
    
    // 在CPU上执行NMS
    std::vector<int> keep_indices;
    std::vector<bool> suppressed(valid_count, false);
    
    for (int i = 0; i < valid_count; ++i) {
        if (suppressed[i]) continue;
        
        keep_indices.push_back(i);
        
        for (int j = i + 1; j < valid_count; ++j) {
            if (suppressed[j]) continue;
            
            const DetectionResult& current = cpu_detections[i];
            const DetectionResult& candidate = cpu_detections[j];
            
            // 计算IOU
            float x1 = max(current.x1, candidate.x1);
            float y1 = max(current.y1, candidate.y1);
            float x2 = min(current.x2, candidate.x2);
            float y2 = min(current.y2, candidate.y2);
            
            if (x2 <= x1 || y2 <= y1) continue;  // 无重叠
            
            float intersection = (x2 - x1) * (y2 - y1);
            float area1 = (current.x2 - current.x1) * (current.y2 - current.y1);
            float area2 = (candidate.x2 - candidate.x1) * (candidate.y2 - candidate.y1);
            float union_area = area1 + area2 - intersection;
            
            if (union_area <= 0) continue;  // 避免除零错误
            
            float iou = intersection / union_area;
            
            if (iou > iou_threshold) {
                suppressed[j] = true;
            }
        }
    }
    
    // printf("NMS completed, kept %d detections out of %d\n", keep_indices.size(), valid_count);
    
    // 将NMS结果复制回GPU
    if (!keep_indices.empty()) {
        std::vector<DetectionResult> nms_detections;
        nms_detections.reserve(keep_indices.size());
        
        for (int idx : keep_indices) {
            nms_detections.push_back(cpu_detections[idx]);
        }
        
        status = cudaMemcpy(gpu_detections, nms_detections.data(),
                           nms_detections.size() * sizeof(DetectionResult),
                           cudaMemcpyHostToDevice);
        if (status != cudaSuccess) {
            printf("Failed to copy NMS results back to GPU: %s\n", cudaGetErrorString(status));
            return valid_count;  // 返回排序后的结果
        }
        
        return nms_detections.size();
    }
    
    return 0;
} 