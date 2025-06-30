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

// 检测结果结构体
struct DetectionResult {
    float x1, y1, x2, y2;  // 边界框坐标
    float confidence;      // 置信度
    int class_id;          // 类别ID
    float keypoints[51];   // 17个关键点 * 3 (x, y, conf)
};

// 简单的比较函数，用于Thrust排序
struct DetectionResultCompare {
    __host__ __device__ bool operator()(const DetectionResult& a, const DetectionResult& b) const {
        return a.confidence > b.confidence;
    }
};

// GPU端置信度过滤和坐标转换内核
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
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = batch_size * num_anchors;
    
    if (idx >= total_threads) return;
    
    int batch_idx = idx / num_anchors;
    int anchor_idx = idx % num_anchors;
    
    // 获取预处理参数
    float ratio = preprocess_params[batch_idx * 5 + 0];
    float padTop = preprocess_params[batch_idx * 5 + 1];
    float padLeft = preprocess_params[batch_idx * 5 + 2];
    float originalWidth = preprocess_params[batch_idx * 5 + 3];
    float originalHeight = preprocess_params[batch_idx * 5 + 4];
    
    // 计算数据起始位置
    int batch_start = batch_idx * feature_dim * num_anchors;
    
    // 获取边界框坐标
    float x = output[batch_start + 0 * num_anchors + anchor_idx];
    float y = output[batch_start + 1 * num_anchors + anchor_idx];
    float w = output[batch_start + 2 * num_anchors + anchor_idx];
    float h = output[batch_start + 3 * num_anchors + anchor_idx];
    
    // 获取类别置信度
    float max_conf = 0.0f;
    int max_class = 0;
    for (int c = 0; c < num_classes; ++c) {
        float conf = output[batch_start + (4 + c) * num_anchors + anchor_idx];
        if (conf > max_conf) {
            max_conf = conf;
            max_class = c;
        }
    }
    
    // 置信度过滤
    if (max_conf < conf_threshold) return;
    
    // 坐标转换 (xywh -> xyxy)
    float x1 = ((x - w / 2) - padLeft) / ratio;
    float y1 = ((y - h / 2) - padTop) / ratio;
    float x2 = ((x + w / 2) - padLeft) / ratio;
    float y2 = ((y + h / 2) - padTop) / ratio;
    
    // 坐标裁剪
    x1 = max(0.0f, min(x1, originalWidth));
    y1 = max(0.0f, min(y1, originalHeight));
    x2 = max(0.0f, min(x2, originalWidth));
    y2 = max(0.0f, min(y2, originalHeight));
    
    // 检查边界框有效性
    float box_width = x2 - x1;
    float box_height = y2 - y1;
    if (box_width < 1 || box_height < 1 || box_width > originalWidth || box_height > originalHeight) {
        return;
    }
    
    // 原子操作获取有效检测索引
    int detection_idx = atomicAdd(valid_count, 1);
    if (detection_idx >= max_detections) return;
    
    // 保存检测结果
    DetectionResult& detection = detections[detection_idx];
    detection.x1 = x1;
    detection.y1 = y1;
    detection.x2 = x2;
    detection.y2 = y2;
    detection.confidence = max_conf;
    detection.class_id = max_class;
    
    // 处理关键点
    for (int k = 0; k < num_keys; ++k) {
        float kpt_x = output[batch_start + (4 + num_classes + k * 3) * num_anchors + anchor_idx];
        float kpt_y = output[batch_start + (4 + num_classes + k * 3 + 1) * num_anchors + anchor_idx];
        float kpt_conf = output[batch_start + (4 + num_classes + k * 3 + 2) * num_anchors + anchor_idx];
        
        // 坐标转换
        kpt_x = (kpt_x - padLeft) / ratio;
        kpt_y = (kpt_y - padTop) / ratio;
        
        if (kpt_conf < conf_threshold) {
            detection.keypoints[k * 3 + 0] = 0.0f;
            detection.keypoints[k * 3 + 1] = 0.0f;
            detection.keypoints[k * 3 + 2] = 0.0f;
        } else {
            detection.keypoints[k * 3 + 0] = kpt_x;
            detection.keypoints[k * 3 + 1] = kpt_y;
            detection.keypoints[k * 3 + 2] = kpt_conf;
        }
    }
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
    
    // 简单的NMS实现：按置信度排序后，逐个检查IOU
    // 这里使用简化的实现，实际应用中可能需要更复杂的并行NMS算法
    
    // 标记是否保留当前检测
    bool keep = true;
    
    for (int i = 0; i < idx; ++i) {
        if (keep_indices[i] == -1) continue;  // 已被抑制
        
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
    // 重置有效检测计数
    cudaMemsetAsync(gpu_valid_count, 0, sizeof(int), stream);
    
    // 计算线程块配置
    int total_threads = batch_size * num_anchors;
    int block_size = 256;
    int grid_size = (total_threads + block_size - 1) / block_size;
    
    // 执行置信度过滤和坐标转换
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
    
    // 获取有效检测数量
    int valid_count;
    cudaMemcpyAsync(&valid_count, gpu_valid_count, sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    
    if (valid_count == 0) return 0;
    
    // 按置信度排序 - 使用简单的比较函数而不是lambda
    thrust::device_ptr<DetectionResult> detections_ptr(gpu_detections);
    thrust::sort(thrust::cuda::par.on(stream), detections_ptr, detections_ptr + valid_count, DetectionResultCompare());
    
    // 执行NMS
    int* gpu_keep_indices;
    int* gpu_keep_count;
    cudaMallocAsync(&gpu_keep_indices, valid_count * sizeof(int), stream);
    cudaMallocAsync(&gpu_keep_count, sizeof(int), stream);
    
    cudaMemsetAsync(gpu_keep_indices, -1, valid_count * sizeof(int), stream);
    cudaMemsetAsync(gpu_keep_count, 0, sizeof(int), stream);
    
    block_size = 256;
    grid_size = (valid_count + block_size - 1) / block_size;
    
    nmsKernel<<<grid_size, block_size, 0, stream>>>(
        gpu_detections,
        gpu_keep_indices,
        gpu_keep_count,
        valid_count,
        iou_threshold
    );
    
    // 获取NMS后的检测数量
    int keep_count;
    cudaMemcpyAsync(&keep_count, gpu_keep_count, sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    
    // 清理GPU内存
    cudaFreeAsync(gpu_keep_indices, stream);
    cudaFreeAsync(gpu_keep_count, stream);
    
    return keep_count;
} 