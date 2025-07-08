/*******************************************************
 文件名：Yolov11PoseGPU.h
 作者：
 描述：GPU加速的YOLOv11姿态估计推理模块
 版本：v1.0
 日期：2025-01-20
 *******************************************************/

#pragma once
#include <memory>
#include <vector>
#include <string>
#include <fstream>
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <numeric>
#include <cstring>
#include <opencv2/opencv.hpp>
#include "log.h"
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>

#include "IBaseModule.h"
#include "ModuleFactory.h"
#include "FunctionHub.h"
#include "CMultiModalSrcData.h"
#include "PoseEstimation_conf.pb.h"
#include "CPoseEstimationAlg.h" 
#include "Yolov11PoseGPU_postprocess.h"  // 添加GPU后处理头文件

namespace nvinfer1 {
    class IRuntime;
    class ICudaEngine;
    class IExecutionContext;
}

// TensorRT日志记录器
class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kINFO)
            std::cout << "[TensorRT] " << msg << std::endl;
    }
};

extern "C" {
    typedef struct CUstream_st* cudaStream_t;
}

class Yolov11PoseGPU : public IBaseModule {
public:
    Yolov11PoseGPU(const std::string& exe_path);
    ~Yolov11PoseGPU() override;

    std::string getModuleName() const override { return "Yolov11PoseGPU"; }
    ModuleType getModuleType() const override { return ModuleType::INFERENCE; }
    bool init(void* p_pAlgParam) override;
    void execute() override;
    void setInput(void* input) override;
    void* getOutput() override;

    void cleanup();
    void initTensorRT();
    std::vector<float> inference();
    void prepareBatchData(size_t batch_start, size_t batch_end);
    void prepareBatchDataGPU(size_t batch_start, size_t batch_end);
    void rescale_coords(std::vector<float>& coords, bool is_keypoint);
    std::vector<std::vector<float>> process_output(const std::vector<float>& output);
    std::vector<std::vector<float>> process_keypoints(const std::vector<float>& output, const std::vector<std::vector<float>>& boxes);
    std::vector<int> nms(const std::vector<std::vector<float>>& boxes, const std::vector<float>& scores);
    CAlgResult formatConverted(std::vector<std::vector<float>> results);
    std::vector<std::vector<CObjectResult>> formatConvertedByImage(std::vector<std::vector<float>> results, size_t batch_size);
    std::string classify_pose(const std::vector<float>& keypoints) const;

private:
    // GPU加速相关函数
    bool initCUDA();
    void cleanupCUDA();
    bool allocateGPUMemory(size_t max_batch_size);
    void freeGPUMemory();
    
    // GPU后处理加速
    void launchNMSKernel(const std::vector<std::vector<float>>& boxes, 
                        const std::vector<float>& scores,
                        std::vector<int>& keep_indices);
    void launchCoordinateTransformKernel(std::vector<float>& coords, 
                                        bool is_keypoint,
                                        float ratio, int dw, int dh);

    posetimation::YOLOModelConfig m_poseConfig;       // 配置参数
    MultiImagePreprocessResult m_inputData;           // 输入数据（多图像预处理结果，CPU版本）
    MultiImagePreprocessResultGPU m_inputDataGPU;     // 输入数据（多图像预处理结果，GPU版本）
    CAlgResult m_outputResult;                        // 输出结果

    // TensorRT相关成员
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;
    Logger logger_;
    std::vector<void*> input_buffers_;
    std::vector<void*> output_buffers_;
    cudaStream_t stream_ = nullptr;
    const char* input_name_;
    const char* output_name_;
    nvinfer1::Dims input_dims_;
    nvinfer1::Dims output_dims_;
    size_t input_size_ = 1;
    size_t output_size_ = 1;
    
    // CUDA相关成员
    cublasHandle_t m_cublasHandle;
    void* m_gpuTempBuffer;           // GPU临时缓冲区
    void* m_gpuNMSBuffer;            // GPU NMS缓冲区
    size_t m_maxGPUBufferSize;       // 最大GPU缓冲区大小
    bool m_cudaInitialized;          // CUDA初始化状态
    
    // 多batch处理相关
    std::vector<std::vector<float>> m_batchInputs;    // 批处理输入数据
    std::vector<std::pair<int, int>> m_batchSizes;    // 批处理中每个图像的尺寸
    int m_maxBatchSize;                               // 最大批处理大小

    // 后处理参数
    int new_unpad_h_;           // 填充后图像高度
    int new_unpad_w_;           // 填充后图像宽度
    int dw_;                    // 宽度填充量
    int dh_;                    // 高度填充量
    float ratio_ = 1;           // 缩放比例
    std::vector<int> stride_;   // 步长
    int batch_size_ = 1;        // 批量大小
    float conf_thres_;          // 置信度阈值
    float iou_thres_;           // 交并比阈值
    int num_classes_;           // 类别数量
    int channels_;              // 通道数量
    int num_keys_;              // 关键点数量
    int max_dets_ = 300;        // 最大检测目标阈值
    int src_width_;             // 原始图像宽度
    int src_height_;            // 原始图像高度
    int target_h_;              // 目标图像高度
    int target_w_;              // 目标图像宽度
    int num_anchors_ = 0;       // 锚框数量
    std::string engine_path_;   // 引擎路径

    // 运行状态
    bool status_;
    
    // GPU后处理加速
    std::unique_ptr<GPUPostProcessor> gpu_postprocessor_;  // GPU后处理器
    bool use_gpu_postprocessing_;                          // 是否使用GPU后处理

    // 姿态特征结构体
    struct PoseFeatures {
        float trunk_angle;           // 躯干角度 (度)
        float leg_angle;             // 腿部角度 (度)
        float head_angle;            // 头部角度 (度)
        float body_height_ratio;     // 身体高宽比
        float shoulder_hip_distance; // 肩臀距离
        float knee_ankle_distance;   // 膝踝距离
        float trunk_length;          // 躯干长度
        float leg_length;            // 腿部长度
        float arm_angle;             // 手臂角度
        float stability_score;       // 稳定性评分
        float symmetry_score;        // 对称性评分
    };
    
    // 姿态权重结构体
    struct PoseWeights {
        float trunk_angle_weight;
        float leg_angle_weight;
        float head_angle_weight;
        float body_ratio_weight;
        float stability_weight;
        float symmetry_weight;
    };
    
    // 数学化姿态分类函数
    PoseFeatures calculate_pose_features(const std::vector<std::pair<float, float>>& points,
                                        const std::vector<float>& confidences) const;
    std::string classify_pose_mathematical(const PoseFeatures& features) const;
    float calculate_pose_match_score(const PoseFeatures& features, 
                                   const PoseWeights& weights,
                                   const std::vector<float>& ideal_values,
                                   const std::vector<float>& tolerances) const;
    
    // 高级几何计算函数
    float calculate_trunk_angle_advanced(const std::vector<std::pair<float, float>>& points,
                                        const std::vector<float>& confidences,
                                        int left_shoulder, int right_shoulder,
                                        int left_hip, int right_hip) const;
    float calculate_joint_angle(const std::vector<std::pair<float, float>>& points,
                               const std::vector<float>& confidences,
                               int joint1, int joint2, int joint3) const;
    float calculate_head_angle_advanced(const std::vector<std::pair<float, float>>& points,
                                       const std::vector<float>& confidences,
                                       int nose, int left_shoulder, int right_shoulder) const;
    float calculate_body_geometry(const std::vector<std::pair<float, float>>& points,
                                 const std::vector<float>& confidences,
                                 int nose, int left_ankle, int right_ankle) const;
    float calculate_euclidean_distance(const std::vector<std::pair<float, float>>& points,
                                      const std::vector<float>& confidences,
                                      int left1, int right1, int left2, int right2) const;
    float calculate_segment_length(const std::vector<std::pair<float, float>>& points,
                                  const std::vector<float>& confidences,
                                  int left1, int right1, int left2, int right2) const;
    float calculate_stability_score(const std::vector<std::pair<float, float>>& points,
                                   const std::vector<float>& confidences) const;
    float calculate_symmetry_score(const std::vector<std::pair<float, float>>& points,
                                  const std::vector<float>& confidences) const;
    
    // 保留原有函数以保持兼容性
    float calculate_trunk_angle(const std::vector<std::pair<float, float>>& points, 
                               const std::vector<float>& confidences,
                               int left_shoulder, int right_shoulder, 
                               int left_hip, int right_hip) const;
    float calculate_leg_angle(const std::vector<std::pair<float, float>>& points,
                             const std::vector<float>& confidences,
                             int hip, int knee, int ankle) const;
    float calculate_head_angle(const std::vector<std::pair<float, float>>& points,
                              const std::vector<float>& confidences,
                              int nose, int left_shoulder, int right_shoulder) const;
    float calculate_body_height_ratio(const std::vector<std::pair<float, float>>& points,
                                     const std::vector<float>& confidences,
                                     int nose, int left_ankle, int right_ankle) const;
    float calculate_distance(const std::vector<std::pair<float, float>>& points,
                            const std::vector<float>& confidences,
                            int left1, int right1, int left2, int right2) const;
    
    // 姿态判断函数 (保留原有实现)
    bool is_standing_walking(const std::vector<std::pair<float, float>>& points,
                            const std::vector<float>& confidences,
                            float trunk_angle, float leg_angle, float head_angle,
                            float body_height_ratio) const;
    bool is_hunched(const std::vector<std::pair<float, float>>& points,
                   const std::vector<float>& confidences,
                   float trunk_angle, float head_angle, float body_height_ratio) const;
    bool is_lying(const std::vector<std::pair<float, float>>& points,
                 const std::vector<float>& confidences,
                 float trunk_angle, float leg_angle, float body_height_ratio) const;
    bool is_sitting(const std::vector<std::pair<float, float>>& points,
                   const std::vector<float>& confidences,
                   float trunk_angle, float leg_angle, 
                   float shoulder_hip_distance, float knee_ankle_distance) const;
    bool is_squatting(const std::vector<std::pair<float, float>>& points,
                     const std::vector<float>& confidences,
                     float trunk_angle, float leg_angle, float body_height_ratio) const;
}; 