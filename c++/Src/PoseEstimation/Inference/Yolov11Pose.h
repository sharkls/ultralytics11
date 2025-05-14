/*******************************************************
 文件名：Yolov11Pose.h
 作者：
 描述：Yolov11姿态估计推理模块
 版本：v1.0
 日期：2025-05-13
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

#include "IBaseModule.h"
#include "ModuleFactory.h"
#include "FunctionHub.h"
#include "CMultiModalSrcData.h"
#include "PoseEstimation_conf.pb.h"

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

class Yolov11Pose : public IBaseModule {
public:
    Yolov11Pose(const std::string& exe_path);
    ~Yolov11Pose() override;

    std::string getModuleName() const override { return "Yolov11Pose"; }
    ModuleType getModuleType() const override { return ModuleType::INFERENCE; }
    bool init(void* p_pAlgParam) override;
    void execute() override;
    void setInput(void* input) override;
    void* getOutput() override;

    void cleanup();
    void initTensorRT();
    std::vector<float> inference();
    void rescale_coords(std::vector<float>& coords, bool is_keypoint);
    std::vector<std::vector<float>> process_output(const std::vector<float>& output);
    std::vector<std::vector<float>> process_keypoints(const std::vector<float>& output, const std::vector<std::vector<float>>& boxes);
    std::vector<int> nms(const std::vector<std::vector<float>>& boxes, const std::vector<float>& scores);
    CAlgResult formatConverted(std::vector<std::vector<float>> results);

private:
    posetimation::YOLOModelConfig m_poseConfig;       // 配置参数
    std::vector<float> m_inputImage;    // 输入图像数据
    CAlgResult m_outputResult;          // 输出结果

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
};