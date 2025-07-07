/*******************************************************
 文件名：Yolov11ClassifyGPU.h
 作者：
 描述：GPU加速的YOLOv11图像分类推理模块
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
#include "CObjectClassifyAlg.h" 

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

class Yolov11ClassifyGPU : public IBaseModule {
public:
    Yolov11ClassifyGPU(const std::string& exe_path);
    ~Yolov11ClassifyGPU() override;

    std::string getModuleName() const override { return "Yolov11ClassifyGPU"; }
    ModuleType getModuleType() const override { return ModuleType::INFERENCE; }
    bool init(void* p_pAlgParam) override;
    void execute() override;
    void setInput(void* input) override;
    void* getOutput() override;

    void cleanup();
    void initTensorRT();
    std::vector<float> inference();
    std::vector<std::vector<float>> process_output_classification_only(const std::vector<float>& output);
    std::string get_class_name(int class_id) const;

private:
    // GPU加速相关函数
    bool initCUDA();
    void cleanupCUDA();

    posetimation::YOLOModelConfig m_poseConfig;       // 配置参数
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
    
    // CUDA相关成员
    cublasHandle_t m_cublasHandle;
    bool m_cudaInitialized;          // CUDA初始化状态
    
    // 多batch处理相关
    int m_maxBatchSize;                               // 最大批处理大小

    // 分类参数
    float conf_thres_;          // 置信度阈值
    int num_classes_;           // 类别数量
    int channels_;              // 通道数量
    int target_h_;              // 目标图像高度
    int target_w_;              // 目标图像宽度
    std::string engine_path_;   // 引擎路径
}; 