/*******************************************************
 文件名：CPoseEstimationAlg.h
 作者：sharkls
 描述：姿态估计算法主类，负责协调各个子模块的运行
 版本：v1.0
 日期：2025-05-09
 *******************************************************/

#pragma once

#include <memory>
#include <vector>
#include <string>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <cuda_runtime.h>
#include "log.h"
#include <google/protobuf/text_format.h>    // 解析prototext格式文本
#include <opencv2/opencv.hpp>
#include "ExportPoseEstimationAlgLib.h"
#include "IBaseModule.h"
#include "AlgorithmConfig.h"
#include "ModuleFactory.h"
#include "PoseEstimation_conf.pb.h"
#include "AlgorithmConfig_conf.pb.h"
#include "CMultiModalSrcData.h"
#include "CAlgResult.h"
#include "GlobalContext.h"
#include "FunctionHub.h"

// 多图像预处理结果结构体
struct MultiImagePreprocessResult {
    std::vector<std::vector<float>> images;           // 多个子图的预处理数据
    std::vector<std::pair<int, int>> imageSizes;      // 每个子图对应的尺寸 (width, height)
    
    // 新增：预处理参数信息
    struct PreprocessParams {
        float ratio;           // 缩放比例
        int padTop;           // 顶部填充
        int padLeft;          // 左侧填充
        int originalWidth;    // 原始图像宽度
        int originalHeight;   // 原始图像高度
        int targetWidth;      // 目标图像宽度
        int targetHeight;     // 目标图像高度
        int dw;               // 左右填充
        int dh;               // 上下填充
        
        PreprocessParams() : ratio(1.0f), padTop(0), padLeft(0), 
                           originalWidth(0), originalHeight(0), 
                           targetWidth(0), targetHeight(0), dw(0), dh(0) {}
    };
    std::vector<PreprocessParams> preprocessParams;  // 每个图像的预处理参数
    
    // 构造函数
    MultiImagePreprocessResult() = default;
    
    // 清空数据
    void clear() {
        images.clear();
        imageSizes.clear();
        preprocessParams.clear();
    }
    
    // 获取图像数量
    size_t size() const {
        return images.size();
    }
    
    // 检查是否为空
    bool empty() const {
        return images.empty();
    }
};

// GPU版本的多图像预处理结果结构体
struct MultiImagePreprocessResultGPU {
    void* gpu_buffer;                                 // GPU内存缓冲区
    std::vector<size_t> image_offsets;               // 每个图像在GPU缓冲区中的偏移
    std::vector<std::pair<int, int>> imageSizes;     // 每个子图对应的尺寸 (width, height)
    
    // 预处理参数信息
    struct PreprocessParams {
        float ratio;           // 缩放比例
        int padTop;           // 顶部填充
        int padLeft;          // 左侧填充
        int originalWidth;    // 原始图像宽度
        int originalHeight;   // 原始图像高度
        int targetWidth;      // 目标图像宽度
        int targetHeight;     // 目标图像高度
        int dw;               // 左右填充
        int dh;               // 上下填充
        
        PreprocessParams() : ratio(1.0f), padTop(0), padLeft(0), 
                           originalWidth(0), originalHeight(0), 
                           targetWidth(0), targetHeight(0), dw(0), dh(0) {}
    };
    std::vector<PreprocessParams> preprocessParams;  // 每个图像的预处理参数
    
    size_t total_size;        // GPU缓冲区总大小
    int max_width;           // 最大宽度
    int max_height;          // 最大高度
    int channels;            // 通道数
    int batch_size;          // 批次大小
    
    // 构造函数
    MultiImagePreprocessResultGPU() : gpu_buffer(nullptr), total_size(0), 
                                     max_width(0), max_height(0), channels(3), batch_size(0) {}
    
    // 析构函数
    ~MultiImagePreprocessResultGPU() {
        if (gpu_buffer) {
            cudaFree(gpu_buffer);
            gpu_buffer = nullptr;
        }
    }
    
    // 清空数据
    void clear() {
        if (gpu_buffer) {
            cudaFree(gpu_buffer);
            gpu_buffer = nullptr;
        }
        image_offsets.clear();
        imageSizes.clear();
        preprocessParams.clear();
        total_size = 0;
        max_width = 0;
        max_height = 0;
        batch_size = 0;
    }
    
    // 获取图像数量
    size_t size() const {
        return image_offsets.size();
    }
    
    // 检查是否为空
    bool empty() const {
        return image_offsets.empty();
    }
    
    // 获取指定图像的GPU指针
    float* getImagePtr(size_t index) const {
        if (index >= image_offsets.size() || !gpu_buffer) {
            return nullptr;
        }
        // 修复：将字节偏移转换为float指针偏移
        size_t byte_offset = image_offsets[index];
        return static_cast<float*>(gpu_buffer) + (byte_offset / sizeof(float));
    }
    
    // 获取指定图像的大小（以float为单位）
    size_t getImageSize(size_t index) const {
        if (index >= imageSizes.size()) {
            return 0;
        }
        return channels * imageSizes[index].second * imageSizes[index].first;
    }
    
    // 获取指定图像的大小（以字节为单位）
    size_t getImageSizeBytes(size_t index) const {
        if (index >= imageSizes.size()) {
            return 0;
        }
        return channels * imageSizes[index].second * imageSizes[index].first * sizeof(float);
    }
};

class PoseEstimationConfig : public AlgorithmConfig {
public:
    bool loadFromFile(const std::string& path) override;
    const google::protobuf::Message* getConfigMessage() const override { return &m_config; }
    posetimation::PoseConfig& getPoseConfig() { return m_config; }
private:
    posetimation::PoseConfig m_config;
};

class CPoseEstimationAlg : public IPoseEstimationAlg {
public:
    CPoseEstimationAlg(const std::string& exePath);
    ~CPoseEstimationAlg() override;

    // 实现IPoseEstimationAlg接口
    bool initAlgorithm(CSelfAlgParam* p_pAlgParam, const AlgCallback& alg_cb, void* hd) override;
    void runAlgorithm(void* p_pSrcData) override;

private:
    // 加载配置文件
    bool loadConfig(const std::string& configPath);
    
    // 创建并初始化模块
    bool initModules();
    
    // 执行模块链
    bool executeModuleChain();

    // 可视化检测结果
    void visualizationResult();

    // 创建合并的可视化图像
    void createCombinedVisualization(const std::vector<CVideoSrcData>& allVideoSrcData, 
                                   const std::vector<CObjectResult>& objResults, 
                                   uint32_t frameId);

    // 坐标转换和结果合并
    void convertCoordinatesAndMergeResults();

private:
    std::string m_exePath;                                    // 工程路径
    std::shared_ptr<PoseEstimationConfig> m_pConfig;          // 配置对象
    std::vector<std::shared_ptr<IBaseModule>> m_moduleChain;  // 模块执行链
    AlgCallback m_algCallback;                                // 算法回调函数
    void* m_callbackHandle;                                   // 回调函数句柄
    CAlgResult* m_currentInput;                               // 当前输入数据
    CAlgResult m_currentOutput;                               // 当前输出数据

    // 离线测试配置
    bool m_run_status{false};
}; 

