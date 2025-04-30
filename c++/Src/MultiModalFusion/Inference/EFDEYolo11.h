/*******************************************************
 EFDEYolo11.h
 作者：
 描述：多模态融合推理算法接口实现，用于多模态融合推理算法的运行及结果数据处理
 版本：v1.0
 日期：2024-1-11
 *******************************************************/

#pragma once

#include <iostream>
#include <vector>
#include <fstream>
#include <memory>
#include "ICommonAlg.h"
#include "CAlgResult.h"
#include "CSelfAlgParam.h"
#include "NvInfer.h"

// TensorRT日志记录器
class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kINFO)
            std::cout << "[TensorRT] " << msg << std::endl;
    }
};

struct LetterBoxInfo {
    float dw;
    float dh;
    float ratio;
};

class EFDEYolo11 : public ICommonAlg
{
public:
    EFDEYolo11();

    EFDEYolo11(const CSelfAlgParam* p_pAlgParam);

    virtual ~EFDEYolo11();

    // 初始化多模态融合推理算法参数
    void init(CSelfAlgParam* p_pAlgParam);

    // 执行多模态融合推理算法
    void execute();

    // 设置通用数据
    void setCommonData(CCommonDataPtr p_commonData) override
    {
        m_pCommonData = p_commonData;
    }

    // 获取通用数据
    CCommonDataPtr getCommonData() override
    {
        return m_pCommonData;
    }

private:
    // 推理相关参数
    int m_nBatchSize;           // 批处理大小
    int m_nInputChannels;       // 输入通道数
    int m_nInputHeight;         // 输入高度
    int m_nInputWidth;          // 输入宽度
    std::string m_strModelPath; // 模型路径

    // 检测相关参数
    float m_fConfThres;         // 置信度阈值
    float m_fIouThres;          // IOU阈值
    int m_nNumClasses;          // 类别数量
    int m_nNumAnchors;          // anchor数量

    // CUDA相关
    cudaStream_t m_pStream;     // CUDA流
    void* m_pInputBuffer;       // 输入缓冲区
    void* m_pOutputBuffer;      // 输出缓冲区
    void* m_pInputBuffer_images0 = nullptr;    // 用于 images0
    void* m_pInputBuffer_images1 = nullptr;    // 用于 images1
    void* m_pInputBuffer_extrinsics = nullptr; // 用于 extrinsics

    // TensorRT相关
    Logger m_logger;            // TensorRT日志记录器
    std::unique_ptr<nvinfer1::IRuntime> m_pRuntime;        // TensorRT运行时
    std::unique_ptr<nvinfer1::ICudaEngine> m_pEngine;      // TensorRT引擎
    std::unique_ptr<nvinfer1::IExecutionContext> m_pContext; // TensorRT执行上下文
    std::vector<nvinfer1::Dims> m_vecInputDims;            // 输入维度
    std::vector<nvinfer1::Dims> m_vecOutputDims;           // 输出维度

    // 辅助函数
    std::vector<std::vector<float>> processOutput(
        const std::vector<float>& output,
        float confThres,
        float iouThres,
        int numClasses,
        const LetterBoxInfo& letterboxInfo);

    std::vector<int> nms(
        const std::vector<std::vector<float>>& boxes,
        const std::vector<float>& scores,
        float iouThres);
};

