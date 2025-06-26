/*******************************************************
 文件名：CObjectDetectionAlg.h
 作者：shark
 描述：目标检测算法主类，负责协调各个子模块的运行
 版本：v1.0
 日期：2025-05-16
 *******************************************************/

#pragma once

#include <memory>
#include <vector>
#include <string>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <algorithm>
#include "log.h"
#include <google/protobuf/text_format.h>    // 解析prototext格式文本
#include <opencv2/opencv.hpp>
#include "ExportObjectDetectionAlgLib.h"
#include "IBaseModule.h"
#include "AlgorithmConfig.h"
#include "ModuleFactory.h"
#include "ObjectDetection_conf.pb.h"
#include "AlgorithmConfig_conf.pb.h"
#include "CMultiModalSrcData.h"
#include "CAlgResult.h"
#include "GlobalContext.h"
#include "FunctionHub.h"

class ObjectDetectionConfig : public AlgorithmConfig {
public:
    bool loadFromFile(const std::string& path) override;
    const google::protobuf::Message* getConfigMessage() const override { return &m_config; }
    objectdetection::ObjectDetectionConfig& getObjectDetectionConfig() { return m_config; }
private:
    objectdetection::ObjectDetectionConfig m_config;
};

class CObjectDetectionAlg : public IObjectDetectionAlg {
public:
    CObjectDetectionAlg(const std::string& exePath);
    ~CObjectDetectionAlg() override;

    // 实现IObjectDetectionAlg接口
    bool initAlgorithm(CSelfAlgParam* p_pAlgParam, const AlgCallback& alg_cb, void* hd) override;
    void runAlgorithm(void* p_pSrcData) override;

private:
    // 加载配置文件
    bool loadConfig(const std::string& configPath);
    
    // 创建并初始化模块
    bool initModules();
    
    // 执行模块链
    bool executeModuleChain();

    void visualizationResult();

    // 保存目标框区域图像到 vecVideoSrcData
    void saveObjectRegionImages();

private:
    std::string m_exePath;                                    // 工程路径
    std::shared_ptr<ObjectDetectionConfig> m_pConfig;         // 配置对象
    CMultiModalSrcData* m_currentInput;                       // 当前输入数据
    void* m_callbackHandle;                                   // 回调函数句柄
    AlgCallback m_algCallback;                                // 算法回调函数
    std::vector<std::shared_ptr<IBaseModule>> m_moduleChain;  // 模块执行链
    CAlgResult m_currentOutput;                               // 当前输出数据
    bool m_run_status;                                        // 运行状态
}; 