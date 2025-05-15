/*******************************************************
 文件名：CObjectLocationAlg.h
 作者：sharkls
 描述：目标定位算法主类，负责协调各个子模块的运行
 版本：v1.0
 日期：2025-05-15
 *******************************************************/

#pragma once

#include <memory>
#include <vector>
#include <string>
#include <iostream>
#include <filesystem>
#include <fstream>
#include "log.h"
#include <google/protobuf/text_format.h>    // 解析prototext格式文本

#include "ExportObjectLocationAlgLib.h"
#include "IBaseModule.h"
#include "AlgorithmConfig.h"
#include "ModuleFactory.h"
#include "ObjectLocation_conf.pb.h"
#include "AlgorithmConfig_conf.pb.h"

#include "CMultiModalSrcData.h"
#include "CAlgResult.h"
#include "GlobalContext.h"
#include "FunctionHub.h"

class ObjectLocationConfig : public AlgorithmConfig {
public:
    bool loadFromFile(const std::string& path) override;
    const google::protobuf::Message* getConfigMessage() const override { return &m_config; }
    objectlocation::ObjectLocationConfig& getObjectLocationConfig() { return m_config; }
private:
    objectlocation::ObjectLocationConfig m_config;
};

class CObjectLocationAlg : public IObjectLocationAlg {
public:
    CObjectLocationAlg(const std::string& exePath);
    ~CObjectLocationAlg() override;

    // 实现IObjectLocationAlg接口
    bool initAlgorithm(CSelfAlgParam* p_pAlgParam, const AlgCallback& alg_cb, void* hd) override;
    void runAlgorithm(void* p_pSrcData) override;

private:
    // 加载配置文件
    bool loadConfig(const std::string& configPath);
    
    // 创建并初始化模块
    bool initModules();
    
    // 执行模块链
    bool executeModuleChain();

private:
    std::string m_exePath;                                    // 工程路径
    std::shared_ptr<ObjectLocationConfig> m_pConfig;          // 配置对象
    std::vector<std::shared_ptr<IBaseModule>> m_moduleChain;  // 模块执行链
    AlgCallback m_algCallback;                                // 算法回调函数
    void* m_callbackHandle;                                   // 回调函数句柄
    CAlgResult* m_currentInput;                               // 当前输入数据
    CAlgResult m_currentOutput;                               // 当前输出数据
}; 