/*******************************************************
 文件名：CObjectLocationAlg.h
 作者：
 描述：姿态估计算法主类，负责协调各个子模块的运行
 版本：v1.0
 日期：2024-03-21
 *******************************************************/

#ifndef COBJECT_LOCATION_ALG_H
#define COBJECT_LOCATION_ALG_H

#include <memory>
#include <vector>
#include <string>
#include <map>
#include "../../Include/Interface/ExportObjectLocationAlgLib.h"
#include "../Common/IBaseModule.h"
#include "Config/AlgorithmConfig.h"

// 回调函数类型定义
using ResultCallback = std::function<void(const void*)>;

class CObjectLocationAlg : public IObjectLocationAlg {
public:
    CObjectLocationAlg(const std::string& exePath);
    ~CObjectLocationAlg() override;

    // 实现IObjectLocationAlg接口
    bool initAlgorithm(CSelfAlgParam* p_pAlgParam, const AlgCallback& alg_cb, void* hd) override;
    void runAlgorithm(void* p_pSrcData) override;

    // 设置结果回调函数
    void setResultCallback(ResultCallback callback);

private:
    // 加载配置文件
    bool loadConfig(const std::string& configPath);
    
    // 创建并初始化模块
    bool initModules();
    
    // 执行模块链
    bool executeModuleChain();

private:
    std::string m_exePath;                                    // 可执行文件路径
    std::shared_ptr<AlgorithmConfig> m_pConfig;               // 配置对象
    std::vector<std::shared_ptr<IBaseModule>> m_moduleChain;  // 模块执行链
    AlgCallback m_algCallback;                               // 算法回调函数
    void* m_callbackHandle;                                  // 回调函数句柄
    void* m_currentInput;                                    // 当前输入数据
    void* m_currentOutput;                                   // 当前输出数据
    ResultCallback m_resultCallback;                              // 结果回调函数
};

#endif // COBJECT_LOCATION_ALG_H 