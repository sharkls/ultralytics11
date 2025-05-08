/*******************************************************
 文件名：CObjectLocationAlg.cpp
 作者：
 描述：姿态估计算法主类实现
 版本：v1.0
 日期：2024-03-21
 *******************************************************/

#include "CObjectLocationAlg.h"
#include "Factory/ModuleFactory.h"
#include <iostream>
#include <filesystem>

CObjectLocationAlg::CObjectLocationAlg(const std::string& exePath)
    : m_exePath(exePath)
    , m_pConfig(std::make_shared<AlgorithmConfig>())
    , m_currentInput(nullptr)
    , m_currentOutput(nullptr)
    , m_callbackHandle(nullptr)
{
}

CObjectLocationAlg::~CObjectLocationAlg()
{
    m_moduleChain.clear();
}

bool CObjectLocationAlg::initAlgorithm(CSelfAlgParam* p_pAlgParam, const AlgCallback& alg_cb, void* hd)
{
    if (!p_pAlgParam) {
        std::cerr << "Algorithm parameters is null" << std::endl;
        return false;
    }

    // 保存回调函数和句柄
    m_algCallback = alg_cb;
    m_callbackHandle = hd;

    // 构建配置文件路径
    std::filesystem::path exePath(m_exePath);
    std::string configPath = (exePath.parent_path() / "config" / "pose_estimation.yaml").string();

    // 加载配置文件
    if (!loadConfig(configPath)) {
        std::cerr << "Failed to load config file: " << configPath << std::endl;
        return false;
    }

    // 初始化模块
    if (!initModules()) {
        std::cerr << "Failed to initialize modules" << std::endl;
        return false;
    }

    return true;
}

void CObjectLocationAlg::runAlgorithm(void* p_pSrcData)
{
    if (!p_pSrcData) {
        std::cerr << "Input data is null" << std::endl;
        if (m_algCallback) {
            m_algCallback(m_callbackHandle, nullptr);
        }
        return;
    }

    m_currentInput = p_pSrcData;
    
    // 执行模块链
    if (!executeModuleChain()) {
        std::cerr << "Failed to execute module chain" << std::endl;
        if (m_algCallback) {
            m_algCallback(m_callbackHandle, nullptr);
        }
        return;
    }

    // 通过回调函数返回结果
    if (m_algCallback) {
        m_algCallback(m_callbackHandle, m_currentOutput);
    }
}

bool CObjectLocationAlg::loadConfig(const std::string& configPath)
{
    return m_pConfig->loadFromFile(configPath);
}

bool CObjectLocationAlg::initModules()
{
    const auto& moduleConfigs = m_pConfig->getModuleConfigs();
    m_moduleChain.clear();

    for (const auto& config : moduleConfigs) {
        // 通过工厂创建模块
        auto module = ModuleFactory::getInstance().createModule(config.moduleName);
        if (!module) {
            std::cerr << "Failed to create module: " << config.moduleName << std::endl;
            return false;
        }

        // 初始化模块
        if (!module->init(nullptr)) {  // 这里需要传入适当的参数
            std::cerr << "Failed to initialize module: " << config.moduleName << std::endl;
            return false;
        }

        m_moduleChain.push_back(module);
    }

    return true;
}

bool CObjectLocationAlg::executeModuleChain()
{
    void* currentData = m_currentInput;

    for (auto& module : m_moduleChain) {
        // 设置输入数据
        module->setInput(currentData);

        // 执行模块
        currentData = module->execute();
        if (!currentData) {
            std::cerr << "Module execution failed: " << module->getModuleName() << std::endl;
            return false;
        }
    }

    m_currentOutput = currentData;
    return true;
} 