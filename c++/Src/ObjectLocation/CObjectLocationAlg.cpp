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

bool ObjectLocationConfig::loadFromFile(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        std::cerr << "Failed to open config file: " << path << std::endl;
        return false;
    }
    if (!m_config.ParseFromIstream(&input)) {
        std::cerr << "Failed to parse protobuf config file: " << path << std::endl;
        return false;
    }
    return true;
}

CObjectLocationAlg::CObjectLocationAlg(const std::string& exePath)
    : m_exePath(exePath)
    , m_pConfig(std::make_shared<ObjectLocationConfig>())
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
    std::string configPath = (exePath.parent_path() / "config" / "PoseEstimationConfig.conf").string();

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
    const auto& poseConfig = m_pConfig->getPoseConfig();
    const auto& modules = poseConfig.modules_config();
    m_moduleChain.clear();

    // 预处理
    if (!modules.preprocess().empty()) {
        auto module = ModuleFactory::getInstance().createModule("ObjectLocation", modules.preprocess());
        if (!module) return false;
        if (!module->init(&poseConfig.yolo_model_config())) return false;
        m_moduleChain.push_back(module);
    }

    // 推理
    if (!modules.inference().empty()) {
        auto module = ModuleFactory::getInstance().createModule("ObjectLocation", modules.inference());
        if (!module) return false;
        if (!module->init(&poseConfig.yolo_model_config())) return false;
        m_moduleChain.push_back(module);
    }

    // 后处理
    if (!modules.postprocess().empty()) {
        auto module = ModuleFactory::getInstance().createModule("ObjectLocation", modules.postprocess());
        if (!module) return false;
        if (!module->init(&poseConfig.yolo_model_config())) return false;
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