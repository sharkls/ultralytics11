/*******************************************************
 文件名：CPoseEstimationAlg.cpp
 作者：
 描述：姿态估计算法主类实现
 版本：v1.0
 日期：2024-03-21
 *******************************************************/

#include "CPoseEstimationAlg.h"

// 加载指定路径的conf配置文件并将其反序列化（解析prototext文件）
bool PoseEstimationConfig::loadFromFile(const std::string& path) 
{
    std::ifstream input(path);
    if (!input) {
        LOG(ERROR) << "Failed to open config file: " << path;
        return false;
    }
    std::stringstream buffer;
    buffer << input.rdbuf();
    std::string content = buffer.str();
    if (!google::protobuf::TextFormat::ParseFromString(content, &m_config)) {
        LOG(ERROR) << "Failed to parse protobuf config file: " << path;
        return false;
    }
    return true;
}

CPoseEstimationAlg::CPoseEstimationAlg(const std::string& exePath)
    : m_exePath(exePath)
    , m_pConfig(std::make_shared<PoseEstimationConfig>())
    , m_currentInput(nullptr)
    , m_callbackHandle(nullptr)
{
}

CPoseEstimationAlg::~CPoseEstimationAlg()
{
    m_moduleChain.clear();
}

bool CPoseEstimationAlg::initAlgorithm(CSelfAlgParam* p_pAlgParam, const AlgCallback& alg_cb, void* hd)
{   
    // 1. 检查参数
    if (!p_pAlgParam) {
        LOG(ERROR) << "Algorithm parameters is null";
        return false;
    }

    // 2. 保存回调函数和句柄
    m_algCallback = alg_cb;
    m_callbackHandle = hd;

    // 3. 构建配置文件路径
    std::filesystem::path exePath(m_exePath);
    LOG(INFO) << "m_exePath : " << exePath.parent_path().string();
    std::string configPath = (exePath.parent_path() / "Configs"/ "Alg" / "PoseEstimationConfig.conf").string();

    // 4. 加载配置文件
    if (!loadConfig(configPath)) {
        LOG(ERROR) << "Failed to load config file: " << configPath;
        return false;
    }

    // 5. 初始化模块
    if (!initModules()) {
        LOG(ERROR) << "Failed to initialize modules";
        return false;
    }
    LOG(INFO) << "CPoseEstimationAlg::initAlgorithm status: success ";
    return true;
}

void CPoseEstimationAlg::runAlgorithm(void* p_pSrcData)
{
    LOG(INFO) << "CPoseEstimationAlg::runAlgorithm status: start ";
    // 0. 每次运行前重置结构体内容
    m_currentOutput = CAlgResult(); // 或者手动清空成员

    // 1. 核验输入数据是否为空
    if (!p_pSrcData) {
        LOG(ERROR) << "Input data is null";
        if (m_algCallback) {
            m_algCallback(m_currentOutput, m_callbackHandle);
        }
        return;
    }

    // 2. 输入数据赋值
    m_currentInput = static_cast<CMultiModalSrcData *>(p_pSrcData);   
    
    // 3. 执行模块链
    if (!executeModuleChain()) {
        LOG(ERROR) << "Failed to execute module chain";
        if (m_algCallback) {
            m_algCallback(m_currentOutput, m_callbackHandle);
        }
        return;
    }

    // 4. 通过回调函数返回结果
    if (m_algCallback) {
        m_algCallback(m_currentOutput, m_callbackHandle);
    }
    LOG(INFO) << "CPoseEstimationAlg::runAlgorithm status: success ";
}

// 加载配置文件
bool CPoseEstimationAlg::loadConfig(const std::string& configPath)
{
    return m_pConfig->loadFromFile(configPath);
}

// 初始化子模块
bool CPoseEstimationAlg::initModules()
{   
    LOG(INFO) << "CPoseEstimationAlg::initModules status: start ";
    auto& poseConfig = m_pConfig->getPoseConfig();
    const common::ModulesConfig& modules = poseConfig.modules_config();
    m_moduleChain.clear();

    // 遍历所有模块，按顺序实例化
    for (const common::ModuleConfig& mod : modules.modules()) {
        auto module = ModuleFactory::getInstance().createModule("PoseEstimation", mod.name(), m_exePath);
        if (!module) {
            LOG(ERROR) << "Failed to create module: " << mod.name();
            return false;
        }
        // 传递可写指针
        if (!module->init((void*)poseConfig.mutable_yolo_model_config())) {
            LOG(ERROR) << "Failed to initialize module: " << mod.name();
            return false;
        }
        m_moduleChain.push_back(module);
    }
    LOG(INFO) << "CPoseEstimationAlg::initModules status: success ";
    return true;
}

// 执行模块链
bool CPoseEstimationAlg::executeModuleChain()
{
    void* currentData = static_cast<void *>(m_currentInput);

    for (auto& module : m_moduleChain) {
        // 设置输入数据
        module->setInput(currentData);

        // 执行模块
        module->execute();
        currentData = module->getOutput();
        if (!currentData) {
            LOG(ERROR) << "Module execution failed: " << module->getModuleName();
            return false;
        }
    }

    // 假设最后一个模块输出CAlgResult结构体
    m_currentOutput = *static_cast<CAlgResult *>(currentData);
    return true;
} 