/*******************************************************
 文件名：CMultiModalFusionAlg.cpp
 作者：sharkls
 描述：多模态融合算法主类实现
 版本：v1.0
 日期：2025-05-13
 *******************************************************/

#include "CMultiModalFusionAlg.h"

// 加载指定路径的conf配置文件并将其反序列化（解析prototext文件）
bool MultiModalFusionConfig::loadFromFile(const std::string& path) 
{
    std::ifstream input(path);
    if (!input) {
        LOG(ERROR) << "Failed to open config file: " << path;
        return false;
    }
    std::stringstream buffer;
    buffer << input.rdbuf();
    std::string content = buffer.str();
    if (!google::protobuf::TextFormat::ParseFromString(content, &m_protoConfig)) {
        LOG(ERROR) << "Failed to parse protobuf config file: " << path;
        return false;
    }
    return true;
}

CMultiModalFusionAlg::CMultiModalFusionAlg(const std::string& exePath)
    : m_exePath(exePath)
    , m_pConfig(std::make_shared<MultiModalFusionConfig>())
    , m_currentInput(nullptr)
    , m_callbackHandle(nullptr)
{
}

CMultiModalFusionAlg::~CMultiModalFusionAlg()
{
    m_moduleChain.clear();
}

bool CMultiModalFusionAlg::initAlgorithm(CSelfAlgParam* p_pAlgParam, const AlgCallback& alg_cb, void* hd)
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
    std::string configPath = (exePath.parent_path() / "Configs"/ "Alg" / "MultiModalFusionConfig.conf").string();

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
    LOG(INFO) << "CMultiModalFusionAlg::initAlgorithm status: success ";
    return true;
}

void CMultiModalFusionAlg::runAlgorithm(void* p_pSrcData)
{
    LOG(INFO) << "CMultiModalFusionAlg::runAlgorithm status: start ";
    // 0. 每次运行前重置结构体内容
    m_currentOutput = CAlgResult(); // 或者手动清空成员
    int64_t startTimeStamp = GetTimeStamp();
    m_currentOutput.mapTimeStamp()[TIMESTAMP_MMALG_BEGIN] = startTimeStamp;

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
    LOG(INFO) << "CMultiModalFusionAlg::runAlgorithm status: success ";
}

// 加载配置文件
bool CMultiModalFusionAlg::loadConfig(const std::string& configPath)
{
    return m_pConfig->loadFromFile(configPath);
}

// 初始化子模块
bool CMultiModalFusionAlg::initModules()
{   
    LOG(INFO) << "CMultiModalFusionAlg::initModules status: start ";
    auto& multiModalConfig = m_pConfig->getMultiModalFusionConfig();
    const common::ModulesConfig& modules = multiModalConfig.modules_config();
    m_moduleChain.clear();

    // 遍历所有模块，按顺序实例化
    for (const common::ModuleConfig& mod : modules.modules()) {
        auto module = ModuleFactory::getInstance().createModule("MultiModalFusion", mod.name(), m_exePath);
        if (!module) {
            LOG(ERROR) << "Failed to create module: " << mod.name();
            return false;
        }
        // 传递可写指针
        if (!module->init((void*)multiModalConfig.mutable_model_config())) {
            LOG(ERROR) << "Failed to initialize module: " << mod.name();
            return false;
        }
        m_moduleChain.push_back(module);
    }
    LOG(INFO) << "CMultiModalFusionAlg::initModules status: success ";
    return true;
}

// 执行模块链
bool CMultiModalFusionAlg::executeModuleChain()
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
    int64_t endTimeStamp = GetTimeStamp();


    std::cout << "MultiModalFusion End : Input TIMESTAMP_TIME_MATCH"<< m_currentInput->mapTimeStamp()[TIMESTAMP_TIME_MATCH] << std::endl;
    // 结果穿透
    if(m_currentOutput.vecFrameResult().size() > 0) 
    {   
        // 输入数据常规信息穿透
        m_currentOutput.vecFrameResult()[0].unFrameId() = m_currentInput->unFrameId();
        m_currentOutput.vecFrameResult()[0].mapTimeStamp() = m_currentInput->mapTimeStamp();
        m_currentOutput.vecFrameResult()[0].mapDelay() = m_currentInput->mapDelay();
        m_currentOutput.vecFrameResult()[0].mapFps() = m_currentInput->mapFps();

        // 独有数据填充
        m_currentOutput.vecFrameResult()[0].tCameraSupplement() = m_currentInput->tDisparityResult();          // 深度图
        m_currentOutput.vecFrameResult()[0].eDataType(DATA_TYPE_MMALG_RESULT);                                 // 数据类型赋值
        m_currentOutput.vecFrameResult()[0].mapTimeStamp()[TIMESTAMP_MMALG_END] = endTimeStamp;                // 多模态结束时间戳
        m_currentOutput.vecFrameResult()[0].mapDelay()[DELAY_TYPE_MMALG] = endTimeStamp - m_currentOutput.mapTimeStamp()[TIMESTAMP_MMALG_BEGIN];    // 多模态算法耗时计算
    }

    return true;
} 