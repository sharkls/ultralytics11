/*******************************************************
 文件名：CBinocularPositioningAlg.cpp
 作者：sharkls
 描述：姿态估计算法主类实现
 版本：v1.0
 日期：2024-03-21
 *******************************************************/

#include "CBinocularPositioningAlg.h"

// 加载指定路径的conf配置文件并将其反序列化（解析prototext文件）
bool BinocularPositioningConfig::loadFromFile(const std::string& path) 
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

CBinocularPositioningAlg::CBinocularPositioningAlg(const std::string& exePath)
    : m_exePath(exePath)
    , m_pConfig(std::make_shared<BinocularPositioningConfig>())
    , m_currentInput(nullptr)
    , m_callbackHandle(nullptr)
{
}

CBinocularPositioningAlg::~CBinocularPositioningAlg()
{
    m_moduleChain.clear();
}

bool CBinocularPositioningAlg::initAlgorithm(CSelfAlgParam* p_pAlgParam, const AlgCallback& alg_cb, void* hd)
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
    std::string configPath = (exePath.parent_path() / "Configs"/ "Alg" / "BinocularPositioningConfig.conf").string();

    // 4. 加载配置文件
    if (!loadConfig(configPath)) {
        LOG(ERROR) << "Failed to load config file: " << configPath;
        return false;
    }

    m_run_status = m_pConfig->getBinocularPositioningConfig().model_config().run_status();

    // 5. 初始化模块
    if (!initModules()) {
        LOG(ERROR) << "Failed to initialize modules";
        return false;
    }
    LOG(INFO) << "CBinocularPositioningAlg::initAlgorithm status: successs ";
    return true;
}

void CBinocularPositioningAlg::runAlgorithm(void* p_pSrcData)
{
    LOG(INFO) << "CBinocularPositioningAlg::runAlgorithm status: start ";
    // 0. 每次运行前重置结构体内容
    m_currentOutput = CAlgResult(); // 或者手动清空成员
    int64_t beginTimeStamp = GetTimeStamp();
    m_currentOutput.mapTimeStamp()[TIMESTAMP_BINOCULARPOSITIONINGALG_BEGIN] = beginTimeStamp;

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
    LOG(INFO) << "CBinocularPositioningAlg::runAlgorithm status: success ";

    return;
}

// 加载配置文件
bool CBinocularPositioningAlg::loadConfig(const std::string& configPath)
{
    return m_pConfig->loadFromFile(configPath);
}

// 初始化子模块
bool CBinocularPositioningAlg::initModules()
{   
    LOG(INFO) << "CBinocularPositioningAlg::initModules status: start ";
    auto& config = m_pConfig->getBinocularPositioningConfig();
    const common::ModulesConfig& modules = config.modules_config();
    m_moduleChain.clear();

    // 遍历所有模块，按顺序实例化
    for (const common::ModuleConfig& mod : modules.modules()) {
        auto module = ModuleFactory::getInstance().createModule("BinocularPositioning", mod.name(), m_exePath);
        if (!module) {
            LOG(ERROR) << "Failed to create module: " << mod.name();
            return false;
        }
        // 传递可写指针
        if (!module->init((void*)config.mutable_model_config())) {
            LOG(ERROR) << "Failed to initialize module: " << mod.name();
            return false;
        }
        m_moduleChain.push_back(module);
    }
    LOG(INFO) << "CBinocularPositioningAlg::initModules status: success ";
    return true;
}

// 执行模块链
bool CBinocularPositioningAlg::executeModuleChain()
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
    CAlgResult* resultPtr = static_cast<CAlgResult *>(currentData);
    int64_t endTimeStamp = GetTimeStamp();

    m_currentOutput = *resultPtr;

    // 结果穿透
    m_currentOutput.lTimeStamp() = m_currentInput->vecVideoSrcData()[0].lTimeStamp();
    if(m_currentOutput.vecFrameResult().size() > 0) 
    {   
        // 输入数据常规信息穿透
        m_currentOutput.vecFrameResult()[0].unFrameId() = m_currentInput->vecVideoSrcData()[0].unFrameId();

        m_currentOutput.vecFrameResult()[0].mapTimeStamp() = m_currentInput->vecVideoSrcData()[0].mapTimeStamp();
        m_currentOutput.vecFrameResult()[0].mapDelay() = m_currentInput->vecVideoSrcData()[0].mapDelay();
        m_currentOutput.vecFrameResult()[0].mapFps() = m_currentInput->vecVideoSrcData()[0].mapFps();

        LOG(INFO) << "原有信息穿透完毕： FrameId : " << m_currentOutput.vecFrameResult()[0].unFrameId() << ", lTimeStamp : " << m_currentOutput.lTimeStamp();

        // 独有数据填充
        m_currentOutput.vecFrameResult()[0].eDataType(DATA_TYPE_BINOCULARPOSITIONINGALG_RESULT);                                 // 数据类型赋值
        m_currentOutput.vecFrameResult()[0].mapTimeStamp()[TIMESTAMP_BINOCULARPOSITIONINGALG_END] = endTimeStamp;
        m_currentOutput.vecFrameResult()[0].mapDelay()[DELAY_TYPE_BINOCULARPOSITIONINGALG] = endTimeStamp - m_currentOutput.mapTimeStamp()[TIMESTAMP_BINOCULARPOSITIONINGALG_BEGIN];
    }

    if(m_run_status)
    {
        visualizationResult();
    }
    return true;
} 


void CBinocularPositioningAlg::visualizationResult()
{
    // 1. 获取深度图数据
    auto& frameResult = m_currentOutput.vecFrameResult()[0];
    int width = frameResult.tCameraSupplement().usWidth();
    int height = frameResult.tCameraSupplement().usHeight();
    const std::vector<int32_t>& depthData = frameResult.tCameraSupplement().vecDistanceInfo();

    if (depthData.empty() || width <= 0 || height <= 0) {
        LOG(ERROR) << "Depth data is empty or size invalid!";
        return;
    }

    // 2. 转为Mat并归一化
    cv::Mat depthMat(height, width, CV_32S, (void*)depthData.data());
    cv::Mat depthMatFloat;
    depthMat.convertTo(depthMatFloat, CV_32F);

    cv::Mat depthVis;
    cv::normalize(depthMatFloat, depthVis, 0, 255, cv::NORM_MINMAX);
    depthVis.convertTo(depthVis, CV_8U);

    // 3. 构造保存目录
    std::string visDir = (std::filesystem::path(m_exePath) / "Vis_BinocularPositioning_Result").string();
    if (!std::filesystem::exists(visDir)) {
        std::filesystem::create_directories(visDir);
    }

    // 4. 构造保存路径
    uint32_t frameId = frameResult.unFrameId();
    std::string savePath = visDir + "/" + std::to_string(frameId) + "_depth.jpg";

    // 5. 保存深度图
    cv::imwrite(savePath, depthVis);

    LOG(INFO) << "深度图已保存到: " << savePath;
}