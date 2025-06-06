/*******************************************************
 文件名：CObjectDetectionAlg.cpp
 作者：sharkls
 描述：目标检测算法主类实现
 版本：v1.0
 日期：2025-05-16
 *******************************************************/

#include "CObjectDetectionAlg.h"

// 加载指定路径的conf配置文件并将其反序列化（解析prototext文件）
bool ObjectDetectionConfig::loadFromFile(const std::string& path) 
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

CObjectDetectionAlg::CObjectDetectionAlg(const std::string& exePath)
    : m_exePath(exePath)
    , m_pConfig(std::make_shared<ObjectDetectionConfig>())
    , m_currentInput(nullptr)
    , m_callbackHandle(nullptr)
{
}

CObjectDetectionAlg::~CObjectDetectionAlg()
{
    m_moduleChain.clear();
}

bool CObjectDetectionAlg::initAlgorithm(CSelfAlgParam* p_pAlgParam, const AlgCallback& alg_cb, void* hd)
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
    std::string configPath = (exePath.parent_path() / "Configs"/ "Alg" / "ObjectDetectionConfig.conf").string();

    // 4. 加载配置文件
    if (!loadConfig(configPath)) {
        LOG(ERROR) << "Failed to load config file: " << configPath;
        return false;
    }

    m_run_status = m_pConfig->getObjectDetectionConfig().yolo_model_config().run_status();

    // 5. 初始化模块
    if (!initModules()) {
        LOG(ERROR) << "Failed to initialize modules";
        return false;
    }
    LOG(INFO) << "CObjectDetectionAlg::initAlgorithm status: success ";
    return true;
}

void CObjectDetectionAlg::runAlgorithm(void* p_pSrcData)
{
    // LOG(INFO) << "CObjectDetectionAlg::runAlgorithm status: start ";
    // 0. 每次运行前重置结构体内容
    m_currentOutput = CAlgResult(); // 或者手动清空成员
    int64_t endTimeStamp = GetTimeStamp();
    m_currentOutput.mapTimeStamp()[TIMESTAMP_POSEALG_BEGIN] = endTimeStamp;

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
    // LOG(INFO) << "CObjectDetectionAlg::runAlgorithm status: success ";
    return;
}

// 加载配置文件
bool CObjectDetectionAlg::loadConfig(const std::string& configPath)
{
    return m_pConfig->loadFromFile(configPath);
}

// 初始化子模块
bool CObjectDetectionAlg::initModules()
{   
    LOG(INFO) << "CObjectDetectionAlg::initModules status: start ";
    auto& objectDetectionConfig = m_pConfig->getObjectDetectionConfig();
    const common::ModulesConfig& modules = objectDetectionConfig.modules_config();
    m_moduleChain.clear();

    // 遍历所有模块，按顺序实例化
    for (const common::ModuleConfig& mod : modules.modules()) {
        auto module = ModuleFactory::getInstance().createModule("ObjectDetection", mod.name(), m_exePath);
        if (!module) {
            LOG(ERROR) << "Failed to create module: " << mod.name();
            return false;
        }
        // 传递可写指针
        if (!module->init((void*)objectDetectionConfig.mutable_yolo_model_config())) {
            LOG(ERROR) << "Failed to initialize module: " << mod.name();
            return false;
        }
        m_moduleChain.push_back(module);
    }
    LOG(INFO) << "CObjectDetectionAlg::initModules status: success ";
    return true;
}

// 执行模块链
bool CObjectDetectionAlg::executeModuleChain()
{
    void* currentData = static_cast<void *>(m_currentInput);

    for (auto& module : m_moduleChain) {
        // 设置输入数据
        module->setInput(currentData);
        
        // 执行模块
        module->execute();
        
        currentData = module->getOutput();
        if (module->getModuleName() == "Yolov11") 
        {
            auto timestamp = m_currentInput->vecVideoSrcData()[0].lTimeStamp();
            CAlgResult* result = static_cast<CAlgResult*>(currentData);
            if (!result->vecFrameResult().empty()) {
                result->vecFrameResult()[0].lTimeStamp(timestamp);
            }
        }
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
    if(m_currentOutput.vecFrameResult().size() > 0) 
    {   
        // 输入数据常规信息穿透
        m_currentOutput.vecFrameResult()[0].unFrameId() = m_currentInput->vecVideoSrcData()[0].unFrameId();
        m_currentOutput.vecFrameResult()[0].mapTimeStamp() = m_currentInput->vecVideoSrcData()[0].mapTimeStamp();
        m_currentOutput.vecFrameResult()[0].mapDelay() = m_currentInput->vecVideoSrcData()[0].mapDelay();
        m_currentOutput.vecFrameResult()[0].mapFps() = m_currentInput->vecVideoSrcData()[0].mapFps();
        m_currentOutput.lTimeStamp() = m_currentInput->vecVideoSrcData()[0].lTimeStamp();

        LOG(INFO) << "原有信息穿透完毕： FrameId : " << m_currentOutput.vecFrameResult()[0].unFrameId() << ", lTimeStamp : " << m_currentOutput.lTimeStamp();

        // 独有数据填充
        m_currentOutput.vecFrameResult()[0].eDataType(DATA_TYPE_POSEALG_RESULT);                                 
        m_currentOutput.vecFrameResult()[0].mapTimeStamp()[TIMESTAMP_POSEALG_END] = endTimeStamp;                
        m_currentOutput.vecFrameResult()[0].mapDelay()[DELAY_TYPE_POSEALG] = endTimeStamp - m_currentOutput.mapTimeStamp()[TIMESTAMP_POSEALG_BEGIN];    
        m_currentOutput.vecFrameResult()[0].tCameraSupplement() = m_currentInput->tDisparityResult();
    }

    if(m_run_status)
    {   
        visualizationResult();
    }

    if(m_currentOutput.vecFrameResult().size() > 0)
    {
        LOG(INFO) << "所有数据完成穿透! vecFrameResult()[0].vecObjectResult().size(): " << m_currentOutput.vecFrameResult()[0].vecObjectResult().size();

        // 添加目标深度值获取逻辑
        auto& frameResult = m_currentOutput.vecFrameResult()[0];
        auto& objResults = frameResult.vecObjectResult();
        const auto& disparity = frameResult.tCameraSupplement();
        int width = disparity.usWidth();
        int height = disparity.usHeight();
        const auto& depthMap = disparity.vecDistanceInfo();
        LOG(INFO) << "depthMap.size() : " << depthMap.size();
        for (auto& obj : objResults) {
            // 1. 计算中心点
            float cx = (obj.fTopLeftX() + obj.fBottomRightX()) / 2.0f;
            float cy = (obj.fTopLeftY() + obj.fBottomRightY()) / 2.0f;

            // 2. 像素坐标转整数下标
            int ix = static_cast<int>(cx + 0.5f);
            int iy = static_cast<int>(cy + 0.5f);

            // 3. 检查边界
            if (ix >= 0 && ix < width && iy >= 0 && iy < height) {
                // 定义一个数组来存储 5×5 区域的深度值
                std::vector<float> depthValues;

                // 遍历中心点周围 5×5 的区域
                std::cout << "5x5区域深度值：" << std::endl;
                for (int dy = -2; dy <= 2; ++dy) {
                    for (int dx = -2; dx <= 2; ++dx) {
                        int currentIx = ix + dx;
                        int currentIy = iy + dy;

                        // 检查当前坐标是否在深度图范围内
                        if (currentIx >= 0 && currentIx < width && currentIy >= 0 && currentIy < height) {
                            int idx = currentIy * width + currentIx;
                            if (idx >= 0 && idx < depthMap.size()) {
                                depthValues.push_back(depthMap[idx]);
                                std::cout << std::fixed << std::setprecision(2) << depthMap[idx] << "\t";
                            }
                        } else {
                            std::cout << "N/A\t";
                        }
                    }
                    std::cout << std::endl;
                }
                std::cout << std::endl;
                

                // 如果收集到足够的深度值
                if (!depthValues.empty()) {
                    // 去除深度值为 0 的点
                    std::vector<float> nonZeroDepthValues;
                    for (float depth : depthValues) {
                        if (depth != 0.0f) {
                            nonZeroDepthValues.push_back(depth);
                        }
                    }

                    // 如果存在非零深度值
                    if (!nonZeroDepthValues.empty()) {
                        // 如果非零深度值足够多（≥20 个），舍弃 10 个最小值和 10 个最大值
                        if (nonZeroDepthValues.size() >= 20) {
                            std::sort(nonZeroDepthValues.begin(), nonZeroDepthValues.end());

                            int startIdx = 10;
                            int endIdx = nonZeroDepthValues.size() - 10;

                            float sum = 0.0f;
                            for (int i = startIdx; i < endIdx; ++i) {
                                sum += nonZeroDepthValues[i];
                            }
                            float averageDepth = sum / (endIdx - startIdx);

                            obj.fDistance() = averageDepth; // 赋值距离
                        } else {
                            // 如果非零深度值不足 20 个，直接求平均值
                            float sum = 0.0f;
                            for (float depth : nonZeroDepthValues) {
                                sum += depth;
                            }
                            float averageDepth = sum / nonZeroDepthValues.size();

                            obj.fDistance() = averageDepth; // 赋值距离
                        }
                        LOG(INFO) << "目标距离 ： " << obj.fDistance();
                    }
                }
            }
        }
    }
    return true;
}

void CObjectDetectionAlg::visualizationResult()
{
    // 1. 获取原始图像
    const auto& videoSrc = m_currentInput->vecVideoSrcData()[0];
    int width = videoSrc.usBmpWidth();
    int height = videoSrc.usBmpLength();
    int totalBytes = videoSrc.unBmpBytes();
    int channels = 0;
    if (width > 0 && height > 0) {
        channels = totalBytes / (width * height);
    }
    cv::Mat srcImage;
    if (channels == 3) {
        srcImage = cv::Mat(height, width, CV_8UC3, (void*)videoSrc.vecImageBuf().data()).clone();
    } else if (channels == 1) {
        srcImage = cv::Mat(height, width, CV_8UC1, (void*)videoSrc.vecImageBuf().data()).clone();
    } else {
        LOG(ERROR) << "Unsupported image channel: " << channels;
        return;
    }
    auto& frameResult = m_currentOutput.vecFrameResult()[0];

    // 2. 绘制检测结果
    for(const auto& obj : frameResult.vecObjectResult())
    {
        cv::Rect rect(
            cv::Point(static_cast<int>(obj.fTopLeftX()), static_cast<int>(obj.fTopLeftY())),
            cv::Point(static_cast<int>(obj.fBottomRightX()), static_cast<int>(obj.fBottomRightY()))
        );
        cv::rectangle(srcImage, rect, cv::Scalar(0, 255, 0), 2);

        // 可选：绘制类别、置信度和ID
        std::string label = obj.strClass() + " " + std::to_string(obj.fVideoConfidence());
        if (obj.usTargetId() > 0) {
            label += " ID:" + std::to_string(obj.usTargetId());
        }
        cv::putText(srcImage, label, rect.tl(), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,255,0), 1);
    }

    // 3. 构造保存目录
    std::string visDir = (std::filesystem::path(m_exePath) / "Vis_Detection_Result").string();
    if (!std::filesystem::exists(visDir)) {
        std::filesystem::create_directories(visDir);
    }

    // 4. 构造保存路径
    uint32_t frameId = frameResult.unFrameId();
    std::string savePath = visDir + "/" + std::to_string(frameId) + ".jpg";

    // 5. 保存图片
    cv::imwrite(savePath, srcImage);

    LOG(INFO) << "检测结果已保存到: " << savePath;
} 