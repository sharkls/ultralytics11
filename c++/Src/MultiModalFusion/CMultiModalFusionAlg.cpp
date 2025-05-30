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

    m_run_status = m_pConfig->getMultiModalFusionConfig().model_config().run_status();

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

    // const auto& tCameraSupplement = m_currentInput->tDisparityResult();
    // int depth_width = tCameraSupplement.usWidth();
    // int depth_height = tCameraSupplement.usHeight();
    // const auto& depth_map = tCameraSupplement.vecDistanceInfo();
    // std::cout << "depth_width0: " << depth_width << ", depth_height0: " << depth_height << std::endl;
    // if (depth_width > 370 && depth_height > 370) {
    //     int idx = 370 * depth_width + 370;
    //     if (idx < depth_map.size()) {
    //         float d_370_370 = static_cast<float>(depth_map[idx]);
    //         LOG(INFO) << "[Fusion] Depth at (370,370): " << d_370_370;
    //     } else {
    //         LOG(INFO) << "[Fusion] Index out of range for depth_map, idx=" << idx << ", size=" << depth_map.size();
    //     }
    // } else {
    //     LOG(INFO) << "[Fusion] Depth map size too small for (370,370), width=" << depth_width << ", height=" << depth_height;
    // }
    
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

    // 结果穿透
    if(m_currentOutput.vecFrameResult().size() > 0) 
    {   
        // 输入数据常规信息穿透
        m_currentOutput.vecFrameResult()[0].unFrameId() = m_currentInput->vecVideoSrcData()[0].unFrameId();
        m_currentOutput.vecFrameResult()[0].mapTimeStamp() = m_currentInput->vecVideoSrcData()[0].mapTimeStamp();
        m_currentOutput.vecFrameResult()[0].mapDelay() = m_currentInput->vecVideoSrcData()[0].mapDelay();
        m_currentOutput.vecFrameResult()[0].mapFps() = m_currentInput->vecVideoSrcData()[0].mapFps();
        m_currentOutput.lTimeStamp() = m_currentInput->vecVideoSrcData()[0].lTimeStamp();

        LOG(INFO) << "原始数据穿透 FrameID ： " << m_currentOutput.vecFrameResult()[0].unFrameId() << ", TimeStamp :" << m_currentOutput.lTimeStamp();
        
        // 独有数据填充
        m_currentOutput.vecFrameResult()[0].tCameraSupplement() = m_currentInput->tDisparityResult();          // 深度图
        m_currentOutput.vecFrameResult()[0].eDataType(DATA_TYPE_MMALG_RESULT);                                 // 数据类型赋值
        m_currentOutput.vecFrameResult()[0].mapTimeStamp()[TIMESTAMP_MMALG_END] = endTimeStamp;                // 多模态结束时间戳
        m_currentOutput.vecFrameResult()[0].mapDelay()[DELAY_TYPE_MMALG] = endTimeStamp - m_currentOutput.mapTimeStamp()[TIMESTAMP_MMALG_BEGIN];    // 多模态算法耗时计算
    }

    // 离线状态时将检测结果绘制到原始图像上
    // std::cout  << "m_run_status: " << m_run_status << "distancemap : " << m_currentOutput.vecFrameResult()[0].tCameraSupplement().vecDistanceInfo().size() << std::endl; 
    // if(m_run_status)
    // {   
    //     visualizationResult();
    // }

    // // 定位
    // if (m_currentOutput.vecFrameResult().size() > 0)
    // {
    //     auto& frameResult = m_currentOutput.vecFrameResult()[0];
    //     auto& objResults = frameResult.vecObjectResult();
    //     const auto& disparity = frameResult.tCameraSupplement();
    //     int width = disparity.usWidth();
    //     int height = disparity.usHeight();
    //     const auto& depthMap = disparity.vecDistanceInfo();

    //     for (auto& obj : objResults) {
    //         // 1. 计算中心点
    //         float cx = (obj.fTopLeftX() + obj.fBottomRightX()) / 2.0f;
    //         float cy = (obj.fTopLeftY() + obj.fBottomRightY()) / 2.0f;

    //         // 2. 像素坐标转整数下标
    //         int ix = static_cast<int>(cx + 0.5f);
    //         int iy = static_cast<int>(cy + 0.5f);

    //         // 3. 检查边界
    //         if (ix >= 0 && ix < width && iy >= 0 && iy < height) {
    //             int idx = iy * width + ix;
    //             if (idx >= 0 && idx < depthMap.size()) {
    //                 obj.fDistance() = depthMap[idx]; // 赋值距离
    //             }
    //         }
    //     }
    // }

    if (m_currentOutput.vecFrameResult().size() > 0)
    {
        auto& frameResult = m_currentOutput.vecFrameResult()[0];
        auto& objResults = frameResult.vecObjectResult();
        const auto& disparity = frameResult.tCameraSupplement();
        int width = disparity.usWidth();
        int height = disparity.usHeight();
        const auto& depthMap = disparity.vecDistanceInfo();

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
                for (int dy = -2; dy <= 2; ++dy) {
                    for (int dx = -2; dx <= 2; ++dx) {
                        int currentIx = ix + dx;
                        int currentIy = iy + dy;

                        // 检查当前坐标是否在深度图范围内
                        if (currentIx >= 0 && currentIx < width && currentIy >= 0 && currentIy < height) {
                            int idx = currentIy * width + currentIx;
                            if (idx >= 0 && idx < depthMap.size()) {
                                depthValues.push_back(depthMap[idx]);
                            }
                        }
                    }
                }

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
                    }
                }
            }
        }
    }

    if(m_currentOutput.vecFrameResult().size() > 0)
    {
        LOG(INFO) << "所有数据完成穿透! vecFrameResult()[0].vecObjectResult().size(): " << m_currentOutput.vecFrameResult()[0].vecObjectResult().size();
    }

    return true;
} 

void CMultiModalFusionAlg::visualizationResult()
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

        // 可选：绘制类别和置信度
        std::string label = obj.strClass() + " " + std::to_string(obj.fVideoConfidence());
        cv::putText(srcImage, label, rect.tl(), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,255,0), 1);
    }

    // 3. 构造保存目录
    std::string visDir = (std::filesystem::path(m_exePath) / "Vis__Result").string();
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