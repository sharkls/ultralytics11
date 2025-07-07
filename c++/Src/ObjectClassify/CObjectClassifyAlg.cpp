/*******************************************************
 文件名：CObjectClassifyAlg.cpp
 作者：
 描述：目标分类算法主类实现
 版本：v1.0
 日期：2024-03-21
 *******************************************************/

#include "CObjectClassifyAlg.h"
#include <algorithm>

// 加载指定路径的conf配置文件并将其反序列化（解析prototext文件）
bool ObjectClassifyConfig::loadFromFile(const std::string& path) 
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

CObjectClassifyAlg::CObjectClassifyAlg(const std::string& exePath)
    : m_exePath(exePath)
    , m_pConfig(std::make_shared<ObjectClassifyConfig>())
    , m_currentInput(nullptr)
    , m_callbackHandle(nullptr)
{
}

CObjectClassifyAlg::~CObjectClassifyAlg()
{
    m_moduleChain.clear();
}

bool CObjectClassifyAlg::initAlgorithm(CSelfAlgParam* p_pAlgParam, const AlgCallback& alg_cb, void* hd)
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
    std::string configPath = (exePath.parent_path() / "Configs"/ "Alg" / "ObjectClassifyConfig.conf").string();

    // 4. 加载配置文件
    if (!loadConfig(configPath)) {
        LOG(ERROR) << "Failed to load config file: " << configPath;
        return false;
    }

    m_run_status = m_pConfig->getPoseConfig().yolo_model_config().run_status();

    // 5. 初始化模块
    if (!initModules()) {
        LOG(ERROR) << "Failed to initialize modules";
        return false;
    }
    LOG(INFO) << "CObjectClassifyAlg::initAlgorithm status: successs ";
    return true;
}

void CObjectClassifyAlg::runAlgorithm(void* p_pSrcData)
{
    LOG(INFO) << "CObjectClassifyAlg::runAlgorithm status: start ";
    
    try {
        // 0. 每次运行前重置结构体内容 - 手动初始化避免构造函数问题
        LOG(INFO) << "开始初始化m_currentOutput...";
        
        // 手动清空成员变量
        m_currentOutput.vecFrameResult().clear();
        m_currentOutput.mapTimeStamp().clear();
        m_currentOutput.mapDelay().clear();
        m_currentOutput.mapFps().clear();
        
        // 设置基本成员变量
        m_currentOutput.eDataType() = 0;
        m_currentOutput.eDataSourceType() = 0;
        m_currentOutput.unFrameId() = 0;
        m_currentOutput.lTimeStamp() = 0;
        
        LOG(INFO) << "m_currentOutput初始化完成";

        // 1. 核验输入数据是否为空
        if (!p_pSrcData) {
            LOG(ERROR) << "Input data is null";
            if (m_algCallback) {
                m_algCallback(m_currentOutput, m_callbackHandle);
            }
            return;
        }
        LOG(INFO) << "输入数据验证完成";

        // 2. 输入数据赋值
        m_currentInput = static_cast<CAlgResult *>(p_pSrcData);   
        LOG(INFO) << "输入数据赋值完成";
        
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
        LOG(INFO) << "CObjectClassifyAlg::runAlgorithm status: success ";
        
    } catch (const std::exception& e) {
        LOG(ERROR) << "CObjectClassifyAlg::runAlgorithm exception: " << e.what();
        // 确保输出结果有基本结构，避免访问空向量
        if (m_currentOutput.vecFrameResult().empty()) {
            CFrameResult frameResult;
            m_currentOutput.vecFrameResult().push_back(frameResult);
        }
        
        // 确保FrameResult有基本结构
        if (m_currentOutput.vecFrameResult().size() > 0) {
            CFrameResult& frameResult = m_currentOutput.vecFrameResult()[0];
            if (frameResult.vecObjectResult().empty()) {
                // 添加一个空的对象结果，避免后续访问时越界
                CObjectResult emptyObj;
                frameResult.vecObjectResult().push_back(emptyObj);
            }
        }
        
        if (m_algCallback) {
            m_algCallback(m_currentOutput, m_callbackHandle);
        }
    } catch (...) {
        LOG(ERROR) << "CObjectClassifyAlg::runAlgorithm unknown exception";
        // 确保输出结果有基本结构，避免访问空向量
        if (m_currentOutput.vecFrameResult().empty()) {
            CFrameResult frameResult;
            m_currentOutput.vecFrameResult().push_back(frameResult);
        }
        
        // 确保FrameResult有基本结构
        if (m_currentOutput.vecFrameResult().size() > 0) {
            CFrameResult& frameResult = m_currentOutput.vecFrameResult()[0];
            if (frameResult.vecObjectResult().empty()) {
                // 添加一个空的对象结果，避免后续访问时越界
                CObjectResult emptyObj;
                frameResult.vecObjectResult().push_back(emptyObj);
            }
        }
        
        if (m_algCallback) {
            m_algCallback(m_currentOutput, m_callbackHandle);
        }
    }

    return;
}

// 加载配置文件
bool CObjectClassifyAlg::loadConfig(const std::string& configPath)
{
    return m_pConfig->loadFromFile(configPath);
}

// 初始化子模块
bool CObjectClassifyAlg::initModules()
{   
    LOG(INFO) << "CObjectClassifyAlg::initModules status: start ";
    auto& poseConfig = m_pConfig->getPoseConfig();
    const common::ModulesConfig& modules = poseConfig.modules_config();
    m_moduleChain.clear();

    // 遍历所有模块，按顺序实例化
    for (const common::ModuleConfig& mod : modules.modules()) {
        auto module = ModuleFactory::getInstance().createModule("ObjectClassify", mod.name(), m_exePath);
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
    LOG(INFO) << "CObjectClassifyAlg::initModules status: success ";
    return true;
}

// 执行模块链
bool CObjectClassifyAlg::executeModuleChain()
{   
    int64_t beginTimeStamp = GetTimeStamp();
    int64_t preprocessEndTimeStamp = 0;
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
        
        // 根据模块类型处理输出数据
        if (module->getModuleName() == "ImagePreProcessGPU") {
            // ImagePreProcessGPU 输出 MultiImagePreprocessResultGPU
            preprocessEndTimeStamp = GetTimeStamp();
            LOG(INFO) << "ImagePreProcessGPU completed, output type: MultiImagePreprocessResultGPU. [DELAY_TYPE_OBJECTCLASSIFYALG_PREPROCESS] : " << preprocessEndTimeStamp - beginTimeStamp;
        } else if (module->getModuleName() == "Yolov11ClassifyGPU") {
            // Yolov11ClassifyGPU 输出 CAlgResult，这是最终结果
            LOG(INFO) << "Yolov11ClassifyGPU completed, output type: CAlgResult. [DELAY_TYPE_OBJECTCLASSIFYALG_INFERENCE] : " << GetTimeStamp() - preprocessEndTimeStamp;
        }
    }

    // 最后一个模块应该是 Yolov11ClassifyGPU，输出 CAlgResult
    int64_t endTimeStamp = GetTimeStamp();    
    if (currentData) {
        CAlgResult* resultPtr = static_cast<CAlgResult *>(currentData);
        m_currentOutput = *resultPtr;
    } else {
        LOG(ERROR) << "No valid output from module chain";
        return false;
    }

    // 结果穿透 - 修正数据访问逻辑
    if (m_currentInput && !m_currentInput->vecFrameResult().empty() && 
        !m_currentInput->vecFrameResult()[0].vecVideoSrcData().empty()) {
        m_currentOutput.lTimeStamp() = m_currentInput->vecFrameResult()[0].vecVideoSrcData()[0].lTimeStamp();
        
        if(m_currentOutput.vecFrameResult().size() > 0) 
        {   
            // 输入数据常规信息穿透
            m_currentOutput.vecFrameResult()[0].unFrameId() = m_currentInput->vecFrameResult()[0].vecVideoSrcData()[0].unFrameId();
            m_currentOutput.vecFrameResult()[0].mapTimeStamp() = m_currentInput->vecFrameResult()[0].vecVideoSrcData()[0].mapTimeStamp();
            m_currentOutput.vecFrameResult()[0].mapDelay() = m_currentInput->vecFrameResult()[0].vecVideoSrcData()[0].mapDelay();
            m_currentOutput.vecFrameResult()[0].mapFps() = m_currentInput->vecFrameResult()[0].vecVideoSrcData()[0].mapFps();

            LOG(INFO) << "原有信息穿透完毕： FrameId : " << m_currentOutput.vecFrameResult()[0].unFrameId() << ", lTimeStamp : " << m_currentOutput.lTimeStamp();

            // 独有数据填充
            m_currentOutput.vecFrameResult()[0].mapTimeStamp()[TIMESTAMP_OBJECTCLASSIFYALG_BEGIN] = beginTimeStamp;            // 目标分类算法开始时间戳
            m_currentOutput.vecFrameResult()[0].eDataType(DATA_TYPE_OBJECTCLASSIFYALG_RESULT);                                 // 数据类型赋值
            m_currentOutput.vecFrameResult()[0].mapTimeStamp()[TIMESTAMP_OBJECTCLASSIFYALG_END] = endTimeStamp;                // 目标分类算法结束时间戳
            m_currentOutput.vecFrameResult()[0].mapDelay()[DELAY_TYPE_OBJECTCLASSIFYALG] = endTimeStamp - beginTimeStamp;    // 目标分类算法耗时计算
            LOG(INFO) << "[DELAY_TYPE_OBJECTCLASSIFYALG] : " << m_currentOutput.vecFrameResult()[0].mapDelay()[DELAY_TYPE_OBJECTCLASSIFYALG];
            // 修正视差数据访问
            if (!m_currentInput->vecFrameResult()[0].tCameraSupplement().vecDistanceInfo().empty()) {
                m_currentOutput.vecFrameResult()[0].tCameraSupplement() = m_currentInput->vecFrameResult()[0].tCameraSupplement();
            }
        }
    }

    // 结果可视化 - 在坐标转换之前进行，使用子图坐标系
    if(m_run_status)
    {
        visualizationResult();
    }

    // 坐标转换和结果合并
    convertCoordinatesAndMergeResults();

    return true;
} 


void CObjectClassifyAlg::visualizationResult()
{
    // 检查输入数据有效性
    if (!m_currentInput || m_currentInput->vecFrameResult().empty() || 
        m_currentInput->vecFrameResult()[0].vecVideoSrcData().empty()) {
        LOG(ERROR) << "No valid input data for visualization";
        return;
    }
    
    if (m_currentOutput.vecFrameResult().empty()) {
        LOG(ERROR) << "No output data for visualization";
        return;
    }
    
    // 获取目标分类结果（多个FrameResult，每个包含一个ObjectResult）
    const auto& classifyFrameResults = m_currentOutput.vecFrameResult();
    
    // 获取所有子图数据
    const auto& allVideoSrcData = m_currentInput->vecFrameResult()[0].vecVideoSrcData();
    LOG(INFO) << "开始可视化，子图数量: " << allVideoSrcData.size() 
              << ", 目标分类结果数量: " << classifyFrameResults.size();
    
    // 创建合并的可视化图像
    createCombinedVisualization(allVideoSrcData, classifyFrameResults, 
                               m_currentInput->vecFrameResult()[0].unFrameId());
}

void CObjectClassifyAlg::createCombinedVisualization(const std::vector<CVideoSrcData>& allVideoSrcData, 
                                                    const std::vector<CFrameResult>& classifyFrameResults, 
                                                    uint32_t frameId)
{
    // 计算合并图像的尺寸
    int totalWidth = 0;
    int maxHeight = 0;
    std::vector<cv::Mat> subImages;
    
    // 收集所有子图并绘制分类结果
    for (size_t subImgIdx = 0; subImgIdx < allVideoSrcData.size(); ++subImgIdx) {
        const auto& videoSrc = allVideoSrcData[subImgIdx];
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
            cv::cvtColor(srcImage, srcImage, cv::COLOR_GRAY2BGR);
        } else {
            LOG(ERROR) << "Unsupported image channel: " << channels << " for sub-image " << subImgIdx;
            continue;
        }
        
        // 查找属于当前子图的分类结果
        const CObjectResult* targetObj = nullptr;
        
        if (subImgIdx < classifyFrameResults.size()) {
            // 按索引分配分类结果
            const auto& frameResult = classifyFrameResults[subImgIdx];
            if (!frameResult.vecObjectResult().empty()) {
                targetObj = &frameResult.vecObjectResult()[0];
                LOG(INFO) << "子图 " << subImgIdx << " 使用分类结果索引 " << subImgIdx;
            } else {
                LOG(WARNING) << "子图 " << subImgIdx << " 的FrameResult中没有ObjectResult";
            }
        } else {
            LOG(WARNING) << "子图 " << subImgIdx << " 没有对应的分类结果";
        }
        
        // 绘制对应的分类结果
        if (targetObj) {
            // 在图像中心绘制分类结果信息
            std::string label = "Class: " + targetObj->strClass();
            if (targetObj->fVideoConfidence() > 0.0f) {
                label += " (" + std::to_string(static_cast<int>(targetObj->fVideoConfidence() * 100)) + "%)";
            }
            
            // 计算文本位置（图像中心）
            int fontFace = cv::FONT_HERSHEY_SIMPLEX;
            double fontScale = 1.0;
            int thickness = 2;
            cv::Size textSize = cv::getTextSize(label, fontFace, fontScale, thickness, nullptr);
            
            int textX = (width - textSize.width) / 2;
            int textY = (height + textSize.height) / 2;
            
            // 绘制文本背景
            cv::Rect textRect(textX - 10, textY - textSize.height - 10, 
                            textSize.width + 20, textSize.height + 20);
            cv::rectangle(srcImage, textRect, cv::Scalar(0, 0, 0), -1);
            cv::rectangle(srcImage, textRect, cv::Scalar(255, 255, 255), 2);
            
            // 绘制分类文本
            cv::putText(srcImage, label, cv::Point(textX, textY), 
                       fontFace, fontScale, cv::Scalar(0, 255, 0), thickness);
            
            LOG(INFO) << "子图 " << subImgIdx << " 分类结果: " << label;
        }
        
        subImages.push_back(srcImage);
        totalWidth += width;
        maxHeight = std::max(maxHeight, height);
    }
    
    // 创建合并图像
    cv::Mat combinedImage(maxHeight, totalWidth, CV_8UC3, cv::Scalar(0, 0, 0));
    
    int currentX = 0;
    for (size_t i = 0; i < subImages.size(); ++i) {
        cv::Mat roi = combinedImage(cv::Rect(currentX, 0, subImages[i].cols, subImages[i].rows));
        subImages[i].copyTo(roi);
        currentX += subImages[i].cols;
    }
    
    // 保存合并图像
    std::string visDir = (std::filesystem::path(m_exePath) / "Vis_ObjectClassify_Result").string();
    if (!std::filesystem::exists(visDir)) {
        std::filesystem::create_directories(visDir);
    }
    
    std::string savePath = visDir + "/" + std::to_string(frameId) + ".jpg";
    cv::imwrite(savePath, combinedImage);
    LOG(INFO) << "合并可视化结果已保存到: " << savePath;
}

void CObjectClassifyAlg::convertCoordinatesAndMergeResults()
{
    LOG(INFO) << "开始结果合并...";
    LOG(INFO) << "输入目标检测结果数量: " << m_currentInput->vecFrameResult()[0].vecObjectResult().size();
    LOG(INFO) << "目标分类结果数量: " << m_currentOutput.vecFrameResult().size();
    
    // 获取目标检测结果（整图上的目标框）
    const auto& detectionResults = m_currentInput->vecFrameResult()[0].vecObjectResult();
    // 获取目标分类结果（多个FrameResult，每个包含一个ObjectResult）
    const auto& classifyResults = m_currentOutput.vecFrameResult();
    
    LOG(INFO) << "检测结果详细信息:";
    for (size_t i = 0; i < detectionResults.size(); ++i) {
        LOG(INFO) << "  检测结果 " << i << ": class=" << detectionResults[i].strClass() 
                  << ", conf=" << detectionResults[i].fVideoConfidence();
    }
    
    LOG(INFO) << "分类结果详细信息:";
    for (size_t i = 0; i < classifyResults.size(); ++i) {
        if (!classifyResults[i].vecObjectResult().empty()) {
            LOG(INFO) << "  分类结果 " << i << ": class=" << classifyResults[i].vecObjectResult()[0].strClass() 
                      << ", conf=" << classifyResults[i].vecObjectResult()[0].fVideoConfidence();
        } else {
            LOG(INFO) << "  分类结果 " << i << ": 空的ObjectResult";
        }
    }
    
    // 在清空容器前，先保存分类结果的副本
    std::vector<CFrameResult> classifyResultsCopy = classifyResults;
    
    // 清空输出容器，准备重新填充
    m_currentOutput.vecFrameResult().clear();
    
    // 创建一个新的FrameResult来存储合并后的结果
    CFrameResult mergedFrameResult;
    mergedFrameResult.eDataType(DATA_TYPE_OBJECTCLASSIFYALG_RESULT);
    
    // 遍历每个目标检测结果
    for (size_t detIdx = 0; detIdx < detectionResults.size(); ++detIdx) {
        const auto& detectionObj = detectionResults[detIdx];
        
        // 创建新的结果对象，基于目标检测结果
        CObjectResult mergedObj = detectionObj;
        
        LOG(INFO) << "处理目标 " << detIdx << ", 检测类别: " << detectionObj.strClass();
        
        // 查找对应的目标分类结果
        // 目标检测和分类的索引是一一对应的，因为每个子图对应一个目标
        if (detIdx < classifyResultsCopy.size()) {
            const auto& classifyFrameResult = classifyResultsCopy[detIdx];
            if (!classifyFrameResult.vecObjectResult().empty()) {
                const auto& classifyObj = classifyFrameResult.vecObjectResult()[0];
                
                // 获取检测和分类的类别信息
                std::string detectionClass = detectionObj.strClass();
                std::string classifyClass = classifyObj.strClass();

                mergedObj.strClass(classifyClass);
                mergedObj.fVideoConfidence(classifyObj.fVideoConfidence());
                
                LOG(INFO) << "目标 " << detIdx << " 分类结果: " << classifyClass 
                          << " (置信度: " << classifyObj.fVideoConfidence() << ")";
            } else {
                LOG(WARNING) << "目标 " << detIdx << " 的FrameResult中没有ObjectResult";
            }
        } else {
            // 如果没有对应的分类结果，保留目标检测结果
            LOG(WARNING) << "目标 " << detIdx << " 没有对应的分类结果 (detIdx=" << detIdx 
                         << ", classifyResultsCopy.size()=" << classifyResultsCopy.size() << ")";
        }
        
        // 添加到合并的FrameResult中
        mergedFrameResult.vecObjectResult().push_back(mergedObj);
    }
    
    // 将合并后的FrameResult添加到输出中
    m_currentOutput.vecFrameResult().push_back(mergedFrameResult);
    
    LOG(INFO) << "结果合并完成，输出结果数量: " 
              << m_currentOutput.vecFrameResult()[0].vecObjectResult().size();
}