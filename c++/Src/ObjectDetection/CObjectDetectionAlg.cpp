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
        m_currentOutput.vecFrameResult()[0].eDataType(DATA_TYPE_MMALG_RESULT);                                 
        m_currentOutput.vecFrameResult()[0].mapTimeStamp()[TIMESTAMP_MMALG_END] = endTimeStamp;                
        m_currentOutput.vecFrameResult()[0].mapDelay()[DELAY_TYPE_MMALG] = endTimeStamp - m_currentOutput.mapTimeStamp()[TIMESTAMP_POSEALG_BEGIN];    
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
                // std::cout << "5x5区域深度值：" << std::endl;
                for (int dy = -2; dy <= 2; ++dy) {
                    for (int dx = -2; dx <= 2; ++dx) {
                        int currentIx = ix + dx;
                        int currentIy = iy + dy;

                        // 检查当前坐标是否在深度图范围内
                        if (currentIx >= 0 && currentIx < width && currentIy >= 0 && currentIy < height) {
                            int idx = currentIy * width + currentIx;
                            if (idx >= 0 && idx < depthMap.size()) {
                                depthValues.push_back(depthMap[idx]);
                                // std::cout << std::fixed << std::setprecision(2) << depthMap[idx] << "\t";
                            }
                        } 
                        // else {
                        //     std::cout << "N/A\t";
                        // }
                    }
                    // std::cout << std::endl;
                }
                // std::cout << std::endl;
                

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

        // 保存目标框区域图像到 vecVideoSrcData
        saveObjectRegionImages();
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

void CObjectDetectionAlg::saveObjectRegionImages()
{
    if (m_currentOutput.vecFrameResult().empty()) {
        LOG(ERROR) << "No frame result available for saving object region images";
        return;
    }

    auto& frameResult = m_currentOutput.vecFrameResult()[0];
    const auto& objResults = frameResult.vecObjectResult();
    
    if (objResults.empty()) {
        LOG(INFO) << "No objects detected, skipping region image extraction";
        return;
    }

    // 获取原始图像
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

    // 清空现有的 vecVideoSrcData
    frameResult.vecVideoSrcData().clear();

    // 为每个检测到的目标提取区域图像
    for (size_t i = 0; i < objResults.size(); ++i) {
        const auto& obj = objResults[i];
        
        // 计算边界框坐标
        int x1 = static_cast<int>(obj.fTopLeftX());
        int y1 = static_cast<int>(obj.fTopLeftY());
        int x2 = static_cast<int>(obj.fBottomRightX());
        int y2 = static_cast<int>(obj.fBottomRightY());

        // 确保坐标在有效范围内
        x1 = std::max(0, std::min(x1, width - 1));
        y1 = std::max(0, std::min(y1, height - 1));
        x2 = std::max(0, std::min(x2, width - 1));
        y2 = std::max(0, std::min(y2, height - 1));

        // 确保边界框有效
        if (x2 <= x1 || y2 <= y1) {
            LOG(WARNING) << "Invalid bounding box for object " << i << ": (" << x1 << "," << y1 << ") to (" << x2 << "," << y2 << ")";
            continue;
        }

        // 提取目标区域图像
        cv::Mat roiImage = srcImage(cv::Rect(x1, y1, x2 - x1, y2 - y1)).clone();

        // 当 m_run_status 为 true 时，保存目标区域图像
        if (m_run_status) {
            // 构造保存目录
            std::string roiVisDir = (std::filesystem::path(m_exePath) / "Vis_Object_Regions").string();
            if (!std::filesystem::exists(roiVisDir)) {
                std::filesystem::create_directories(roiVisDir);
            }

            // 在图像上添加标注信息
            cv::Mat annotatedImage = roiImage.clone();
            
            // 添加目标信息文本
            std::string infoText = obj.strClass() + " Conf:" + std::to_string(static_cast<int>(obj.fVideoConfidence()));
            if (obj.fDistance() > 0) {
                infoText += " Dist:" + std::to_string(static_cast<int>(obj.fDistance())) + "cm";
            }
            
            // 计算文本位置和大小
            int fontFace = cv::FONT_HERSHEY_SIMPLEX;
            double fontScale = 0.5;
            int thickness = 1;
            cv::Size textSize = cv::getTextSize(infoText, fontFace, fontScale, thickness, nullptr);
            
            // 绘制背景矩形
            cv::Rect textRect(5, 5, textSize.width + 10, textSize.height + 10);
            cv::rectangle(annotatedImage, textRect, cv::Scalar(0, 0, 0), -1);
            cv::rectangle(annotatedImage, textRect, cv::Scalar(255, 255, 255), 1);
            
            // 绘制文本
            cv::putText(annotatedImage, infoText, cv::Point(10, 20), fontFace, fontScale, 
                       cv::Scalar(255, 255, 255), thickness);

            // 构造保存路径：帧ID_目标索引_类别.jpg
            uint32_t frameId = m_currentInput->vecVideoSrcData()[0].unFrameId();
            std::string className = obj.strClass();
            // 替换类别名称中的特殊字符，避免文件名问题
            std::replace(className.begin(), className.end(), ' ', '_');
            std::string savePath = roiVisDir + "/" + std::to_string(frameId) + "_" + 
                                  std::to_string(i) + "_" + className + ".jpg";

            // 保存图片
            if (cv::imwrite(savePath, annotatedImage)) {
                LOG(INFO) << "目标区域图像已保存到: " << savePath;
            } else {
                LOG(ERROR) << "保存目标区域图像失败: " << savePath;
            }
        }

        // 创建 CVideoSrcData 对象
        CVideoSrcData videoData;
        videoData.ucCameraId(static_cast<uint8_t>(i));  // 使用目标索引作为相机ID
        videoData.usBmpWidth(roiImage.cols);
        videoData.usBmpLength(roiImage.rows);
        videoData.unBmpBytes(roiImage.total() * roiImage.elemSize());
        videoData.unFrameId(m_currentInput->vecVideoSrcData()[0].unFrameId());
        videoData.lTimeStamp(m_currentInput->vecVideoSrcData()[0].lTimeStamp());

        // 设置 CDataBase 继承的字段
        videoData.eDataType(DATA_TYPE_RGB_IMAGE);  // 设置为RGB图像数据类型
        videoData.eDataSourceType(0);  // 设置为双目数据源类型
        videoData.mapTimeStamp() = m_currentInput->vecVideoSrcData()[0].mapTimeStamp();
        videoData.mapDelay() = m_currentInput->vecVideoSrcData()[0].mapDelay();
        videoData.mapFps() = m_currentInput->vecVideoSrcData()[0].mapFps();

        // 将图像数据转换为 vector<uint8_t>
        std::vector<uint8_t> imageData(roiImage.data, roiImage.data + roiImage.total() * roiImage.elemSize());
        videoData.vecImageBuf(imageData);

        // 将提取的区域图像添加到 vecVideoSrcData
        frameResult.vecVideoSrcData().push_back(videoData);

        LOG(INFO) << "Extracted region image for object " << i 
                  << " (" << obj.strClass() << "): " 
                  << roiImage.cols << "x" << roiImage.rows 
                  << " from (" << x1 << "," << y1 << ") to (" << x2 << "," << y2 << ")";
    }

    LOG(INFO) << "Successfully extracted " << frameResult.vecVideoSrcData().size() 
              << " object region images for secondary detection";
    
    // 如果启用了离线保存，显示保存信息
    if (m_run_status && !objResults.empty()) {
        LOG(INFO) << "Saved " << objResults.size() << " object region images to Vis_Object_Regions directory";
    }
} 