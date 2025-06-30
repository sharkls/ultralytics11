/*******************************************************
 文件名：CPoseEstimationAlg.cpp
 作者：
 描述：姿态估计算法主类实现
 版本：v1.0
 日期：2024-03-21
 *******************************************************/

#include "CPoseEstimationAlg.h"
#include <algorithm>

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
    std::string configPath = (exePath.parent_path() / "Configs"/ "Alg" / "PoseEstimationConfigv2.conf").string();

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
    LOG(INFO) << "CPoseEstimationAlg::initAlgorithm status: successs ";
    return true;
}

void CPoseEstimationAlg::runAlgorithm(void* p_pSrcData)
{
    LOG(INFO) << "CPoseEstimationAlg::runAlgorithm status: start ";
    
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
        LOG(INFO) << "CPoseEstimationAlg::runAlgorithm status: success ";
        
    } catch (const std::exception& e) {
        LOG(ERROR) << "CPoseEstimationAlg::runAlgorithm exception: " << e.what();
        if (m_algCallback) {
            m_algCallback(m_currentOutput, m_callbackHandle);
        }
    } catch (...) {
        LOG(ERROR) << "CPoseEstimationAlg::runAlgorithm unknown exception";
        if (m_algCallback) {
            m_algCallback(m_currentOutput, m_callbackHandle);
        }
    }

    return;
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
    int64_t beginTimeStamp = GetTimeStamp();
    m_currentOutput.mapTimeStamp()[TIMESTAMP_POSEALG_BEGIN] = beginTimeStamp;
    void* currentData = static_cast<void *>(m_currentInput);

    int64_t preprocessEndTimeStamp = 0;

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
        if (module->getModuleName() == "ImagePreProcess" || module->getModuleName() == "ImagePreProcessGPU") {
            // ImagePreProcess 输出 MultiImagePreprocessResult，直接传递给下一个模块
            preprocessEndTimeStamp = GetTimeStamp();
            LOG(INFO) << "ImagePreProcess completed, output type: MultiImagePreprocessResult. [DELAY_TYPE_POSEALG_PREPROCESS] : " << preprocessEndTimeStamp - beginTimeStamp;
            // m_currentOutput.mapTimeStamp()[DELAY_TYPE_POSEALG_PREPROCESS] = preprocessEndTimeStamp - beginTimeStamp;
        } else if (module->getModuleName() == "Yolov11Pose" || module->getModuleName() == "Yolov11PoseGPU") {
            // Yolov11Pose 输出 CAlgResult，这是最终结果
            LOG(INFO) << "Yolov11Pose completed, output type: CAlgResult. [DELAY_TYPE_POSEALG_INFERENCE] : " << GetTimeStamp() - preprocessEndTimeStamp;
            // m_currentOutput.mapTimeStamp()[DELAY_TYPE_POSEALG_INFERENCE] = GetTimeStamp() - m_currentOutput.mapTimeStamp()[DELAY_TYPE_POSEALG_PREPROCESS];
        }
    }

    // 最后一个模块应该是 Yolov11Pose，输出 CAlgResult
    int64_t endTimeStamp = GetTimeStamp();    
    // 保存开始时间戳，避免被覆盖
    int64_t savedBeginTimeStamp = m_currentOutput.mapTimeStamp()[TIMESTAMP_POSEALG_BEGIN];
    
    if (currentData) {
        CAlgResult* resultPtr = static_cast<CAlgResult *>(currentData);
        m_currentOutput = *resultPtr;
        
        // 恢复开始时间戳
        m_currentOutput.mapTimeStamp()[TIMESTAMP_POSEALG_BEGIN] = savedBeginTimeStamp;
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
            m_currentOutput.vecFrameResult()[0].eDataType(DATA_TYPE_POSEALG_RESULT);                                 // 数据类型赋值
            m_currentOutput.vecFrameResult()[0].mapTimeStamp()[TIMESTAMP_POSEALG_END] = endTimeStamp;                // 姿态估计算法结束时间戳
            m_currentOutput.vecFrameResult()[0].mapDelay()[DELAY_TYPE_POSEALG] = endTimeStamp - m_currentOutput.mapTimeStamp()[TIMESTAMP_POSEALG_BEGIN];    // 姿态估计算法耗时计算
            LOG(INFO) << "[DELAY_TYPE_POSEALG] : " << m_currentOutput.vecFrameResult()[0].mapDelay()[DELAY_TYPE_POSEALG];
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

    // 添加目标深度值获取逻辑
    if (m_currentOutput.vecFrameResult().size() > 0) {
        LOG(INFO) << "开始计算关键点深度值...";
        auto& frameResult = m_currentOutput.vecFrameResult()[0];  // 现在只有一个FrameResult
        auto& objResults = frameResult.vecObjectResult();
        
        // 检查是否有视差数据
        if (!frameResult.tCameraSupplement().vecDistanceInfo().empty()) {
            const auto& disparity = frameResult.tCameraSupplement();
            int width = disparity.usWidth();
            int height = disparity.usHeight();
            const auto& depthMap = disparity.vecDistanceInfo();
            LOG(INFO) << "depthMap.size() : " << depthMap.size();
            
            for (auto& obj : objResults) {
                // 获取所有关键点的深度值
                std::vector<float> allKeypointDepths;
                const auto& keypoints = obj.vecKeypoints();
                
                for (const auto& kp : keypoints) {
                    // 获取关键点坐标（已经是整图坐标）
                    float kx = kp.x();
                    float ky = kp.y();
                    
                    // 像素坐标转整数下标
                    int kix = static_cast<int>(kx + 0.5f);
                    int kiy = static_cast<int>(ky + 0.5f);
                    
                    // 检查边界
                    if (kix >= 0 && kix < width && kiy >= 0 && kiy < height) {
                        // 遍历关键点周围 5×5 的区域
                        for (int dy = -2; dy <= 2; ++dy) {
                            for (int dx = -2; dx <= 2; ++dx) {
                                int currentIx = kix + dx;
                                int currentIy = kiy + dy;
                                
                                // 检查当前坐标是否在深度图范围内
                                if (currentIx >= 0 && currentIx < width && currentIy >= 0 && currentIy < height) {
                                    int idx = currentIy * width + currentIx;
                                    if (idx >= 0 && idx < depthMap.size()) {
                                        float depth = depthMap[idx];
                                        if (depth > 0.0f) {  // 只收集有效的深度值
                                            allKeypointDepths.push_back(depth);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                
                // 如果收集到足够的深度值
                if (!allKeypointDepths.empty()) {
                    // 对深度值进行排序
                    std::sort(allKeypointDepths.begin(), allKeypointDepths.end());
                    
                    // 计算四分位数 - 添加边界检查
                    size_t n = allKeypointDepths.size();
                    if (n >= 4) {  // 确保有足够的数据点
                        size_t q1_idx = static_cast<size_t>(n * 0.25);
                        size_t q3_idx = static_cast<size_t>(n * 0.75);
                        
                        // 确保索引在有效范围内
                        q1_idx = std::min(q1_idx, n - 1);
                        q3_idx = std::min(q3_idx, n - 1);
                        
                        float q1 = allKeypointDepths[q1_idx];
                        float q3 = allKeypointDepths[q3_idx];
                        float iqr = q3 - q1;
                        
                        // 定义异常值的界限
                        float lower_bound = q1 - 1.5 * iqr;
                        float upper_bound = q3 + 1.5 * iqr;
                        
                        // 过滤掉异常值
                        std::vector<float> filtered_depths;
                        for (float depth : allKeypointDepths) {
                            if (depth >= lower_bound && depth <= upper_bound) {
                                filtered_depths.push_back(depth);
                            }
                        }
                        
                        if (!filtered_depths.empty()) {
                            // 计算中位数作为最终深度值
                            size_t mid = filtered_depths.size() / 2;
                            float median_depth;
                            if (filtered_depths.size() % 2 == 0) {
                                median_depth = (filtered_depths[mid - 1] + filtered_depths[mid]) / 2.0f;
                            } else {
                                median_depth = filtered_depths[mid];
                            }
                            
                            obj.fDistance() = median_depth;  // 赋值距离
                            LOG(INFO) << "目标距离（基于关键点）： " << obj.fDistance() << " mm";
                        }
                    } else {
                        // 数据点不足时，使用平均值
                        float avg_depth = 0.0f;
                        for (float depth : allKeypointDepths) {
                            avg_depth += depth;
                        }
                        avg_depth /= allKeypointDepths.size();
                        obj.fDistance() = avg_depth;
                        LOG(INFO) << "目标距离（基于关键点平均值）： " << obj.fDistance() << " mm";
                    }
                }
            }
        } else {
            LOG(WARNING) << "没有可用的视差数据，跳过深度值计算";
        }
    }

    return true;
} 


void CPoseEstimationAlg::visualizationResult()
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
    
    // 获取姿态估计结果（子图坐标系）
    auto& frameResult = m_currentOutput.vecFrameResult()[0];
    const auto& objResults = frameResult.vecObjectResult();
    
    // 获取所有子图数据
    const auto& allVideoSrcData = m_currentInput->vecFrameResult()[0].vecVideoSrcData();
    LOG(INFO) << "开始可视化，子图数量: " << allVideoSrcData.size() << ", 姿态估计结果数量: " << objResults.size();
    
    // 创建合并的可视化图像
    createCombinedVisualization(allVideoSrcData, objResults, frameResult.unFrameId());
}

void CPoseEstimationAlg::createCombinedVisualization(const std::vector<CVideoSrcData>& allVideoSrcData, 
                                                    const std::vector<CObjectResult>& objResults, 
                                                    uint32_t frameId)
{
    // 计算合并图像的尺寸
    int totalWidth = 0;
    int maxHeight = 0;
    std::vector<cv::Mat> subImages;
    
    // 收集所有子图并绘制检测结果
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
        
        // 绘制对应的检测结果
        if (subImgIdx < objResults.size()) {
            const auto& obj = objResults[subImgIdx];
            
            // 直接使用姿态估计结果中的边界框坐标（已经是子图坐标系）
            float bbox_x1 = obj.fTopLeftX();
            float bbox_y1 = obj.fTopLeftY();
            float bbox_x2 = obj.fBottomRightX();
            float bbox_y2 = obj.fBottomRightY();
            
            // 添加调试信息
            LOG(INFO) << "子图 " << subImgIdx << " 边界框坐标: (" << bbox_x1 << ", " << bbox_y1 << ") - (" << bbox_x2 << ", " << bbox_y2 << ")";
            LOG(INFO) << "子图 " << subImgIdx << " 图像尺寸: " << width << "x" << height;
            
            // 改进的边界检查：允许部分超出边界的边界框，只要中心点在图像内即可
            float bbox_center_x = (bbox_x1 + bbox_x2) / 2.0f;
            float bbox_center_y = (bbox_y1 + bbox_y2) / 2.0f;
            
            // 检查边界框中心是否在图像内，或者边界框是否与图像有重叠
            bool bbox_valid = (bbox_center_x >= 0 && bbox_center_x < width && 
                              bbox_center_y >= 0 && bbox_center_y < height) ||
                             (bbox_x1 < width && bbox_x2 > 0 && 
                              bbox_y1 < height && bbox_y2 > 0);
            
            if (bbox_valid) {
                // 裁剪边界框到图像范围内
                float clamped_x1 = std::max(0.0f, bbox_x1);
                float clamped_y1 = std::max(0.0f, bbox_y1);
                float clamped_x2 = std::min(static_cast<float>(width), bbox_x2);
                float clamped_y2 = std::min(static_cast<float>(height), bbox_y2);
                
                // 绘制目标框
                cv::Rect rect(
                    cv::Point(static_cast<int>(clamped_x1), static_cast<int>(clamped_y1)),
                    cv::Point(static_cast<int>(clamped_x2), static_cast<int>(clamped_y2))
                );
                cv::rectangle(srcImage, rect, cv::Scalar(0, 255, 0), 2);

                // 绘制类别、置信度和深度值
                std::string label = obj.strClass() + " " + std::to_string(obj.fVideoConfidence());
                if (obj.fDistance() > 0.0f) {
                    label += " " + std::to_string(static_cast<int>(obj.fDistance())) + "mm";
                }
                cv::putText(srcImage, label, rect.tl() + cv::Point(0, -10), 
                           cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,255,0), 1);

                // 绘制人体关键点（子图坐标系）
                const auto& keypoints = obj.vecKeypoints();
                for(const auto& kp : keypoints) {
                    // 边界检查（相对于子图）- 允许关键点稍微超出边界
                    if (kp.x() < -10 || kp.y() < -10 || kp.x() >= width + 10 || kp.y() >= height + 10) {
                        continue;
                    }
                    
                    // 根据置信度决定是否绘制关键点
                    if (kp.confidence() > 0.1f) {  // 只绘制置信度大于0.1的关键点
                        cv::Point2f pt(static_cast<int>(kp.x()), static_cast<int>(kp.y()));
                        
                        // 根据置信度调整颜色和大小
                        int radius = static_cast<int>(3 + kp.confidence() * 5);  // 半径根据置信度调整
                        cv::Scalar color;
                        if (kp.confidence() > 0.7f) {
                            color = cv::Scalar(0, 0, 255);  // 红色 - 高置信度
                        } else if (kp.confidence() > 0.4f) {
                            color = cv::Scalar(0, 255, 255);  // 黄色 - 中等置信度
                        } else {
                            color = cv::Scalar(128, 128, 128);  // 灰色 - 低置信度
                        }
                        
                        cv::circle(srcImage, pt, radius, color, -1);
                    }
                }
                
                // 绘制关键点连线（人体骨架）
                if (keypoints.size() >= 17) {  // COCO格式有17个关键点
                    // 定义关键点连接关系（COCO格式）- 正确的骨架连线
                    std::vector<std::pair<int, int>> connections = {
                        {16, 14}, {14, 12}, {17, 15}, {15, 13}, {12, 13},  // 躯干和腿部
                        {6, 12}, {7, 13}, {6, 7},                          // 肩部和躯干
                        {6, 8}, {7, 9}, {8, 10}, {9, 11},                  // 手臂
                        {2, 3}, {1, 2}, {1, 3}, {2, 4}, {3, 5}, {4, 6}, {5, 7}  // 头部和颈部
                    };
                    
                    for (const auto& connection : connections) {
                        int idx1 = connection.first;
                        int idx2 = connection.second;
                        
                        if (idx1 < keypoints.size() && idx2 < keypoints.size()) {
                            const auto& kp1 = keypoints[idx1];
                            const auto& kp2 = keypoints[idx2];
                            
                            // 只连接置信度足够高的关键点
                            if (kp1.confidence() > 0.3f && kp2.confidence() > 0.3f) {
                                cv::Point2f pt1(static_cast<int>(kp1.x()), static_cast<int>(kp1.y()));
                                cv::Point2f pt2(static_cast<int>(kp2.x()), static_cast<int>(kp2.y()));
                                
                                // 边界检查（相对于子图）- 允许连线稍微超出边界
                                if (pt1.x >= -10 && pt1.y >= -10 && pt1.x < width + 10 && pt1.y < height + 10 &&
                                    pt2.x >= -10 && pt2.y >= -10 && pt2.x < width + 10 && pt2.y < height + 10) {
                                    cv::line(srcImage, pt1, pt2, cv::Scalar(255, 0, 0), 2);
                                }
                            }
                        }
                    }
                }
                
                LOG(INFO) << "子图 " << subImgIdx << " 绘制完成，关键点数: " << keypoints.size();
            } else {
                LOG(WARNING) << "子图 " << subImgIdx << " 目标框完全超出图像边界，跳过绘制";
                LOG(WARNING) << "边界框中心: (" << bbox_center_x << ", " << bbox_center_y << ")";
            }
        } else {
            LOG(WARNING) << "子图 " << subImgIdx << " 没有对应的检测结果";
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
    std::string visDir = (std::filesystem::path(m_exePath) / "Vis_PoseEstimation_Result").string();
    if (!std::filesystem::exists(visDir)) {
        std::filesystem::create_directories(visDir);
    }
    
    std::string savePath = visDir + "/" + std::to_string(frameId) + ".jpg";
    cv::imwrite(savePath, combinedImage);
    LOG(INFO) << "合并可视化结果已保存到: " << savePath;
}

void CPoseEstimationAlg::convertCoordinatesAndMergeResults()
{
    LOG(INFO) << "开始坐标转换和结果合并...";
    LOG(INFO) << "输入目标检测结果数量: " << m_currentInput->vecFrameResult()[0].vecObjectResult().size();
    LOG(INFO) << "姿态估计结果数量: " << m_currentOutput.vecFrameResult()[0].vecObjectResult().size();
    
    // 获取目标检测结果（整图上的目标框）
    const auto& detectionResults = m_currentInput->vecFrameResult()[0].vecObjectResult();
    // 获取姿态估计结果（子图上的关键点）
    auto& poseResults = m_currentOutput.vecFrameResult()[0].vecObjectResult();
    
    // 在清空容器前，先保存姿态估计结果的副本
    std::vector<CObjectResult> poseResultsCopy = poseResults;
    
    // 清空输出容器，准备重新填充
    m_currentOutput.vecFrameResult()[0].vecObjectResult().clear();
    
    // 遍历每个目标检测结果
    for (size_t detIdx = 0; detIdx < detectionResults.size(); ++detIdx) {
        const auto& detectionObj = detectionResults[detIdx];
        
        // 创建新的结果对象，基于目标检测结果
        CObjectResult mergedObj = detectionObj;
        
        // 查找对应的姿态估计结果
        // 目标检测和姿态估计的索引是一一对应的，因为每个子图对应一个目标
        if (detIdx < poseResultsCopy.size()) {
            const auto& poseObj = poseResultsCopy[detIdx];
            
            // 获取目标框信息（用于坐标转换）
            float bbox_x1 = detectionObj.fTopLeftX();
            float bbox_y1 = detectionObj.fTopLeftY();
            float bbox_x2 = detectionObj.fBottomRightX();
            float bbox_y2 = detectionObj.fBottomRightY();
            
            // 转换关键点坐标从子图到整图
            std::vector<Keypoint> convertedKeypoints;
            const auto& originalKeypoints = poseObj.vecKeypoints();
            
            for (const auto& kp : originalKeypoints) {
                Keypoint convertedKp;
                
                // 姿态估计输出的坐标是相对于子图的绝对像素坐标
                float sub_x = kp.x();
                float sub_y = kp.y();
                
                // 将子图坐标转换为整图坐标
                // 子图坐标是相对于子图的，需要加上子图在整图中的偏移
                // 由于每个子图都是独立的，子图坐标就是相对于子图左上角的坐标
                // 而目标检测的边界框是相对于整图的，所以需要加上边界框的左上角坐标
                float full_x = bbox_x1 + sub_x;
                float full_y = bbox_y1 + sub_y;
                
                convertedKp.x(full_x);
                convertedKp.y(full_y);
                convertedKp.confidence(kp.confidence());
                convertedKeypoints.push_back(convertedKp);
            }
            
            // 设置转换后的关键点
            mergedObj.vecKeypoints(convertedKeypoints);
            
            // 计算平均置信度
            float avg_confidence = (detectionObj.fVideoConfidence() + poseObj.fVideoConfidence()) / 2.0f;
            mergedObj.fVideoConfidence(avg_confidence);
        } else {
            // 如果没有对应的姿态估计结果，保留目标检测结果，关键点为空
            LOG(WARNING) << "目标 " << detIdx << " 没有对应的姿态估计结果";
        }
        
        // 添加到输出结果中
        m_currentOutput.vecFrameResult()[0].vecObjectResult().push_back(mergedObj);
    }
    
    LOG(INFO) << "坐标转换和结果合并完成，输出结果数量: " 
              << m_currentOutput.vecFrameResult()[0].vecObjectResult().size();
}