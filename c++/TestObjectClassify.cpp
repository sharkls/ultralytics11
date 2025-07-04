#include <iostream>
#include <iomanip>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <vector>
#include <sstream>
#include "log.h"

#include "CAlgResult.h"
#include "CMultiModalSrcData.h"
#include "ExportObjectClassifyAlgLib.h"
#include "GlobalContext.h"
#include "FunctionHub.h"

// 全局变量
std::string g_save_dir;

// 目标框信息结构体
struct DetectionBox {
    std::string className;
    float confidence;
    float x1, y1, x2, y2;
    int targetId;
    float distance;
};

// 解析目标检测的txt文件
std::vector<DetectionBox> parseDetectionTxt(const std::string& txtPath) {
    std::vector<DetectionBox> detections;
    std::ifstream file(txtPath);
    
    if (!file.is_open()) {
        LOG(ERROR) << "无法打开检测结果文件: " << txtPath;
        return detections;
    }
    
    std::string line;
    int lineCount = 0;
    
    while (std::getline(file, line)) {
        lineCount++;
        
        // 跳过注释行和空行
        if (line.empty() || line[0] == '#') {
            continue;
        }
        
        std::istringstream iss(line);
        DetectionBox box;
        
        if (iss >> box.className >> box.confidence >> box.x1 >> box.y1 >> box.x2 >> box.y2 >> box.targetId >> box.distance) {
            detections.push_back(box);
            LOG(INFO) << "解析目标框 " << detections.size() - 1 << ": " 
                      << box.className << " conf=" << box.confidence 
                      << " box=(" << box.x1 << "," << box.y1 << "," << box.x2 << "," << box.y2 << ")";
        } else {
            LOG(WARNING) << "解析第 " << lineCount << " 行失败: " << line;
        }
    }
    
    file.close();
    LOG(INFO) << "成功解析 " << detections.size() << " 个目标框";
    return detections;
}

// 加载目标检测输出的数据
CAlgResult loadDetectionOutput(const std::string& detectionOutputPath, int frameId) {
    CAlgResult result;
    
    // 1. 解析目标框txt文件
    std::filesystem::path txtPath = std::filesystem::path(detectionOutputPath) / "Vis_Detection_Result" / std::to_string(frameId) / (std::to_string(frameId) + ".txt");
    auto detections = parseDetectionTxt(txtPath.string());
    
    if (detections.empty()) {
        LOG(ERROR) << "没有找到有效的检测结果";
        return result;
    }
    
    // 2. 加载子图数据
    std::filesystem::path regionsPath = std::filesystem::path(detectionOutputPath) / "Vis_Object_Regions" / std::to_string(frameId);
    
    // 创建FrameResult
    CFrameResult frameResult;
    frameResult.unFrameId(frameId);
    frameResult.eDataType(DATA_TYPE_POSEALG_RESULT);
    
    // 预分配内存
    frameResult.vecObjectResult().reserve(detections.size());
    frameResult.vecVideoSrcData().reserve(detections.size());
    
    // 为每个检测结果创建ObjectResult
    for (size_t i = 0; i < detections.size(); ++i) {
        const auto& det = detections[i];
        
        // 创建ObjectResult
        CObjectResult objResult;
        objResult.strClass(det.className);
        objResult.fVideoConfidence(det.confidence);
        objResult.fTopLeftX(det.x1);
        objResult.fTopLeftY(det.y1);
        objResult.fBottomRightX(det.x2);
        objResult.fBottomRightY(det.y2);
        objResult.usTargetId(det.targetId);
        objResult.fDistance(det.distance);
        
        // 加载对应的子图
        std::filesystem::path subImagePath = regionsPath / (std::to_string(i) + "_" + det.className + ".jpg");
        cv::Mat subImage = cv::imread(subImagePath.string());
        
        if (!subImage.empty()) {
            // 创建CVideoSrcData
            CVideoSrcData videoData;
            videoData.ucCameraId(static_cast<uint8_t>(i));
            videoData.usBmpWidth(subImage.cols);
            videoData.usBmpLength(subImage.rows);
            videoData.unBmpBytes(subImage.total() * subImage.elemSize());
            videoData.unFrameId(frameId);
            videoData.lTimeStamp(GetTimeStamp());
            videoData.eDataType(DATA_TYPE_RGB_IMAGE);
            videoData.eDataSourceType(0);
            
            // 直接使用图像数据，避免额外的内存复制
            videoData.vecImageBuf().assign(subImage.data, subImage.data + subImage.total() * subImage.elemSize());
            
            // 添加到vecVideoSrcData
            frameResult.vecVideoSrcData().push_back(std::move(videoData));
            
            LOG(INFO) << "加载子图 " << i << ": " << subImagePath.string() << " 尺寸: " << subImage.cols << "x" << subImage.rows;
        } else {
            LOG(ERROR) << "无法加载子图: " << subImagePath.string();
        }
        
        // 添加到ObjectResult列表
        frameResult.vecObjectResult().push_back(std::move(objResult));
    }
    
    // 设置时间戳
    result.lTimeStamp(GetTimeStamp());
    result.vecFrameResult().push_back(std::move(frameResult));
    
    LOG(INFO) << "成功加载检测输出数据，帧ID: " << frameId 
              << ", 目标数量: " << result.vecFrameResult()[0].vecObjectResult().size()
              << ", 子图数量: " << result.vecFrameResult()[0].vecVideoSrcData().size();
    
    return result;
}

void testObjectClassifyAlg(const CAlgResult& alg_result, void* p_handle)
{
    // 获取检测结果
    const auto& detections = alg_result.vecFrameResult().at(0).vecObjectResult();
    
    LOG(INFO) << "图像分类完成，检测到的目标数量: " << detections.size();
    
    // 输出每个目标的关键点信息
    for (size_t i = 0; i < detections.size(); ++i) {
        const auto& det = detections[i];
        const auto& keypoints = det.vecKeypoints();
        
        LOG(INFO) << "目标 " << i << " (" << det.strClass() << "):";
        LOG(INFO) << "  置信度: " << det.fVideoConfidence();
        LOG(INFO) << "  距离: " << det.fDistance() << " mm";
        LOG(INFO) << "  关键点数量: " << keypoints.size();
    }
    
    LOG(INFO) << "目标分类推理完成";
}

int main(int argc, char** argv) {
    try {
        // 设置默认路径
        std::string deploy_path = "/ultralytics/c++/Output/";
        g_save_dir = deploy_path + "vis/";
        std::string detection_output_path = "/ultralytics/c++/Output/";  // 目标检测输出路径

        // 算法接口调用流程基本如下：
        IObjectClassifyAlg* l_pObj = CreateObjectClassifyAlgObj(deploy_path);

        // 准备算法参数
        CSelfAlgParam *l_stTestAlgParam = new CSelfAlgParam();
        l_stTestAlgParam->m_strRootPath = deploy_path;
        
        // 初始化算法接口对象
        l_pObj->initAlgorithm(l_stTestAlgParam, testObjectClassifyAlg, nullptr);

        // 测试1-4号图像
        for (int frameId = 1; frameId <= 4; frameId++)
        {
            LOG(INFO) << "开始处理帧 " << frameId;
            
            // 加载目标检测输出数据
            CAlgResult detectionResult = loadDetectionOutput(detection_output_path, frameId);
            
            // 检查数据是否有效
            if (detectionResult.vecFrameResult().empty() || 
                detectionResult.vecFrameResult()[0].vecObjectResult().empty()) {
                LOG(ERROR) << "帧 " << frameId << " 数据加载失败，跳过";
                continue;
            }
            
            // 调用图像分类算法
            l_pObj->runAlgorithm(&detectionResult);
            
            LOG(INFO) << "帧 " << frameId << " 处理完成";
        }
        
        LOG(INFO) << "所有帧处理完成";
        
    } catch (const std::exception& e) {
        LOG(ERROR) << "错误: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}