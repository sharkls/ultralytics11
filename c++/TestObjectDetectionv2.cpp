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
#include "ExportObjectDetectionAlgLib.h"
#include "FunctionHub.h"

// 全局变量
std::string g_rgb_path;
std::string g_save_dir;

CMultiModalSrcData loadOfflineData(std::string data_path, int index) 
{   
    CMultiModalSrcData data;
    std::string rgb_path = data_path + "/" + std::to_string(index) + "/" + std::to_string(index) + ".jpg";
    
    // 加载可见光图像
    cv::Mat rgb_img = cv::imread(rgb_path);
    if (rgb_img.empty()) {
        LOG(ERROR) << "Failed to load RGB image: " << rgb_path << std::endl;
        return data;
    }
    
    LOG(INFO) << "成功加载图像: " << rgb_path << ", 尺寸: " << rgb_img.cols << "x" << rgb_img.rows;
    
    // 设置可见光图像数据
    CVideoSrcData rgb_data;
    rgb_data.ucCameraId(0);  // 0表示可见光相机
    rgb_data.usBmpWidth(rgb_img.cols);
    rgb_data.usBmpLength(rgb_img.rows);
    rgb_data.unBmpBytes(rgb_img.total() * rgb_img.elemSize());
    rgb_data.unFrameId(index);
    std::vector<uint8_t> rgb_vec(rgb_img.data, rgb_img.data + rgb_img.total() * rgb_img.elemSize());
    rgb_data.vecImageBuf(rgb_vec);
    
    // 设置时间匹配数据
    std::vector<CVideoSrcData> video_data = {rgb_data};

    data.vecVideoSrcData(video_data);
    int64_t endTimeStamp = GetTimeStamp();
    data.vecVideoSrcData()[0].lTimeStamp(endTimeStamp);
    
    return data;
}

void testObjectDetectionAlg(const CAlgResult& alg_result, void* p_handle)
{
    // 获取检测结果
    const auto& detections = alg_result.vecFrameResult().at(0).vecObjectResult();
    
    // 加载原始图像用于可视化
    cv::Mat rgb_img = cv::imread(g_rgb_path);
    if (rgb_img.empty()) {
        LOG(ERROR) << "无法加载原始图像用于可视化" << std::endl;
        return;
    }

    // 可视化结果
    for (const auto& det : detections) {
        // 获取目标框坐标
        float x1 = det.fTopLeftX();
        float y1 = det.fTopLeftY();
        float x2 = det.fBottomRightX();
        float y2 = det.fBottomRightY();
        float conf = det.fVideoConfidence();
        std::string cls_name = det.strClass();

        // 绘制边界框
        cv::rectangle(rgb_img, cv::Point(x1, y1), cv::Point(x2, y2),
                     cv::Scalar(0, 255, 0), 2);

        // 准备标签文本
        std::stringstream ss;
        ss << cls_name << " " << std::fixed << std::setprecision(2) << conf;
        if (det.fDistance() > 0) {
            ss << " " << std::fixed << std::setprecision(0) << det.fDistance() << "mm";
        }
        std::string label = ss.str();

        // 计算文本大小
        int baseline = 0;
        cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 2, &baseline);

        // 绘制标签背景
        cv::rectangle(rgb_img, 
                     cv::Point(x1, y1 - text_size.height - 10),
                     cv::Point(x1 + text_size.width, y1),
                     cv::Scalar(0, 255, 0), -1);

        // 绘制标签文本
        cv::putText(rgb_img, label, 
                   cv::Point(x1, y1 - 5),
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 2);
    }

    // 保存结果
    std::string save_path = g_save_dir + 
        std::filesystem::path(g_rgb_path).stem().string() + "_det.jpg";
        
    // 检查保存目录是否存在
    if (!std::filesystem::exists(g_save_dir)) {
        std::filesystem::create_directories(g_save_dir);
        LOG(INFO) << "创建保存目录: " << g_save_dir;
    }
    
    // 保存图像
    bool save_success = cv::imwrite(save_path, rgb_img);
    if (!save_success) {
        LOG(ERROR) << "保存图像失败: " << save_path;
        return;
    }
    
    LOG(INFO) << "推理完成，结果已保存到: " << save_path;
    LOG(INFO) << "检测到的目标数量: " << detections.size();
}

int main(int argc, char** argv) {
    try {
        // 设置默认路径
        std::string deploy_path = "/ultralytics/c++/Output/";
        g_save_dir = deploy_path + "vis/";
        std::string data_path = "/ultralytics/c++/Data/SrcData/";
        

        // 算法接口调用流程基本如下：
        IObjectDetectionAlg* l_pObj = CreateObjectDetectionAlgObj(deploy_path);

        // 准备算法参数
        CSelfAlgParam *l_stTestAlgParam = new CSelfAlgParam();
        l_stTestAlgParam->m_strRootPath = deploy_path;
        
        // 初始化算法接口对象
        l_pObj->initAlgorithm(l_stTestAlgParam, testObjectDetectionAlg, nullptr);

        // 测试1-4号图像
        for (int i = 1; i <= 4; i++)
        {
            LOG(INFO) << "开始处理图像 " << i;
            CMultiModalSrcData multi_modal_data = loadOfflineData(data_path, i);
            
            // 检查数据是否有效
            if (multi_modal_data.vecVideoSrcData().empty()) {
                LOG(ERROR) << "图像 " << i << " 数据加载失败，跳过";
                continue;
            }
            
            // 更新全局变量
            g_rgb_path = data_path + std::to_string(i) + "/" + std::to_string(i) + ".jpg";
            l_pObj->runAlgorithm(&multi_modal_data);
            
            LOG(INFO) << "图像 " << i << " 处理完成";
        }
        
        LOG(INFO) << "所有图像处理完成";
        
    } catch (const std::exception& e) {
        LOG(ERROR) << "错误: " << e.what() << std::endl;
        return -1;
    }

    return 0;
} 