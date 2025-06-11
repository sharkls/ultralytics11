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
#include "ExportBinocularPositioningAlgLib.h"
#include "FunctionHub.h"

// 全局变量
std::string g_left_path;
std::string g_save_dir;
cv::Mat g_depthMatFloat;  // 用于存储深度图数据

// 鼠标回调函数
void onMouse(int event, int x, int y, int flags, void* userdata) {
    if (event == cv::EVENT_LBUTTONDOWN) {
        float depth = g_depthMatFloat.at<float>(y, x);
        if(depth > 0) {
            std::cout << "位置(" << x << "," << y << ")的深度值: " << depth << "mm" << std::endl;
        } else {
            std::cout << "位置(" << x << "," << y << ")的深度值无效" << std::endl;
        }
    }
}

// 离线数据加载函数（加载双目图像）
CMultiModalSrcData loadOfflineData(const std::string& data_path, int index) 
{   
    CMultiModalSrcData data;
    std::string left_path = data_path + "/" + "left.jpg";
    std::string right_path = data_path + "/" + "right.jpg";
    
    // 加载左图
    cv::Mat left_img = cv::imread(left_path);
    if (left_img.empty()) {
        LOG(ERROR) << "Failed to load left image: " << left_path;
        return data;
    }
    // 加载右图
    cv::Mat right_img = cv::imread(right_path);
    if (right_img.empty()) {
        LOG(ERROR) << "Failed to load right image: " << right_path;
        return data;
    }

    // 设置左图数据
    CVideoSrcData left_data;
    left_data.ucCameraId(0);
    left_data.usBmpWidth(left_img.cols);
    left_data.usBmpLength(left_img.rows);
    left_data.unBmpBytes(left_img.total() * left_img.elemSize());
    left_data.unFrameId(index);
    std::vector<uint8_t> left_vec(left_img.data, left_img.data + left_img.total() * left_img.elemSize());
    left_data.vecImageBuf(left_vec);

    // 设置右图数据
    CVideoSrcData right_data;
    right_data.ucCameraId(1);
    right_data.usBmpWidth(right_img.cols);
    right_data.usBmpLength(right_img.rows);
    right_data.unBmpBytes(right_img.total() * right_img.elemSize());
    right_data.unFrameId(index);
    std::vector<uint8_t> right_vec(right_img.data, right_img.data + right_img.total() * right_img.elemSize());
    right_data.vecImageBuf(right_vec);

    // 设置时间戳
    int64_t endTimeStamp = GetTimeStamp();
    left_data.lTimeStamp(endTimeStamp);
    right_data.lTimeStamp(endTimeStamp);

    // 组装
    std::vector<CVideoSrcData> video_data = {left_data, right_data};
    data.vecVideoSrcData(video_data);

    return data;
}

// 算法回调函数
void testBinocularPositioningAlg(const CAlgResult& alg_result, void* p_handle)
{
    // 获取深度图数据
    const auto& frameResult = alg_result.vecFrameResult().at(0);
    int width = frameResult.tCameraSupplement().usWidth();
    int height = frameResult.tCameraSupplement().usHeight();
    const std::vector<int>& depthData = frameResult.tCameraSupplement().vecDistanceInfo();

    if (depthData.empty() || width <= 0 || height <= 0) {
        LOG(ERROR) << "Depth data is empty or size invalid!";
        return;
    }

    // 转为Mat
    cv::Mat depthMat(height, width, CV_32S, (void*)depthData.data());
    cv::Mat depthMatFloat;
    depthMat.convertTo(depthMatFloat, CV_32F);

    // 创建可视化图像
    cv::Mat depthVis;
    cv::normalize(depthMatFloat, depthVis, 0, 255, cv::NORM_MINMAX);
    depthVis.convertTo(depthVis, CV_8U);
    cv::cvtColor(depthVis, depthVis, cv::COLOR_GRAY2BGR);

    // 计算深度值统计信息
    double minVal, maxVal, meanVal, stdVal;
    cv::minMaxLoc(depthMatFloat, &minVal, &maxVal);
    cv::Scalar mean, stddev;
    cv::meanStdDev(depthMatFloat, mean, stddev);
    meanVal = mean[0];
    stdVal = stddev[0];

    // 在图像上显示统计信息
    std::string stats = cv::format("Min: %.1f, Max: %.1f, Mean: %.1f, Std: %.1f", 
                                  minVal, maxVal, meanVal, stdVal);
    cv::putText(depthVis, stats, cv::Point(10, 30), 
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);

    // 在图像上显示深度值
    int step = 50;  // 每隔50个像素显示一个深度值
    for(int i = step; i < height; i += step) {
        for(int j = step; j < width; j += step) {
            float depth = depthMatFloat.at<float>(i, j);
            if(depth > 0) {
                std::string depthText = cv::format("%.1f", depth);
                cv::putText(depthVis, depthText, cv::Point(j, i), 
                           cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
            }
        }
    }

    // 保存深度图
    std::string save_path = g_save_dir + 
        std::filesystem::path(g_left_path).stem().string() + "_depth.jpg";
        
    if (!std::filesystem::exists(g_save_dir)) {
        std::filesystem::create_directories(g_save_dir);
        LOG(INFO) << "创建保存目录: " << g_save_dir;
    }
    
    bool save_success = cv::imwrite(save_path, depthVis);
    if (!save_success) {
        LOG(ERROR) << "保存深度图失败: " << save_path;
        return;
    }

    // 保存深度值信息到文本文件
    std::string depth_info_path = g_save_dir + 
        std::filesystem::path(g_left_path).stem().string() + "_depth_info.txt";
    std::ofstream depth_info_file(depth_info_path);
    if (depth_info_file.is_open()) {
        depth_info_file << "深度图统计信息：" << std::endl;
        depth_info_file << stats << std::endl << std::endl;
        
        depth_info_file << "采样点深度值（每隔" << step << "个像素）：" << std::endl;
        for(int i = step; i < height; i += step) {
            for(int j = step; j < width; j += step) {
                float depth = depthMatFloat.at<float>(i, j);
                if(depth > 0) {
                    depth_info_file << "位置(" << j << "," << i << "): " << depth << "mm" << std::endl;
                }
            }
        }
        depth_info_file.close();
        LOG(INFO) << "深度值信息已保存到: " << depth_info_path;
    }
    
    LOG(INFO) << "推理完成，深度图已保存到: " << save_path;
}

int main(int argc, char** argv) {
    try {
        // 设置默认路径
        std::string deploy_path = "/ultralytics/c++/Output/";
        g_save_dir = deploy_path + "vis/";
        std::string data_path = "/ultralytics/data/Data/BibocularPositioning"; // 假设此目录下有xxx_left.jpg和xxx_right.jpg

        // 算法接口调用流程
        IBinocularPositioningAlg* l_pAlg = CreateBinocularPositioningAlgObj(deploy_path);

        // 准备算法参数
        CSelfAlgParam *l_stTestAlgParam = new CSelfAlgParam();
        l_stTestAlgParam->m_strRootPath = deploy_path;
        
        // 初始化算法接口对象
        l_pAlg->initAlgorithm(l_stTestAlgParam, testBinocularPositioningAlg, nullptr);

        int start_idx = 0, end_idx = 100; // 根据实际数据调整
        for (int i = start_idx; i < end_idx; i++)
        {
            CMultiModalSrcData multi_modal_data = loadOfflineData(data_path, i);
            // 更新全局变量
            g_left_path = data_path + "/" + "left.jpg";
            l_pAlg->runAlgorithm(&multi_modal_data);
        }
    } catch (const std::exception& e) {
        LOG(ERROR) << "错误: " << e.what();
        return -1;
    }

    return 0;
}