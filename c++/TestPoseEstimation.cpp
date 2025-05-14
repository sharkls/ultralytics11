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
#include "ExportPoseEstimationAlgLib.h"

// 全局变量
std::string g_rgb_path;
std::string g_save_dir;

CMultiModalSrcData loadOfflineData(std::string data_path, int index) 
{   
    CMultiModalSrcData data;
    std::string rgb_path = data_path + "images/visible/test/19000" + std::to_string(index) + ".jpg";
    std::string ir_path = data_path + "images/infrared/test/19000" + std::to_string(index) + ".jpg";
    std::string homography_path = data_path + "extrinsics/test/19000" + std::to_string(index) + ".txt";
    
    // 加载可见光图像
    cv::Mat rgb_img = cv::imread(rgb_path);
    if (rgb_img.empty()) {
        LOG(ERROR) << "Failed to load RGB image: " << rgb_path << std::endl;
        return data;
    }
    
    // 加载红外图像
    cv::Mat ir_img = cv::imread(ir_path);
    if (ir_img.empty()) {
        LOG(ERROR) << "Failed to load IR image: " << ir_path << std::endl;
        return data;
    }
    
    // 加载单应性矩阵
    std::ifstream homography_file(homography_path);
    if (!homography_file.is_open()) {
        LOG(ERROR) << "Failed to open homography file: " << homography_path << std::endl;
        return data;
    }
    
    // 读取单应性矩阵
    std::vector<float> homography_matrix(9);
    for (int i = 0; i < 9; ++i) {
        if (!(homography_file >> homography_matrix[i])) {
            LOG(ERROR) << "Failed to read homography matrix" << std::endl;
            return data;
        }
    }
    
    // 设置可见光图像数据
    CVideoSrcData rgb_data;
    rgb_data.ucCameraId(0);  // 0表示可见光相机
    rgb_data.usBmpWidth(rgb_img.cols);
    rgb_data.usBmpLength(rgb_img.rows);
    rgb_data.unBmpBytes(rgb_img.total() * rgb_img.elemSize());
    std::vector<uint8_t> rgb_vec(rgb_img.data, rgb_img.data + rgb_img.total() * rgb_img.elemSize());
    rgb_data.vecImageBuf(rgb_vec);
    
    // 设置红外图像数据
    CVideoSrcData ir_data;
    ir_data.ucCameraId(1);  // 1表示红外相机
    ir_data.usBmpWidth(ir_img.cols);
    ir_data.usBmpLength(ir_img.rows);
    ir_data.unBmpBytes(ir_img.total() * ir_img.elemSize());
    std::vector<uint8_t> ir_vec(ir_img.data, ir_img.data + ir_img.total() * ir_img.elemSize());
    ir_data.vecImageBuf(ir_vec);
    
    // 设置时间匹配数据
    std::vector<CVideoSrcData> video_data = {rgb_data, ir_data};
    data.vecVideoSrcData(video_data);
    data.vecfHomography(homography_matrix);
    
    return data;
}

void testPoseEstimationAlg(const CAlgResult& alg_result, void* p_handle)
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

        // 绘制关键点
        const auto& keypoints = det.vecKeypoints();
        for (size_t j = 0; j < keypoints.size(); ++j) {
            float kpt_x = keypoints[j].x();
            float kpt_y = keypoints[j].y();
            float kpt_conf = keypoints[j].confidence();
            if (kpt_conf > 0.3) {
                cv::circle(rgb_img, cv::Point(kpt_x, kpt_y), 3, cv::Scalar(0, 255, 0), -1);
            }
        }

        // // （可选）骨架连线
        // static const int skeleton[][2] = {
        //     {0,1},{1,2},{2,3},{3,4},
        //     {0,5},{5,6},{6,7},{7,8},
        //     {0,9},{9,10},{10,11},
        //     {0,12},{12,13},{13,14},
        //     {0,15},{15,16}
        // };
        // for (const auto& pair : skeleton) {
        //     int idx1 = pair[0], idx2 = pair[1];
        //     if (idx1 < keypoints.size() && idx2 < keypoints.size() &&
        //         keypoints[idx1].confidence() > 0.3 && keypoints[idx2].confidence() > 0.3) {
        //         cv::line(rgb_img, 
        //                  cv::Point(keypoints[idx1].x(), keypoints[idx1].y()),
        //                  cv::Point(keypoints[idx2].x(), keypoints[idx2].y()),
        //                  cv::Scalar(255, 0, 0), 2);
        //     }
        // }
    }

    // 保存结果
    std::string save_path = g_save_dir + 
        std::filesystem::path(g_rgb_path).stem().string() + "_pose.jpg";
        
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
        std::string data_path = "/ultralytics/data/Test_1/";

        // 算法接口调用流程基本如下：
        IPoseEstimationAlg* l_pObj = CreatePoseEstimationAlgObj(deploy_path);

        // 准备算法参数
        CSelfAlgParam *l_stTestAlgParam = new CSelfAlgParam();
        l_stTestAlgParam->m_strRootPath = deploy_path;
        
        // 初始化算法接口对象
        l_pObj->initAlgorithm(l_stTestAlgParam, testPoseEstimationAlg, nullptr);

        int size_data = 10;
        for (int i = 1; i < size_data; i++)
        {
            CMultiModalSrcData multi_modal_data = loadOfflineData(data_path, i);
            // 更新全局变量
            g_rgb_path = data_path + "images/visible/test/19000" + std::to_string(i) + ".jpg";
            l_pObj->runAlgorithm(&multi_modal_data);
        }
    } catch (const std::exception& e) {
        LOG(ERROR) << "错误: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}