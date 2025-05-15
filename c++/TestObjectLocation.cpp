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
#include "ExportObjectLocationAlgLib.h"

// 全局变量
std::string g_rgb_path;
std::string g_save_dir;

CAlgResult loadOfflineData(std::string data_path, int index) 
{   
    CMultiModalSrcData data;
    std::string multi_modal_result = data_path + "multimodal/result/19000" + std::to_string(index) + ".bin";
    std::string pose_result = data_path + "pose/result/19000" + std::to_string(index) + ".bin";
    std::string depth_map = data_path + "multimodal/depth/19000" + std::to_string(index) + ".bin";

    CAlgResult alg_result;

    // 加载多模态融合感知结果
    std::vector<float> buffer;
    std::ifstream fin(multi_modal_result, std::ios::binary);
    fin.seekg(0, std::ios::end);
    size_t num_bytes = fin.tellg();
    fin.seekg(0, std::ios::beg);
    buffer.resize(num_bytes / sizeof(float));
    fin.read(reinterpret_cast<char*>(buffer.data()), num_bytes);
    fin.close();

    int M = 6; // 每行6个float(left, top, right, bottom, score, cls)
    int N = buffer.size() / M;
    CFrameResult frame_result;
    std::vector<CObjectResult> objects;
    for (int i = 0; i < N; ++i) {
        CObjectResult obj;
        obj.fTopLeftX(buffer[i*M + 0]);
        obj.fTopLeftY(buffer[i*M + 1]);
        obj.fBottomRightX(buffer[i*M + 2]);
        obj.fBottomRightY(buffer[i*M + 3]);
        obj.fVideoConfidence(buffer[i*M + 4]);
        obj.strClass(std::to_string(static_cast<int>(buffer[i*M + 5])));
        objects.push_back(obj);
    }
    frame_result.vecObjectResult(objects);

    // 加载深度图
    std::vector<float> depth_buffer;
    std::ifstream depth_fin(depth_map, std::ios::binary);
    depth_fin.seekg(0, std::ios::end);
    size_t depth_num_bytes = depth_fin.tellg();
    depth_fin.seekg(0, std::ios::beg);
    depth_buffer.resize(depth_num_bytes / sizeof(float));
    depth_fin.read(reinterpret_cast<char*>(depth_buffer.data()), depth_num_bytes);
    depth_fin.close();  

    CDisparityResult disparity_result;
    disparity_result.vecDistanceInfo(depth_buffer);
    disparity_result.usWidth(1920);
    disparity_result.usHeight(1080);
    frame_result.tCameraSupplement(disparity_result);
    alg_result.vecFrameResult().push_back(frame_result);

    // 加载姿态估计任务结果
    std::vector<float> pose_buffer;
    std::ifstream pose_fin(pose_result, std::ios::binary);
    pose_fin.seekg(0, std::ios::end);
    size_t pose_num_bytes = pose_fin.tellg();
    pose_fin.seekg(0, std::ios::beg);
    pose_buffer.resize(pose_num_bytes / sizeof(float));
    pose_fin.read(reinterpret_cast<char*>(pose_buffer.data()), pose_num_bytes);
    pose_fin.close();

    int num_keypoints = 17;
    int M_pose = 6 + num_keypoints * 3; // 每行6个float(left, top, right, bottom, score, cls) + 17*3个float(x, y, confidence)
    int N_pose = pose_buffer.size() / M_pose;
    CFrameResult frame_result_pose;
    std::vector<CObjectResult> objects_pose;
    for (int i = 0; i < N_pose; ++i) {
        CObjectResult obj;
        obj.fTopLeftX(pose_buffer[i*M_pose + 0]);
        obj.fTopLeftY(pose_buffer[i*M_pose + 1]);
        obj.fBottomRightX(pose_buffer[i*M_pose + 2]);
        obj.fBottomRightY(pose_buffer[i*M_pose + 3]);
        obj.fVideoConfidence(pose_buffer[i*M_pose + 4]);
        obj.strClass(std::to_string(static_cast<int>(pose_buffer[i*M_pose + 5])));

        // 关键点
        std::vector<Keypoint> keypoints;
        for (int k = 0; k < num_keypoints; ++k) {
            float kx = pose_buffer[i*M_pose + 6 + k*3 + 0];
            float ky = pose_buffer[i*M_pose + 6 + k*3 + 1];
            float kc = pose_buffer[i*M_pose + 6 + k*3 + 2];
            Keypoint kp;
            kp.x(kx);
            kp.y(ky);
            kp.confidence(kc);
            keypoints.push_back(kp);
        }
        obj.vecKeypoints(keypoints);

        objects.push_back(obj);
    }
    frame_result_pose.vecObjectResult(objects);
    alg_result.vecFrameResult().push_back(frame_result_pose);
    
    return alg_result;
}

void testObjectLocationAlg(const CAlgResult& alg_result, void* p_handle)
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
        float depth = det.fDistance();

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

        // 绘制深度值文本
        std::stringstream ss_depth;
        ss_depth << "Depth: " << std::fixed << std::setprecision(2) << depth;
        std::string depth_label = ss_depth.str();
        cv::putText(rgb_img, depth_label, 
                   cv::Point(x1, y1 + text_size.height + 10),
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 2);
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
        IObjectLocationAlg* l_pObj = CreateObjectLocationAlgObj(deploy_path);

        // 准备算法参数
        CSelfAlgParam *l_stTestAlgParam = new CSelfAlgParam();
        l_stTestAlgParam->m_strRootPath = deploy_path;
        
        // 初始化算法接口对象
        l_pObj->initAlgorithm(l_stTestAlgParam, testObjectLocationAlg, nullptr);

        int size_data = 10;
        for (int i = 1; i < size_data; i++)
        {   
            std::cout << "loadOfflineData status: start" << std::endl;
            CAlgResult multi_modal_data = loadOfflineData(data_path, i);
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