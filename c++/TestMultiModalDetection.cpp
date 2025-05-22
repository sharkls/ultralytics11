#include <iostream>
#include <iomanip>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <vector>
#include <sstream>
#include <chrono>
// #include "FunctionHub.h"
#include "GlobalContext.h"
#include "CAlgResult.h"
#include "CMultiModalSrcData.h"
#include "ExportMultiModalFusionAlgLib.h"

// 全局变量
std::string g_rgb_path;
std::string g_save_dir;

/**
 * 获取当前ms UTC时间
 * 参数：
 * 返回值：ms UTC时间
 */
inline int64_t GetTimeStamp()
{
    std::chrono::time_point<std::chrono::system_clock, std::chrono::milliseconds> tp =
        std::chrono::time_point_cast<std::chrono::milliseconds>(std::chrono::system_clock::now());

    auto tmp = std::chrono::duration_cast<std::chrono::milliseconds>(tp.time_since_epoch());
    return tmp.count();
}

CMultiModalSrcData loadOfflineData(std::string data_path, int index) 
{   
    CMultiModalSrcData data;
    std::string rgb_path = data_path + "images/visible/test/19000" + std::to_string(index) + ".jpg";
    std::string ir_path = data_path + "images/infrared/test/19000" + std::to_string(index) + ".jpg";
    std::string homography_path = data_path + "extrinsics/test/19000" + std::to_string(index) + ".txt";
    // std::cout << "rgb_path: " << rgb_path << std::endl;
    // std::cout << "ir_path: " << ir_path << std::endl;
    // std::cout << "homography_path: " << homography_path << std::endl;
    
    // 加载可见光图像
    cv::Mat rgb_img = cv::imread(rgb_path);
    if (rgb_img.empty()) {
        std::cerr << "Failed to load RGB image: " << rgb_path << std::endl;
        return data;
    }
    
    // 加载红外图像
    cv::Mat ir_img = cv::imread(ir_path);
    if (ir_img.empty()) {
        std::cerr << "Failed to load IR image: " << ir_path << std::endl;
        return data;
    }
    
    // 加载单应性矩阵
    std::ifstream homography_file(homography_path);
    if (!homography_file.is_open()) {
        std::cerr << "Failed to open homography file: " << homography_path << std::endl;
        return data;
    }
    
    // 读取单应性矩阵
    std::vector<float> homography_matrix(9);
    for (int i = 0; i < 9; ++i) {
        if (!(homography_file >> homography_matrix[i])) {
            std::cerr << "Failed to read homography matrix" << std::endl;
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
    data.mapTimeStamp()[TIMESTAMP_TIME_MATCH] = GetTimeStamp();

    std::cout << " ReadData time_match: " << data.mapTimeStamp()[TIMESTAMP_TIME_MATCH] << std::endl;
    return data;
}

void testMultiModalFusionAlg(const CAlgResult& alg_result, void* p_handle)
{
    // 获取检测结果
    const auto& detections = alg_result.vecFrameResult().at(0).vecObjectResult();
    
    // 加载原始图像用于可视化
    cv::Mat rgb_img = cv::imread(g_rgb_path);
    if (rgb_img.empty()) {
        std::cerr << "无法加载原始图像用于可视化" << std::endl;
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

        // // 打印详细信息到控制台
        // std::cout << "\n检测目标信息:" << std::endl;
        // std::cout << "类别: " << cls_name << std::endl;
        // std::cout << "置信度: " << std::fixed << std::setprecision(4) << conf << std::endl;
        // std::cout << "位置: [" << x1 << ", " << y1 << ", " << x2 << ", " << y2 << "]" << std::endl;
        // std::cout << "目标ID: " << det.usTargetId() << std::endl;
        // std::cout << "速度: " << det.sSpeed() << " m/s" << std::endl;
    }

    // 保存结果
    std::string save_path = g_save_dir + "/" + 
        std::filesystem::path(g_rgb_path).stem().string() + "_det.jpg";
        
    // 检查保存目录是否存在
    if (!std::filesystem::exists(g_save_dir)) {
        std::filesystem::create_directories(g_save_dir);
        std::cout << "创建保存目录: " << g_save_dir << std::endl;
    }
    
    // 保存图像
    bool save_success = cv::imwrite(save_path, rgb_img);
    if (!save_success) {
        std::cerr << "保存图像失败: " << save_path << std::endl;
        return;
    }
    
    std::cout << "推理完成，结果已保存到: " << save_path << std::endl;
    std::cout << "检测到的目标数量: " << detections.size() << std::endl;
}

int main(int argc, char** argv) {
    try {
        // 设置默认路径
        std::string deploy_path = "/ultralytics/c++/Output/";
        g_save_dir = deploy_path + "vis";
        std::string data_path = "/ultralytics/data/Test_unmatch/";

        // 算法接口调用流程基本如下：
        IMultiModalFusionAlg* l_pObj = CreateMultiModalFusionAlgObj(deploy_path);

        // 准备算法参数
        CSelfAlgParam *l_stTestAlgParam = new CSelfAlgParam();
        l_stTestAlgParam->m_strRootPath = deploy_path;

        // 初始化算法接口对象
        l_pObj->initAlgorithm(l_stTestAlgParam, testMultiModalFusionAlg, nullptr);

        int size_data = 10;
        for (int i = 1; i < size_data; i++)
        {
            CMultiModalSrcData multi_modal_data = loadOfflineData(data_path, i);
            // 更新全局变量
            g_rgb_path = data_path + "images/visible/test/19000" + std::to_string(i) + ".jpg";
            l_pObj->runAlgorithm(&multi_modal_data);
        }
    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}