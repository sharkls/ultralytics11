#include "src/Detection/trt_infer.h"
#include <iostream>
#include <iomanip>
#include <filesystem>
#include <chrono>
#include <vector>

int main(int argc, char** argv) {
    try {
        // 设置默认路径
        std::string engine_path = "/ultralytics/runs/multimodal/train6/weights/last.engine";
        std::string rgb_path = "/ultralytics/data/Test_1/images/visible/test/190001.jpg";
        std::string ir_path = "/ultralytics/data/Test_1/images/infrared/test/190001.jpg";
        std::string homography_path = "/ultralytics/data/Test_1/extrinsics/test/190001.txt";
        std::string save_dir = "/ultralytics/runs/tensorrt_vis";

        // 创建保存目录
        std::filesystem::create_directories(save_dir);

        // 初始化推理器
        TRTInference trt_infer(engine_path);

        // 读取图像
        cv::Mat rgb_img = cv::imread(rgb_path);
        cv::Mat ir_img = cv::imread(ir_path);
        if (rgb_img.empty() || ir_img.empty()) {
            throw std::runtime_error("无法读取图像");
        }

        // 读取单应性矩阵
        std::vector<float> homography(9);
        std::ifstream h_file(homography_path);
        if (!h_file.is_open()) {
            throw std::runtime_error("无法打开单应性矩阵文件");
        }
        for (int i = 0; i < 9; ++i) {
            h_file >> homography[i];
        }

        // 模型预热
        std::cout << "开始模型预热..." << std::endl;
        for (int i = 0; i < 3; i++) {
            LetterBoxInfo letterbox_info;
            trt_infer.inference(rgb_img, ir_img, homography, letterbox_info);
        }
        std::cout << "模型预热完成" << std::endl;

        // 执行推理
        LetterBoxInfo letterbox_info;
        std::vector<float> total_time;
        
        for(int i = 0; i < 10; i++) {
            // 开始计时
            auto start_time = std::chrono::high_resolution_clock::now();
            
            std::vector<float> output = trt_infer.inference(rgb_img, ir_img, homography, letterbox_info);
            
            // 结束计时
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
            total_time.push_back(duration.count() / 1000.0f);  // 转换为毫秒
            
            // 后处理
            float conf_thres = 0.25f;
            float iou_thres = 0.45f;
            int num_classes = 1;
            std::vector<std::vector<float>> detections = trt_infer.process_output(
                output, conf_thres, iou_thres, num_classes, letterbox_info);

            // 可视化结果
            for (const auto& det : detections) {
                float x1 = det[0], y1 = det[1], x2 = det[2], y2 = det[3];
                float conf = det[4];
                int cls_id = static_cast<int>(det[5]);

                // 绘制边界框
                cv::rectangle(rgb_img, cv::Point(x1, y1), cv::Point(x2, y2),
                             cv::Scalar(0, 255, 0), 2);

                // 添加标签
                std::string label = "cls" + std::to_string(cls_id) + " " + 
                                  std::to_string(conf).substr(0, 4);
                cv::putText(rgb_img, label, cv::Point(x1, y1 - 10),
                           cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
            }

            // 保存结果
            std::string save_path = save_dir + "/" + 
                std::filesystem::path(rgb_path).stem().string() + "_det.jpg";
                
            // 检查图像是否有效
            if (rgb_img.empty()) {
                throw std::runtime_error("保存图像失败：图像为空");
            }
            
            // 检查图像尺寸
            std::cout << "图像尺寸: " << rgb_img.cols << "x" << rgb_img.rows << std::endl;
            
            // 检查保存目录权限
            if (!std::filesystem::exists(save_dir)) {
                std::filesystem::create_directories(save_dir);
                std::cout << "创建保存目录: " << save_dir << std::endl;
            }
            
            // 尝试保存图像
            bool save_success = cv::imwrite(save_path, rgb_img);
            if (!save_success) {
                throw std::runtime_error("保存图像失败：无法写入文件 " + save_path);
            }
            
            std::cout << "推理完成，结果已保存到: " << save_path << std::endl;

            // 打印检测结果信息
            std::cout << "\n检测结果信息:" << std::endl;
            std::cout << "检测到的目标数量: " << detections.size() << std::endl;
            for (size_t i = 0; i < detections.size(); ++i) {
                const auto& det = detections[i];
                float x1 = det[0], y1 = det[1], x2 = det[2], y2 = det[3];
                float conf = det[4];
                int cls_id = static_cast<int>(det[5]);

                // std::cout << "目标 " << i + 1 << ":" << std::endl;
                // std::cout << "  类别: " << cls_id << std::endl;
                // std::cout << "  置信度: " << std::fixed << std::setprecision(4) << conf << std::endl;
                // std::cout << "  边界框: [" << x1 << ", " << y1 << ", " << x2 << ", " << y2 << "]" << std::endl;
                // std::cout << "  宽度: " << x2 - x1 << std::endl;
                // std::cout << "  高度: " << y2 - y1 << std::endl;
            }
        }

        // 计算平均推理时间
        float avg_time = 0.0f;
        for (float t : total_time) {
            avg_time += t;
        }
        avg_time /= total_time.size();
        
        std::cout << "\n性能统计:" << std::endl;
        std::cout << "平均推理时间: " << std::fixed << std::setprecision(2) << avg_time << " 毫秒" << std::endl;
        std::cout << "FPS: " << std::fixed << std::setprecision(2) << 1000.0f / avg_time << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}