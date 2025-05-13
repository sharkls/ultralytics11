#include "src/Pose/trt_infer.h"
#include <iostream>
#include <iomanip>
#include <filesystem>
#include <chrono>
#include <vector>

int main(int argc, char** argv) {
    try {
        // 设置默认路径
        std::string engine_path = "/ultralytics/ckpt/yolo11m-pose.engine";
        std::string image_path = "/ultralytics/data/Test_1/images/visible/test/190009.jpg";
        std::string save_dir = "/ultralytics/runs/pose";

        // 创建保存目录
        std::filesystem::create_directories(save_dir);

        // 初始化推理器
        TRTInference trt_infer(engine_path);

        // 读取图像
        cv::Mat img = cv::imread(image_path);
        if (img.empty()) {
            throw std::runtime_error("无法读取图像");
        }

        // 打印原始图像信息
        std::cout << "原始图像尺寸: " << img.cols << "x" << img.rows << std::endl;
        std::cout << "图像类型: " << img.type() << std::endl;

        // 确保图像是有效的
        if (img.cols <= 0 || img.rows <= 0) {
            throw std::runtime_error("无效的图像尺寸");
        }

        // 模型预热
        std::cout << "开始模型预热..." << std::endl;
        for (int i = 0; i < 3; i++) {
            LetterBoxInfo letterbox_info;
            try {
                trt_infer.inference(img, letterbox_info);
            } catch (const std::exception& e) {
                std::cerr << "预热过程中出错: " << e.what() << std::endl;
                throw;
            }
        }
        std::cout << "模型预热完成" << std::endl;

        // 执行推理
        LetterBoxInfo letterbox_info;
        std::vector<float> total_time;
        
        for(int i = 0; i < 1; i++) {
            // 开始计时
            auto start_time = std::chrono::high_resolution_clock::now();
            
            std::vector<float> output = trt_infer.inference(img, letterbox_info);
            
            // 结束计时
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
            total_time.push_back(duration.count() / 1000.0f);  // 转换为毫秒
            
            // 后处理
            float conf_thres = 0.25f;  // 与Python代码中的conf_thres保持一致
            float iou_thres = 0.7f;    // 与Python代码中的iou_thres保持一致
            int num_classes = 1;       // 姿态估计只有一个类别
            std::vector<std::vector<float>> results = trt_infer.process_output(
                output, conf_thres, iou_thres, num_classes, letterbox_info);

            // 可视化结果
            cv::Mat vis_img = img.clone();
            for (const auto& result : results) {
                // 绘制边界框
                float x1 = result[0], y1 = result[1], x2 = result[2], y2 = result[3];
                float conf = result[4];
                int cls_id = static_cast<int>(result[5]);

                // 确保坐标在有效范围内
                x1 = std::max(0.0f, std::min(static_cast<float>(img.cols), x1));
                y1 = std::max(0.0f, std::min(static_cast<float>(img.rows), y1));
                x2 = std::max(0.0f, std::min(static_cast<float>(img.cols), x2));
                y2 = std::max(0.0f, std::min(static_cast<float>(img.rows), y2));

                cv::rectangle(vis_img, cv::Point(x1, y1), cv::Point(x2, y2),
                             cv::Scalar(0, 0, 255), 2);  // 使用红色绘制边界框，与Python代码保持一致

                // 绘制关键点
                // 从结果中提取关键点数据（从第6个元素开始，每个检测框有6个基础元素）
                for (int j = 0; j < 17; ++j) {
                    // 获取关键点坐标和置信度
                    float kpt_x = result[6 + j * 3];
                    float kpt_y = result[6 + j * 3 + 1];
                    float kpt_conf = result[6 + j * 3 + 2];
                    
                    // 确保关键点坐标在有效范围内
                    kpt_x = std::max(0.0f, std::min(static_cast<float>(img.cols), kpt_x));
                    kpt_y = std::max(0.0f, std::min(static_cast<float>(img.rows), kpt_y));
                    
                    if (kpt_conf > 0.3) {  // 与Python代码中的关键点置信度阈值保持一致
                        cv::circle(vis_img, cv::Point(kpt_x, kpt_y), 3,
                                 cv::Scalar(0, 255, 0), -1);  // 使用绿色绘制关键点，与Python代码保持一致
                    }
                }

                // 添加标签
                std::string label = "cls" + std::to_string(cls_id) + " " + 
                                  std::to_string(conf).substr(0, 4);
                cv::putText(vis_img, label, cv::Point(x1, y1 - 10),
                           cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
            }

            // 保存结果
            std::string save_path = save_dir + "/result_engine_C++.jpg";  // 与Python代码中的save_engine保持一致
                
            // 检查图像是否有效
            if (vis_img.empty()) {
                throw std::runtime_error("保存图像失败：图像为空");
            }
            
            // 检查图像尺寸
            std::cout << "可视化图像尺寸: " << vis_img.cols << "x" << vis_img.rows << std::endl;
            
            // 检查保存目录权限
            if (!std::filesystem::exists(save_dir)) {
                std::filesystem::create_directories(save_dir);
                std::cout << "创建保存目录: " << save_dir << std::endl;
            }
            
            // 尝试保存图像
            bool save_success = cv::imwrite(save_path, vis_img);
            if (!save_success) {
                throw std::runtime_error("保存图像失败：无法写入文件 " + save_path);
            }
            
            std::cout << "推理完成，结果已保存到: " << save_path << std::endl;

            // 打印检测结果信息
            std::cout << "\n检测结果信息:" << std::endl;
            std::cout << "检测到的目标数量: " << results.size() << std::endl;
            for (size_t i = 0; i < results.size(); ++i) {
                const auto& result = results[i];
                float x1 = result[0], y1 = result[1], x2 = result[2], y2 = result[3];
                float conf = result[4];
                int cls_id = static_cast<int>(result[5]);

                // std::cout << "目标 " << i + 1 << ":" << std::endl;
                // std::cout << "  类别: " << cls_id << std::endl;
                // std::cout << "  置信度: " << std::fixed << std::setprecision(4) << conf << std::endl;
                // std::cout << "  边界框: [" << x1 << ", " << y1 << ", " << x2 << ", " << y2 << "]" << std::endl;
                // std::cout << "  关键点数量: 17" << std::endl;
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