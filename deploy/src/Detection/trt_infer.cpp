#include "trt_infer.h"
#include <fstream>
#include <algorithm>
#include <cmath>

TRTInference::TRTInference(const std::string& engine_path) {
    // 初始化CUDA流
    cudaStreamCreate(&stream_);

    // 加载engine
    std::ifstream file(engine_path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("无法打开engine文件: " + engine_path);
    }

    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> engineData(size);
    file.read(engineData.data(), size);

    runtime_ = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger_));
    engine_ = std::unique_ptr<nvinfer1::ICudaEngine>(
        runtime_->deserializeCudaEngine(engineData.data(), size));
    context_ = std::unique_ptr<nvinfer1::IExecutionContext>(
        engine_->createExecutionContext());

    // 获取输入输出信息
    int num_bindings = engine_->getNbIOTensors();
    for (int i = 0; i < num_bindings; ++i) {
    auto dims = engine_->getTensorShape(engine_->getIOTensorName(i));  // 使用新API
    if (engine_->getTensorIOMode(engine_->getIOTensorName(i)) == nvinfer1::TensorIOMode::kINPUT) {  // 使用新API
        input_dims_.push_back(dims);
        input_buffers_.push_back(nullptr);
    } else {
        output_dims_.push_back(dims);
        output_buffers_.push_back(nullptr);
    }
}

    // 设置输入尺寸
    input_h_ = input_dims_[0].d[2];
    input_w_ = input_dims_[0].d[3];
    stride_ = 32;

    // 分配GPU内存
    for (size_t i = 0; i < input_dims_.size(); ++i) {
        size_t size = 1;
        for (int j = 0; j < input_dims_[i].nbDims; ++j) {
            size *= input_dims_[i].d[j];
        }
        cudaMalloc(&input_buffers_[i], size * sizeof(float));
    }

    for (size_t i = 0; i < output_dims_.size(); ++i) {
        size_t size = 1;
        for (int j = 0; j < output_dims_[i].nbDims; ++j) {
            size *= output_dims_[i].d[j];
        }
        cudaMalloc(&output_buffers_[i], size * sizeof(float));
    }
}

TRTInference::~TRTInference() {
    cleanup();
}

void TRTInference::cleanup() {
    for (auto& buf : input_buffers_) {
        if (buf) cudaFree(buf);
    }
    for (auto& buf : output_buffers_) {
        if (buf) cudaFree(buf);
    }
    if (stream_) cudaStreamDestroy(stream_);

    // 使用智能指针自动管理资源
    context_.reset();
    engine_.reset();
    runtime_.reset();
}

void TRTInference::preprocess(const cv::Mat& rgb_img, const cv::Mat& ir_img,
                            const std::vector<float>& homography,
                            std::vector<float>& rgb_input,
                            std::vector<float>& ir_input,
                            std::vector<float>& h_input,
                            LetterBoxInfo& letterbox_info) {
    // 计算letterbox参数
    float r_rgb = std::min(static_cast<float>(input_h_) / rgb_img.rows,
                          static_cast<float>(input_w_) / rgb_img.cols);
    float r_ir = std::min(static_cast<float>(input_h_) / ir_img.rows,
                         static_cast<float>(input_w_) / ir_img.cols);

    int new_unpad_rgb_w = static_cast<int>(rgb_img.cols * r_rgb);
    int new_unpad_rgb_h = static_cast<int>(rgb_img.rows * r_rgb);
    int new_unpad_ir_w = static_cast<int>(ir_img.cols * r_ir);
    int new_unpad_ir_h = static_cast<int>(ir_img.rows * r_ir);

    float dw_rgb = (input_w_ - new_unpad_rgb_w) / 2.0f;
    float dh_rgb = (input_h_ - new_unpad_rgb_h) / 2.0f;
    float dw_ir = (input_w_ - new_unpad_ir_w) / 2.0f;
    float dh_ir = (input_h_ - new_unpad_ir_h) / 2.0f;

    // 更新单应性矩阵
    // 构建变换矩阵
    std::vector<float> S_rgb = {r_rgb, 0, 0, 0, r_rgb, 0, 0, 0, 1};
    std::vector<float> S_ir = {r_ir, 0, 0, 0, r_ir, 0, 0, 0, 1};
    std::vector<float> T_rgb = {1, 0, dw_rgb, 0, 1, dh_rgb, 0, 0, 1};
    std::vector<float> T_ir = {1, 0, dw_ir, 0, 1, dh_ir, 0, 0, 1};

    // 计算更新后的单应性矩阵
    // H_new = T_rgb @ S_rgb @ H @ S_ir^(-1) @ T_ir^(-1)
    // 这里简化处理，实际需要实现矩阵乘法
    h_input = homography; // 简化处理，实际需要完整矩阵运算

    // 保存letterbox信息
    letterbox_info.dw = dw_rgb;
    letterbox_info.dh = dh_rgb;
    letterbox_info.ratio = r_rgb;

    // 预处理RGB图像
    cv::Mat rgb_resized;
    cv::resize(rgb_img, rgb_resized, cv::Size(new_unpad_rgb_w, new_unpad_rgb_h));
    cv::Mat rgb_padded(input_h_, input_w_, CV_8UC3, cv::Scalar(114, 114, 114));
    rgb_resized.copyTo(rgb_padded(cv::Rect(dw_rgb, dh_rgb, new_unpad_rgb_w, new_unpad_rgb_h)));
    
    // 预处理IR图像
    cv::Mat ir_resized;
    cv::resize(ir_img, ir_resized, cv::Size(new_unpad_ir_w, new_unpad_ir_h));
    cv::Mat ir_padded(input_h_, input_w_, CV_8UC3, cv::Scalar(114, 114, 114));
    ir_resized.copyTo(ir_padded(cv::Rect(dw_ir, dh_ir, new_unpad_ir_w, new_unpad_ir_h)));

    // 转换为float并归一化
    rgb_padded.convertTo(rgb_padded, CV_32FC3, 1.0/255.0);
    ir_padded.convertTo(ir_padded, CV_32FC3, 1.0/255.0);

    // HWC to CHW
    rgb_input.resize(3 * input_h_ * input_w_);
    ir_input.resize(3 * input_h_ * input_w_);
    
    std::vector<cv::Mat> rgb_channels(3);
    std::vector<cv::Mat> ir_channels(3);
    cv::split(rgb_padded, rgb_channels);
    cv::split(ir_padded, ir_channels);

    for (int c = 0; c < 3; ++c) {
        memcpy(rgb_input.data() + c * input_h_ * input_w_,
               rgb_channels[c].data,
               input_h_ * input_w_ * sizeof(float));
        memcpy(ir_input.data() + c * input_h_ * input_w_,
               ir_channels[c].data,
               input_h_ * input_w_ * sizeof(float));
    }
}

std::vector<float> TRTInference::inference(const cv::Mat& rgb_img,
                                         const cv::Mat& ir_img,
                                         const std::vector<float>& homography,
                                         LetterBoxInfo& letterbox_info) {
    // 预处理
    std::vector<float> rgb_input, ir_input, h_input;
    preprocess(rgb_img, ir_img, homography, rgb_input, ir_input, h_input, letterbox_info);

    // 拷贝数据到GPU
    cudaMemcpyAsync(input_buffers_[0], rgb_input.data(),
                   rgb_input.size() * sizeof(float),
                   cudaMemcpyHostToDevice, stream_);
    cudaMemcpyAsync(input_buffers_[1], ir_input.data(),
                   ir_input.size() * sizeof(float),
                   cudaMemcpyHostToDevice, stream_);
    cudaMemcpyAsync(input_buffers_[2], h_input.data(),
                   h_input.size() * sizeof(float),
                   cudaMemcpyHostToDevice, stream_);

    // 设置输入张量地址
    for (int i = 0; i < input_dims_.size(); ++i) {
        context_->setTensorAddress(engine_->getIOTensorName(i), input_buffers_[i]);
    }

    // 设置输出张量地址
    for (int i = 0; i < output_dims_.size(); ++i) {
        context_->setTensorAddress(engine_->getIOTensorName(i + input_dims_.size()), output_buffers_[i]);
    }

    // 执行推理
    bool status = context_->enqueueV3(stream_);
    if (!status) {
        throw std::runtime_error("TensorRT inference failed");
    }
    cudaStreamSynchronize(stream_);

    // 获取输出
    size_t output_size = 1;
    for (int i = 0; i < output_dims_[0].nbDims; ++i) {
        output_size *= output_dims_[0].d[i];
    }
    std::vector<float> output(output_size);
    cudaMemcpyAsync(output.data(), output_buffers_[0],
                   output_size * sizeof(float),
                   cudaMemcpyDeviceToHost, stream_);
    cudaStreamSynchronize(stream_);

    return output;
}

std::vector<int> TRTInference::nms(const std::vector<std::vector<float>>& boxes,
                                 const std::vector<float>& scores,
                                 float iou_threshold) {
    std::vector<int> indices(scores.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(),
              [&scores](int i1, int i2) { return scores[i1] > scores[i2]; });

    std::vector<int> keep;
    while (!indices.empty()) {
        int idx = indices[0];
        keep.push_back(idx);
        indices.erase(indices.begin());

        std::vector<int> tmp_indices;
        for (int i : indices) {
            float iou = 0.0f;
            // 计算IOU
            float xx1 = std::max(boxes[idx][0], boxes[i][0]);
            float yy1 = std::max(boxes[idx][1], boxes[i][1]);
            float xx2 = std::min(boxes[idx][2], boxes[i][2]);
            float yy2 = std::min(boxes[idx][3], boxes[i][3]);

            float w = std::max(0.0f, xx2 - xx1);
            float h = std::max(0.0f, yy2 - yy1);
            float intersection = w * h;

            float area1 = (boxes[idx][2] - boxes[idx][0]) * (boxes[idx][3] - boxes[idx][1]);
            float area2 = (boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1]);
            float union_area = area1 + area2 - intersection;

            iou = intersection / (union_area + 1e-16f);

            if (iou <= iou_threshold) {
                tmp_indices.push_back(i);
            }
        }
        indices = tmp_indices;
    }
    return keep;
}

std::vector<std::vector<float>> TRTInference::process_output(
    const std::vector<float>& output,
    float conf_thres,
    float iou_thres,
    int num_classes,
    const LetterBoxInfo& letterbox_info) {
    
    // 打印输出信息
    std::cout << "\n输出信息:" << std::endl;
    std::cout << "输出大小: " << output.size() << std::endl;
    std::cout << "输出范围: [" << *std::min_element(output.begin(), output.end()) 
              << ", " << *std::max_element(output.begin(), output.end()) << "]" << std::endl;
    
    // 转置输出 [bs, 4+nc, 8400] -> [8400, 4+nc]
    std::vector<std::vector<float>> boxes;
    std::vector<float> scores;
    std::vector<int> class_ids;

    // 解析输出
    for (int i = 0; i < 8400; ++i) {
        float max_conf = 0.0f;
        int max_class = 0;
        
        // 获取类别分数
        for (int c = 0; c < num_classes; ++c) {
            // 正确计算索引：output[4 + c][i]
            float conf = output[(4 + c) * 8400 + i];
            if (conf > max_conf) {
                max_conf = conf;
                max_class = c;
            }
        }

        if (max_conf > conf_thres) {
            // 获取边界框坐标
            float x = output[i];  // output[0][i]
            float y = output[8400 + i];  // output[1][i]
            float w = output[2 * 8400 + i];  // output[2][i]
            float h = output[3 * 8400 + i];  // output[3][i]

            // 转换为xyxy格式
            float x1 = x - w/2;
            float y1 = y - h/2;
            float x2 = x + w/2;
            float y2 = y + h/2;

            // 应用letterbox变换
            x1 = (x1 - letterbox_info.dw) / letterbox_info.ratio;
            y1 = (y1 - letterbox_info.dh) / letterbox_info.ratio;
            x2 = (x2 - letterbox_info.dw) / letterbox_info.ratio;
            y2 = (y2 - letterbox_info.dh) / letterbox_info.ratio;

            // 检查边界框是否有效
            if (x1 >= 0 && y1 >= 0 && x2 > x1 && y2 > y1) {
                boxes.push_back({x1, y1, x2, y2});
                scores.push_back(max_conf);
                class_ids.push_back(max_class);
            }
        }
    }

    std::cout << "置信度阈值过滤后检测框数量: " << boxes.size() << std::endl;

    // 对每个类别分别应用NMS
    std::vector<std::vector<float>> results;
    for (int cls_id = 0; cls_id < num_classes; ++cls_id) {
        std::vector<std::vector<float>> cls_boxes;
        std::vector<float> cls_scores;
        std::vector<int> indices;
        
        // 收集当前类别的检测框
        for (size_t i = 0; i < class_ids.size(); ++i) {
            if (class_ids[i] == cls_id) {
                cls_boxes.push_back(boxes[i]);
                cls_scores.push_back(scores[i]);
                indices.push_back(i);
            }
        }
        
        if (cls_boxes.empty()) continue;
        
        std::cout << "类别 " << cls_id << " 的检测框数量: " << cls_boxes.size() << std::endl;
        
        // 应用NMS
        std::vector<int> keep = nms(cls_boxes, cls_scores, iou_thres);
        
        std::cout << "NMS后类别 " << cls_id << " 的检测框数量: " << keep.size() << std::endl;
        
        // 添加结果
        for (int i : keep) {
            results.push_back({cls_boxes[i][0], cls_boxes[i][1], cls_boxes[i][2], cls_boxes[i][3],
                             cls_scores[i], static_cast<float>(cls_id)});
        }
    }

    std::cout << "最终检测结果数量: " << results.size() << std::endl;
    return results;
}

// 新增：融合与保存函数
void TRTInference::fuse_and_save_padded_images(const cv::Mat& rgb_img, const cv::Mat& ir_img,
                                               const std::vector<float>& homography,
                                               const std::string& save_dir,
                                               const std::string& save_name) {
    // 计算letterbox参数
    float r_rgb = std::min(static_cast<float>(input_h_) / rgb_img.rows,
                          static_cast<float>(input_w_) / rgb_img.cols);
    float r_ir = std::min(static_cast<float>(input_h_) / ir_img.rows,
                         static_cast<float>(input_w_) / ir_img.cols);

    int new_unpad_rgb_w = static_cast<int>(rgb_img.cols * r_rgb);
    int new_unpad_rgb_h = static_cast<int>(rgb_img.rows * r_rgb);
    int new_unpad_ir_w = static_cast<int>(ir_img.cols * r_ir);
    int new_unpad_ir_h = static_cast<int>(ir_img.rows * r_ir);

    float dw_rgb = (input_w_ - new_unpad_rgb_w) / 2.0f;
    float dh_rgb = (input_h_ - new_unpad_rgb_h) / 2.0f;
    float dw_ir = (input_w_ - new_unpad_ir_w) / 2.0f;
    float dh_ir = (input_h_ - new_unpad_ir_h) / 2.0f;

    // 更新单应性矩阵
    std::vector<float> S_rgb = {r_rgb, 0, 0, 0, r_rgb, 0, 0, 0, 1};
    std::vector<float> S_ir = {r_ir, 0, 0, 0, r_ir, 0, 0, 0, 1};
    std::vector<float> T_rgb = {1, 0, dw_rgb, 0, 1, dh_rgb, 0, 0, 1};
    std::vector<float> T_ir = {1, 0, dw_ir, 0, 1, dh_ir, 0, 0, 1};
    // 这里简化处理，实际应为 H_new = T_rgb @ S_rgb @ H @ S_ir^(-1) @ T_ir^(-1)
    std::vector<float> h_input = homography;

    // 生成padded图像
    cv::Mat rgb_resized, ir_resized;
    cv::resize(rgb_img, rgb_resized, cv::Size(new_unpad_rgb_w, new_unpad_rgb_h));
    cv::resize(ir_img, ir_resized, cv::Size(new_unpad_ir_w, new_unpad_ir_h));
    cv::Mat rgb_padded(input_h_, input_w_, CV_8UC3, cv::Scalar(114, 114, 114));
    cv::Mat ir_padded(input_h_, input_w_, CV_8UC3, cv::Scalar(114, 114, 114));
    rgb_resized.copyTo(rgb_padded(cv::Rect(dw_rgb, dh_rgb, new_unpad_rgb_w, new_unpad_rgb_h)));
    ir_resized.copyTo(ir_padded(cv::Rect(dw_ir, dh_ir, new_unpad_ir_w, new_unpad_ir_h)));

    // 单应性变换（注意h_input为float型vector）
    cv::Mat H = cv::Mat(3, 3, CV_32F, h_input.data()).clone();
    cv::Mat ir_warped;
    cv::warpPerspective(ir_padded, ir_warped, H, rgb_padded.size());

    // 融合
    cv::Mat fused_img;
    cv::addWeighted(rgb_padded, 0.6, ir_warped, 0.4, 0, fused_img);

    // 保存
    std::string fused_save_path = save_dir + "/" + save_name;
    cv::imwrite(fused_save_path, fused_img);
    std::cout << "融合图像已保存到: " << fused_save_path << std::endl;
}