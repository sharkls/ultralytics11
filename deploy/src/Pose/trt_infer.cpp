#include "trt_infer.h"
#include <fstream>
#include <algorithm>
#include <cmath>

TRTInference::TRTInference(const std::string& engine_path) {
    // 初始化成员变量
    input_buffers_.resize(1, nullptr);
    output_buffers_.resize(1, nullptr);

    // 初始化CUDA流
    cudaError_t cuda_status = cudaStreamCreate(&stream_);
    if (cuda_status != cudaSuccess) {
        throw std::runtime_error("创建CUDA流失败: " + std::string(cudaGetErrorString(cuda_status)));
    }

    // 加载engine
    std::ifstream file(engine_path, std::ios::binary);
    if (!file.good()) {
        throw std::runtime_error("无法打开engine文件");
    }

    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> engineData(size);
    file.read(engineData.data(), size);
    file.close();

    std::cout << "engine_path: " << engine_path << std::endl;
    std::cout << "[TensorRT] Loaded engine size: " << size / (1024 * 1024) << " MiB" << std::endl;

    // 创建runtime和engine
    runtime_.reset(nvinfer1::createInferRuntime(logger_));
    if (!runtime_) {
        throw std::runtime_error("创建TensorRT runtime失败");
    }

    engine_.reset(runtime_->deserializeCudaEngine(engineData.data(), size));
    if (!engine_) {
        throw std::runtime_error("反序列化engine失败");
    }

    // 创建执行上下文
    context_.reset(engine_->createExecutionContext());
    if (!context_) {
        throw std::runtime_error("创建执行上下文失败");
    }

    // 获取输入输出信息
    const char* input_name = engine_->getIOTensorName(0);
    const char* output_name = engine_->getIOTensorName(1);

    std::cout << "TensorRT模型绑定数量: " << engine_->getNbIOTensors() << std::endl;
    std::cout << "输入张量名称: " << input_name << std::endl;
    std::cout << "输出张量名称: " << output_name << std::endl;

    // 获取输入维度信息
    nvinfer1::Dims input_dims = engine_->getTensorShape(input_name);
    std::cout << "输入维度数量: " << input_dims.nbDims << std::endl;
    for (int i = 0; i < input_dims.nbDims; ++i) {
        std::cout << "输入维度 " << i << ": " << input_dims.d[i] << std::endl;
    }

    // 获取输出维度信息
    nvinfer1::Dims output_dims = engine_->getTensorShape(output_name);
    std::cout << "输出维度数量: " << output_dims.nbDims << std::endl;
    for (int i = 0; i < output_dims.nbDims; ++i) {
        std::cout << "输出维度 " << i << ": " << output_dims.d[i] << std::endl;
    }

    // 检查是否支持动态形状
    bool input_dynamic = false;
    bool output_dynamic = false;
    for (int i = 0; i < input_dims.nbDims; ++i) {
        if (input_dims.d[i] == -1) {
            input_dynamic = true;
            break;
        }
    }
    for (int i = 0; i < output_dims.nbDims; ++i) {
        if (output_dims.d[i] == -1) {
            output_dynamic = true;
            break;
        }
    }
    std::cout << "输入是否支持动态形状: " << (input_dynamic ? "是" : "否") << std::endl;
    std::cout << "输出是否支持动态形状: " << (output_dynamic ? "是" : "否") << std::endl;

    // 设置初始输入形状
    nvinfer1::Dims dims;
    dims.nbDims = 4;
    dims.d[0] = 1;  // batch size
    dims.d[1] = 3;  // channels
    dims.d[2] = 640;  // height
    dims.d[3] = 640;  // width

    std::cout << "设置初始输入形状: " << dims.d[0] << "x" << dims.d[1] 
              << "x" << dims.d[2] << "x" << dims.d[3] << std::endl;

    if (!context_->setInputShape(input_name, dims)) {
        throw std::runtime_error("设置输入形状失败");
    }

    // 计算输入输出大小
    size_t input_size = 1;
    for (int i = 0; i < input_dims.nbDims; ++i) {
        input_size *= (input_dims.d[i] == -1 ? dims.d[i] : input_dims.d[i]);
    }

    // 使用固定的输出大小
    size_t output_size = 1 * 56 * 8400;  // batch_size * num_classes+4+num_masks * num_anchors

    std::cout << "输入张量大小: " << input_size * sizeof(float) << " 字节" << std::endl;
    std::cout << "输出张量大小: " << output_size * sizeof(float) << " 字节" << std::endl;

    // 检查输入大小是否合理
    if (input_size == 0 || input_size > 1000000000) {  // 设置一个合理的上限
        throw std::runtime_error("输入张量大小不合理: " + std::to_string(input_size));
    }

    // 检查输出大小是否合理
    if (output_size == 0 || output_size > 1000000000) {  // 设置一个合理的上限
        throw std::runtime_error("输出张量大小不合理: " + std::to_string(output_size));
    }

    // 分配GPU内存
    void* input_buffer = nullptr;
    cuda_status = cudaMalloc(&input_buffer, input_size * sizeof(float));
    if (cuda_status != cudaSuccess) {
        throw std::runtime_error("分配输入GPU内存失败: " + std::string(cudaGetErrorString(cuda_status)));
    }
    input_buffers_[0] = input_buffer;

    void* output_buffer = nullptr;
    cuda_status = cudaMalloc(&output_buffer, output_size * sizeof(float));
    if (cuda_status != cudaSuccess) {
        // 清理已分配的内存
        cudaFree(input_buffer);
        throw std::runtime_error("分配输出GPU内存失败: " + std::string(cudaGetErrorString(cuda_status)));
    }
    output_buffers_[0] = output_buffer;

    // 设置绑定
    if (!context_->setTensorAddress(input_name, input_buffers_[0])) {
        // 清理已分配的内存
        cudaFree(input_buffer);
        cudaFree(output_buffer);
        throw std::runtime_error("设置输入张量地址失败");
    }
    if (!context_->setTensorAddress(output_name, output_buffers_[0])) {
        // 清理已分配的内存
        cudaFree(input_buffer);
        cudaFree(output_buffer);
        throw std::runtime_error("设置输出张量地址失败");
    }

    std::cout << "input_name: " << input_name << ", output_name: " << output_name << std::endl;
    std::cout << "input_dims: ";
    for (int i = 0; i < input_dims.nbDims; ++i) std::cout << input_dims.d[i] << " ";
    std::cout << std::endl;
    std::cout << "output_dims: ";
    for (int i = 0; i < output_dims.nbDims; ++i) std::cout << output_dims.d[i] << " ";
    std::cout << std::endl;
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

    context_.reset();
    engine_.reset();
    runtime_.reset();
}

void save_bin(const std::vector<float>& input_data, const std::string& filename) {
    std::ofstream ofs(filename, std::ios::binary);
    ofs.write(reinterpret_cast<const char*>(input_data.data()), input_data.size() * sizeof(float));
    ofs.close();
}

void TRTInference::preprocess(const cv::Mat& img,
                            std::vector<float>& input,
                            LetterBoxInfo& letterbox_info) {
    // 检查输入图像
    if (img.empty()) {
        throw std::runtime_error("输入图像为空");
    }
    if (img.cols <= 0 || img.rows <= 0) {
        throw std::runtime_error("输入图像尺寸无效");
    }

    // 打印调试信息
    std::cout << "预处理 - 输入图像尺寸: " << img.cols << "x" << img.rows << std::endl;

    // 计算目标尺寸
    float r = std::min(static_cast<float>(input_h_) / static_cast<float>(img.rows),
                      static_cast<float>(input_w_) / static_cast<float>(img.cols));

    // 计算缩放后的尺寸
    int new_unpad_h = static_cast<int>(img.rows * r);
    int new_unpad_w = static_cast<int>(img.cols * r);

    // 确保尺寸是stride的整数倍
    new_unpad_h = (new_unpad_h / stride_) * stride_;
    new_unpad_w = (new_unpad_w / stride_) * stride_;

    // 计算实际需要的填充尺寸
    int target_h = new_unpad_h;
    int target_w = new_unpad_w;
    if (img.rows > img.cols) {
        // 高度大于宽度，宽度需要填充
        target_w = ((new_unpad_w + stride_ - 1) / stride_) * stride_;
    } else {
        // 宽度大于高度，高度需要填充
        target_h = ((new_unpad_h + stride_ - 1) / stride_) * stride_;
    }

    // 计算填充
    float dw = (target_w - new_unpad_w) / 2.0f;
    float dh = (target_h - new_unpad_h) / 2.0f;

    // 保存letterbox信息
    letterbox_info.dw = dw;
    letterbox_info.dh = dh;
    letterbox_info.ratio = r;
    letterbox_info.new_unpad_h = new_unpad_h;
    letterbox_info.new_unpad_w = new_unpad_w;
    letterbox_info.stride = stride_;

    // 预处理图像
    cv::Mat resized;
    try {
        cv::resize(img, resized, cv::Size(new_unpad_w, new_unpad_h), 0, 0, cv::INTER_LINEAR);
    } catch (const cv::Exception& e) {
        throw std::runtime_error("图像缩放失败: " + std::string(e.what()));
    }

    // 创建填充后的图像
    cv::Mat padded(target_h, target_w, CV_8UC3, cv::Scalar(114, 114, 114));
    
    // 确保ROI有效
    if (dw >= 0 && dh >= 0 && new_unpad_w > 0 && new_unpad_h > 0) {
        resized.copyTo(padded(cv::Rect(dw, dh, resized.cols, resized.rows)));
    } else {
        throw std::runtime_error("无效的ROI参数");
    }

    // 转换为float并归一化
    padded.convertTo(padded, CV_32FC3, 1.0/255.0);

    // HWC to CHW
    input.resize(3 * target_h * target_w);
    std::vector<cv::Mat> channels(3);
    cv::split(padded, channels);

    for (int c = 0; c < 3; ++c) {
        memcpy(input.data() + c * target_h * target_w,
               channels[c].data,
               target_h * target_w * sizeof(float));
    }

    std::cout << "预处理 - 缩放比例: " << r << std::endl;
    std::cout << "预处理 - 缩放后尺寸: " << new_unpad_w << "x" << new_unpad_h << std::endl;
    std::cout << "预处理 - 目标尺寸: " << target_w << "x" << target_h << std::endl;
    std::cout << "预处理 - 填充尺寸: " << dw << "," << dh << std::endl;

    // 新增：保存bin文件
    save_bin(input, "preprocess_trt_infer.bin");
}

void TRTInference::rescale_coords(std::vector<float>& coords, 
                                const LetterBoxInfo& letterbox_info,
                                bool is_keypoint) {
    if (coords.empty()) return;

    float r = letterbox_info.ratio;
    float dw = letterbox_info.dw;
    float dh = letterbox_info.dh;

    if (is_keypoint) {
        // 关键点坐标还原
        // 每个关键点有3个值(x, y, conf)，所以步长为3
        for (size_t i = 0; i < coords.size(); i += 3) {
            // x坐标：先减去左填充，再除以缩放比例
            coords[i] = (coords[i] - dw) / r;
            // y坐标：先减去上填充，再除以缩放比例
            coords[i + 1] = (coords[i + 1] - dh) / r;
            // 保持置信度不变
        }
    } else {
        // 检测框坐标还原
        for (size_t i = 0; i < coords.size(); i += 4) {
            // x1坐标：先减去左填充，再除以缩放比例
            coords[i] = (coords[i] - dw) / r;
            // y1坐标：先减去上填充，再除以缩放比例
            coords[i + 1] = (coords[i + 1] - dh) / r;
            // x2坐标：先减去左填充，再除以缩放比例
            coords[i + 2] = (coords[i + 2] - dw) / r;
            // y2坐标：先减去上填充，再除以缩放比例
            coords[i + 3] = (coords[i + 3] - dh) / r;
        }
    }
}

std::vector<std::vector<float>> TRTInference::process_keypoints(
    const std::vector<float>& output,
    const std::vector<std::vector<float>>& boxes,
    const LetterBoxInfo& letterbox_info) {
    
    std::vector<std::vector<float>> keypoints;
    int num_keypoints = 17;  // COCO格式的关键点数量
    // int num_anchors = 5460;  // 固定anchor数量
    
    for (size_t i = 0; i < boxes.size(); ++i) {
        std::vector<float> kpts;
        // 从输出中获取关键点坐标（从第6个元素开始，每个检测框有6个基础元素）
        int kpt_start = 6;  // 4(bbox) + 1(conf) + 1(cls)
        for (int j = 0; j < num_keypoints; ++j) {
            // 获取关键点坐标
            float x = output[kpt_start + j * 3];
            float y = output[kpt_start + j * 3 + 1];
            float conf = output[kpt_start + j * 3 + 2];
            
            kpts.push_back(x);
            kpts.push_back(y);
            kpts.push_back(conf);
        }
        // 还原关键点坐标
        rescale_coords(kpts, letterbox_info, true);
        keypoints.push_back(kpts);
    }
    
    return keypoints;
}

std::vector<float> TRTInference::inference(const cv::Mat& img, LetterBoxInfo& letterbox_info) 
{
    // 预处理
    std::vector<float> input;
    preprocess(img, input, letterbox_info);
    std::cout << "preprocess output input.size() = " << input.size() << std::endl;

    std::cout << "推理输入 shape: 1x3x" << letterbox_info.new_unpad_h << "x" << letterbox_info.new_unpad_w << std::endl;
    for (int i = 0; i < std::min(10, (int)input.size()); ++i) {
        std::cout << "input[" << i << "] = " << input[i] << std::endl;
    }

    // 检查输入缓冲区
    if (input_buffers_.empty() || input_buffers_[0] == nullptr) {
        throw std::runtime_error("输入缓冲区未正确初始化");
    }

    // 设置动态输入尺寸
    const char* input_name = engine_->getIOTensorName(0);
    nvinfer1::Dims input_dims;
    input_dims.nbDims = 4;
    input_dims.d[0] = 1;  // batch size
    input_dims.d[1] = 3;  // channels
    input_dims.d[2] = letterbox_info.new_unpad_h;  // height
    input_dims.d[3] = letterbox_info.new_unpad_w;  // width
    
    std::cout << "设置动态输入形状: " << input_dims.d[0] << "x" << input_dims.d[1] 
              << "x" << input_dims.d[2] << "x" << input_dims.d[3] << std::endl;
    
    if (!context_->setInputShape(input_name, input_dims)) {
        throw std::runtime_error("设置输入形状失败");
    }

    // 拷贝数据到GPU
    size_t input_size = input.size() * sizeof(float);
    cudaError_t cuda_status = cudaMemcpyAsync(input_buffers_[0], input.data(),
                                            input_size,
                                            cudaMemcpyHostToDevice, stream_);
    if (cuda_status != cudaSuccess) {
        throw std::runtime_error("CUDA内存拷贝失败: " + std::string(cudaGetErrorString(cuda_status)));
    }

    // 执行推理
    bool status = context_->enqueueV3(stream_);
    if (!status) {
        throw std::runtime_error("TensorRT推理失败");
    }
    cudaStreamSynchronize(stream_);

    // 获取输出形状
    const char* output_name = engine_->getIOTensorName(1);
    nvinfer1::Dims output_dims = context_->getTensorShape(output_name);
    
    // 计算输出大小
    size_t output_size = 1;  // batch size
    for (int i = 0; i < output_dims.nbDims; ++i) {
        output_size *= output_dims.d[i];
    }
    std::cout << "inference output_size = " << output_size << std::endl;
    std::cout << "推理 - 输入尺寸: " << letterbox_info.new_unpad_w << "x" << letterbox_info.new_unpad_h << std::endl;
    std::cout << "推理 - 输出维度: ";
    for (int i = 0; i < output_dims.nbDims; ++i) {
        std::cout << output_dims.d[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "推理 - 输出大小: " << output_size << std::endl;

    // 获取输出
    std::vector<float> output(output_size);
    cuda_status = cudaMemcpyAsync(output.data(), output_buffers_[0],
                                 output_size * sizeof(float),
                                 cudaMemcpyDeviceToHost, stream_);
    if (cuda_status != cudaSuccess) {
        throw std::runtime_error("CUDA输出内存拷贝失败: " + std::string(cudaGetErrorString(cuda_status)));
    }
    cudaStreamSynchronize(stream_);

    // 新增：保存推理输出为bin文件
    save_bin(output, "output_trt_infer.bin");

    std::cout << "推理输出 shape: " << output.size() << std::endl;
    for (int i = 0; i < std::min(10, (int)output.size()); ++i) {
        std::cout << "output[" << i << "] = " << output[i] << std::endl;
    }

    return output;
}

std::vector<std::vector<float>> TRTInference::process_output(
    const std::vector<float>& output,
    float conf_thres,
    float iou_thres,
    int num_classes,
    const LetterBoxInfo& letterbox_info)
{
    // 获取输出形状
    const char* output_name = engine_->getIOTensorName(1);
    nvinfer1::Dims output_dims = context_->getTensorShape(output_name);
    
    // 计算特征图尺寸对应的anchor数量
    int num_anchors = 0;
    std::vector<std::pair<int, int>> feature_maps;
    
    // 计算每个stride对应的特征图尺寸
    for (int stride : {8, 16, 32}) {  // YOLOv8使用的stride
        int h = letterbox_info.new_unpad_h / stride;
        int w = letterbox_info.new_unpad_w / stride;
        feature_maps.push_back({h, w});
        num_anchors += h * w;
    }
    
    std::cout << "处理输出 - 特征图尺寸: ";
    for (const auto& [h, w] : feature_maps) {
        std::cout << h << "x" << w << " ";
    }
    std::cout << std::endl;
    std::cout << "处理输出 - 总anchor数量: " << num_anchors << std::endl;

    // 将输出重塑为(batch_size, num_anchors, num_classes + 4 + num_masks)格式
    // 注意：TensorRT输出格式是(1, 56, 5460)，需要转置为(1, 5460, 56)
    std::vector<std::vector<float>> reshaped_output(1, std::vector<float>(num_anchors * 56));
    for (int i = 0; i < num_anchors; ++i) {
        for (int j = 0; j < 56; ++j) {
            // 转置操作：从(j, i)到(i, j)
            reshaped_output[0][i * 56 + j] = output[j * num_anchors + i];
        }
    }
    std::cout << "重塑输出 - 维度0: " << reshaped_output.size() << "x" << reshaped_output[0].size() << std::endl;

    // 获取类别置信度
    std::vector<float> class_scores(num_anchors);
    std::vector<int> class_ids(num_anchors);
    for (int i = 0; i < num_anchors; ++i) {
        float max_conf = 0.0f;
        int max_class = 0;
        for (int c = 0; c < num_classes; ++c) {
            // 注意：这里使用转置后的索引
            float conf = reshaped_output[0][i * 56 + 4 + c];
            if (conf > max_conf) {
                max_conf = conf;
                max_class = c;
            }
        }
        class_scores[i] = max_conf;
        class_ids[i] = max_class;
    }

    // 置信度过滤
    std::vector<int> candidates;
    for (int i = 0; i < num_anchors; ++i) {
        if (class_scores[i] > conf_thres) {
            candidates.push_back(i);
        }
    }

    std::cout << "置信度过滤后剩余框数量: " << candidates.size() << std::endl;

    // 准备NMS输入
    std::vector<std::vector<float>> boxes;
    std::vector<float> scores;
    std::vector<int> class_ids_filtered;
    std::vector<std::vector<float>> keypoints;

    for (int idx : candidates) {
        // 获取边界框坐标 - 使用转置后的索引
        float x = reshaped_output[0][idx * 56];
        float y = reshaped_output[0][idx * 56 + 1];
        float w = reshaped_output[0][idx * 56 + 2];
        float h = reshaped_output[0][idx * 56 + 3];

        // 转换为xyxy格式
        float x1 = x - w/2;
        float y1 = y - h/2;
        float x2 = x + w/2;
        float y2 = y + h/2;

        // 检查边界框是否有效
        if (x1 >= 0 && y1 >= 0 && x2 > x1 && y2 > y1) {
            std::vector<float> box = {x1, y1, x2, y2};
            rescale_coords(box, letterbox_info, false);
            boxes.push_back(box);
            scores.push_back(class_scores[idx]);
            class_ids_filtered.push_back(class_ids[idx]);

            // 提取关键点 - 使用转置后的索引
            std::vector<float> kpts;
            int kpt_start = 5;  // 4(bbox) + 1(conf) + 1(cls)
            int kpt_dim = 51;   // 17*3
            for (int j = 0; j < kpt_dim; j += 3) {
                float kpt_x = reshaped_output[0][idx * 56 + kpt_start + j];
                float kpt_y = reshaped_output[0][idx * 56 + kpt_start + j + 1];
                float kpt_conf = reshaped_output[0][idx * 56 + kpt_start + j + 2];
                kpts.push_back(kpt_x);
                kpts.push_back(kpt_y);
                kpts.push_back(kpt_conf);
                // std::cout << "关键点 " << j << ": (" << kpt_x << ", " << kpt_y << "), 置信度: " << kpt_conf << std::endl;
            }
            rescale_coords(kpts, letterbox_info, true);
            keypoints.push_back(kpts);
        }
    }

    // 在添加结果之前，打印预处理参数
    std::cout << "\n预处理参数信息:" << std::endl;
    std::cout << "缩放比例(ratio): " << letterbox_info.ratio << std::endl;
    std::cout << "左填充(dw): " << letterbox_info.dw << std::endl;
    std::cout << "上填充(dh): " << letterbox_info.dh << std::endl;
    std::cout << "缩放后高度(new_unpad_h): " << letterbox_info.new_unpad_h << std::endl;
    std::cout << "缩放后宽度(new_unpad_w): " << letterbox_info.new_unpad_w << std::endl;

    // 对每个类别分别应用NMS
    std::vector<std::vector<float>> results;
    for (int cls_id = 0; cls_id < num_classes; ++cls_id) {
        std::vector<std::vector<float>> cls_boxes;
        std::vector<float> cls_scores;
        std::vector<int> indices;
        std::vector<std::vector<float>> cls_keypoints;
        
        // 收集当前类别的检测框和关键点
        for (size_t i = 0; i < class_ids_filtered.size(); ++i) {
            if (class_ids_filtered[i] == cls_id) {
                cls_boxes.push_back(boxes[i]);
                cls_scores.push_back(scores[i]);
                cls_keypoints.push_back(keypoints[i]);
                indices.push_back(i);
            }
        }
        
        if (cls_boxes.empty()) continue;
        
        // 应用NMS
        std::vector<int> keep = nms(cls_boxes, cls_scores, iou_thres);
        
        std::cout << "\n类别 " << cls_id << " NMS后剩余框数量: " << keep.size() << std::endl;
        
        // 添加结果
        for (int i : keep) {
            std::vector<float> result = {
                cls_boxes[i][0], cls_boxes[i][1], cls_boxes[i][2], cls_boxes[i][3],
                cls_scores[i], static_cast<float>(cls_id)
            };
            
            // 添加关键点信息
            result.insert(result.end(), cls_keypoints[i].begin(), cls_keypoints[i].end());
            results.push_back(result);

            // 打印每个检测框的详细信息
            std::cout << "\n检测框 " << results.size() - 1 << " 信息:" << std::endl;
            std::cout << "边界框坐标: [" << cls_boxes[i][0] << ", " << cls_boxes[i][1] 
                      << ", " << cls_boxes[i][2] << ", " << cls_boxes[i][3] << "]" << std::endl;
            std::cout << "置信度: " << cls_scores[i] << std::endl;
            
            // 打印关键点信息
            std::cout << "关键点信息:" << std::endl;
            // COCO数据集17个关键点的含义
            const std::vector<std::string> keypoint_names = {
                "nose",          // 0: 鼻子
                "left_eye",      // 1: 左眼
                "right_eye",     // 2: 右眼
                "left_ear",      // 3: 左耳
                "right_ear",     // 4: 右耳
                "left_shoulder", // 5: 左肩
                "right_shoulder",// 6: 右肩
                "left_elbow",    // 7: 左肘
                "right_elbow",   // 8: 右肘
                "left_wrist",    // 9: 左手腕
                "right_wrist",   // 10: 右手腕
                "left_hip",      // 11: 左髋
                "right_hip",     // 12: 右髋
                "left_knee",     // 13: 左膝
                "right_knee",    // 14: 右膝
                "left_ankle",    // 15: 左踝
                "right_ankle"    // 16: 右踝
            };
            
            for (int j = 0; j < 17; ++j) {
                // 从原始输出中获取关键点信息
                int kpt_start = 5;  // 4(bbox) + 1(conf) + 1(cls)
                float kpt_x = reshaped_output[0][indices[i] * 56 + kpt_start + j * 3];
                float kpt_y = reshaped_output[0][indices[i] * 56 + kpt_start + j * 3 + 1];
                float kpt_conf = reshaped_output[0][indices[i] * 56 + kpt_start + j * 3 + 2];
                
                // 还原关键点坐标
                std::vector<float> kpt = {kpt_x, kpt_y, kpt_conf};
                rescale_coords(kpt, letterbox_info, true);
                
                std::cout << keypoint_names[j] << " (" << kpt[0] << ", " << kpt[1] 
                          << "), 置信度: " << kpt[2] << std::endl;
            }
        }

        for (size_t i = 0; i < cls_boxes.size(); ++i) {
            std::cout << "[DEBUG] NMS前 box " << i << ": [" << cls_boxes[i][0] << ", " << cls_boxes[i][1] << ", " << cls_boxes[i][2] << ", " << cls_boxes[i][3] << "], score = " << cls_scores[i] << std::endl;
        }
        for (size_t i = 0; i < keep.size(); ++i) {
            int idx = keep[i];
            std::cout << "[DEBUG] NMS后 box " << i << ": [" << cls_boxes[idx][0] << ", " << cls_boxes[idx][1] << ", " << cls_boxes[idx][2] << ", " << cls_boxes[idx][3] << "], score = " << cls_scores[idx] << std::endl;
        }
    }

    // 对结果进行排序（按置信度降序）
    std::sort(results.begin(), results.end(),
              [](const std::vector<float>& a, const std::vector<float>& b) {
                  return a[4] > b[4];  // 按置信度排序
              });

    // 限制最大检测数量
    const int max_detections = 300;  // 与Python中的max_det一致
    if (results.size() > max_detections) {
        results.resize(max_detections);
    }

    std::cout << "\n最终检测结果统计:" << std::endl;
    std::cout << "总检测数量: " << results.size() << std::endl;
    if (!results.empty()) {
        std::cout << "最高置信度: " << results[0][4] << std::endl;
        std::cout << "最低置信度: " << results.back()[4] << std::endl;
    }

    return results;
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