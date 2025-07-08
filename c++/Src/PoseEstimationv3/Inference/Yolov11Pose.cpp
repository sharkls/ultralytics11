#include "Yolov11Pose.h"

// 注册模块
REGISTER_MODULE("PoseEstimation", Yolov11Pose, Yolov11Pose)

Yolov11Pose::Yolov11Pose(const std::string& exe_path) : IBaseModule(exe_path) 
{
    // 构造函数初始化
    input_buffers_.resize(1, nullptr);
    output_buffers_.resize(1, nullptr);
    m_maxBatchSize = 8; // 设置最大批处理大小

    // 初始化CUDA流
    cudaError_t cuda_status = cudaStreamCreate(&stream_);
    if (cuda_status != cudaSuccess) {
        throw std::runtime_error("创建CUDA流失败: " + std::string(cudaGetErrorString(cuda_status)));
    }
}

Yolov11Pose::~Yolov11Pose() {
    cleanup();
}

bool Yolov11Pose::init(void* p_pAlgParam) 
{   
    LOG(INFO) << "Yolov11Pose::init status: start ";
    // 1. 配置参数核验
    if (!p_pAlgParam) return false;
    m_poseConfig = *static_cast<posetimation::YOLOModelConfig*>(p_pAlgParam);

    // 2. 配置参数获取
    engine_path_ = m_poseConfig.engine_path();
    conf_thres_ = m_poseConfig.conf_thres();
    iou_thres_ = m_poseConfig.iou_thres();
    num_classes_ = m_poseConfig.num_class();
    new_unpad_h_ = m_poseConfig.new_unpad_h();
    new_unpad_w_ = m_poseConfig.new_unpad_w();
    dw_ = m_poseConfig.dw();
    dh_ = m_poseConfig.dh();
    ratio_ = m_poseConfig.resize_ratio();
    stride_.assign(m_poseConfig.stride().begin(), m_poseConfig.stride().end());
    channels_ = m_poseConfig.channels();
    batch_size_ = m_poseConfig.batch_size();
    num_keys_ = m_poseConfig.num_keys();
    max_dets_ = m_poseConfig.max_dets();
    src_width_ = m_poseConfig.src_width();
    src_height_ = m_poseConfig.src_height();
    status_ = m_poseConfig.run_status();

    // 3. 如果new_unpad_w_或new_unpad_h_为0，使用配置中的width和height作为默认值
    if (new_unpad_w_ <= 0 || new_unpad_h_ <= 0) {
        new_unpad_w_ = m_poseConfig.width();
        new_unpad_h_ = m_poseConfig.height();
        LOG(INFO) << "Using default dimensions from config: " << new_unpad_w_ << "x" << new_unpad_h_;
    }

    // 计算anchor_nums
    for(int stride : stride_) {
        num_anchors_ += (new_unpad_h_ / stride) * (new_unpad_w_ / stride);
    }

    // 初始化TensorRT相关配置
    initTensorRT();
    LOG(INFO) << "Yolov11Pose::init status: success ";
    return true;
}

void Yolov11Pose::initTensorRT()
{
    // 1. 读取engine文件
    std::ifstream file(engine_path_, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("无法打开engine文件: " + engine_path_);
    }

    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> engineData(size);
    file.read(engineData.data(), size);
    file.close();
    LOG(INFO) << "engine_path_: " << engine_path_;
    LOG(INFO) << "[TensorRT] Loaded engine size: " << size / (1024 * 1024) << " MiB";

    // 2. 创建runtime和engine
    runtime_.reset(nvinfer1::createInferRuntime(logger_));
    if (!runtime_) {
        throw std::runtime_error("创建TensorRT runtime失败");
    }

    engine_.reset(runtime_->deserializeCudaEngine(engineData.data(), size));
    if (!engine_) {
        throw std::runtime_error("反序列化engine失败");
    }

    // 3. 创建执行上下文
    context_.reset(engine_->createExecutionContext());
    if (!context_) {
        throw std::runtime_error("创建执行上下文失败");
    }

    // 4. 获取输入输出信息
    input_name_ = engine_->getIOTensorName(0);
    output_name_ = engine_->getIOTensorName(1);
    
    // 5. 设置初始输入形状（使用配置中的默认值）
    input_dims_.nbDims = 4;
    input_dims_.d[0] = batch_size_;
    input_dims_.d[1] = channels_;
    input_dims_.d[2] = new_unpad_h_;
    input_dims_.d[3] = new_unpad_w_;
    
    if (!context_->setInputShape(input_name_, input_dims_)) {
        throw std::runtime_error("设置初始输入形状失败");
    }

    output_dims_ = context_->getTensorShape(output_name_);

    // 6. 计算最大可能的内存需求（用于预分配）
    size_t max_batch_size = m_maxBatchSize;
    size_t max_input_size = max_batch_size * channels_ * new_unpad_h_ * new_unpad_w_;
    size_t max_output_size = max_batch_size * (4 + num_classes_ + num_keys_ * 3) * num_anchors_;
    
    LOG(INFO) << "最大batch_size: " << max_batch_size;
    LOG(INFO) << "最大输入大小: " << max_input_size << " (约 " << (max_input_size * sizeof(float)) / (1024*1024) << "MB)";
    LOG(INFO) << "最大输出大小: " << max_output_size << " (约 " << (max_output_size * sizeof(float)) / (1024*1024) << "MB)";

    // 7. 检查GPU内存
    size_t free_mem, total_mem;
    cudaError_t cuda_status = cudaMemGetInfo(&free_mem, &total_mem);
    if (cuda_status != cudaSuccess) {
        throw std::runtime_error("获取GPU内存信息失败: " + std::string(cudaGetErrorString(cuda_status)));
    }
    
    size_t required_mem = (max_input_size + max_output_size) * sizeof(float);
    LOG(INFO) << "GPU内存信息 - 总内存: " << total_mem / (1024*1024) << "MB, 可用内存: " << free_mem / (1024*1024) << "MB";
    LOG(INFO) << "需要分配内存: " << required_mem / (1024*1024) << "MB";
    
    if (free_mem < required_mem) {
        throw std::runtime_error("GPU内存不足，需要 " + std::to_string(required_mem / (1024*1024)) + 
                                "MB，但只有 " + std::to_string(free_mem / (1024*1024)) + "MB 可用");
    }

    // 8. 预分配最大可能的GPU内存
    try {
        // 8.1 分配输入GPU内存
        void* input_buffer = nullptr;
        cuda_status = cudaMalloc(&input_buffer, max_input_size * sizeof(float));
        if (cuda_status != cudaSuccess) {
            throw std::runtime_error("分配输入GPU内存失败: " + std::string(cudaGetErrorString(cuda_status)));
        }
        input_buffers_[0] = input_buffer;
        LOG(INFO) << "成功分配输入GPU内存: " << (max_input_size * sizeof(float)) / (1024*1024) << "MB";

        // 8.2 分配输出GPU内存
        void* output_buffer = nullptr;
        cuda_status = cudaMalloc(&output_buffer, max_output_size * sizeof(float));
        if (cuda_status != cudaSuccess) {
            // 清理已分配的内存
            cudaFree(input_buffer);
            input_buffers_[0] = nullptr;
            throw std::runtime_error("分配输出GPU内存失败: " + std::string(cudaGetErrorString(cuda_status)));
        }
        output_buffers_[0] = output_buffer;
        LOG(INFO) << "成功分配输出GPU内存: " << (max_output_size * sizeof(float)) / (1024*1024) << "MB";

        // 9. 设置张量地址
        if (!context_->setTensorAddress(input_name_, input_buffers_[0])) {
            // 清理已分配的内存
            cudaFree(input_buffer);
            cudaFree(output_buffer);
            input_buffers_[0] = nullptr;
            output_buffers_[0] = nullptr;
            throw std::runtime_error("设置输入张量地址失败");
        }
        if (!context_->setTensorAddress(output_name_, output_buffers_[0])) {
            // 清理已分配的内存
            cudaFree(input_buffer);
            cudaFree(output_buffer);
            input_buffers_[0] = nullptr;
            output_buffers_[0] = nullptr;
            throw std::runtime_error("设置输出张量地址失败");
        }
        
        LOG(INFO) << "TensorRT初始化成功完成";
        
    } catch (const std::exception& e) {
        // 确保清理所有已分配的资源
        cleanup();
        throw;
    }
}

void Yolov11Pose::setInput(void* input) 
{   
    // 核验输入数据的合法性并进行类型转换和保存
    if (!input) {
        LOG(ERROR) << "输入为空";
        return;
    }
    m_inputData = *static_cast<MultiImagePreprocessResult*>(input);
    LOG(INFO) << "Received " << m_inputData.size() << " preprocessed images";
}

void* Yolov11Pose::getOutput() {
    return &m_outputResult;
}

void Yolov11Pose::execute() 
{
    LOG(INFO) << "Yolov11Pose::execute status: start with " << m_inputData.size() << " images";
    
    if (m_inputData.empty()) {
        LOG(ERROR) << "No input data available";
        return;
    }
    
    // 清空之前的输出
    m_outputResult.vecFrameResult().clear();
    
    // 创建单个FrameResult来存储所有结果
    CFrameResult allFrameResult;
    
    // 分批处理图像
    for (size_t batch_start = 0; batch_start < m_inputData.size(); batch_start += m_maxBatchSize) {
        size_t batch_end = std::min(batch_start + m_maxBatchSize, m_inputData.size());
        size_t batch_size = batch_end - batch_start;
        
        LOG(INFO) << "Processing batch " << (batch_start / m_maxBatchSize + 1) 
                  << " with " << batch_size << " images";
        
        // 准备批处理数据
        prepareBatchData(batch_start, batch_end);
        
        // 执行推理
        std::vector<float> output = inference();
        
        // 结果后处理
        std::vector<std::vector<float>> results = process_output(output);
        
        // 格式转换并添加到总结果中
        CAlgResult batchResult = formatConverted(results);
        
        // 合并结果到单个FrameResult中
        if (!batchResult.vecFrameResult().empty()) {
            const auto& batchFrameResult = batchResult.vecFrameResult()[0];
            const auto& batchObjectResults = batchFrameResult.vecObjectResult();
            allFrameResult.vecObjectResult().insert(
                allFrameResult.vecObjectResult().end(),
                batchObjectResults.begin(),
                batchObjectResults.end()
            );
        }
    }
    
    // 将合并后的FrameResult添加到输出中
    m_outputResult.vecFrameResult().push_back(allFrameResult);
    
    LOG(INFO) << "Yolov11Pose::execute status: success, total results: " 
              << allFrameResult.vecObjectResult().size();
}

void Yolov11Pose::prepareBatchData(size_t batch_start, size_t batch_end)
{
    m_batchInputs.clear();
    m_batchSizes.clear();
    
    // 收集当前批次的图像数据和尺寸
    for (size_t i = batch_start; i < batch_end; ++i) {
        if (i < m_inputData.images.size()) {
            m_batchInputs.push_back(m_inputData.images[i]);
            m_batchSizes.push_back(m_inputData.imageSizes[i]);
        }
    }
    
    LOG(INFO) << "Prepared batch data with " << m_batchInputs.size() << " images";
}

void Yolov11Pose::cleanup() 
{
    LOG(INFO) << "开始清理TensorRT资源...";
    
    // 清理GPU内存
    for (auto& buf : input_buffers_) {
        if (buf) { 
            cudaError_t status = cudaFree(buf);
            if (status != cudaSuccess) {
                LOG(WARNING) << "清理输入GPU内存失败: " << cudaGetErrorString(status);
            }
            buf = nullptr; 
        }
    }
    for (auto& buf : output_buffers_) {
        if (buf) { 
            cudaError_t status = cudaFree(buf);
            if (status != cudaSuccess) {
                LOG(WARNING) << "清理输出GPU内存失败: " << cudaGetErrorString(status);
            }
            buf = nullptr; 
        }
    }
    
    // 清理CUDA流
    if (stream_) { 
        cudaError_t status = cudaStreamDestroy(stream_);
        if (status != cudaSuccess) {
            LOG(WARNING) << "销毁CUDA流失败: " << cudaGetErrorString(status);
        }
        stream_ = nullptr; 
    }
    
    // 清理TensorRT资源
    context_.reset();
    engine_.reset();
    runtime_.reset();
    
    LOG(INFO) << "TensorRT资源清理完成";
}

// 将模型输出结果转换为CAlgResult
CAlgResult Yolov11Pose::formatConverted(std::vector<std::vector<float>> results)
{
    CAlgResult alg_result;
    CFrameResult frame_result;

    for (const auto& result : results) {
        CObjectResult obj_result;

        // 边界框
        obj_result.fTopLeftX(result[0]);
        obj_result.fTopLeftY(result[1]);
        obj_result.fBottomRightX(result[2]);
        obj_result.fBottomRightY(result[3]);
        
        // 置信度 - 确保在合理范围内
        float confidence = result[4];
        if (confidence > 1.0f) {
            LOG(WARNING) << "置信度值异常高: " << confidence << "，限制为1.0";
            confidence = 1.0f;
        }
        obj_result.fVideoConfidence(confidence);
        
        // 类别
        // obj_result.strClass(std::to_string(static_cast<int>(result[5]))); // 注释原有类别设置

        // 关键点
        std::vector<Keypoint> keypoints;
        std::vector<float> keypoints_vec;
        for (int j = 0; j < num_keys_; ++j) 
        {
            Keypoint kp;
            float x = result[6 + j * 3];
            float y = result[6 + j * 3 + 1];
            float conf = result[6 + j * 3 + 2];
            
            kp.x(x);
            kp.y(y);
            kp.confidence(conf);
            keypoints.push_back(kp);
            keypoints_vec.push_back(x);
            keypoints_vec.push_back(y);
            keypoints_vec.push_back(conf);
        }
        obj_result.vecKeypoints(keypoints);

        // --- 新增：根据关键点分类并设置类别 ---
        std::string pose_state = classify_pose(keypoints_vec);
        if (pose_state == "躺着" || pose_state == "坐着或佝偻" || pose_state == "未知") {
            obj_result.strClass("0");
        } else {
            obj_result.strClass("1");
        }
        // --- 新增结束 ---

        frame_result.vecObjectResult().push_back(obj_result);
    }

    alg_result.vecFrameResult({frame_result});

    LOG(INFO) << "formatConverted: alg_result.vecFrameResult().size() = " << alg_result.vecFrameResult().size();
    if (alg_result.vecFrameResult().size() > 0)
        LOG(INFO) << "formatConverted: frame_result.vecObjectResult().size() = " << alg_result.vecFrameResult()[0].vecObjectResult().size();
    return alg_result;
}

std::vector<float> Yolov11Pose::inference()
{
    LOG(INFO) << "Yolov11Pose::inference status: start with batch size: " << m_batchInputs.size();
    
    if (m_batchInputs.empty()) {
        LOG(ERROR) << "No batch data available";
        return std::vector<float>();
    }
    
    // 计算当前批次的实际大小
    int current_batch_size = static_cast<int>(m_batchInputs.size());
    
    // 检查batch_size是否超出限制
    if (current_batch_size > m_maxBatchSize) {
        LOG(ERROR) << "Batch size " << current_batch_size << " exceeds maximum " << m_maxBatchSize;
        return std::vector<float>();
    }
    
    // 1. 统一图像尺寸 - 找到最大尺寸并填充所有图像
    int max_width = 0, max_height = 0;
    std::vector<std::pair<int, int>> original_sizes;
    
    // 收集所有图像的原始尺寸
    for (const auto& size : m_batchSizes) {
        max_width = std::max(max_width, size.first);
        max_height = std::max(max_height, size.second);
        original_sizes.push_back(size);
    }
    
    // 确保尺寸是stride的整数倍
    max_width = (max_width / stride_[2]) * stride_[2];
    max_height = (max_height / stride_[2]) * stride_[2];
    
    LOG(INFO) << "统一图像尺寸: " << max_width << "x" << max_height;
    LOG(INFO) << "当前批次大小: " << current_batch_size;
    
    // 2. 设置动态输入尺寸
    input_dims_.nbDims = 4;
    input_dims_.d[0] = current_batch_size;  // batch size
    input_dims_.d[1] = channels_;           // channels
    input_dims_.d[2] = max_height;          // height
    input_dims_.d[3] = max_width;           // width
    
    LOG(INFO) << "设置TensorRT输入形状: [" << input_dims_.d[0] << ", " << input_dims_.d[1] 
              << ", " << input_dims_.d[2] << ", " << input_dims_.d[3] << "]";
    
    if (!context_->setInputShape(input_name_, input_dims_)) {
        throw std::runtime_error("设置输入形状失败");
    }

    // 3. 重新绑定输入输出 buffer
    if (!context_->setTensorAddress(input_name_, input_buffers_[0])) {
        throw std::runtime_error("设置输入张量地址失败");
    }
    if (!context_->setTensorAddress(output_name_, output_buffers_[0])) {
        throw std::runtime_error("设置输出张量地址失败");
    }

    // 4. 准备批处理输入数据 - 统一尺寸并填充
    std::vector<float> batch_input;
    size_t single_image_size = channels_ * max_height * max_width;
    batch_input.reserve(current_batch_size * single_image_size);
    
    LOG(INFO) << "单张图像大小: " << single_image_size << " (约 " 
              << (single_image_size * sizeof(float)) / (1024*1024) << "MB)";
    
    for (size_t i = 0; i < m_batchInputs.size(); ++i) {
        const auto& image_data = m_batchInputs[i];
        const auto& original_size = original_sizes[i];
        
        // 计算当前图像的实际大小
        size_t actual_size = channels_ * original_size.second * original_size.first;
        
        if (image_data.size() == actual_size) {
            // 图像尺寸与原始尺寸匹配，需要填充到统一尺寸
            std::vector<float> padded_image(single_image_size, 0.0f);
            
            // 复制原始数据到填充图像
            for (int h = 0; h < original_size.second; ++h) {
                for (int w = 0; w < original_size.first; ++w) {
                    for (int c = 0; c < channels_; ++c) {
                        int src_idx = c * original_size.second * original_size.first + h * original_size.first + w;
                        int dst_idx = c * max_height * max_width + h * max_width + w;
                        padded_image[dst_idx] = image_data[src_idx];
                    }
                }
            }
            
            batch_input.insert(batch_input.end(), padded_image.begin(), padded_image.end());
            
            LOG(INFO) << "Image " << i << ": " << original_size.first << "x" << original_size.second 
                      << " -> padded to " << max_width << "x" << max_height;
        } else {
            LOG(WARNING) << "Image " << i << " data size mismatch: expected " << actual_size 
                         << ", got " << image_data.size() << ", skipping";
            // 用零填充
            batch_input.resize(batch_input.size() + single_image_size, 0.0f);
        }
    }

    // 5. 拷贝输入数据到GPU
    size_t input_size = batch_input.size() * sizeof(float);
    LOG(INFO) << "拷贝输入数据到GPU: " << input_size << " bytes (约 " 
              << input_size / (1024*1024) << "MB)";
    
    cudaError_t cuda_status = cudaMemcpyAsync(input_buffers_[0], batch_input.data(),
                                              input_size,
                                              cudaMemcpyHostToDevice, stream_);
    if (cuda_status != cudaSuccess) {
        throw std::runtime_error("CUDA内存拷贝失败: " + std::string(cudaGetErrorString(cuda_status)));
    }
    cudaStreamSynchronize(stream_);

    // 6. 执行推理
    LOG(INFO) << "开始TensorRT推理...";
    bool status = context_->enqueueV3(stream_);
    if (!status) {
        throw std::runtime_error("TensorRT推理失败");
    }
    cudaStreamSynchronize(stream_);
    LOG(INFO) << "TensorRT推理完成";

    // 7. 获取输出 shape
    nvinfer1::Dims output_dims = context_->getTensorShape(output_name_);
    size_t output_size = 1;
    for (int i = 0; i < output_dims.nbDims; ++i) {
        output_size *= output_dims.d[i];
    }
    std::vector<float> output(output_size);

    // 8. 拷贝输出数据到CPU
    cuda_status = cudaMemcpyAsync(output.data(), output_buffers_[0],
                                  output_size * sizeof(float),
                                  cudaMemcpyDeviceToHost, stream_);
    if (cuda_status != cudaSuccess) {
        throw std::runtime_error("CUDA输出内存拷贝失败: " + std::string(cudaGetErrorString(cuda_status)));
    }
    cudaStreamSynchronize(stream_);

    LOG(INFO) << "Yolov11Pose::inference status: success, output size: " << output.size();
    return output;
}

void Yolov11Pose::rescale_coords(std::vector<float>& coords, bool is_keypoint) 
{
    if (coords.empty()) return;
    if (is_keypoint) {
        for (size_t i = 0; i < coords.size(); i += 3) {
            coords[i] = (coords[i] - dw_) / ratio_;
            coords[i + 1] = (coords[i + 1] - dh_) / ratio_;
        }
    } else {
        for (size_t i = 0; i < coords.size(); i += 4) {
            coords[i] = (coords[i] - dw_) / ratio_;
            coords[i + 1] = (coords[i + 1] - dh_) / ratio_;
            coords[i + 2] = (coords[i + 2] - dw_) / ratio_;
            coords[i + 3] = (coords[i + 3] - dh_) / ratio_;
        }
    }
}

std::vector<std::vector<float>> Yolov11Pose::process_keypoints(const std::vector<float>& output, const std::vector<std::vector<float>>& boxes) {
    std::vector<std::vector<float>> keypoints;
    int num_keypoints = num_keys_; // 可根据模型实际调整
    for (size_t i = 0; i < boxes.size(); ++i) {
        std::vector<float> kpts;
        int kpt_start = 6; // 可根据模型实际调整
        for (int j = 0; j < num_keypoints; ++j) {
            float x = output[kpt_start + j * 3];
            float y = output[kpt_start + j * 3 + 1];
            float conf = output[kpt_start + j * 3 + 2];
            kpts.push_back(x);
            kpts.push_back(y);
            kpts.push_back(conf);
        }
        rescale_coords(kpts, true);
        keypoints.push_back(kpts);
    }
    return keypoints;
}

std::vector<std::vector<float>> Yolov11Pose::process_output(const std::vector<float>& output) 
{   
    LOG(INFO) << "Yolov11Pose::process_output status: start ";
    
    if (m_batchInputs.empty()) {
        LOG(ERROR) << "No batch data available for processing";
        return std::vector<std::vector<float>>();
    }
    
    int current_batch_size = static_cast<int>(m_batchInputs.size());
    
    // 2. 正确处理TensorRT输出数据格式
    // 输出形状: [batch_size, feature_dim, num_anchors] -> [batch_size * num_anchors, feature_dim]
    int feature_dim = 4 + num_classes_ + num_keys_ * 3; 
    std::vector<std::vector<float>> results;
    
    // 获取当前推理使用的统一尺寸
    nvinfer1::Dims input_dims = context_->getTensorShape(input_name_);
    int max_height = input_dims.d[2];
    int max_width = input_dims.d[3];
    
    // 从当前推理的实际输出形状获取正确的num_anchors
    nvinfer1::Dims output_dims = context_->getTensorShape(output_name_);
    int num_anchors = output_dims.d[2];  // 输出形状: [batch_size, feature_dim, num_anchors]
    
    LOG(INFO) << "正确处理输出数据: num_anchors=" << num_anchors << ", feature_dim=" << feature_dim;
    LOG(INFO) << "原始输出大小: " << output.size();
    LOG(INFO) << "期望输出大小: " << (current_batch_size * feature_dim * num_anchors);
    LOG(INFO) << "统一推理尺寸: " << max_width << "x" << max_height;
    
    // 验证输出大小是否正确
    if (output.size() != current_batch_size * feature_dim * num_anchors) {
        LOG(ERROR) << "输出大小不匹配！期望: " << (current_batch_size * feature_dim * num_anchors) 
                   << ", 实际: " << output.size();
        return std::vector<std::vector<float>>();
    }
    
    // 3. 为每个batch中的图像处理结果
    for (int batch_idx = 0; batch_idx < current_batch_size; ++batch_idx) {
        // 获取当前图像的预处理参数
        float ratio = 1.0f;
        int padTop = 0, padLeft = 0;
        int original_width = 0;   // 修复：初始化为0，确保必须从正确来源获取
        int original_height = 0;  // 修复：初始化为0，确保必须从正确来源获取
        
        // 从预处理结果中获取参数
        if (batch_idx < m_inputData.preprocessParams.size()) {
            const auto& params = m_inputData.preprocessParams[batch_idx];
            ratio = params.ratio;
            padTop = params.padTop;
            padLeft = params.padLeft;
            original_width = params.originalWidth;   // 使用预处理保存的原始图像宽度
            original_height = params.originalHeight; // 使用预处理保存的原始图像高度
            
            LOG(INFO) << "Image " << batch_idx << ": using saved params - original " 
                      << original_width << "x" << original_height 
                      << ", ratio: " << ratio << ", pad: (" << padTop << "," << padLeft << ")";
        } else {
            // 如果没有保存的参数，使用默认计算（兼容性）
            // 注意：这里应该从其他地方获取原始图像尺寸，而不是使用预处理后的尺寸
            LOG(WARNING) << "Image " << batch_idx << ": no saved preprocessing params, using fallback";
            
            // 修复：如果没有预处理参数，应该从输入数据中获取原始图像信息
            // 这里暂时使用一个合理的默认值，但建议确保预处理参数总是可用
            if (batch_idx < m_batchSizes.size()) {
                // 注意：m_batchSizes存储的是预处理后的尺寸，不是原始尺寸
                // 这里需要根据实际情况调整
                original_width = m_batchSizes[batch_idx].first;
                original_height = m_batchSizes[batch_idx].second;
                LOG(WARNING) << "Using preprocessed size as original size (may be incorrect): " 
                             << original_width << "x" << original_height;
            } else {
                // 最后的fallback：使用配置中的默认值
                original_width = new_unpad_w_;
                original_height = new_unpad_h_;
                LOG(WARNING) << "Using config default size as original size: " 
                             << original_width << "x" << original_height;
            }
            
            // 计算当前图像的预处理参数 - 修复：与Python脚本保持一致
            // 从统一推理尺寸和原始尺寸计算缩放比例和填充
            float r = std::min(static_cast<float>(max_height) / original_height, 
                              static_cast<float>(max_width) / original_width);
            int new_unpad_w = static_cast<int>(original_width * r);
            int new_unpad_h = static_cast<int>(original_height * r);
            
            // 获取stride值（使用第一个stride）
            int current_stride = stride_.empty() ? 32 : stride_[0];
            new_unpad_w = (new_unpad_w / current_stride) * current_stride;
            new_unpad_h = (new_unpad_h / current_stride) * current_stride;
            
            // 计算填充参数
            int dh = max_height - new_unpad_h;
            int dw = max_width - new_unpad_w;
            padTop = dh / 2;
            padLeft = dw / 2;
            ratio = r;
            
            LOG(INFO) << "Image " << batch_idx << ": computed params - original " 
                      << original_width << "x" << original_height 
                      << ", ratio: " << ratio << ", pad: (" << padTop << "," << padLeft << ")";
        }
        
        // 验证原始尺寸的有效性
        if (original_width <= 0 || original_height <= 0) {
            LOG(ERROR) << "Image " << batch_idx << ": invalid original dimensions " 
                       << original_width << "x" << original_height << ", skipping";
            continue;
        }
        
        // 计算当前图像在数据中的起始位置
        int batch_start = batch_idx * feature_dim * num_anchors;
        
        // 遍历当前图像的所有anchor，筛选置信度大于阈值的候选框
        std::vector<std::vector<float>> boxes;
        std::vector<float> scores;
        std::vector<int> class_ids;
        std::vector<std::vector<float>> keypoints;
        
        int valid_detections = 0;
        
        for (int i = 0; i < num_anchors; ++i) 
        {
            // 正确计算索引: [batch_idx, feature, anchor_i]
            int bbox_start = batch_start + i;  // 每个feature的起始位置
            
            // 取bbox
            float x = output[batch_start + 0 * num_anchors + i];
            float y = output[batch_start + 1 * num_anchors + i];
            float w = output[batch_start + 2 * num_anchors + i];
            float h = output[batch_start + 3 * num_anchors + i];

            // 取类别分数 - 修复：TensorRT engine输出已经是sigmoid激活后的值
            float max_conf = 0.0f;
            int max_class = 0;
            for (int c = 0; c < num_classes_; ++c) {
                float conf = output[batch_start + (4 + c) * num_anchors + i];
                // 直接使用模型输出的置信度值，因为已经是sigmoid激活后的值
                
                if (conf > max_conf) {
                    max_conf = conf;
                    max_class = c;
                }
            }
            
            // 修复：使用sigmoid激活后的置信度值与阈值比较
            if (max_conf < conf_thres_) {
                continue;
            }

            // 坐标还原 - 修复：使用正确的坐标转换逻辑，与Python的xywh2xyxy保持一致
            // Python: xywh2xyxy(x) -> x1 = x - w/2, y1 = y - h/2, x2 = x + w/2, y2 = y + h/2
            // 然后考虑填充偏移：(coord - pad) / ratio
            float x1 = ((x - w / 2) - padLeft) / ratio;
            float y1 = ((y - h / 2) - padTop) / ratio;
            float x2 = ((x + w / 2) - padLeft) / ratio;
            float y2 = ((y + h / 2) - padTop) / ratio;
            
            // 修复：确保坐标在合理范围内，与Python的clip_boxes保持一致
            x1 = std::max(0.0f, std::min(x1, static_cast<float>(original_width)));
            y1 = std::max(0.0f, std::min(y1, static_cast<float>(original_height)));
            x2 = std::max(0.0f, std::min(x2, static_cast<float>(original_width)));
            y2 = std::max(0.0f, std::min(y2, static_cast<float>(original_height)));
            
            // 修复：检查边界框大小的合理性，但放宽条件
            float box_width = x2 - x1;
            float box_height = y2 - y1;
            if (box_width < 1 || box_height < 1 || box_width > original_width || box_height > original_height) {
                if (valid_detections < 10) {  // 只打印前10个被过滤的
                    LOG(INFO) << "Image " << batch_idx << " anchor " << i << ": 边界框尺寸不合理，跳过";
                }
                continue;
            }

            boxes.push_back({x1, y1, x2, y2});
            scores.push_back(max_conf);
            class_ids.push_back(max_class);

            // 关键点 - 修复：使用正确的坐标转换和置信度处理，与Python脚本保持一致
            std::vector<float> kpts;
            for (int j = 0; j < num_keys_ * 3; j += 3) {
                // 修复：直接使用模型输出的坐标和置信度值，因为已经是sigmoid激活后的值
                float kpt_x = output[batch_start + (4 + num_classes_ + j) * num_anchors + i];
                float kpt_y = output[batch_start + (4 + num_classes_ + j + 1) * num_anchors + i];
                float kpt_conf = output[batch_start + (4 + num_classes_ + j + 2) * num_anchors + i];
                
                // 修复：将坐标转换到原始图像尺寸，考虑填充偏移
                kpt_x = (kpt_x - padLeft) / ratio;
                kpt_y = (kpt_y - padTop) / ratio;
                
                if (kpt_conf < conf_thres_) {   // 使用配置的置信度阈值
                    kpts.push_back(0.0f);
                    kpts.push_back(0.0f);
                    kpts.push_back(0.0f);
                } else {
                    kpts.push_back(kpt_x);
                    kpts.push_back(kpt_y);
                    kpts.push_back(kpt_conf);
                }
            }
            keypoints.push_back(kpts);
            valid_detections++;
        }
        
        LOG(INFO) << "Image " << batch_idx << ": 有效检测数=" << valid_detections 
                  << ", 置信度阈值=" << conf_thres_;

        // 4. 按类别分组做NMS
        for (int cls = 0; cls < num_classes_; ++cls) {
            std::vector<std::vector<float>> cls_boxes;
            std::vector<float> cls_scores;
            std::vector<std::vector<float>> cls_keypoints;
            for (size_t i = 0; i < class_ids.size(); ++i) {
                if (class_ids[i] == cls) {
                    cls_boxes.push_back(boxes[i]);
                    cls_scores.push_back(scores[i]);
                    cls_keypoints.push_back(keypoints[i]);
                }
            }
            if (cls_boxes.empty()) continue;
            
            std::vector<int> keep = nms(cls_boxes, cls_scores);
            
            for (int idx : keep) {
                std::vector<float> result = cls_boxes[idx];
                result.push_back(cls_scores[idx]);
                result.push_back(static_cast<float>(cls));  // 添加类别信息
                result.insert(result.end(), cls_keypoints[idx].begin(), cls_keypoints[idx].end());
                results.push_back(result);
            }
        }
    }

    // 5. 按置信度排序，截断最大检测数
    std::sort(results.begin(), results.end(), [](const std::vector<float>& a, const std::vector<float>& b) {
        return a[4] > b[4];
    });
    
    LOG(INFO) << "排序后检测数: " << results.size();
    
    // 修复：使用配置中的置信度阈值进行过滤，与Python脚本保持一致
    std::vector<std::vector<float>> filtered_results;
    for (const auto& result : results) {
        if (result[4] >= conf_thres_) {  // 使用配置的置信度阈值
            filtered_results.push_back(result);
        }
    }
    
    LOG(INFO) << "置信度过滤后检测数: " << filtered_results.size();
    
    // 如果过滤后的结果仍然超过max_dets_，则截断
    if (filtered_results.size() > max_dets_) {
        LOG(INFO) << "截断检测数从 " << filtered_results.size() << " 到 " << max_dets_;
        filtered_results.resize(max_dets_);
    }
    
    results = filtered_results;

    if(status_)
    {
        save_bin(results, "./Save_Data/pose/result/processed_output_yolov11pose.bin"); // Yolov11Pose/Inference
    }
    
    LOG(INFO) << "Yolov11Pose::process_output status: success, found " << results.size() << " detections";
    return results;
}

std::vector<int> Yolov11Pose::nms(const std::vector<std::vector<float>>& boxes, const std::vector<float>& scores) 
{
    float iou_threshold = iou_thres_; // 可根据成员变量或配置调整
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

std::string Yolov11Pose::classify_pose(const std::vector<float>& keypoints) const
{
    if (keypoints.size() < num_keys_ * 3) return "未知";

    // 关键点下标（COCO格式）
    int LShoulder = 5, RShoulder = 6, LHip = 11, RHip = 12, LKnee = 13, RKnee = 14, Nose = 0;

    auto get_pt = [&](int idx) {
        return cv::Point2f(keypoints[idx * 3], keypoints[idx * 3 + 1]);
    };

    cv::Point2f shoulder_mid = (get_pt(LShoulder) + get_pt(RShoulder)) * 0.5f;
    cv::Point2f hip_mid = (get_pt(LHip) + get_pt(RHip)) * 0.5f;
    cv::Point2f knee_mid = (get_pt(LKnee) + get_pt(RKnee)) * 0.5f;

    // 主轴角度（修正：与y轴夹角，站立为0°，躺着为90°）
    cv::Point2f axis = shoulder_mid - hip_mid;
    float axis_angle = std::atan2(axis.x, axis.y) * 180.0f / CV_PI;
    axis_angle = std::abs(axis_angle);
    if (axis_angle > 90.0f) axis_angle = 180.0f - axis_angle;

    // 躯干夹角
    auto angle = [](cv::Point2f a, cv::Point2f b, cv::Point2f c) {
        cv::Point2f v1 = a - b, v2 = c - b;
        float dot = v1.dot(v2);
        float norm = cv::norm(v1) * cv::norm(v2);
        if (norm < 1e-3f) return 180.0f;
        float cos_theta = dot / norm;
        cos_theta = std::max(-1.0f, std::min(1.0f, cos_theta));
        return static_cast<float>(std::acos(cos_theta) * 180.0 / CV_PI);
    };
    float trunk_angle = angle(shoulder_mid, hip_mid, knee_mid);

    // 判断
    if (axis_angle < 30 && trunk_angle > 160)
        return "站立/行走";
    else if (axis_angle > 60)
        return "躺着";
    else if (trunk_angle < 160)
        return "坐着或佝偻";
    else
        return "未知";
}