/*******************************************************
 文件名：Yolov11ClassifyGPU.cpp
 作者：sharkls
 描述：GPU加速的YOLOv11图像分类推理模块实现
 版本：v1.0
 日期：2025-01-20
 *******************************************************/

#include "Yolov11ClassifyGPU.h"

// 注册模块
REGISTER_MODULE("ObjectClassify", Yolov11ClassifyGPU, Yolov11ClassifyGPU)

Yolov11ClassifyGPU::Yolov11ClassifyGPU(const std::string& exe_path) : IBaseModule(exe_path) 
{
    // 构造函数初始化
    input_buffers_.resize(1, nullptr);
    output_buffers_.resize(1, nullptr);
    m_maxBatchSize = 8; // 设置最大批处理大小
    m_cudaInitialized = false;

    // 初始化CUDA流
    cudaError_t cuda_status = cudaStreamCreate(&stream_);
    if (cuda_status != cudaSuccess) {
        throw std::runtime_error("创建CUDA流失败: " + std::string(cudaGetErrorString(cuda_status)));
    }
}

Yolov11ClassifyGPU::~Yolov11ClassifyGPU() {
    cleanup();
}

bool Yolov11ClassifyGPU::init(void* p_pAlgParam) 
{   
    LOG(INFO) << "Yolov11ClassifyGPU::init status: start ";
    // 1. 配置参数核验
    if (!p_pAlgParam) return false;
    m_poseConfig = *static_cast<posetimation::YOLOModelConfig*>(p_pAlgParam);

    // 2. 配置参数获取
    engine_path_ = m_poseConfig.engine_path();
    conf_thres_ = m_poseConfig.conf_thres();
    num_classes_ = m_poseConfig.num_class();
    channels_ = m_poseConfig.channels();
    target_h_ = m_poseConfig.height();
    target_w_ = m_poseConfig.width();

    LOG(INFO) << "Configuration: engine_path=" << engine_path_ 
              << ", conf_thres=" << conf_thres_
              << ", num_classes=" << num_classes_
              << ", channels=" << channels_
              << ", target_h=" << target_h_
              << ", target_w=" << target_w_;

    // 3. 初始化CUDA
    if (!initCUDA()) {
        LOG(ERROR) << "Failed to initialize CUDA";
        return false;
    }

    // 4. 初始化TensorRT相关配置
    initTensorRT();
    
    LOG(INFO) << "Yolov11ClassifyGPU::init status: success ";
    return true;
}

bool Yolov11ClassifyGPU::initCUDA() {
    if (m_cudaInitialized) {
        return true;
    }

    // 初始化CUDA
    cudaError_t cuda_status = cudaSetDevice(0);
    if (cuda_status != cudaSuccess) {
        LOG(ERROR) << "Failed to set CUDA device: " << cudaGetErrorString(cuda_status);
        return false;
    }

    // 初始化cuBLAS
    cublasStatus_t cublas_status = cublasCreate(&m_cublasHandle);
    if (cublas_status != CUBLAS_STATUS_SUCCESS) {
        LOG(ERROR) << "Failed to create cuBLAS handle";
        return false;
    }

    // 设置cuBLAS流
    cublas_status = cublasSetStream(m_cublasHandle, stream_);
    if (cublas_status != CUBLAS_STATUS_SUCCESS) {
        LOG(ERROR) << "Failed to set cuBLAS stream";
        return false;
    }

    m_cudaInitialized = true;
    LOG(INFO) << "CUDA initialized successfully";
    return true;
}

void Yolov11ClassifyGPU::cleanupCUDA() {
    if (m_cublasHandle) {
        cublasDestroy(m_cublasHandle);
        m_cublasHandle = nullptr;
    }
    m_cudaInitialized = false;
}

void Yolov11ClassifyGPU::initTensorRT()
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
    
    // 5. 设置初始输入形状（最小batch size）
    input_dims_.nbDims = 4;
    input_dims_.d[0] = 1;  // 最小batch size
    input_dims_.d[1] = channels_;
    input_dims_.d[2] = target_h_;
    input_dims_.d[3] = target_w_;
    
    if (!context_->setInputShape(input_name_, input_dims_)) {
        throw std::runtime_error("设置初始输入形状失败");
    }

    output_dims_ = context_->getTensorShape(output_name_);

    // 6. 初始化GPU内存指针为空（在inference时动态分配）
    input_buffers_[0] = nullptr;
    output_buffers_[0] = nullptr;
    
    LOG(INFO) << "TensorRT initialized successfully, GPU memory will be allocated dynamically during inference";
}

void Yolov11ClassifyGPU::setInput(void* input) 
{   
    // 只接受GPU版本的输入数据
    if (!input) {
        LOG(ERROR) << "Yolo11ClassifyGPU 输入为空";
        return;
    }
    
    MultiImagePreprocessResultGPU* gpu_input = static_cast<MultiImagePreprocessResultGPU*>(input);
    if (gpu_input && !gpu_input->empty()) {
        m_inputDataGPU = *gpu_input;
        LOG(INFO) << "Yolov11ClassifyGPU::setInput: 接收到GPU版本 " << m_inputDataGPU.size() << " 张图像";
    } else {
        LOG(ERROR) << "输入数据为空或格式不支持";
    }
}

void* Yolov11ClassifyGPU::getOutput() {
    return &m_outputResult;
}

void Yolov11ClassifyGPU::execute() 
{
    LOG(INFO) << "Yolov11ClassifyGPU::execute status: start ";
    
    // 检查输入数据
    if (m_inputDataGPU.empty()) {
        LOG(ERROR) << "No GPU input images available";
        return;
    }
    
    LOG(INFO) << "Using GPU input data with " << m_inputDataGPU.size() << " images";

    // 清空之前的输出结果
    m_outputResult.vecFrameResult().clear();
    
    // 直接处理所有图像（固定尺寸target_w_*target_h_）
    size_t total_images = m_inputDataGPU.size();
    
    // 检查所有图像的尺寸是否一致且为目标尺寸
    if (!m_inputDataGPU.allImagesSameSize()) {
        LOG(ERROR) << "All images must have the same size (" << target_w_ << "x" << target_h_ << ")";
        return;
    }
    
    auto unified_size = m_inputDataGPU.getUnifiedImageSize();
    if (unified_size.first != target_w_ || unified_size.second != target_h_) {
        LOG(ERROR) << "Image size must be " << target_w_ << "x" << target_h_ 
                   << ", got " << unified_size.first << "x" << unified_size.second;
        return;
    }
    
    LOG(INFO) << "Processing " << total_images << " images with fixed size " << target_w_ << "x" << target_h_;
    
    // 执行推理
    std::vector<float> output = inference();
    
    // 结果后处理
    std::vector<std::vector<float>> results = process_output_classification_only(output);
    
    // 为每个图像创建单独的FrameResult
    for (size_t i = 0; i < results.size(); ++i) {
        LOG(INFO) << "Creating FrameResult for image " << i;
        
        // 为每个图像创建一个FrameResult
        CFrameResult frameResult;
        frameResult.eDataType(DATA_TYPE_OBJECTCLASSIFYALG_RESULT);
        
        // 创建一个CObjectResult来存储该图像的分类结果
        CObjectResult obj_result;
        
        if (!results[i].empty()) {
            float confidence = results[i][0];
            int class_id = static_cast<int>(results[i][1]);
            
            if (confidence >= conf_thres_) {
                obj_result.fVideoConfidence(confidence);
                obj_result.strClass(get_class_name(class_id));
                
                LOG(INFO) << "Image " << i << " classified as: " << get_class_name(class_id) 
                          << " with probability: " << confidence;
            } else {
                // 如果置信度不够，设置为未知类别
                obj_result.fVideoConfidence(0.1f);
                obj_result.strClass("class_unknown");
                
                LOG(INFO) << "Image " << i << " confidence too low: " << confidence;
            }
        } else {
            // 如果没有结果，设置为未知类别
            obj_result.fVideoConfidence(0.1f);
            obj_result.strClass("class_unknown");
            
            LOG(INFO) << "Image " << i << " no classification result";
        }
        
        // 将CObjectResult添加到FrameResult中
        frameResult.vecObjectResult().push_back(obj_result);
        
        // 将当前图像的FrameResult添加到输出中
        m_outputResult.vecFrameResult().push_back(frameResult);
    }
    
    LOG(INFO) << "Yolov11ClassifyGPU::execute status: success, total FrameResults: " 
              << m_outputResult.vecFrameResult().size();
}



void Yolov11ClassifyGPU::cleanup() 
{
    LOG(INFO) << "开始清理TensorRT和CUDA资源...";
    
    // 清理GPU内存（如果存在）
    if (input_buffers_[0]) { 
        cudaError_t status = cudaFree(input_buffers_[0]);
        if (status != cudaSuccess) {
            LOG(WARNING) << "清理输入GPU内存失败: " << cudaGetErrorString(status);
        }
        input_buffers_[0] = nullptr; 
    }
    if (output_buffers_[0]) { 
        cudaError_t status = cudaFree(output_buffers_[0]);
        if (status != cudaSuccess) {
            LOG(WARNING) << "清理输出GPU内存失败: " << cudaGetErrorString(status);
        }
        output_buffers_[0] = nullptr; 
    }
    
    // 清理CUDA流
    if (stream_) { 
        cudaError_t status = cudaStreamDestroy(stream_);
        if (status != cudaSuccess) {
            LOG(WARNING) << "销毁CUDA流失败: " << cudaGetErrorString(status);
        }
        stream_ = nullptr; 
    }
    
    // 清理CUDA资源
    cleanupCUDA();
    
    // 清理TensorRT资源
    context_.reset();
    engine_.reset();
    runtime_.reset();
    
    LOG(INFO) << "TensorRT和CUDA资源清理完成";
}

std::vector<float> Yolov11ClassifyGPU::inference()
{
    LOG(INFO) << "Yolov11ClassifyGPU::inference status: start with total images: " << m_inputDataGPU.size();
    
    if (m_inputDataGPU.empty()) {
        LOG(ERROR) << "No GPU input data available";
        return std::vector<float>();
    }
    
    // 获取所有图像数量
    int total_images = static_cast<int>(m_inputDataGPU.size());
    
    // 检查图像尺寸是否为target_w_*target_h_
    auto unified_size = m_inputDataGPU.getUnifiedImageSize();
    if (unified_size.first != target_w_ || unified_size.second != target_h_) {
        LOG(ERROR) << "Image size must be " << target_w_ << "x" << target_h_ << ", got " << unified_size.first << "x" << unified_size.second;
        return std::vector<float>();
    }
    
    LOG(INFO) << "Processing " << total_images << " images with fixed size " << target_w_ << "x" << target_h_;
    
    // 设置输入尺寸为所有图像
    input_dims_.nbDims = 4;
    input_dims_.d[0] = total_images;  // batch size = total images
    input_dims_.d[1] = channels_;     // channels = 3
    input_dims_.d[2] = target_h_;           // height = target_h_
    input_dims_.d[3] = target_w_;           // width = target_w_
    
    LOG(INFO) << "设置TensorRT输入形状: [" << input_dims_.d[0] << ", " << input_dims_.d[1] 
              << ", " << input_dims_.d[2] << ", " << input_dims_.d[3] << "]";
    
    if (!context_->setInputShape(input_name_, input_dims_)) {
        throw std::runtime_error("设置输入形状失败");
    }

    // 动态分配GPU内存
    size_t input_size = total_images * channels_ * target_h_ * target_w_;
    size_t output_size = total_images * num_classes_;
    
    LOG(INFO) << "Allocating GPU memory: input_size=" << input_size 
              << ", output_size=" << output_size;
    
    // 释放之前的GPU内存（如果存在）
    if (input_buffers_[0]) {
        cudaFree(input_buffers_[0]);
        input_buffers_[0] = nullptr;
    }
    if (output_buffers_[0]) {
        cudaFree(output_buffers_[0]);
        output_buffers_[0] = nullptr;
    }
    
    // 分配新的GPU内存
    void* input_buffer = nullptr;
    cudaError_t cuda_status = cudaMalloc(&input_buffer, input_size * sizeof(float));
    if (cuda_status != cudaSuccess) {
        throw std::runtime_error("分配输入GPU内存失败: " + std::string(cudaGetErrorString(cuda_status)));
    }
    input_buffers_[0] = input_buffer;

    void* output_buffer = nullptr;
    cuda_status = cudaMalloc(&output_buffer, output_size * sizeof(float));
    if (cuda_status != cudaSuccess) {
        cudaFree(input_buffer);
        throw std::runtime_error("分配输出GPU内存失败: " + std::string(cudaGetErrorString(cuda_status)));
    }
    output_buffers_[0] = output_buffer;

    // 设置绑定
    if (!context_->setTensorAddress(input_name_, input_buffers_[0])) {
        cudaFree(input_buffer);
        cudaFree(output_buffer);
        throw std::runtime_error("设置输入张量地址失败");
    }
    if (!context_->setTensorAddress(output_name_, output_buffers_[0])) {
        cudaFree(input_buffer);
        cudaFree(output_buffer);
        throw std::runtime_error("设置输出张量地址失败");
    }

    // 准备所有图像数据 - 直接使用GPU内存数据
    size_t single_image_size = channels_ * target_w_ * target_h_;  // 固定尺寸
    
    LOG(INFO) << "单张图像大小: " << single_image_size << " (约 " 
              << (single_image_size * sizeof(float)) / (1024*1024) << "MB)";
    
    // 直接使用GPU内存数据，避免CPU-GPU转换
    LOG(INFO) << "Using GPU input data directly, copying all images to TensorRT buffer";
    
    // 将所有GPU数据复制到TensorRT输入缓冲区
    for (size_t i = 0; i < total_images; ++i) {
        float* gpu_image_ptr = m_inputDataGPU.getImagePtr(i);
        if (gpu_image_ptr) {
            // 直接复制GPU数据到TensorRT输入缓冲区
            size_t image_size_float = m_inputDataGPU.getImageSize(i);
            size_t image_size_bytes = m_inputDataGPU.getImageSizeBytes(i);
            
            // 验证图像大小是否与期望的单张图像大小匹配
            if (image_size_float != single_image_size) {
                LOG(ERROR) << "Image " << i << " size mismatch: expected " << single_image_size 
                           << " floats, got " << image_size_float << " floats";
                continue;
            }
            
            cudaError_t cuda_status = cudaMemcpyAsync(
                static_cast<float*>(input_buffers_[0]) + i * single_image_size,
                gpu_image_ptr,
                image_size_bytes,
                cudaMemcpyDeviceToDevice,
                stream_
            );
            if (cuda_status != cudaSuccess) {
                LOG(ERROR) << "Failed to copy GPU image " << i << " to TensorRT buffer: " 
                           << cudaGetErrorString(cuda_status);
            } else {
                LOG(INFO) << "Successfully copied GPU image " << i << " to TensorRT buffer, size: " 
                          << image_size_bytes << " bytes";
            }
        } else {
            LOG(ERROR) << "Failed to get GPU pointer for image " << i;
        }
    }
    
    // 同步流
    cudaStreamSynchronize(stream_);

    // 执行推理
    LOG(INFO) << "开始TensorRT推理...";
    bool status = context_->enqueueV3(stream_);
    if (!status) {
        throw std::runtime_error("TensorRT推理失败");
    }
    cudaStreamSynchronize(stream_);
    LOG(INFO) << "TensorRT推理完成";

    // 获取输出 shape
    nvinfer1::Dims output_dims = context_->getTensorShape(output_name_);
    size_t actual_output_size = 1;
    for (int i = 0; i < output_dims.nbDims; ++i) {
        actual_output_size *= output_dims.d[i];
    }
    
    LOG(INFO) << "Output size: " << actual_output_size;
    LOG(INFO) << "Output dimensions: [" << output_dims.d[0] << ", " << output_dims.d[1] << "]";

    // 拷贝数据到CPU进行后处理
    std::vector<float> output(actual_output_size);
    
    cuda_status = cudaMemcpyAsync(output.data(), output_buffers_[0],
                                  actual_output_size * sizeof(float),
                                  cudaMemcpyDeviceToHost, stream_);
    if (cuda_status != cudaSuccess) {
        throw std::runtime_error("CUDA输出内存拷贝失败: " + std::string(cudaGetErrorString(cuda_status)));
    }
    cudaStreamSynchronize(stream_);

    LOG(INFO) << "Yolov11ClassifyGPU::inference status: success, CPU output size: " << output.size();
    return output;
}

std::vector<std::vector<float>> Yolov11ClassifyGPU::process_output_classification_only(const std::vector<float>& output)
{
    LOG(INFO) << "Yolov11ClassifyGPU::process_output_classification_only status: start ";
    
    if (m_inputDataGPU.empty()) {
        LOG(ERROR) << "No GPU batch data available for processing";
        return std::vector<std::vector<float>>();
    }
    
    int current_batch_size = static_cast<int>(m_inputDataGPU.size());
    LOG(INFO) << "Processing GPU batch data with " << current_batch_size << " images";
    
    // 获取输出维度信息 - 分类任务输出形状为 [batch_size, num_classes]
    nvinfer1::Dims output_dims = context_->getTensorShape(output_name_);
    int actual_batch_size = output_dims.d[0];  // 实际batch size
    int num_classes = output_dims.d[1];        // 实际类别数
    
    LOG(INFO) << "Classification output: actual_batch_size=" << actual_batch_size 
              << ", num_classes=" << num_classes;
    LOG(INFO) << "Raw output size: " << output.size();
    
    // 验证输出大小 - 使用实际的TensorRT输出形状
    size_t expected_size = actual_batch_size * num_classes;
    if (output.size() != expected_size) {
        LOG(ERROR) << "Output size mismatch! Expected: " << expected_size 
                   << ", Actual: " << output.size();
        return std::vector<std::vector<float>>();
    }
    
    // 验证batch size是否匹配
    if (actual_batch_size != current_batch_size) {
        LOG(WARNING) << "Batch size mismatch! Expected: " << current_batch_size 
                     << ", Actual: " << actual_batch_size;
    }
    
    std::vector<std::vector<float>> results;
    results.reserve(actual_batch_size);
    
    // 为每个batch中的图像处理分类结果
    for (int batch_idx = 0; batch_idx < actual_batch_size; ++batch_idx) {
        // 计算当前batch的输出偏移
        int batch_output_offset = batch_idx * num_classes;
        
        // 找到最高概率的类别
        float max_prob = 0.0f;
        int max_class = 0;
        
        // 遍历所有类别概率
        for (int class_idx = 0; class_idx < num_classes; ++class_idx) {
            float prob = output[batch_output_offset + class_idx];
            if (prob > max_prob) {
                max_prob = prob;
                max_class = class_idx;
            }
        }
        
        // 创建分类结果 [probability, class_id]
        results.push_back({max_prob, static_cast<float>(max_class)});
        LOG(INFO) << "Image " << batch_idx << " classified as class " << max_class 
                  << " with probability: " << max_prob;
    }
    
    LOG(INFO) << "Classification processing completed, processed " << results.size() << " images";
    return results;
}

std::string Yolov11ClassifyGPU::get_class_name(int class_id) const
{
    // 根据类别ID返回类别名称
    // 这里可以根据实际需求修改类别名称
    switch (class_id) {
        case 0:
            return "class_0";  // 类别0
        case 1:
            return "class_1";  // 类别1
        case 2:
            return "class_2";  // 类别2
        case 3:
            return "class_3";  // 类别3
        case 4:
            return "class_4";  // 类别4
        case 5:
            return "class_5";  // 类别5
        case 6:
            return "class_6";  // 类别6
        case 7:
            return "class_7";  // 类别7
        case 8:
            return "class_8";  // 类别8
        case 9:
            return "class_9";  // 类别9
        default:
            return "class_unknown";  // 未知类别
    }
}