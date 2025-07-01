/*******************************************************
 文件名：Yolov11PoseGPU.cpp
 作者：sharkls
 描述：GPU加速的YOLOv11姿态估计推理模块实现
 版本：v1.0
 日期：2025-01-20
 *******************************************************/

#include "Yolov11PoseGPU.h"

// 注册模块
REGISTER_MODULE("PoseEstimation", Yolov11PoseGPU, Yolov11PoseGPU)

Yolov11PoseGPU::Yolov11PoseGPU(const std::string& exe_path) : IBaseModule(exe_path) 
{
    // 构造函数初始化
    input_buffers_.resize(1, nullptr);
    output_buffers_.resize(1, nullptr);
    m_maxBatchSize = 8; // 设置最大批处理大小
    m_cudaInitialized = false;
    m_gpuTempBuffer = nullptr;
    m_gpuNMSBuffer = nullptr;
    m_maxGPUBufferSize = 0;
    use_gpu_postprocessing_ = true;  // 默认启用GPU后处理

    // 初始化CUDA流
    cudaError_t cuda_status = cudaStreamCreate(&stream_);
    if (cuda_status != cudaSuccess) {
        throw std::runtime_error("创建CUDA流失败: " + std::string(cudaGetErrorString(cuda_status)));
    }
}

Yolov11PoseGPU::~Yolov11PoseGPU() {
    cleanup();
}

bool Yolov11PoseGPU::init(void* p_pAlgParam) 
{   
    LOG(INFO) << "Yolov11PoseGPU::init status: start ";
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
    target_h_ = m_poseConfig.height();
    target_w_ = m_poseConfig.width();

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

    // 4. 初始化CUDA
    if (!initCUDA()) {
        LOG(ERROR) << "Failed to initialize CUDA";
        return false;
    }

    // 5. 初始化TensorRT相关配置
    initTensorRT();
    
    // 6. 初始化GPU后处理器
    if (use_gpu_postprocessing_) {
        gpu_postprocessor_ = std::make_unique<GPUPostProcessor>();
        if (!gpu_postprocessor_->initialize(m_maxBatchSize, max_dets_)) {
            LOG(ERROR) << "Failed to initialize GPU post-processor";
            return false;
        }
        LOG(INFO) << "GPU post-processor initialized successfully";
    }
    
    LOG(INFO) << "Yolov11PoseGPU::init status: success ";
    return true;
}

bool Yolov11PoseGPU::initCUDA() {
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

void Yolov11PoseGPU::cleanupCUDA() {
    if (m_cublasHandle) {
        cublasDestroy(m_cublasHandle);
        m_cublasHandle = nullptr;
    }

    if (m_gpuTempBuffer) {
        cudaFree(m_gpuTempBuffer);
        m_gpuTempBuffer = nullptr;
    }

    if (m_gpuNMSBuffer) {
        cudaFree(m_gpuNMSBuffer);
        m_gpuNMSBuffer = nullptr;
    }

    m_cudaInitialized = false;
}

bool Yolov11PoseGPU::allocateGPUMemory(size_t max_batch_size) {
    // 计算需要的GPU内存大小
    size_t temp_buffer_size = max_batch_size * channels_ * target_h_ * target_w_ * sizeof(float);
    size_t nms_buffer_size = max_batch_size * max_dets_ * sizeof(float);

    // 分配临时缓冲区
    if (temp_buffer_size > m_maxGPUBufferSize) {
        if (m_gpuTempBuffer) {
            cudaFree(m_gpuTempBuffer);
        }
        cudaError_t status = cudaMalloc(&m_gpuTempBuffer, temp_buffer_size);
        if (status != cudaSuccess) {
            LOG(ERROR) << "Failed to allocate GPU temp buffer: " << cudaGetErrorString(status);
            return false;
        }
        m_maxGPUBufferSize = temp_buffer_size;
    }

    // 分配NMS缓冲区
    if (m_gpuNMSBuffer) {
        cudaFree(m_gpuNMSBuffer);
    }
    cudaError_t status = cudaMalloc(&m_gpuNMSBuffer, nms_buffer_size);
    if (status != cudaSuccess) {
        LOG(ERROR) << "Failed to allocate GPU NMS buffer: " << cudaGetErrorString(status);
        return false;
    }

    return true;
}

void Yolov11PoseGPU::freeGPUMemory() {
    if (m_gpuTempBuffer) {
        cudaFree(m_gpuTempBuffer);
        m_gpuTempBuffer = nullptr;
    }
    if (m_gpuNMSBuffer) {
        cudaFree(m_gpuNMSBuffer);
        m_gpuNMSBuffer = nullptr;
    }
    m_maxGPUBufferSize = 0;
}

void Yolov11PoseGPU::initTensorRT()
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

    // 6. 计算输入输出大小
    input_size_ = batch_size_ * channels_ * new_unpad_h_ * new_unpad_w_;
    output_size_ = batch_size_ * (4 + num_classes_ + num_keys_ * 3) * num_anchors_;
    LOG(INFO) << "input_size_ = " << input_size_ << ", output_size_ = " << output_size_ << ", num_anchors_ : " << num_anchors_;

    // 7. 分配GPU内存
    void* input_buffer = nullptr;
    cudaError_t cuda_status = cudaMalloc(&input_buffer, input_size_ * sizeof(float));
    if (cuda_status != cudaSuccess) {
        throw std::runtime_error("分配输入GPU内存失败: " + std::string(cudaGetErrorString(cuda_status)));
    }
    input_buffers_[0] = input_buffer;

    void* output_buffer = nullptr;
    cuda_status = cudaMalloc(&output_buffer, output_size_ * sizeof(float));
    if (cuda_status != cudaSuccess) {
        cudaFree(input_buffer);
        throw std::runtime_error("分配输出GPU内存失败: " + std::string(cudaGetErrorString(cuda_status)));
    }
    output_buffers_[0] = output_buffer;

    // 8. 设置绑定
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

    // 9. 分配GPU内存
    if (!allocateGPUMemory(m_maxBatchSize)) {
        cudaFree(input_buffer);
        cudaFree(output_buffer);
        throw std::runtime_error("分配GPU内存失败");
    }
}

void Yolov11PoseGPU::setInput(void* input) 
{   
    // 核验输入数据的合法性并进行类型转换和保存
    if (!input) {
        LOG(ERROR) << "输入为空";
        return;
    }
    
    // 尝试转换为GPU版本
    MultiImagePreprocessResultGPU* gpu_input = static_cast<MultiImagePreprocessResultGPU*>(input);
    if (gpu_input && !gpu_input->empty()) {
        m_inputDataGPU = *gpu_input;
        LOG(INFO) << "Yolov11PoseGPU::setInput: 接收到GPU版本 " << m_inputDataGPU.size() << " 张图像";
        return;
    }
    
    // 如果不是GPU版本，尝试CPU版本
    MultiImagePreprocessResult* cpu_input = static_cast<MultiImagePreprocessResult*>(input);
    if (cpu_input && !cpu_input->empty()) {
        m_inputData = *cpu_input;
        LOG(INFO) << "Yolov11PoseGPU::setInput: 接收到CPU版本 " << m_inputData.images.size() << " 张图像";
        return;
    }
    
    LOG(ERROR) << "输入数据类型不支持或为空";
}

void* Yolov11PoseGPU::getOutput() {
    return &m_outputResult;
}

void Yolov11PoseGPU::execute() 
{
    LOG(INFO) << "Yolov11PoseGPU::execute status: start ";
    
    // 检查输入数据
    bool use_gpu_input = !m_inputDataGPU.empty();
    bool use_cpu_input = !m_inputData.images.empty();
    
    if (!use_gpu_input && !use_cpu_input) {
        LOG(ERROR) << "No input images available";
        return;
    }
    
    if (use_gpu_input) {
        LOG(INFO) << "Using GPU input data with " << m_inputDataGPU.size() << " images";
    } else {
        LOG(INFO) << "Using CPU input data with " << m_inputData.images.size() << " images";
    }

    // 创建单个FrameResult来合并所有batch的结果
    CFrameResult allFrameResult;
    allFrameResult.eDataType(DATA_TYPE_POSEALG_RESULT);
    
    // 分批处理图像
    size_t total_images = use_gpu_input ? m_inputDataGPU.size() : m_inputData.images.size();
    
    for (size_t batch_start = 0; batch_start < total_images; batch_start += m_maxBatchSize) {
        size_t batch_end = std::min(batch_start + m_maxBatchSize, total_images);
        size_t batch_size = batch_end - batch_start;
        
        LOG(INFO) << "Processing batch " << (batch_start / m_maxBatchSize + 1) 
                  << " with " << batch_size << " images";
        
        // 准备批处理数据
        if (use_gpu_input) {
            prepareBatchDataGPU(batch_start, batch_end);
        } else {
            prepareBatchData(batch_start, batch_end);
        }
        
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
    
    LOG(INFO) << "Yolov11PoseGPU::execute status: success, total results: " 
              << allFrameResult.vecObjectResult().size();
}

void Yolov11PoseGPU::prepareBatchData(size_t batch_start, size_t batch_end)
{
    LOG(INFO) << "Yolov11PoseGPU::prepareBatchData status: start ";
    
    // 清空之前的批处理数据
    m_batchInputs.clear();
    
    // 获取当前批次的图像数据
    size_t batch_size = batch_end - batch_start;
    for (size_t i = batch_start; i < batch_end && i < m_inputData.images.size(); ++i) {
        m_batchInputs.push_back(m_inputData.images[i]);
    }
    
    LOG(INFO) << "Prepared batch data with " << m_batchInputs.size() << " images";
    
    // 检查所有图像的尺寸是否一致
    if (m_batchInputs.empty()) {
        LOG(ERROR) << "No images in batch";
        return;
    }
    
    size_t first_image_size = m_batchInputs[0].size();
    bool all_same_size = true;
    for (const auto& image : m_batchInputs) {
        if (image.size() != first_image_size) {
            all_same_size = false;
            break;
        }
    }
    
    if (!all_same_size) {
        LOG(ERROR) << "All images in batch must have the same size";
        return;
    }
    
    // 根据实际图像尺寸计算输入形状
    int actual_height = 0, actual_width = 0;
    if (first_image_size > 0) {
        // 假设图像是CHW格式，计算高度和宽度
        actual_height = static_cast<int>(sqrt(first_image_size / channels_));
        actual_width = static_cast<int>(first_image_size / (channels_ * actual_height));
        
        // 验证计算是否正确
        if (actual_height * actual_width * channels_ != first_image_size) {
            LOG(ERROR) << "Invalid image size calculation: " << first_image_size 
                       << " != " << (actual_height * actual_width * channels_);
            return;
        }
    }
    
    LOG(INFO) << "Batch image size: " << actual_width << "x" << actual_height 
              << " (total pixels: " << first_image_size << ")";
    
    // 动态调整TensorRT输入形状
    if (actual_height > 0 && actual_width > 0) {
        nvinfer1::Dims new_input_dims;
        new_input_dims.nbDims = 4;
        new_input_dims.d[0] = static_cast<int>(batch_size);  // batch size
        new_input_dims.d[1] = channels_;                     // channels
        new_input_dims.d[2] = actual_height;                 // height
        new_input_dims.d[3] = actual_width;                  // width
        
        // 检查新形状是否与当前形状不同
        nvinfer1::Dims current_dims = context_->getTensorShape(input_name_);
        bool shape_changed = false;
        if (current_dims.nbDims != new_input_dims.nbDims) {
            shape_changed = true;
        } else {
            for (int i = 0; i < current_dims.nbDims; ++i) {
                if (current_dims.d[i] != new_input_dims.d[i]) {
                    shape_changed = true;
                    break;
                }
            }
        }
        
        if (shape_changed) {
            LOG(INFO) << "Resizing TensorRT input from " 
                      << current_dims.d[3] << "x" << current_dims.d[2] 
                      << " to " << new_input_dims.d[3] << "x" << new_input_dims.d[2];
            
            if (!context_->setInputShape(input_name_, new_input_dims)) {
                LOG(ERROR) << "Failed to set new input shape";
                return;
            }
            
            // 更新输出形状
            output_dims_ = context_->getTensorShape(output_name_);
            
            // 重新计算输出大小
            size_t new_output_size = 1;
            for (int i = 0; i < output_dims_.nbDims; ++i) {
                new_output_size *= output_dims_.d[i];
            }
            
            LOG(INFO) << "New output size: " << new_output_size;
        }
    }
}

void Yolov11PoseGPU::prepareBatchDataGPU(size_t batch_start, size_t batch_end)
{
    LOG(INFO) << "Yolov11PoseGPU::prepareBatchDataGPU status: start ";
    
    // 清空之前的批处理数据
    m_batchInputs.clear();
    
    // 获取当前批次的图像数据
    size_t batch_size = batch_end - batch_start;
    
    // 直接使用GPU内存数据，避免CPU-GPU转换
    // 将GPU数据直接复制到TensorRT输入缓冲区
    size_t total_size = 0;
    std::vector<size_t> image_sizes;
    
    for (size_t i = batch_start; i < batch_end && i < m_inputDataGPU.size(); ++i) {
        size_t image_size = m_inputDataGPU.getImageSize(i);
        image_sizes.push_back(image_size);
        total_size += image_size;
    }
    
    LOG(INFO) << "Prepared GPU batch data with " << batch_size << " images, total size: " << total_size;
    
    // 检查所有图像的尺寸是否一致
    if (image_sizes.empty()) {
        LOG(ERROR) << "No images in GPU batch";
        return;
    }
    
    size_t first_image_size = image_sizes[0];
    bool all_same_size = true;
    for (size_t image_size : image_sizes) {
        if (image_size != first_image_size) {
            all_same_size = false;
            break;
        }
    }
    
    if (!all_same_size) {
        LOG(ERROR) << "All images in GPU batch must have the same size";
        return;
    }
    
    // 根据实际图像尺寸计算输入形状
    int actual_height = 0, actual_width = 0;
    if (first_image_size > 0) {
        // 假设图像是CHW格式，计算高度和宽度
        actual_height = static_cast<int>(sqrt(first_image_size / channels_));
        actual_width = static_cast<int>(first_image_size / (channels_ * actual_height));
        
        // 验证计算是否正确
        if (actual_height * actual_width * channels_ != first_image_size) {
            LOG(ERROR) << "Invalid GPU image size calculation: " << first_image_size 
                       << " != " << (actual_height * actual_width * channels_);
            // 使用GPU结果中的实际尺寸
            if (!m_inputDataGPU.imageSizes.empty()) {
                actual_width = m_inputDataGPU.max_width;
                actual_height = m_inputDataGPU.max_height;
                LOG(INFO) << "Using GPU result dimensions: " << actual_width << "x" << actual_height;
            } else {
                return;
            }
        }
    }
    
    LOG(INFO) << "GPU batch image size: " << actual_width << "x" << actual_height 
              << " (total pixels: " << first_image_size << ")";
    
    // 动态调整TensorRT输入形状
    if (actual_height > 0 && actual_width > 0) {
        nvinfer1::Dims new_input_dims;
        new_input_dims.nbDims = 4;
        new_input_dims.d[0] = static_cast<int>(batch_size);  // batch size
        new_input_dims.d[1] = channels_;                     // channels
        new_input_dims.d[2] = actual_height;                 // height
        new_input_dims.d[3] = actual_width;                  // width
        
        // 检查新形状是否与当前形状不同
        nvinfer1::Dims current_dims = context_->getTensorShape(input_name_);
        bool shape_changed = false;
        if (current_dims.nbDims != new_input_dims.nbDims) {
            shape_changed = true;
        } else {
            for (int i = 0; i < current_dims.nbDims; ++i) {
                if (current_dims.d[i] != new_input_dims.d[i]) {
                    shape_changed = true;
                    break;
                }
            }
        }
        
        if (shape_changed) {
            LOG(INFO) << "Resizing TensorRT input from " 
                      << current_dims.d[3] << "x" << current_dims.d[2] 
                      << " to " << new_input_dims.d[3] << "x" << new_input_dims.d[2];
            
            if (!context_->setInputShape(input_name_, new_input_dims)) {
                LOG(ERROR) << "Failed to set new input shape";
                return;
            }
            
            // 更新输出形状
            output_dims_ = context_->getTensorShape(output_name_);
            
            // 重新计算输出大小
            size_t new_output_size = 1;
            for (int i = 0; i < output_dims_.nbDims; ++i) {
                new_output_size *= output_dims_.d[i];
            }
            
            LOG(INFO) << "New output size: " << new_output_size;
        }
    }
}

void Yolov11PoseGPU::cleanup() 
{
    LOG(INFO) << "开始清理TensorRT和CUDA资源...";
    
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
    
    // 清理CUDA资源
    cleanupCUDA();
    
    // 清理TensorRT资源
    context_.reset();
    engine_.reset();
    runtime_.reset();
    
    // 清理GPU后处理器
    if (gpu_postprocessor_) {
        gpu_postprocessor_->cleanup();
        gpu_postprocessor_.reset();
        LOG(INFO) << "GPU post-processor cleanup completed";
    }
    
    LOG(INFO) << "TensorRT和CUDA资源清理完成";
}

std::vector<float> Yolov11PoseGPU::inference()
{
    // 检查输入数据类型
    bool use_gpu_input = !m_inputDataGPU.empty();
    
    if (use_gpu_input) {
        LOG(INFO) << "Yolov11PoseGPU::inference status: start with GPU batch size: " << m_inputDataGPU.size();
        
        if (m_inputDataGPU.empty()) {
            LOG(ERROR) << "No GPU batch data available";
            return std::vector<float>();
        }
        
        // 计算当前批次的实际大小
        int current_batch_size = static_cast<int>(m_inputDataGPU.size());
        
        // 检查batch_size是否超出限制
        if (current_batch_size > m_maxBatchSize) {
            LOG(ERROR) << "Batch size " << current_batch_size << " exceeds maximum " << m_maxBatchSize;
            return std::vector<float>();
        }
    } else {
        LOG(INFO) << "Yolov11PoseGPU::inference status: start with CPU batch size: " << m_batchInputs.size();
        
        if (m_batchInputs.empty()) {
            LOG(ERROR) << "No CPU batch data available";
            return std::vector<float>();
        }
        
        // 计算当前批次的实际大小
        int current_batch_size = static_cast<int>(m_batchInputs.size());
        
        // 检查batch_size是否超出限制
        if (current_batch_size > m_maxBatchSize) {
            LOG(ERROR) << "Batch size " << current_batch_size << " exceeds maximum " << m_maxBatchSize;
            return std::vector<float>();
        }
    }
    
    // 1. 统一图像尺寸 - 找到最大尺寸并填充所有图像（与CPU版本保持一致）
    int max_width = 0, max_height = 0;
    std::vector<std::pair<int, int>> original_sizes;
    int current_batch_size = 0;
    
    if (use_gpu_input) {
        // 使用GPU版本的输入数据
        max_width = m_inputDataGPU.max_width;
        max_height = m_inputDataGPU.max_height;
        original_sizes = m_inputDataGPU.imageSizes;
        current_batch_size = static_cast<int>(m_inputDataGPU.size());
        LOG(INFO) << "Using GPU input dimensions: " << max_width << "x" << max_height;
    } else {
        // 使用CPU版本的输入数据
        for (const auto& size : m_inputData.imageSizes) {
            max_width = std::max(max_width, size.first);
            max_height = std::max(max_height, size.second);
            original_sizes.push_back(size);
        }
        current_batch_size = static_cast<int>(m_batchInputs.size());
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

    // 4. 准备批处理输入数据 - 统一尺寸并填充（与CPU版本保持一致）
    std::vector<float> batch_input;
    size_t single_image_size = channels_ * max_height * max_width;
    batch_input.reserve(current_batch_size * single_image_size);
    
    LOG(INFO) << "单张图像大小: " << single_image_size << " (约 " 
              << (single_image_size * sizeof(float)) / (1024*1024) << "MB)";
    
    if (use_gpu_input) {
        // 直接使用GPU内存数据，避免CPU-GPU转换
        LOG(INFO) << "Using GPU input data directly, skipping CPU-GPU conversion";
        
        // 设置正确的batch大小
        size_t actual_batch_size = std::min(static_cast<size_t>(current_batch_size), m_inputDataGPU.size());
        
        // 将GPU数据直接复制到TensorRT输入缓冲区
        for (size_t i = 0; i < actual_batch_size; ++i) {
            float* gpu_image_ptr = m_inputDataGPU.getImagePtr(i);
            if (gpu_image_ptr) {
                // 直接复制GPU数据到TensorRT输入缓冲区
                size_t image_size = m_inputDataGPU.getImageSize(i);
                cudaError_t cuda_status = cudaMemcpyAsync(
                    static_cast<float*>(input_buffers_[0]) + i * single_image_size,
                    gpu_image_ptr,
                    image_size * sizeof(float),
                    cudaMemcpyDeviceToDevice,
                    stream_
                );
                if (cuda_status != cudaSuccess) {
                    LOG(ERROR) << "Failed to copy GPU image " << i << " to TensorRT buffer: " 
                               << cudaGetErrorString(cuda_status);
                } else {
                    LOG(INFO) << "Successfully copied GPU image " << i << " to TensorRT buffer";
                }
            }
        }
        
        // 同步流
        cudaStreamSynchronize(stream_);
        
    } else {
        // 使用CPU版本的输入数据（原有逻辑）
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
    }

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
    
    LOG(INFO) << "Dynamic output size: " << output_size;

    // 8. 根据是否使用GPU后处理决定是否拷贝数据到CPU
    if (use_gpu_postprocessing_ && gpu_postprocessor_) {
        // 使用GPU后处理，直接返回GPU内存指针
        LOG(INFO) << "Using GPU post-processing, keeping output in GPU memory";
        // 返回一个特殊的向量，第一个元素存储GPU指针的地址值
        std::vector<float> gpu_output_info;
        gpu_output_info.push_back(static_cast<float>(reinterpret_cast<size_t>(output_buffers_[0])));  // GPU指针地址
        gpu_output_info.push_back(static_cast<float>(output_size));              // 输出大小
        gpu_output_info.push_back(1.0f);  // 标记为GPU数据
        
        LOG(INFO) << "Yolov11PoseGPU::inference status: success, GPU output size: " << output_size;
        return gpu_output_info;
    } else {
        // 使用CPU后处理，拷贝数据到CPU
        std::vector<float> output(output_size);
        
        cudaError_t cuda_status = cudaMemcpyAsync(output.data(), output_buffers_[0],
                                      output_size * sizeof(float),
                                      cudaMemcpyDeviceToHost, stream_);
        if (cuda_status != cudaSuccess) {
            throw std::runtime_error("CUDA输出内存拷贝失败: " + std::string(cudaGetErrorString(cuda_status)));
        }
        cudaStreamSynchronize(stream_);

        LOG(INFO) << "Yolov11PoseGPU::inference status: success, CPU output size: " << output.size();
        return output;
    }
}

void Yolov11PoseGPU::rescale_coords(std::vector<float>& coords, bool is_keypoint) 
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

std::vector<std::vector<float>> Yolov11PoseGPU::process_keypoints(const std::vector<float>& output, const std::vector<std::vector<float>>& boxes) {
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

std::vector<std::vector<float>> Yolov11PoseGPU::process_output(const std::vector<float>& output)
{
    LOG(INFO) << "Yolov11PoseGPU::process_output status: start ";
    
    // 检查是否使用GPU后处理
    if (use_gpu_postprocessing_ && gpu_postprocessor_) {
        LOG(INFO) << "Using GPU post-processing for output";
        
        // 检查output是否是GPU数据标记
        if (output.size() >= 3 && output[2] == 1.0f) {
            // 这是GPU数据，提取GPU指针和大小
            size_t gpu_ptr_addr = static_cast<size_t>(output[0]);
            size_t output_size = static_cast<size_t>(output[1]);
            float* gpu_output_ptr = reinterpret_cast<float*>(gpu_ptr_addr);
            
            LOG(INFO) << "Extracted GPU output pointer: " << gpu_output_ptr 
                      << ", size: " << output_size;
            
            // 获取当前批次大小
            int current_batch_size = 0;
            if (!m_inputDataGPU.empty()) {
                current_batch_size = static_cast<int>(m_inputDataGPU.size());
            } else if (!m_batchInputs.empty()) {
                current_batch_size = static_cast<int>(m_batchInputs.size());
            } else {
                LOG(ERROR) << "No input data available for batch size calculation";
                return std::vector<std::vector<float>>();
            }
            
            // 准备预处理参数
            std::vector<float> preprocess_params;
            preprocess_params.reserve(current_batch_size * 5);
            
            for (int i = 0; i < current_batch_size; ++i) {
                float ratio = 1.0f;
                int padTop = 0, padLeft = 0;
                int originalWidth = 0, originalHeight = 0;
                
                if (!m_inputDataGPU.empty() && i < m_inputDataGPU.preprocessParams.size()) {
                    const auto& params = m_inputDataGPU.preprocessParams[i];
                    ratio = params.ratio;
                    padTop = params.padTop;
                    padLeft = params.padLeft;
                    originalWidth = params.originalWidth;
                    originalHeight = params.originalHeight;
                } else if (!m_batchInputs.empty() && i < m_inputData.preprocessParams.size()) {
                    const auto& params = m_inputData.preprocessParams[i];
                    ratio = params.ratio;
                    padTop = params.padTop;
                    padLeft = params.padLeft;
                    originalWidth = params.originalWidth;
                    originalHeight = params.originalHeight;
                }
                
                preprocess_params.push_back(ratio);
                preprocess_params.push_back(static_cast<float>(padTop));
                preprocess_params.push_back(static_cast<float>(padLeft));
                preprocess_params.push_back(static_cast<float>(originalWidth));
                preprocess_params.push_back(static_cast<float>(originalHeight));
            }
            
            // 获取输出维度信息
            nvinfer1::Dims output_dims = context_->getTensorShape(output_name_);
            int feature_dim = 4 + num_classes_ + num_keys_ * 3;
            int num_anchors = output_dims.d[2];
            
            LOG(INFO) << "GPU post-processing: batch_size=" << current_batch_size 
                      << ", feature_dim=" << feature_dim << ", num_anchors=" << num_anchors;
            
            // 执行GPU后处理
            auto batch_results = gpu_postprocessor_->processOutput(
                gpu_output_ptr,  // 使用真正的GPU指针
                current_batch_size,
                feature_dim,
                num_anchors,
                num_classes_,
                num_keys_,
                conf_thres_,
                iou_thres_,
                preprocess_params,
                stream_
            );
            
            // 合并所有batch的结果
            std::vector<std::vector<float>> all_results;
            int total_detections = 0;
            for (const auto& batch_result : batch_results) {
                total_detections += batch_result.size();
            }
            all_results.reserve(total_detections);
            
            for (const auto& batch_result : batch_results) {
                all_results.insert(all_results.end(), batch_result.begin(), batch_result.end());
            }
            
            LOG(INFO) << "GPU post-processing completed, found " << all_results.size() << " detections across " << current_batch_size << " batches";
            return all_results;
        } else {
            LOG(WARNING) << "Output is not GPU data, falling back to CPU processing";
        }
    } else {
        LOG(WARNING) << "GPU post-processing not available, falling back to CPU processing";
    }

    // 原有的CPU后处理逻辑
    // 检查输入数据类型
    bool use_gpu_input = !m_inputDataGPU.empty();
    
    int current_batch_size = 0;
    
    if (use_gpu_input) {
        if (m_inputDataGPU.empty()) {
            LOG(ERROR) << "No GPU batch data available for processing";
            return std::vector<std::vector<float>>();
        }
        current_batch_size = static_cast<int>(m_inputDataGPU.size());
        LOG(INFO) << "Processing GPU batch data with " << current_batch_size << " images";
    } else {
        if (m_batchInputs.empty()) {
            LOG(ERROR) << "No CPU batch data available for processing";
            return std::vector<std::vector<float>>();
        }
        current_batch_size = static_cast<int>(m_batchInputs.size());
        LOG(INFO) << "Processing CPU batch data with " << current_batch_size << " images";
    }
    
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
        // 计算当前batch的输出偏移
        int batch_output_offset = batch_idx * feature_dim * num_anchors;
        
        // 处理当前batch的所有anchor
        for (int anchor_idx = 0; anchor_idx < num_anchors; ++anchor_idx) {
            // 获取边界框坐标
            float x = output[batch_output_offset + 0 * num_anchors + anchor_idx];
            float y = output[batch_output_offset + 1 * num_anchors + anchor_idx];
            float w = output[batch_output_offset + 2 * num_anchors + anchor_idx];
            float h = output[batch_output_offset + 3 * num_anchors + anchor_idx];
            
            // 获取类别置信度
            float max_conf = 0.0f;
            int max_class = 0;
            for (int c = 0; c < num_classes_; ++c) {
                float conf = output[batch_output_offset + (4 + c) * num_anchors + anchor_idx];
                if (conf > max_conf) {
                    max_conf = conf;
                    max_class = c;
                }
            }
            
            // 置信度过滤
            if (max_conf < conf_thres_) continue;
            
            // 检查边界框坐标的有效性
            if (w <= 0 || h <= 0) continue;
            
            // 创建检测结果
            std::vector<float> result;
            result.reserve(4 + 1 + 1 + num_keys_ * 3);  // bbox + confidence + class_id + keypoints
            
            // 添加边界框坐标
            result.push_back(x);
            result.push_back(y);
            result.push_back(w);
            result.push_back(h);
            
            // 添加置信度
            result.push_back(max_conf);
            
            // 添加类别ID
            result.push_back(static_cast<float>(max_class));
            
            // 添加关键点
            for (int k = 0; k < num_keys_; ++k) {
                float kpt_x = output[batch_output_offset + (4 + num_classes_ + k * 3) * num_anchors + anchor_idx];
                float kpt_y = output[batch_output_offset + (4 + num_classes_ + k * 3 + 1) * num_anchors + anchor_idx];
                float kpt_conf = output[batch_output_offset + (4 + num_classes_ + k * 3 + 2) * num_anchors + anchor_idx];
                
                result.push_back(kpt_x);
                result.push_back(kpt_y);
                result.push_back(kpt_conf);
            }
            
            results.push_back(std::move(result));
        }
    }
    
    LOG(INFO) << "CPU post-processing completed, found " << results.size() << " detections";
    return results;
}

std::vector<int> Yolov11PoseGPU::nms(const std::vector<std::vector<float>>& boxes, const std::vector<float>& scores)
{
    if (boxes.empty()) return std::vector<int>();
    
    std::vector<int> indices(boxes.size());
    std::iota(indices.begin(), indices.end(), 0);
    
    // 按置信度排序
    std::sort(indices.begin(), indices.end(), [&scores](int a, int b) {
        return scores[a] > scores[b];
    });
    
    std::vector<int> keep;
    std::vector<bool> suppressed(boxes.size(), false);
    
    for (size_t i = 0; i < indices.size(); ++i) {
        if (suppressed[indices[i]]) continue;
        
        keep.push_back(indices[i]);
        
        for (size_t j = i + 1; j < indices.size(); ++j) {
            if (suppressed[indices[j]]) continue;
            
            // 计算IoU
            const auto& box1 = boxes[indices[i]];
            const auto& box2 = boxes[indices[j]];
            
            float x1 = std::max(box1[0], box2[0]);
            float y1 = std::max(box1[1], box2[1]);
            float x2 = std::min(box1[2], box2[2]);
            float y2 = std::min(box1[3], box2[3]);
            
            if (x2 <= x1 || y2 <= y1) continue;
            
            float intersection = (x2 - x1) * (y2 - y1);
            float area1 = (box1[2] - box1[0]) * (box1[3] - box1[1]);
            float area2 = (box2[2] - box2[0]) * (box2[3] - box2[1]);
            float union_area = area1 + area2 - intersection;
            
            float iou = intersection / union_area;
            
            if (iou > iou_thres_) {
                suppressed[indices[j]] = true;
            }
        }
    }
    
    return keep;
}

CAlgResult Yolov11PoseGPU::formatConverted(std::vector<std::vector<float>> results)
{
    CAlgResult alg_result;
    CFrameResult frame_result;
    frame_result.eDataType(DATA_TYPE_POSEALG_RESULT);
    
    for (const auto& result : results) {
        if (result.size() < 6) continue;  // 至少需要边界框和置信度信息
        
        CObjectResult obj_result;
        obj_result.fVideoConfidence(result[4]);  // 置信度
        obj_result.strClass("person");  // 类别名称
        
        // 设置边界框
        obj_result.fTopLeftX(result[0]);
        obj_result.fTopLeftY(result[1]);
        obj_result.fBottomRightX(result[2]);
        obj_result.fBottomRightY(result[3]);
        
        // 设置关键点
        if (result.size() >= 6 + num_keys_ * 3) {
            std::vector<Keypoint> keypoints;
            for (int i = 0; i < num_keys_; ++i) {
                Keypoint kp;
                kp.x(result[6 + i * 3 + 0]);
                kp.y(result[6 + i * 3 + 1]);
                kp.confidence(result[6 + i * 3 + 2]);
                keypoints.push_back(kp);
            }
            obj_result.vecKeypoints(keypoints);
        }
        
        frame_result.vecObjectResult().push_back(obj_result);
    }

    alg_result.vecFrameResult({frame_result});

    LOG(INFO) << "formatConverted: alg_result.vecFrameResult().size() = " << alg_result.vecFrameResult().size();
    if (alg_result.vecFrameResult().size() > 0)
        LOG(INFO) << "formatConverted: frame_result.vecObjectResult().size() = " << alg_result.vecFrameResult()[0].vecObjectResult().size();
    return alg_result;
}

std::string Yolov11PoseGPU::classify_pose(const std::vector<float>& keypoints) const
{
    // 简单的姿态分类逻辑，可以根据需要扩展
    if (keypoints.size() < num_keys_ * 3) {
        return "unknown";
    }
    
    // 这里可以添加更复杂的姿态分类逻辑
    // 例如基于关键点位置和置信度的分类
    
    return "standing";  // 默认分类
}

void Yolov11PoseGPU::launchNMSKernel(const std::vector<std::vector<float>>& boxes, 
                                    const std::vector<float>& scores,
                                    std::vector<int>& keep_indices)
{
    // GPU加速的NMS实现
    // 这里可以添加CUDA内核来加速NMS计算
    // 暂时使用CPU实现
    keep_indices = nms(boxes, scores);
}

void Yolov11PoseGPU::launchCoordinateTransformKernel(std::vector<float>& coords, 
                                                    bool is_keypoint,
                                                    float ratio, int dw, int dh)
{
    // GPU加速的坐标转换实现
    // 这里可以添加CUDA内核来加速坐标转换
    // 暂时使用CPU实现
    rescale_coords(coords, is_keypoint);
} 