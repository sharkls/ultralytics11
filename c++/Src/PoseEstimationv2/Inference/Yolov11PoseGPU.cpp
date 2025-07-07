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
    m_outputResult.vecFrameResult().clear();
    m_outputResult.mapTimeStamp().clear();
    m_outputResult.mapDelay().clear();
    m_outputResult.mapFps().clear();
    bool use_gpu_input = !m_inputDataGPU.empty();
    bool use_cpu_input = !m_inputData.images.empty();
    if (!use_gpu_input && !use_cpu_input) {
        LOG(ERROR) << "No input images available";
        return;
    }
    size_t total_images = use_gpu_input ? m_inputDataGPU.size() : m_inputData.images.size();
    std::vector<CFrameResult> frameResults;
    frameResults.reserve(total_images);
    for (size_t batch_start = 0; batch_start < total_images; batch_start += m_maxBatchSize) {
        size_t batch_end = std::min(batch_start + m_maxBatchSize, total_images);
        size_t batch_size = batch_end - batch_start;
        if (use_gpu_input) prepareBatchDataGPU(batch_start, batch_end);
        else prepareBatchData(batch_start, batch_end);
        std::vector<float> output = inference();
        std::vector<std::vector<float>> results = process_output(output);
        // 新调用
        auto all_image_results = formatConvertedByImage(results, batch_size);
        for (size_t i = 0; i < batch_size; ++i) {
            CFrameResult frameResult;
            frameResult.eDataType(DATA_TYPE_POSEALG_RESULT);
            const auto& image_results = all_image_results[i];
            if (!image_results.empty()) {
                // 只保留置信度最高的目标
                auto best_it = std::max_element(image_results.begin(), image_results.end(),
                    [](const CObjectResult& a, const CObjectResult& b) {
                        return a.fVideoConfidence() < b.fVideoConfidence();
                    });
                frameResult.vecObjectResult().push_back(*best_it);
                LOG(INFO) << "Image " << (batch_start + i) << " has detection result with confidence: " << best_it->fVideoConfidence();
            } else {
                LOG(INFO) << "Image " << (batch_start + i) << " has no detection result, creating empty FrameResult";
            }
            frameResults.push_back(frameResult);
        }
    }
    for (const auto& frameResult : frameResults) {
        m_outputResult.vecFrameResult().push_back(frameResult);
    }
    LOG(INFO) << "Yolov11PoseGPU::execute status: success, total FrameResults: " 
              << m_outputResult.vecFrameResult().size() 
              << " (expected: " << total_images << ")";
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
        
        // 直接使用output_buffers_[0]作为GPU指针，不通过vector传递
        // 返回一个特殊的向量，第一个元素存储GPU指针的地址值
        std::vector<float> gpu_output_info;
        
        // 将64位指针地址转换为两个32位值
        size_t ptr_addr = reinterpret_cast<size_t>(output_buffers_[0]);
        uint32_t ptr_low = static_cast<uint32_t>(ptr_addr & 0xFFFFFFFF);
        uint32_t ptr_high = static_cast<uint32_t>((ptr_addr >> 32) & 0xFFFFFFFF);
        
        gpu_output_info.push_back(static_cast<float>(ptr_low));   // 指针低32位
        gpu_output_info.push_back(static_cast<float>(ptr_high));  // 指针高32位
        gpu_output_info.push_back(static_cast<float>(output_size)); // 输出大小
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
        if (output.size() >= 4 && output[3] == 1.0f) {
            // 这是GPU数据，提取GPU指针和大小
            uint32_t ptr_low = static_cast<uint32_t>(output[0]);
            uint32_t ptr_high = static_cast<uint32_t>(output[1]);
            size_t output_size = static_cast<size_t>(output[2]);
            
            // 重建64位指针地址
            size_t gpu_ptr_addr = (static_cast<size_t>(ptr_high) << 32) | static_cast<size_t>(ptr_low);
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

    // CPU后处理逻辑
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
    
    // 获取实际的TensorRT输出大小
    nvinfer1::Dims output_dims = context_->getTensorShape(output_name_);
    size_t actual_output_size = 1;
    for (int i = 0; i < output_dims.nbDims; ++i) {
        actual_output_size *= output_dims.d[i];
    }
    
    LOG(INFO) << "Actual TensorRT output size: " << actual_output_size;
    LOG(INFO) << "Output dimensions: [" << output_dims.d[0] << ", " << output_dims.d[1] << ", " << output_dims.d[2] << "]";
    
    // 从GPU下载输出数据到CPU
    std::vector<float> cpu_output(actual_output_size);
    
    cudaError_t status = cudaMemcpyAsync(cpu_output.data(), output_buffers_[0], 
                                        actual_output_size * sizeof(float),
                                        cudaMemcpyDeviceToHost, stream_);
    if (status != cudaSuccess) {
        LOG(ERROR) << "Failed to copy GPU output to CPU: " << cudaGetErrorString(status);
        return std::vector<std::vector<float>>();
    }
    
    // 同步流
    cudaStreamSynchronize(stream_);
    
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
    int feature_dim = 4 + num_classes_ + num_keys_ * 3;
    int num_anchors = output_dims.d[2];
    
    LOG(INFO) << "CPU post-processing: batch_size=" << current_batch_size 
              << ", feature_dim=" << feature_dim << ", num_anchors=" << num_anchors;
    
    // 使用CPU后处理
    auto batch_results = gpu_postprocessor_->processOutputCPU(
        cpu_output.data(),
        current_batch_size,
        feature_dim,
        num_anchors,
        num_classes_,
        num_keys_,
        conf_thres_,
        iou_thres_,
        preprocess_params
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
    
    LOG(INFO) << "CPU post-processing completed, found " << all_results.size() << " detections across " << current_batch_size << " batches";
    return all_results;
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

// 新版formatConverted：按图像分组
std::vector<std::vector<CObjectResult>> Yolov11PoseGPU::formatConvertedByImage(std::vector<std::vector<float>> results, size_t batch_size) {
    // 假设results.size() <= batch_size，每个result属于一张图像
    std::vector<std::vector<CObjectResult>> all_image_results(batch_size);
    // 这里假设每个result按顺序属于每个图像（如2图像2结果，第0个属于0，第1个属于1）
    for (size_t i = 0; i < results.size(); ++i) {
        const auto& result = results[i];
        if (result.size() < 6) continue;
        CObjectResult obj_result;
        obj_result.fVideoConfidence(result[4]);
        obj_result.strClass("person");
        obj_result.fTopLeftX(result[0]);
        obj_result.fTopLeftY(result[1]);
        obj_result.fBottomRightX(result[2]);
        obj_result.fBottomRightY(result[3]);
        if (result.size() >= 6 + num_keys_ * 3) {
            std::vector<Keypoint> keypoints;
            std::vector<float> keypoint_data;
            for (int k = 0; k < num_keys_; ++k) {
                Keypoint kp;
                kp.x(result[6 + k * 3 + 0]);
                kp.y(result[6 + k * 3 + 1]);
                kp.confidence(result[6 + k * 3 + 2]);
                keypoints.push_back(kp);
                keypoint_data.push_back(result[6 + k * 3 + 0]);
                keypoint_data.push_back(result[6 + k * 3 + 1]);
                keypoint_data.push_back(result[6 + k * 3 + 2]);
            }
            obj_result.vecKeypoints(keypoints);
            std::string pose_type = classify_pose(keypoint_data);
            if (pose_type == "standing_walking") obj_result.strClass("pose_0");
            else if (pose_type == "hunched") obj_result.strClass("pose_1");
            else if (pose_type == "lying") obj_result.strClass("pose_2");
            else if (pose_type == "sitting") obj_result.strClass("pose_3");
            else if (pose_type == "squatting") obj_result.strClass("pose_4");
            else obj_result.strClass("pose_unknown");
        }
        // 分配到对应图像
        size_t img_idx = i < batch_size ? i : batch_size-1; // 防止越界
        all_image_results[img_idx].push_back(obj_result);
    }
    return all_image_results;
}

std::string Yolov11PoseGPU::classify_pose(const std::vector<float>& keypoints) const
{
    // 检查关键点数据完整性
    if (keypoints.size() < num_keys_ * 3) {
        return "unknown";
    }
    
    // 定义关键点索引 (COCO格式)
    const int NOSE = 0;
    const int LEFT_EYE = 1;
    const int RIGHT_EYE = 2;
    const int LEFT_EAR = 3;
    const int RIGHT_EAR = 4;
    const int LEFT_SHOULDER = 5;
    const int RIGHT_SHOULDER = 6;
    const int LEFT_ELBOW = 7;
    const int RIGHT_ELBOW = 8;
    const int LEFT_WRIST = 9;
    const int RIGHT_WRIST = 10;
    const int LEFT_HIP = 11;
    const int RIGHT_HIP = 12;
    const int LEFT_KNEE = 13;
    const int RIGHT_KNEE = 14;
    const int LEFT_ANKLE = 15;
    const int RIGHT_ANKLE = 16;
    
    // 提取关键点坐标和置信度
    std::vector<std::pair<float, float>> points(num_keys_);
    std::vector<float> confidences(num_keys_);
    
    for (int i = 0; i < num_keys_; ++i) {
        points[i] = {keypoints[i * 3], keypoints[i * 3 + 1]};
        confidences[i] = keypoints[i * 3 + 2];
    }
    
    // 计算几何特征向量
    PoseFeatures features = calculate_pose_features(points, confidences);
    
    // 使用数学化的姿态分类算法
    return classify_pose_mathematical(features);
}

// 姿态特征结构体
struct PoseFeatures {
    float trunk_angle;           // 躯干角度 (度)
    float leg_angle;             // 腿部角度 (度)
    float head_angle;            // 头部角度 (度)
    float body_height_ratio;     // 身体高宽比
    float shoulder_hip_distance; // 肩臀距离
    float knee_ankle_distance;   // 膝踝距离
    float trunk_length;          // 躯干长度
    float leg_length;            // 腿部长度
    float arm_angle;             // 手臂角度
    float stability_score;       // 稳定性评分
    float symmetry_score;        // 对称性评分
};

Yolov11PoseGPU::PoseFeatures Yolov11PoseGPU::calculate_pose_features(const std::vector<std::pair<float, float>>& points,
                                                                    const std::vector<float>& confidences) const
{
    PoseFeatures features = {0.0f};
    
    // 定义关键点索引
    const int NOSE = 0, LEFT_SHOULDER = 5, RIGHT_SHOULDER = 6, LEFT_ELBOW = 7, RIGHT_ELBOW = 8;
    const int LEFT_WRIST = 9, RIGHT_WRIST = 10, LEFT_HIP = 11, RIGHT_HIP = 12;
    const int LEFT_KNEE = 13, RIGHT_KNEE = 14, LEFT_ANKLE = 15, RIGHT_ANKLE = 16;
    
    // 1. 计算躯干角度 (使用向量叉积和点积)
    features.trunk_angle = calculate_trunk_angle_advanced(points, confidences, LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP);
    
    // 2. 计算腿部角度 (使用关节角度计算)
    float left_leg_angle = calculate_joint_angle(points, confidences, LEFT_HIP, LEFT_KNEE, LEFT_ANKLE);
    float right_leg_angle = calculate_joint_angle(points, confidences, RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE);
    features.leg_angle = (left_leg_angle + right_leg_angle) / 2.0f;
    
    // 3. 计算头部角度
    features.head_angle = calculate_head_angle_advanced(points, confidences, NOSE, LEFT_SHOULDER, RIGHT_SHOULDER);
    
    // 4. 计算身体几何特征
    features.body_height_ratio = calculate_body_geometry(points, confidences, NOSE, LEFT_ANKLE, RIGHT_ANKLE);
    
    // 5. 计算距离特征
    features.shoulder_hip_distance = calculate_euclidean_distance(points, confidences, LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP);
    features.knee_ankle_distance = calculate_euclidean_distance(points, confidences, LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE);
    
    // 6. 计算长度特征
    features.trunk_length = calculate_segment_length(points, confidences, LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP);
    features.leg_length = calculate_segment_length(points, confidences, LEFT_HIP, RIGHT_HIP, LEFT_ANKLE, RIGHT_ANKLE);
    
    // 7. 计算手臂角度
    float left_arm_angle = calculate_joint_angle(points, confidences, LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST);
    float right_arm_angle = calculate_joint_angle(points, confidences, RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST);
    features.arm_angle = (left_arm_angle + right_arm_angle) / 2.0f;
    
    // 8. 计算稳定性评分 (基于重心和支撑点)
    features.stability_score = calculate_stability_score(points, confidences);
    
    // 9. 计算对称性评分
    features.symmetry_score = calculate_symmetry_score(points, confidences);
    
    return features;
}

std::string Yolov11PoseGPU::classify_pose_mathematical(const Yolov11PoseGPU::PoseFeatures& features) const
{
    // 使用多维度特征向量进行姿态分类
    // 采用加权评分和阈值判断的方法
    
    // 直立行走的权重配置
    Yolov11PoseGPU::PoseWeights standing_weights = {
        0.25f,  // 躯干角度权重
        0.20f,  // 腿部角度权重
        0.15f,  // 头部角度权重
        0.20f,  // 身体比例权重
        0.15f,  // 稳定性权重
        0.05f   // 对称性权重
    };
    
    // 计算各姿态类型的匹配度评分
    float standing_score = calculate_pose_match_score(features, standing_weights, 
        {0.0f, 0.0f, 0.0f, 2.5f, 0.8f, 0.7f},  // 理想特征值
        {15.0f, 20.0f, 15.0f, 1.0f, 0.2f, 0.3f}); // 容忍范围
    
    Yolov11PoseGPU::PoseWeights hunched_weights = {0.30f, 0.10f, 0.25f, 0.15f, 0.15f, 0.05f};
    float hunched_score = calculate_pose_match_score(features, hunched_weights,
        {40.0f, 0.0f, 30.0f, 2.0f, 0.6f, 0.6f},
        {20.0f, 30.0f, 20.0f, 0.5f, 0.3f, 0.3f});
    
    Yolov11PoseGPU::PoseWeights lying_weights = {0.35f, 0.25f, 0.10f, 0.20f, 0.05f, 0.05f};
    float lying_score = calculate_pose_match_score(features, lying_weights,
        {90.0f, 90.0f, 0.0f, 0.8f, 0.3f, 0.5f},
        {30.0f, 45.0f, 30.0f, 0.4f, 0.3f, 0.4f});
    
    Yolov11PoseGPU::PoseWeights sitting_weights = {0.20f, 0.30f, 0.10f, 0.20f, 0.15f, 0.05f};
    float sitting_score = calculate_pose_match_score(features, sitting_weights,
        {10.0f, 60.0f, 10.0f, 1.8f, 0.7f, 0.6f},
        {20.0f, 30.0f, 20.0f, 0.6f, 0.3f, 0.3f});
    
    Yolov11PoseGPU::PoseWeights squatting_weights = {0.25f, 0.35f, 0.10f, 0.20f, 0.05f, 0.05f};
    float squatting_score = calculate_pose_match_score(features, squatting_weights,
        {30.0f, 90.0f, 10.0f, 1.2f, 0.5f, 0.6f},
        {15.0f, 30.0f, 20.0f, 0.4f, 0.3f, 0.3f});
    
    // 找到最高评分的姿态类型
    std::vector<std::pair<std::string, float>> scores = {
        {"standing_walking", standing_score},
        {"hunched", hunched_score},
        {"lying", lying_score},
        {"sitting", sitting_score},
        {"squatting", squatting_score}
    };
    
    auto max_score = std::max_element(scores.begin(), scores.end(),
        [](const auto& a, const auto& b) { return a.second < b.second; });
    
    // 如果最高评分低于阈值，返回unknown
    if (max_score->second < 0.6f) {
        return "unknown";
    }
    
    return max_score->first;
}

float Yolov11PoseGPU::calculate_pose_match_score(const Yolov11PoseGPU::PoseFeatures& features, 
                                                const Yolov11PoseGPU::PoseWeights& weights,
                                                const std::vector<float>& ideal_values,
                                                const std::vector<float>& tolerances) const
{
    // 计算特征向量与理想值的匹配度评分
    // 使用高斯函数计算相似度
    
    float total_score = 0.0f;
    float total_weight = 0.0f;
    
    // 计算各特征的匹配度
    std::vector<float> feature_values = {
        features.trunk_angle, features.leg_angle, features.head_angle,
        features.body_height_ratio, features.stability_score, features.symmetry_score
    };
    
    std::vector<float> weight_values = {
        weights.trunk_angle_weight, weights.leg_angle_weight, weights.head_angle_weight,
        weights.body_ratio_weight, weights.stability_weight, weights.symmetry_weight
    };
    
    for (size_t i = 0; i < feature_values.size(); ++i) {
        if (weight_values[i] > 0.0f) {
            // 使用高斯函数计算相似度
            float diff = std::abs(feature_values[i] - ideal_values[i]);
            float similarity = std::exp(-(diff * diff) / (2.0f * tolerances[i] * tolerances[i]));
            total_score += similarity * weight_values[i];
            total_weight += weight_values[i];
        }
    }
    
    return total_weight > 0.0f ? total_score / total_weight : 0.0f;
}

// 计算躯干角度 (肩膀到臀部的角度)
float Yolov11PoseGPU::calculate_trunk_angle(const std::vector<std::pair<float, float>>& points, 
                                           const std::vector<float>& confidences,
                                           int left_shoulder, int right_shoulder, 
                                           int left_hip, int right_hip) const
{
    if (confidences[left_shoulder] < 0.3f || confidences[right_shoulder] < 0.3f ||
        confidences[left_hip] < 0.3f || confidences[right_hip] < 0.3f) {
        return 0.0f;
    }
    
    // 计算肩膀中点
    float shoulder_center_x = (points[left_shoulder].first + points[right_shoulder].first) / 2.0f;
    float shoulder_center_y = (points[left_shoulder].second + points[right_shoulder].second) / 2.0f;
    
    // 计算臀部中点
    float hip_center_x = (points[left_hip].first + points[right_hip].first) / 2.0f;
    float hip_center_y = (points[left_hip].second + points[right_hip].second) / 2.0f;
    
    // 计算角度 (相对于垂直方向)
    float dx = shoulder_center_x - hip_center_x;
    float dy = shoulder_center_y - hip_center_y;
    
    if (std::abs(dy) < 1e-6f) return 0.0f;
    
    float angle = std::atan2(dx, dy) * 180.0f / M_PI;
    return angle;
}

// 计算腿部角度
float Yolov11PoseGPU::calculate_leg_angle(const std::vector<std::pair<float, float>>& points,
                                         const std::vector<float>& confidences,
                                         int hip, int knee, int ankle) const
{
    if (confidences[hip] < 0.3f || confidences[knee] < 0.3f || confidences[ankle] < 0.3f) {
        return 0.0f;
    }
    
    // 计算大腿角度 (臀部到膝盖)
    float thigh_dx = points[knee].first - points[hip].first;
    float thigh_dy = points[knee].second - points[hip].second;
    float thigh_angle = std::atan2(thigh_dx, thigh_dy) * 180.0f / M_PI;
    
    // 计算小腿角度 (膝盖到脚踝)
    float shin_dx = points[ankle].first - points[knee].first;
    float shin_dy = points[ankle].second - points[knee].second;
    float shin_angle = std::atan2(shin_dx, shin_dy) * 180.0f / M_PI;
    
    // 返回平均角度
    return (thigh_angle + shin_angle) / 2.0f;
}

// 计算头部角度
float Yolov11PoseGPU::calculate_head_angle(const std::vector<std::pair<float, float>>& points,
                                          const std::vector<float>& confidences,
                                          int nose, int left_shoulder, int right_shoulder) const
{
    if (confidences[nose] < 0.3f || confidences[left_shoulder] < 0.3f || confidences[right_shoulder] < 0.3f) {
        return 0.0f;
    }
    
    // 计算肩膀中点
    float shoulder_center_x = (points[left_shoulder].first + points[right_shoulder].first) / 2.0f;
    float shoulder_center_y = (points[left_shoulder].second + points[right_shoulder].second) / 2.0f;
    
    // 计算头部角度
    float dx = points[nose].first - shoulder_center_x;
    float dy = points[nose].second - shoulder_center_y;
    
    if (std::abs(dy) < 1e-6f) return 0.0f;
    
    float angle = std::atan2(dx, dy) * 180.0f / M_PI;
    return angle;
}

// 计算身体高度比例
float Yolov11PoseGPU::calculate_body_height_ratio(const std::vector<std::pair<float, float>>& points,
                                                 const std::vector<float>& confidences,
                                                 int nose, int left_ankle, int right_ankle) const
{
    if (confidences[nose] < 0.3f || confidences[left_ankle] < 0.3f || confidences[right_ankle] < 0.3f) {
        return 1.0f;
    }
    
    // 计算身体高度
    float body_height = std::abs(points[nose].second - (points[left_ankle].second + points[right_ankle].second) / 2.0f);
    
    // 计算身体宽度 (肩膀宽度作为参考)
    float body_width = std::abs(points[left_ankle].first - points[right_ankle].first);
    
    if (body_width < 1e-6f) return 1.0f;
    
    return body_height / body_width;
}

// 计算两点间距离
float Yolov11PoseGPU::calculate_distance(const std::vector<std::pair<float, float>>& points,
                                        const std::vector<float>& confidences,
                                        int left1, int right1, int left2, int right2) const
{
    if (confidences[left1] < 0.3f || confidences[right1] < 0.3f ||
        confidences[left2] < 0.3f || confidences[right2] < 0.3f) {
        return 0.0f;
    }
    
    // 计算中点
    float center1_x = (points[left1].first + points[right1].first) / 2.0f;
    float center1_y = (points[left1].second + points[right1].second) / 2.0f;
    float center2_x = (points[left2].first + points[right2].first) / 2.0f;
    float center2_y = (points[left2].second + points[right2].second) / 2.0f;
    
    float dx = center1_x - center2_x;
    float dy = center1_y - center2_y;
    
    return std::sqrt(dx * dx + dy * dy);
}

// 判断是否为直立行走
bool Yolov11PoseGPU::is_standing_walking(const std::vector<std::pair<float, float>>& points,
                                        const std::vector<float>& confidences,
                                        float trunk_angle, float leg_angle, float head_angle,
                                        float body_height_ratio) const
{
    // 直立行走的特征：
    // 1. 躯干角度接近垂直 (0度附近)
    // 2. 腿部角度接近垂直
    // 3. 头部角度接近垂直
    // 4. 身体高度比例较大
    
    return (std::abs(trunk_angle) < 15.0f && 
            std::abs(leg_angle) < 20.0f && 
            std::abs(head_angle) < 15.0f && 
            body_height_ratio > 2.0f);
}

// 判断是否为佝偻
bool Yolov11PoseGPU::is_hunched(const std::vector<std::pair<float, float>>& points,
                               const std::vector<float>& confidences,
                               float trunk_angle, float head_angle, float body_height_ratio) const
{
    // 佝偻的特征：
    // 1. 躯干角度较大 (向前弯曲)
    // 2. 头部角度较大 (向前弯曲)
    // 3. 身体高度比例中等
    
    return (trunk_angle > 20.0f && trunk_angle < 60.0f && 
            head_angle > 15.0f && 
            body_height_ratio > 1.5f && body_height_ratio < 2.5f);
}

// 判断是否为躺着
bool Yolov11PoseGPU::is_lying(const std::vector<std::pair<float, float>>& points,
                             const std::vector<float>& confidences,
                             float trunk_angle, float leg_angle, float body_height_ratio) const
{
    // 躺着的特征：
    // 1. 躯干角度接近水平 (90度或-90度)
    // 2. 腿部角度接近水平
    // 3. 身体高度比例很小
    
    return ((std::abs(trunk_angle) > 60.0f && std::abs(trunk_angle) < 120.0f) && 
            std::abs(leg_angle) > 45.0f && 
            body_height_ratio < 1.0f);
}

// 判断是否为坐着
bool Yolov11PoseGPU::is_sitting(const std::vector<std::pair<float, float>>& points,
                               const std::vector<float>& confidences,
                               float trunk_angle, float leg_angle, 
                               float shoulder_hip_distance, float knee_ankle_distance) const
{
    // 坐着的特征：
    // 1. 躯干角度接近垂直
    // 2. 腿部角度较大 (膝盖弯曲)
    // 3. 肩膀到臀部距离适中
    // 4. 膝盖到脚踝距离较小
    
    return (std::abs(trunk_angle) < 20.0f && 
            leg_angle > 30.0f && leg_angle < 90.0f && 
            shoulder_hip_distance > 0.0f && 
            knee_ankle_distance < shoulder_hip_distance * 0.8f);
}

// 判断是否为蹲着
bool Yolov11PoseGPU::is_squatting(const std::vector<std::pair<float, float>>& points,
                                 const std::vector<float>& confidences,
                                 float trunk_angle, float leg_angle, float body_height_ratio) const
{
    // 蹲着的特征：
    // 1. 躯干角度较大 (向前倾斜)
    // 2. 腿部角度很大 (膝盖严重弯曲)
    // 3. 身体高度比例较小
    
    return (trunk_angle > 15.0f && trunk_angle < 45.0f && 
            leg_angle > 60.0f && 
            body_height_ratio < 1.5f);
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

// 高级躯干角度计算 (使用向量叉积和点积)
float Yolov11PoseGPU::calculate_trunk_angle_advanced(const std::vector<std::pair<float, float>>& points,
                                                    const std::vector<float>& confidences,
                                                    int left_shoulder, int right_shoulder,
                                                    int left_hip, int right_hip) const
{
    if (confidences[left_shoulder] < 0.3f || confidences[right_shoulder] < 0.3f ||
        confidences[left_hip] < 0.3f || confidences[right_hip] < 0.3f) {
        return 0.0f;
    }
    
    // 计算肩膀向量和臀部向量
    float shoulder_dx = points[right_shoulder].first - points[left_shoulder].first;
    float shoulder_dy = points[right_shoulder].second - points[left_shoulder].second;
    float hip_dx = points[right_hip].first - points[left_hip].first;
    float hip_dy = points[right_hip].second - points[left_hip].second;
    
    // 计算躯干向量 (肩膀中点到臀部中点)
    float trunk_dx = (points[left_shoulder].first + points[right_shoulder].first) / 2.0f - 
                     (points[left_hip].first + points[right_hip].first) / 2.0f;
    float trunk_dy = (points[left_shoulder].second + points[right_shoulder].second) / 2.0f - 
                     (points[left_hip].second + points[right_hip].second) / 2.0f;
    
    // 使用向量叉积计算角度
    float cross_product = shoulder_dx * hip_dy - shoulder_dy * hip_dx;
    float dot_product = shoulder_dx * hip_dx + shoulder_dy * hip_dy;
    
    float angle = std::atan2(cross_product, dot_product) * 180.0f / M_PI;
    
    // 计算躯干相对于垂直方向的角度
    float trunk_vertical_angle = std::atan2(trunk_dx, trunk_dy) * 180.0f / M_PI;
    
    return trunk_vertical_angle;
}

// 关节角度计算 (三点角度)
float Yolov11PoseGPU::calculate_joint_angle(const std::vector<std::pair<float, float>>& points,
                                           const std::vector<float>& confidences,
                                           int joint1, int joint2, int joint3) const
{
    if (confidences[joint1] < 0.3f || confidences[joint2] < 0.3f || confidences[joint3] < 0.3f) {
        return 0.0f;
    }
    
    // 计算两个向量
    float vec1_dx = points[joint1].first - points[joint2].first;
    float vec1_dy = points[joint1].second - points[joint2].second;
    float vec2_dx = points[joint3].first - points[joint2].first;
    float vec2_dy = points[joint3].second - points[joint2].second;
    
    // 计算向量长度
    float len1 = std::sqrt(vec1_dx * vec1_dx + vec1_dy * vec1_dy);
    float len2 = std::sqrt(vec2_dx * vec2_dx + vec2_dy * vec2_dy);
    
    if (len1 < 1e-6f || len2 < 1e-6f) return 0.0f;
    
    // 计算点积和叉积
    float dot_product = vec1_dx * vec2_dx + vec1_dy * vec2_dy;
    float cross_product = vec1_dx * vec2_dy - vec1_dy * vec2_dx;
    
    // 计算角度
    float cos_angle = dot_product / (len1 * len2);
    cos_angle = std::max(-1.0f, std::min(1.0f, cos_angle)); // 限制在[-1, 1]范围内
    
    float angle = std::acos(cos_angle) * 180.0f / M_PI;
    
    // 根据叉积符号确定角度方向
    if (cross_product < 0) {
        angle = 360.0f - angle;
    }
    
    return angle;
}

// 高级头部角度计算
float Yolov11PoseGPU::calculate_head_angle_advanced(const std::vector<std::pair<float, float>>& points,
                                                   const std::vector<float>& confidences,
                                                   int nose, int left_shoulder, int right_shoulder) const
{
    if (confidences[nose] < 0.3f || confidences[left_shoulder] < 0.3f || confidences[right_shoulder] < 0.3f) {
        return 0.0f;
    }
    
    // 计算肩膀向量
    float shoulder_dx = points[right_shoulder].first - points[left_shoulder].first;
    float shoulder_dy = points[right_shoulder].second - points[left_shoulder].second;
    
    // 计算头部向量 (鼻子到肩膀中点)
    float head_dx = points[nose].first - (points[left_shoulder].first + points[right_shoulder].first) / 2.0f;
    float head_dy = points[nose].second - (points[left_shoulder].second + points[right_shoulder].second) / 2.0f;
    
    // 计算头部相对于肩膀的角度
    float shoulder_angle = std::atan2(shoulder_dy, shoulder_dx) * 180.0f / M_PI;
    float head_angle = std::atan2(head_dy, head_dx) * 180.0f / M_PI;
    
    // 计算相对角度
    float relative_angle = head_angle - shoulder_angle;
    
    // 标准化到[-180, 180]范围
    while (relative_angle > 180.0f) relative_angle -= 360.0f;
    while (relative_angle < -180.0f) relative_angle += 360.0f;
    
    return relative_angle;
}

// 身体几何特征计算
float Yolov11PoseGPU::calculate_body_geometry(const std::vector<std::pair<float, float>>& points,
                                             const std::vector<float>& confidences,
                                             int nose, int left_ankle, int right_ankle) const
{
    if (confidences[nose] < 0.3f || confidences[left_ankle] < 0.3f || confidences[right_ankle] < 0.3f) {
        return 1.0f;
    }
    
    // 计算身体高度 (鼻子到脚踝中点)
    float body_height = std::abs(points[nose].second - (points[left_ankle].second + points[right_ankle].second) / 2.0f);
    
    // 计算身体宽度 (脚踝间距)
    float body_width = std::abs(points[left_ankle].first - points[right_ankle].first);
    
    // 计算身体面积 (近似为矩形)
    float body_area = body_height * body_width;
    
    // 计算身体周长
    float body_perimeter = 2.0f * (body_height + body_width);
    
    // 计算身体紧凑度 (面积与周长平方的比值)
    float compactness = body_area / (body_perimeter * body_perimeter);
    
    // 返回高宽比和紧凑度的组合
    return body_width > 1e-6f ? (body_height / body_width) * (1.0f + compactness) : 1.0f;
}

// 欧几里得距离计算
float Yolov11PoseGPU::calculate_euclidean_distance(const std::vector<std::pair<float, float>>& points,
                                                  const std::vector<float>& confidences,
                                                  int left1, int right1, int left2, int right2) const
{
    if (confidences[left1] < 0.3f || confidences[right1] < 0.3f ||
        confidences[left2] < 0.3f || confidences[right2] < 0.3f) {
        return 0.0f;
    }
    
    // 计算两个中点
    float center1_x = (points[left1].first + points[right1].first) / 2.0f;
    float center1_y = (points[left1].second + points[right1].second) / 2.0f;
    float center2_x = (points[left2].first + points[right2].first) / 2.0f;
    float center2_y = (points[left2].second + points[right2].second) / 2.0f;
    
    // 计算欧几里得距离
    float dx = center1_x - center2_x;
    float dy = center1_y - center2_y;
    
    return std::sqrt(dx * dx + dy * dy);
}

// 线段长度计算
float Yolov11PoseGPU::calculate_segment_length(const std::vector<std::pair<float, float>>& points,
                                              const std::vector<float>& confidences,
                                              int left1, int right1, int left2, int right2) const
{
    if (confidences[left1] < 0.3f || confidences[right1] < 0.3f ||
        confidences[left2] < 0.3f || confidences[right2] < 0.3f) {
        return 0.0f;
    }
    
    // 计算两个中点
    float center1_x = (points[left1].first + points[right1].first) / 2.0f;
    float center1_y = (points[left1].second + points[right1].second) / 2.0f;
    float center2_x = (points[left2].first + points[right2].first) / 2.0f;
    float center2_y = (points[left2].second + points[right2].second) / 2.0f;
    
    // 计算线段长度
    float dx = center1_x - center2_x;
    float dy = center1_y - center2_y;
    
    return std::sqrt(dx * dx + dy * dy);
}

// 稳定性评分计算
float Yolov11PoseGPU::calculate_stability_score(const std::vector<std::pair<float, float>>& points,
                                               const std::vector<float>& confidences) const
{
    const int LEFT_SHOULDER = 5, RIGHT_SHOULDER = 6, LEFT_HIP = 11, RIGHT_HIP = 12;
    const int LEFT_KNEE = 13, RIGHT_KNEE = 14, LEFT_ANKLE = 15, RIGHT_ANKLE = 16;
    
    // 检查关键点置信度
    if (confidences[LEFT_SHOULDER] < 0.3f || confidences[RIGHT_SHOULDER] < 0.3f ||
        confidences[LEFT_HIP] < 0.3f || confidences[RIGHT_HIP] < 0.3f ||
        confidences[LEFT_KNEE] < 0.3f || confidences[RIGHT_KNEE] < 0.3f ||
        confidences[LEFT_ANKLE] < 0.3f || confidences[RIGHT_ANKLE] < 0.3f) {
        return 0.5f; // 默认中等稳定性
    }
    
    // 计算重心位置 (肩膀和臀部的加权平均)
    float center_x = (points[LEFT_SHOULDER].first + points[RIGHT_SHOULDER].first + 
                     points[LEFT_HIP].first + points[RIGHT_HIP].first) / 4.0f;
    float center_y = (points[LEFT_SHOULDER].second + points[RIGHT_SHOULDER].second + 
                     points[LEFT_HIP].second + points[RIGHT_HIP].second) / 4.0f;
    
    // 计算支撑点 (脚踝)
    float support_x = (points[LEFT_ANKLE].first + points[RIGHT_ANKLE].first) / 2.0f;
    float support_y = (points[LEFT_ANKLE].second + points[RIGHT_ANKLE].second) / 2.0f;
    
    // 计算重心到支撑点的距离
    float dx = center_x - support_x;
    float dy = center_y - support_y;
    float distance = std::sqrt(dx * dx + dy * dy);
    
    // 计算支撑面积 (脚踝间距)
    float support_width = std::abs(points[LEFT_ANKLE].first - points[RIGHT_ANKLE].first);
    
    // 稳定性评分：距离越小，支撑面积越大，稳定性越高
    float distance_score = std::exp(-distance / 100.0f); // 距离评分
    float support_score = std::min(1.0f, support_width / 50.0f); // 支撑面积评分
    
    return (distance_score + support_score) / 2.0f;
}

// 对称性评分计算
float Yolov11PoseGPU::calculate_symmetry_score(const std::vector<std::pair<float, float>>& points,
                                              const std::vector<float>& confidences) const
{
    const int LEFT_SHOULDER = 5, RIGHT_SHOULDER = 6, LEFT_ELBOW = 7, RIGHT_ELBOW = 8;
    const int LEFT_WRIST = 9, RIGHT_WRIST = 10, LEFT_HIP = 11, RIGHT_HIP = 12;
    const int LEFT_KNEE = 13, RIGHT_KNEE = 14, LEFT_ANKLE = 15, RIGHT_ANKLE = 16;
    
    // 计算身体中轴线 (肩膀中点到臀部中点)
    float shoulder_center_x = (points[LEFT_SHOULDER].first + points[RIGHT_SHOULDER].first) / 2.0f;
    float shoulder_center_y = (points[LEFT_SHOULDER].second + points[RIGHT_SHOULDER].second) / 2.0f;
    float hip_center_x = (points[LEFT_HIP].first + points[RIGHT_HIP].first) / 2.0f;
    float hip_center_y = (points[LEFT_HIP].second + points[RIGHT_HIP].second) / 2.0f;
    
    // 计算对称点对的距离差异
    std::vector<std::pair<int, int>> symmetric_pairs = {
        {LEFT_SHOULDER, RIGHT_SHOULDER},
        {LEFT_ELBOW, RIGHT_ELBOW},
        {LEFT_WRIST, RIGHT_WRIST},
        {LEFT_HIP, RIGHT_HIP},
        {LEFT_KNEE, RIGHT_KNEE},
        {LEFT_ANKLE, RIGHT_ANKLE}
    };
    
    float total_symmetry = 0.0f;
    int valid_pairs = 0;
    
    for (const auto& pair : symmetric_pairs) {
        if (confidences[pair.first] >= 0.3f && confidences[pair.second] >= 0.3f) {
            // 计算对称点对到中轴线的距离
            float left_dist = std::abs(points[pair.first].first - shoulder_center_x);
            float right_dist = std::abs(points[pair.second].first - shoulder_center_x);
            
            // 计算对称性 (距离差异越小，对称性越高)
            float max_dist = std::max(left_dist, right_dist);
            if (max_dist > 1e-6f) {
                float symmetry = 1.0f - std::abs(left_dist - right_dist) / max_dist;
                total_symmetry += symmetry;
                valid_pairs++;
            }
        }
    }
    
    return valid_pairs > 0 ? total_symmetry / valid_pairs : 0.5f;
} 