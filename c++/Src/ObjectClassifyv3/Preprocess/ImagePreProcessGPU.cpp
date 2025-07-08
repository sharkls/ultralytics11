/*******************************************************
 文件名：ImagePreProcessGPU.cpp
 作者：sharkls
 描述：GPU加速的图像预处理模块实现
 版本：v1.0
 日期：2025-01-20
 *******************************************************/

#include "ImagePreProcessGPU.h"
#include <chrono>

// 注册模块
REGISTER_MODULE("ObjectClassify", ImagePreProcessGPU, ImagePreProcessGPU)

ImagePreProcessGPU::ImagePreProcessGPU(const std::string& exe_path) : IBaseModule(exe_path) {
    m_cudaStream = nullptr;
    m_cublasHandle = nullptr;
    m_gpuInputBuffer = nullptr;
    m_gpuOutputBuffer = nullptr;
    m_gpuTempBuffer = nullptr;
    m_maxGPUBufferSize = 0;
    m_cudaInitialized = false;
    channels_ = 3;  // 默认3通道（RGB）
}

ImagePreProcessGPU::~ImagePreProcessGPU() {
    cleanupCUDA();
}

bool ImagePreProcessGPU::init(void* p_pAlgParam) {
    LOG(INFO) << "ImagePreProcessGPU::init status: start ";
    
    // 1. 从配置参数中读取预处理参数
    if (!p_pAlgParam) {
        return false;
    }
    
    // 2. 参数格式转换
    posetimation::YOLOModelConfig* yoloConfig = static_cast<posetimation::YOLOModelConfig*>(p_pAlgParam);
    max_model_size_ = yoloConfig->width();
    stride_ = yoloConfig->stride(2);
    status_ = yoloConfig->run_status();

    if (stride_ <= 0) {
        LOG(ERROR) << "stride value error!";
        return false;
    }

    m_poseConfig = *yoloConfig;
    
    // 设置固定输出尺寸
    target_w_ = yoloConfig->width();
    target_h_ = yoloConfig->height();
    
    LOG(INFO) << "Fixed output size: " << target_w_ << "x" << target_h_;
    
    // 3. 初始化CUDA
    if (!initCUDA()) {
        LOG(ERROR) << "Failed to initialize CUDA";
        return false;
    }
    
    LOG(INFO) << "ImagePreProcessGPU::init status: success, max_model_size_: " << max_model_size_ << ", stride_: " << stride_;
    return true;
}

bool ImagePreProcessGPU::initCUDA() {
    // 1. 初始化CUDA运行时
    cudaError_t cuda_status = cudaSetDevice(0);  // 使用第一个GPU
    if (cuda_status != cudaSuccess) {
        LOG(ERROR) << "Failed to set CUDA device: " << cudaGetErrorString(cuda_status);
        return false;
    }
    
    // 2. 创建CUDA流
    cuda_status = cudaStreamCreate(&m_cudaStream);
    if (cuda_status != cudaSuccess) {
        LOG(ERROR) << "Failed to create CUDA stream: " << cudaGetErrorString(cuda_status);
        return false;
    }
    
    // 3. 创建cuBLAS句柄
    cublasStatus_t cublas_status = cublasCreate(&m_cublasHandle);
    if (cublas_status != CUBLAS_STATUS_SUCCESS) {
        LOG(ERROR) << "Failed to create cuBLAS handle";
        cudaStreamDestroy(m_cudaStream);
        m_cudaStream = nullptr;
        return false;
    }
    
    // 4. 设置cuBLAS流
    cublas_status = cublasSetStream(m_cublasHandle, m_cudaStream);
    if (cublas_status != CUBLAS_STATUS_SUCCESS) {
        LOG(ERROR) << "Failed to set cuBLAS stream";
        cublasDestroy(m_cublasHandle);
        cudaStreamDestroy(m_cudaStream);
        m_cublasHandle = nullptr;
        m_cudaStream = nullptr;
        return false;
    }
    
    // 5. 分配GPU内存
    // 计算最大内存需求：
    // - 输入缓冲区：uchar3格式，最大原始图像尺寸
    // - 输出缓冲区：float格式，CHW格式，目标尺寸
    // - 临时缓冲区：float格式，中间处理结果
    size_t max_input_size = max_model_size_ * max_model_size_ * 3;  // uchar3
    size_t max_output_size = max_model_size_ * max_model_size_ * 3 * sizeof(float);  // float CHW
    size_t max_buffer_size = std::max(max_input_size, max_output_size);
    
    // 增加缓冲区大小以确保安全
    max_buffer_size = std::max(max_buffer_size, (size_t)(1280 * 720 * 3 * sizeof(float))); // 支持1280x720图像
    
    if (!allocateGPUMemory(max_buffer_size)) {
        LOG(ERROR) << "Failed to allocate GPU memory";
        cleanupCUDA();
        return false;
    }
    
    m_cudaInitialized = true;
    LOG(INFO) << "CUDA initialization successful";
    return true;
}

void ImagePreProcessGPU::cleanupCUDA() {
    if (m_cudaInitialized) {
        freeGPUMemory();
        
        if (m_cublasHandle) {
            cublasDestroy(m_cublasHandle);
            m_cublasHandle = nullptr;
        }
        
        if (m_cudaStream) {
            cudaStreamDestroy(m_cudaStream);
            m_cudaStream = nullptr;
        }
        
        m_cudaInitialized = false;
        LOG(INFO) << "CUDA cleanup completed";
    }
}

bool ImagePreProcessGPU::allocateGPUMemory(size_t max_image_size) {
    m_maxGPUBufferSize = max_image_size;
    
    // 分配输入缓冲区
    cudaError_t status = cudaMalloc(&m_gpuInputBuffer, max_image_size);
    if (status != cudaSuccess) {
        LOG(ERROR) << "Failed to allocate GPU input buffer: " << cudaGetErrorString(status);
        return false;
    }
    
    // 分配输出缓冲区
    status = cudaMalloc(&m_gpuOutputBuffer, max_image_size);
    if (status != cudaSuccess) {
        LOG(ERROR) << "Failed to allocate GPU output buffer: " << cudaGetErrorString(status);
        cudaFree(m_gpuInputBuffer);
        m_gpuInputBuffer = nullptr;
        return false;
    }
    
    // 分配临时缓冲区
    status = cudaMalloc(&m_gpuTempBuffer, max_image_size);
    if (status != cudaSuccess) {
        LOG(ERROR) << "Failed to allocate GPU temp buffer: " << cudaGetErrorString(status);
        cudaFree(m_gpuInputBuffer);
        cudaFree(m_gpuOutputBuffer);
        m_gpuInputBuffer = nullptr;
        m_gpuOutputBuffer = nullptr;
        return false;
    }
    
    LOG(INFO) << "GPU memory allocation successful, size: " << max_image_size / (1024 * 1024) << " MB";
    return true;
}

void ImagePreProcessGPU::freeGPUMemory() {
    if (m_gpuInputBuffer) {
        cudaFree(m_gpuInputBuffer);
        m_gpuInputBuffer = nullptr;
    }
    if (m_gpuOutputBuffer) {
        cudaFree(m_gpuOutputBuffer);
        m_gpuOutputBuffer = nullptr;
    }
    if (m_gpuTempBuffer) {
        cudaFree(m_gpuTempBuffer);
        m_gpuTempBuffer = nullptr;
    }
    m_maxGPUBufferSize = 0;
}

void ImagePreProcessGPU::setInput(void* input) {
    if (!input) {
        LOG(ERROR) << "Input is null";
        return;
    }
    m_inputData = *static_cast<CAlgResult*>(input);
}

void* ImagePreProcessGPU::getOutput() {
    // 只返回GPU版本的结果
    if (!m_outputResultGPU.empty()) {
        return &m_outputResultGPU;
    }
    return nullptr;
}

void ImagePreProcessGPU::execute() {
    LOG(INFO) << "ImagePreProcessGPU::execute status: start ";
    
    // 性能监控开始
    m_start_time = std::chrono::high_resolution_clock::now();
    
    // 检查GPU缓冲区状态
    LOG(INFO) << "GPU buffer status check: input=" << m_gpuInputBuffer 
              << ", output=" << m_gpuOutputBuffer << ", temp=" << m_gpuTempBuffer;
    
    // 清空之前的输出
    m_outputResultGPU.clear();
    
    if (m_inputData.vecFrameResult().empty()) {
        LOG(ERROR) << "Input data is empty";
        return;
    }
    
    const auto& frameResult = m_inputData.vecFrameResult()[0];
    const auto& videoSrcData = frameResult.vecVideoSrcData();
    
    if (videoSrcData.empty()) {
        LOG(ERROR) << "No video source data available";
        return;
    }
    
    LOG(INFO) << "Input frame results count: " << m_inputData.vecFrameResult().size();
    LOG(INFO) << "Processing frame result, video source data count: " << videoSrcData.size();
    
    LOG(INFO) << "Processing " << videoSrcData.size() << " sub-images with fixed size " << target_w_ << "x" << target_h_;
    
    // 第一步：提取所有图像，使用固定目标尺寸
    std::vector<cv::Mat> srcImages;
    
    for (size_t i = 0; i < videoSrcData.size(); ++i) {
        const auto& videoData = videoSrcData[i];
        
        if (videoData.vecImageBuf().empty()) {
            LOG(WARNING) << "Empty image data for sub-image " << i;
            continue;
        }
        
        try {
            // 创建OpenCV Mat对象
            cv::Mat srcImage;
            int channels = videoData.unBmpBytes() / (videoData.usBmpWidth() * videoData.usBmpLength());
            
            if (channels == 3) {
                srcImage = cv::Mat(videoData.usBmpLength(), videoData.usBmpWidth(), CV_8UC3, 
                                  (void*)videoData.vecImageBuf().data());
            } else if (channels == 1) {
                srcImage = cv::Mat(videoData.usBmpLength(), videoData.usBmpWidth(), CV_8UC1, 
                                  (void*)videoData.vecImageBuf().data());
                cv::Mat colorImage;
                cv::cvtColor(srcImage, colorImage, cv::COLOR_GRAY2BGR);
                srcImage = colorImage;
            } else {
                LOG(ERROR) << "Unsupported image channels: " << channels << " for sub-image " << i;
                continue;
            }
            
            srcImages.push_back(srcImage);
            LOG(INFO) << "Sub-image " << i << " original size: " << srcImage.cols << "x" << srcImage.rows;
            
        } catch (const std::exception& e) {
            LOG(ERROR) << "Exception while processing sub-image " << i << ": " << e.what();
            continue;
        }
    }
    
    LOG(INFO) << "Fixed target size: " << target_w_ << "x" << target_h_;
    
    // 第二步：批量GPU预处理
    if (!srcImages.empty()) {
        // 使用GPU内存直接处理，固定尺寸
        if (!processBatchImagesGPUInPlace(srcImages, target_w_, target_h_, m_outputResultGPU)) {
            LOG(ERROR) << "Failed to process images in GPU memory";
            return;
        }
        
        LOG(INFO) << "GPU preprocessing completed, results kept in GPU memory";
    }
    
    LOG(INFO) << "ImagePreProcessGPU::execute status: success, processed " << m_outputResultGPU.size() << " images";
    
    // 计算GPU预处理总耗时
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - m_start_time);
    LOG(INFO) << "GPU预处理总耗时: " << duration.count() / 1000.0 << " ms";
}

bool ImagePreProcessGPU::uploadImageToGPU(const cv::Mat& image, void* gpu_buffer) {
    if (!gpu_buffer) {
        LOG(ERROR) << "GPU buffer is null";
        return false;
    }
    
    // 确保图像是BGR格式
    cv::Mat bgrImage;
    if (image.channels() == 3) {
        bgrImage = image;
    } else if (image.channels() == 1) {
        cv::cvtColor(image, bgrImage, cv::COLOR_GRAY2BGR);
    } else {
        LOG(ERROR) << "Unsupported image format, channels: " << image.channels();
        return false;
    }
    
    // 上传到GPU
    cudaError_t status = cudaMemcpyAsync(gpu_buffer, bgrImage.data, 
                                        bgrImage.total() * bgrImage.elemSize(),
                                        cudaMemcpyHostToDevice, m_cudaStream);
    if (status != cudaSuccess) {
        LOG(ERROR) << "Failed to upload image to GPU: " << cudaGetErrorString(status);
        return false;
    }
    
    return true;
}

// GPU内核函数调用（调用外部C函数）
void ImagePreProcessGPU::callResizeKernel(void* src, void* dst, int src_width, int src_height, 
                                         int dst_width, int dst_height, cudaStream_t stream) {
    launchResizeKernel(static_cast<uchar3*>(src), static_cast<uchar3*>(dst),
                      src_width, src_height, dst_width, dst_height, stream);
}

void ImagePreProcessGPU::callNormalizeKernel(void* src, void* dst, int width, int height, 
                                            float scale, cudaStream_t stream) {
    launchNormalizeKernel(static_cast<uchar3*>(src), static_cast<float*>(dst),
                         width, height, scale, stream);
}

void ImagePreProcessGPU::callHWCtoCHWKernel(void* src, void* dst, int width, int height, 
                                           cudaStream_t stream) {
    launchHWCtoCHWKernel(static_cast<float*>(src), static_cast<float*>(dst),
                        width, height, stream);
}

void ImagePreProcessGPU::callPadImageKernel(void* src, void* dst, int src_width, int src_height, 
                                           int dst_width, int dst_height, int pad_top, int pad_left, 
                                           float pad_value, cudaStream_t stream) {
    launchPadImageKernel(static_cast<float*>(src), static_cast<float*>(dst),
                        src_width, src_height, dst_width, dst_height,
                        pad_top, pad_left, pad_value, stream);
}

void ImagePreProcessGPU::callBgrToRgbKernel(void* src, void* dst, int width, int height, 
                                           cudaStream_t stream) {
    launchBgrToRgbKernel(static_cast<uchar3*>(src), static_cast<uchar3*>(dst),
                        width, height, stream);
}

// GPU内存直接处理函数
bool ImagePreProcessGPU::processBatchImagesGPUInPlace(const std::vector<cv::Mat>& srcImages, int targetWidth, int targetHeight,
                                                     MultiImagePreprocessResultGPU& gpuResult) {
    LOG(INFO) << "processBatchImagesGPUInPlace start: batch size=" << srcImages.size() 
              << ", fixed target size=" << targetWidth << "x" << targetHeight;
    
    if (srcImages.empty()) {
        LOG(ERROR) << "No source images provided";
        return false;
    }
    
    // 清空之前的结果
    gpuResult.clear();
    
    // 计算总GPU内存需求 - 修复：以字节为单位计算
    size_t total_gpu_size_bytes = 0;
    std::vector<size_t> image_sizes_float;  // 以float为单位的大小
    std::vector<size_t> image_sizes_bytes;  // 以字节为单位的大小
    
    for (const auto& srcImage : srcImages) {
        size_t image_size_float = channels_ * targetHeight * targetWidth;
        size_t image_size_bytes = image_size_float * sizeof(float);
        
        image_sizes_float.push_back(image_size_float);
        image_sizes_bytes.push_back(image_size_bytes);
        total_gpu_size_bytes += image_size_bytes;
    }
    
    // 检查GPU内存是否足够
    size_t free_memory, total_memory;
    cudaError_t cuda_status = cudaMemGetInfo(&free_memory, &total_memory);
    if (cuda_status != cudaSuccess) {
        LOG(ERROR) << "Failed to get GPU memory info: " << cudaGetErrorString(cuda_status);
        
        // 尝试重置GPU设备
        cuda_status = cudaDeviceReset();
        if (cuda_status != cudaSuccess) {
            LOG(ERROR) << "Failed to reset GPU device: " << cudaGetErrorString(cuda_status);
            return false;
        }
        
        // 重新初始化GPU资源
        if (!allocateGPUMemory(m_maxGPUBufferSize)) {
            LOG(ERROR) << "Failed to reallocate GPU memory after reset";
            return false;
        }
        
        // 再次尝试获取内存信息
        cuda_status = cudaMemGetInfo(&free_memory, &total_memory);
        if (cuda_status != cudaSuccess) {
            LOG(ERROR) << "Failed to get GPU memory info after reset: " << cudaGetErrorString(cuda_status);
            return false;
        }
    }
    
    size_t required_memory = total_gpu_size_bytes;
    if (required_memory > free_memory) {
        LOG(ERROR) << "Insufficient GPU memory: required " << required_memory 
                   << " bytes, available " << free_memory << " bytes";
        
        // 尝试清理GPU内存并重试
        cuda_status = cudaDeviceReset();
        if (cuda_status != cudaSuccess) {
            LOG(ERROR) << "Failed to reset GPU device: " << cudaGetErrorString(cuda_status);
            return false;
        }
        
        // 重新初始化GPU资源
        if (!allocateGPUMemory(m_maxGPUBufferSize)) {
            LOG(ERROR) << "Failed to reallocate GPU memory after reset";
            return false;
        }
        
        // 再次检查内存
        cuda_status = cudaMemGetInfo(&free_memory, &total_memory);
        if (cuda_status != cudaSuccess) {
            LOG(ERROR) << "Failed to get GPU memory info after second reset: " << cudaGetErrorString(cuda_status);
            return false;
        }
        
        if (required_memory > free_memory) {
            LOG(ERROR) << "Still insufficient GPU memory after reset: required " << required_memory 
                       << " bytes, available " << free_memory << " bytes";
            return false;
        }
    }
    
    LOG(INFO) << "GPU memory check: required " << required_memory 
              << " bytes, available " << free_memory << " bytes";
    
    // 分配GPU内存
    cuda_status = cudaMalloc(&gpuResult.gpu_buffer, required_memory);
    if (cuda_status != cudaSuccess) {
        LOG(ERROR) << "Failed to allocate GPU buffer: " << cudaGetErrorString(cuda_status);
        
        // 尝试清理GPU内存并重试
        cuda_status = cudaDeviceReset();
        if (cuda_status != cudaSuccess) {
            LOG(ERROR) << "Failed to reset GPU device: " << cudaGetErrorString(cuda_status);
            return false;
        }
        
        // 重新初始化GPU资源
        if (!allocateGPUMemory(m_maxGPUBufferSize)) {
            LOG(ERROR) << "Failed to reallocate GPU memory after reset";
            return false;
        }
        
        // 再次尝试分配
        cuda_status = cudaMalloc(&gpuResult.gpu_buffer, required_memory);
        if (cuda_status != cudaSuccess) {
            LOG(ERROR) << "Failed to allocate GPU buffer after reset: " << cudaGetErrorString(cuda_status);
            return false;
        }
    }
    
    gpuResult.total_size = total_gpu_size_bytes;
    gpuResult.channels = channels_;
    
    // 计算每个图像的偏移
    size_t current_offset = 0;
    for (size_t image_size : image_sizes_bytes) {
        gpuResult.image_offsets.push_back(current_offset);
        current_offset += image_size;
    }
    
    // 处理每个图像
    for (size_t i = 0; i < srcImages.size(); ++i) {
        const cv::Mat& srcImage = srcImages[i];
        float* gpu_dst = gpuResult.getImagePtr(i);
        
        if (!gpu_dst) {
            LOG(ERROR) << "Failed to get GPU pointer for image " << i;
            continue;
        }
        
        // 计算预处理参数（固定尺寸）
        float ratio;
        int padTop, padLeft;
        
        // 计算缩放比例
        ratio = std::min(static_cast<float>(targetHeight) / srcImage.rows, 
                        static_cast<float>(targetWidth) / srcImage.cols);
        
        // 计算缩放后的尺寸
        int newWidth = static_cast<int>(srcImage.cols * ratio);
        int newHeight = static_cast<int>(srcImage.rows * ratio);
        
        // 确保尺寸是stride的整数倍
        newWidth = (newWidth / stride_) * stride_;
        newHeight = (newHeight / stride_) * stride_;
        
        // 计算填充
        int dh = targetHeight - newHeight;
        int dw = targetWidth - newWidth;
        padTop = dh / 2;
        padLeft = dw / 2;
        
        // 保存固定图像尺寸
        gpuResult.imageSizes.push_back(std::make_pair(targetWidth, targetHeight));
        
        LOG(INFO) << "Processing image " << i << ": " << srcImage.cols << "x" << srcImage.rows 
                  << " -> " << newWidth << "x" << newHeight << " -> " << targetWidth << "x" << targetHeight;
        
        // 直接在GPU内存中处理图像
        if (!processSingleImageGPUInPlace(srcImage, targetWidth, targetHeight, gpu_dst, 0, ratio, padTop, padLeft)) {
            LOG(ERROR) << "Failed to process image " << i << " in place";
            continue;
        }
        
        LOG(INFO) << "Successfully processed image " << i << " in GPU memory";
    }
    
    LOG(INFO) << "Batch GPU preprocessing in place completed, processed " << gpuResult.size() << " images";
    return true;
}

bool ImagePreProcessGPU::processSingleImageGPUInPlace(const cv::Mat& srcImage, int targetWidth, int targetHeight,
                                                     float* gpu_dst, size_t dst_offset, float& ratio, int& padTop, int& padLeft) {
    LOG(INFO) << "processSingleImageGPUInPlace start: " << srcImage.cols << "x" << srcImage.rows 
              << " -> " << targetWidth << "x" << targetHeight;
    
    // 检查输入参数
    if (srcImage.empty() || !gpu_dst) {
        LOG(ERROR) << "Invalid input parameters";
        return false;
    }
    
    // 检查GPU缓冲区状态
    if (!m_gpuInputBuffer || !m_gpuTempBuffer) {
        LOG(ERROR) << "GPU buffers not initialized";
        return false;
    }
    
    // 计算缩放比例和尺寸
    ratio = std::min(static_cast<float>(targetHeight) / srcImage.rows, 
                    static_cast<float>(targetWidth) / srcImage.cols);
    
    int newWidth = static_cast<int>(srcImage.cols * ratio);
    int newHeight = static_cast<int>(srcImage.rows * ratio);
    newWidth = (newWidth / stride_) * stride_;
    newHeight = (newHeight / stride_) * stride_;
    
    int dh = targetHeight - newHeight;
    int dw = targetWidth - newWidth;
    padTop = dh / 2;
    padLeft = dw / 2;
    
    LOG(INFO) << "GPU preprocessing in place: " << srcImage.cols << "x" << srcImage.rows 
              << " -> " << newWidth << "x" << newHeight << " -> " << targetWidth << "x" << targetHeight;
    
    // 上传原始图像到GPU（uchar3格式）
    if (!uploadImageToGPU(srcImage, m_gpuInputBuffer)) {
        LOG(ERROR) << "Failed to upload image to GPU";
        return false;
    }
    
    // 执行GPU resize（uchar3 -> uchar3）
    callResizeKernel(m_gpuInputBuffer, m_gpuTempBuffer, 
                    srcImage.cols, srcImage.rows, newWidth, newHeight, m_cudaStream);
    
    // 执行GPU BGR到RGB转换（uchar3 -> uchar3）
    callBgrToRgbKernel(m_gpuTempBuffer, m_gpuInputBuffer, newWidth, newHeight, m_cudaStream);
    
    // 执行GPU归一化（uchar3 -> float）
    callNormalizeKernel(m_gpuInputBuffer, m_gpuTempBuffer, 
                       newWidth, newHeight, 1.0f / 255.0f, m_cudaStream);
    
    // 执行GPU填充（float -> float）
    callPadImageKernel(m_gpuTempBuffer, m_gpuOutputBuffer, 
                      newWidth, newHeight, targetWidth, targetHeight,
                      padTop, padLeft, 114.0f / 255.0f, m_cudaStream);
    
    // 执行HWC到CHW转换（float -> float）
    callHWCtoCHWKernel(m_gpuOutputBuffer, m_gpuTempBuffer, 
                      targetWidth, targetHeight, m_cudaStream);
    
    // 将结果复制到目标GPU内存位置
    size_t result_size_float = channels_ * targetHeight * targetWidth;
    size_t result_size_bytes = result_size_float * sizeof(float);
    cudaError_t cuda_status = cudaMemcpyAsync(gpu_dst + dst_offset, m_gpuTempBuffer, 
                                             result_size_bytes, cudaMemcpyDeviceToDevice, m_cudaStream);
    if (cuda_status != cudaSuccess) {
        LOG(ERROR) << "Failed to copy result to target GPU memory: " << cudaGetErrorString(cuda_status);
        return false;
    }
    
    // 同步流
    cuda_status = cudaStreamSynchronize(m_cudaStream);
    if (cuda_status != cudaSuccess) {
        LOG(ERROR) << "Failed to synchronize CUDA stream: " << cudaGetErrorString(cuda_status);
        return false;
    }
    
    LOG(INFO) << "GPU preprocessing in place completed successfully";
    return true;
} 