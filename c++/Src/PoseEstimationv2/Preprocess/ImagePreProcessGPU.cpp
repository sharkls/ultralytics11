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
REGISTER_MODULE("PoseEstimation", ImagePreProcessGPU, ImagePreProcessGPU)

ImagePreProcessGPU::ImagePreProcessGPU(const std::string& exe_path) : IBaseModule(exe_path) {
    m_cudaStream = nullptr;
    m_cublasHandle = nullptr;
    m_gpuInputBuffer = nullptr;
    m_gpuOutputBuffer = nullptr;
    m_gpuTempBuffer = nullptr;
    m_maxGPUBufferSize = 0;
    m_cudaInitialized = false;
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
    return &m_outputResult;
}

void ImagePreProcessGPU::execute() {
    LOG(INFO) << "ImagePreProcessGPU::execute status: start ";
    
    // 性能监控开始
    m_start_time = std::chrono::high_resolution_clock::now();
    
    // 检查GPU缓冲区状态
    LOG(INFO) << "GPU buffer status check: input=" << m_gpuInputBuffer 
              << ", output=" << m_gpuOutputBuffer << ", temp=" << m_gpuTempBuffer;
    
    // 清空之前的输出
    m_outputResult.clear();
    
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
    LOG(INFO) << "Output results cleared";
    LOG(INFO) << "Processing frame result, video source data count: " << videoSrcData.size();
    
    LOG(INFO) << "Processing " << videoSrcData.size() << " sub-images";
    
    // 第一步：计算所有图像的目标尺寸，取最大值确保一致性（与CPU版本保持一致）
    int targetHeight = 0, targetWidth = 0;
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
            
            // 计算该图像的目标尺寸 - 与CPU版本保持一致
            int src_w = srcImage.cols;
            int src_h = srcImage.rows;
            float r = std::min(static_cast<float>(max_model_size_) / src_h, static_cast<float>(max_model_size_) / src_w);
            int new_unpad_w = static_cast<int>(src_w * r);
            int new_unpad_h = static_cast<int>(src_h * r);
            new_unpad_w = (new_unpad_w / stride_) * stride_;
            new_unpad_h = (new_unpad_h / stride_) * stride_;
            
            targetWidth = std::max(targetWidth, new_unpad_w);
            targetHeight = std::max(targetHeight, new_unpad_h);
            
            LOG(INFO) << "Sub-image " << i << " target size: " << new_unpad_w << "x" << new_unpad_h;
            
        } catch (const std::exception& e) {
            LOG(ERROR) << "Exception while processing sub-image " << i << ": " << e.what();
            continue;
        }
    }
    
    LOG(INFO) << "Unified target size: " << targetWidth << "x" << targetHeight;
    
    // 预分配输出容器
    m_outputResult.images.reserve(srcImages.size());
    m_outputResult.imageSizes.reserve(srcImages.size());
    m_outputResult.preprocessParams.reserve(srcImages.size());
    
    // 第二步：批量GPU预处理 - 优化版本
    if (!srcImages.empty()) {
        std::vector<float> ratios;
        std::vector<int> padTops, padLefts;
        std::vector<std::vector<float>> processedImages = processBatchImagesGPU(
            srcImages, targetWidth, targetHeight, ratios, padTops, padLefts
        );
        
        // 处理结果
        for (size_t i = 0; i < processedImages.size(); ++i) {
            if (!processedImages[i].empty()) {
                m_outputResult.images.push_back(std::move(processedImages[i]));
                m_outputResult.imageSizes.push_back(std::make_pair(targetWidth, targetHeight));
                
                // 保存预处理参数
                MultiImagePreprocessResult::PreprocessParams params;
                params.ratio = ratios[i];
                params.padTop = padTops[i];
                params.padLeft = padLefts[i];
                params.originalWidth = srcImages[i].cols;
                params.originalHeight = srcImages[i].rows;
                params.targetWidth = targetWidth;
                params.targetHeight = targetHeight;
                m_outputResult.preprocessParams.push_back(params);
                
                LOG(INFO) << "Successfully processed sub-image " << i 
                          << " -> " << targetWidth << "x" << targetHeight
                          << " (ratio: " << ratios[i] << ", pad: " << padTops[i] << "," << padLefts[i] << ")";
            } else {
                LOG(ERROR) << "Failed to process sub-image " << i;
            }
        }
    }
    
    LOG(INFO) << "ImagePreProcessGPU::execute status: success, processed " << m_outputResult.size() << " images";
    
    // 计算GPU预处理总耗时
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - m_start_time);
    LOG(INFO) << "GPU预处理总耗时: " << duration.count() / 1000.0 << " ms";
}

std::vector<float> ImagePreProcessGPU::processSingleImageGPU(const cv::Mat& srcImage, int& outWidth, int& outHeight) {
    // 简单的GPU预处理，不进行填充
    outWidth = srcImage.cols;
    outHeight = srcImage.rows;
    
    // 上传图像到GPU
    if (!uploadImageToGPU(srcImage, m_gpuInputBuffer)) {
        LOG(ERROR) << "Failed to upload image to GPU";
        return std::vector<float>();
    }
    
    // 执行GPU预处理
    // 这里可以添加更多的GPU处理步骤
    
    // 下载结果
    std::vector<float> result;
    if (!downloadImageFromGPU(m_gpuOutputBuffer, result, outWidth, outHeight)) {
        LOG(ERROR) << "Failed to download image from GPU";
        return std::vector<float>();
    }
    
    return result;
}

std::vector<float> ImagePreProcessGPU::processSingleImageGPUWithPadding(const cv::Mat& srcImage, int targetWidth, int targetHeight, 
                                                                       float& ratio, int& padTop, int& padLeft) {
    LOG(INFO) << "processSingleImageGPUWithPadding start: " << srcImage.cols << "x" << srcImage.rows 
              << " -> " << targetWidth << "x" << targetHeight;
    
    // 检查输入参数
    if (srcImage.empty()) {
        LOG(ERROR) << "Source image is empty";
        return std::vector<float>();
    }
    
    if (targetWidth <= 0 || targetHeight <= 0) {
        LOG(ERROR) << "Invalid target dimensions: " << targetWidth << "x" << targetHeight;
        return std::vector<float>();
    }
    
    // 检查GPU缓冲区状态
    if (!m_gpuInputBuffer || !m_gpuOutputBuffer || !m_gpuTempBuffer) {
        LOG(ERROR) << "GPU buffers not initialized: input=" << m_gpuInputBuffer 
                   << ", output=" << m_gpuOutputBuffer << ", temp=" << m_gpuTempBuffer;
        return std::vector<float>();
    }
    
    // 检查GPU内存大小是否足够
    size_t required_input_size = srcImage.total() * srcImage.elemSize();
    size_t required_output_size = targetWidth * targetHeight * 3 * sizeof(float);
    
    if (required_input_size > m_maxGPUBufferSize || required_output_size > m_maxGPUBufferSize) {
        LOG(ERROR) << "GPU buffer size insufficient: required=" << std::max(required_input_size, required_output_size) 
                   << ", available=" << m_maxGPUBufferSize;
        return std::vector<float>();
    }
    
    // 计算缩放比例 - 与CPU版本保持一致
    ratio = std::min(static_cast<float>(targetHeight) / srcImage.rows, static_cast<float>(targetWidth) / srcImage.cols);
    
    // 计算缩放后的尺寸 - 与CPU版本保持一致
    int newWidth = static_cast<int>(srcImage.cols * ratio);
    int newHeight = static_cast<int>(srcImage.rows * ratio);
    
    // 确保尺寸是stride的整数倍 - 与CPU版本保持一致
    newWidth = (newWidth / stride_) * stride_;
    newHeight = (newHeight / stride_) * stride_;
    
    // 计算填充 - 与CPU版本保持一致
    int dh = targetHeight - newHeight;
    int dw = targetWidth - newWidth;
    padTop = dh / 2;
    padLeft = dw / 2;
    
    LOG(INFO) << "GPU preprocessing: " << srcImage.cols << "x" << srcImage.rows 
              << " -> " << newWidth << "x" << newHeight << " -> " << targetWidth << "x" << targetHeight;
    
    // 上传原始图像到GPU（uchar3格式）
    if (!uploadImageToGPU(srcImage, m_gpuInputBuffer)) {
        LOG(ERROR) << "Failed to upload image to GPU";
        return std::vector<float>();
    }
    
    LOG(INFO) << "Image uploaded to GPU successfully";
    
    // 执行GPU resize（uchar3 -> uchar3）
    callResizeKernel(m_gpuInputBuffer, m_gpuTempBuffer, 
                    srcImage.cols, srcImage.rows, newWidth, newHeight, m_cudaStream);
    
    LOG(INFO) << "GPU resize completed";
    
    // 执行GPU BGR到RGB转换（uchar3 -> uchar3）- 与CPU版本保持一致
    callBgrToRgbKernel(m_gpuTempBuffer, m_gpuInputBuffer, newWidth, newHeight, m_cudaStream);
    
    LOG(INFO) << "GPU BGR to RGB conversion completed";
    
    // 执行GPU归一化（uchar3 -> float）
    callNormalizeKernel(m_gpuInputBuffer, m_gpuOutputBuffer, 
                       newWidth, newHeight, 1.0f / 255.0f, m_cudaStream);
    
    LOG(INFO) << "GPU normalization completed";
    
    // 执行GPU填充（float -> float）
    callPadImageKernel(m_gpuOutputBuffer, m_gpuTempBuffer, 
                      newWidth, newHeight, targetWidth, targetHeight,
                      padTop, padLeft, 114.0f / 255.0f, m_cudaStream);
    
    LOG(INFO) << "GPU padding completed";
    
    // 执行HWC到CHW转换（float -> float）
    callHWCtoCHWKernel(m_gpuTempBuffer, m_gpuOutputBuffer, 
                      targetWidth, targetHeight, m_cudaStream);
    
    LOG(INFO) << "GPU HWC to CHW conversion completed";
    
    // 下载结果
    std::vector<float> result;
    if (!downloadImageFromGPU(m_gpuOutputBuffer, result, targetWidth, targetHeight)) {
        LOG(ERROR) << "Failed to download image from GPU";
        return std::vector<float>();
    }
    
    LOG(INFO) << "GPU preprocessing completed, result size: " << result.size();
    return result;
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

bool ImagePreProcessGPU::downloadImageFromGPU(void* gpu_buffer, std::vector<float>& output, int width, int height) {
    if (!gpu_buffer) {
        LOG(ERROR) << "GPU buffer is null";
        return false;
    }
    
    if (width <= 0 || height <= 0) {
        LOG(ERROR) << "Invalid dimensions: " << width << "x" << height;
        return false;
    }
    
    size_t size = width * height * 3 * sizeof(float);
    output.resize(width * height * 3);
    
    // 检查CUDA错误
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        LOG(ERROR) << "CUDA error before download: " << cudaGetErrorString(error);
        return false;
    }
    
    cudaError_t status = cudaMemcpyAsync(output.data(), gpu_buffer, size,
                                        cudaMemcpyDeviceToHost, m_cudaStream);
    if (status != cudaSuccess) {
        LOG(ERROR) << "Failed to download image from GPU: " << cudaGetErrorString(status);
        return false;
    }
    
    // 同步流
    status = cudaStreamSynchronize(m_cudaStream);
    if (status != cudaSuccess) {
        LOG(ERROR) << "Failed to synchronize CUDA stream: " << cudaGetErrorString(status);
        return false;
    }
    
    // 验证下载的数据
    if (output.empty()) {
        LOG(ERROR) << "Downloaded data is empty";
        return false;
    }
    
    LOG(INFO) << "Successfully downloaded " << output.size() << " float values from GPU";
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

// 批量GPU预处理函数 - 优化版本
std::vector<std::vector<float>> ImagePreProcessGPU::processBatchImagesGPU(
    const std::vector<cv::Mat>& srcImages, int targetWidth, int targetHeight,
    std::vector<float>& ratios, std::vector<int>& padTops, std::vector<int>& padLefts) {
    
    LOG(INFO) << "processBatchImagesGPU start: batch size=" << srcImages.size() 
              << ", target size=" << targetWidth << "x" << targetHeight;
    
    std::vector<std::vector<float>> results;
    results.reserve(srcImages.size());
    ratios.reserve(srcImages.size());
    padTops.reserve(srcImages.size());
    padLefts.reserve(srcImages.size());
    
    if (srcImages.empty()) {
        return results;
    }
    
    // 检查GPU缓冲区状态
    if (!m_gpuInputBuffer || !m_gpuOutputBuffer || !m_gpuTempBuffer) {
        LOG(ERROR) << "GPU buffers not initialized";
        return results;
    }
    
    // 简化实现：逐个处理图像，但使用优化的GPU流程
    for (size_t i = 0; i < srcImages.size(); ++i) {
        const cv::Mat& srcImage = srcImages[i];
        
        // 计算缩放比例 - 与CPU版本保持一致
        float ratio = std::min(static_cast<float>(targetHeight) / srcImage.rows, 
                              static_cast<float>(targetWidth) / srcImage.cols);
        ratios.push_back(ratio);
        
        // 计算缩放后的尺寸 - 与CPU版本保持一致
        int newWidth = static_cast<int>(srcImage.cols * ratio);
        int newHeight = static_cast<int>(srcImage.rows * ratio);
        
        // 确保尺寸是stride的整数倍 - 与CPU版本保持一致
        newWidth = (newWidth / stride_) * stride_;
        newHeight = (newHeight / stride_) * stride_;
        
        // 计算填充 - 与CPU版本保持一致
        int dh = targetHeight - newHeight;
        int dw = targetWidth - newWidth;
        padTops.push_back(dh / 2);
        padLefts.push_back(dw / 2);
        
        LOG(INFO) << "Processing image " << i << ": " << srcImage.cols << "x" << srcImage.rows 
                  << " -> " << newWidth << "x" << newHeight << " -> " << targetWidth << "x" << targetHeight;
        
        // 使用现有的单图像处理函数，但优化内存使用
        std::vector<float> result = processSingleImageGPUWithPadding(
            srcImage, targetWidth, targetHeight, ratios[i], padTops[i], padLefts[i]
        );
        
        if (!result.empty()) {
            results.push_back(std::move(result));
            LOG(INFO) << "Successfully processed image " << i << ", result size: " << results.back().size();
        } else {
            LOG(ERROR) << "Failed to process image " << i;
            results.push_back(std::vector<float>());
        }
    }
    
    LOG(INFO) << "Batch GPU preprocessing completed, processed " << results.size() << " images";
    return results;
} 