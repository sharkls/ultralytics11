/*******************************************************
 文件名：ImagePreProcessGPU.cpp
 作者：sharkls
 描述：GPU加速的图像预处理模块实现
 版本：v1.0
 日期：2025-01-20
 *******************************************************/

#include "ImagePreProcessGPU.h"

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
    
    // 检查GPU缓冲区状态
    LOG(INFO) << "GPU buffer status check: input=" << m_gpuInputBuffer 
               << ", output=" << m_gpuOutputBuffer << ", temp=" << m_gpuTempBuffer;
    
    if (m_inputData.vecFrameResult().empty()) {
        LOG(ERROR) << "No input frame results";
        return;
    }
    
    LOG(INFO) << "Input frame results count: " << m_inputData.vecFrameResult().size();
    
    // 清空输出结果
    m_outputResult.images.clear();
    m_outputResult.imageSizes.clear();
    m_outputResult.preprocessParams.clear();
    
    LOG(INFO) << "Output results cleared";
    
    // 处理每个帧结果
    for (const auto& frameResult : m_inputData.vecFrameResult()) {
        LOG(INFO) << "Processing frame result, video source data count: " << frameResult.vecVideoSrcData().size();
        
        const auto& videoSrcData = frameResult.vecVideoSrcData();
        
        if (videoSrcData.empty()) {
            LOG(ERROR) << "No video source data available";
            return;
        }
        
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
                
                // 计算该图像的目标尺寸
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
        
        // 第二步：使用统一的目标尺寸处理所有图像
        for (size_t i = 0; i < srcImages.size(); ++i) {
            try {
                const cv::Mat& srcImage = srcImages[i];
                
                LOG(INFO) << "Processing video source data: " << srcImage.cols << "x" << srcImage.rows;
                
                // GPU预处理
                float ratio;
                int padTop, padLeft;
                std::vector<float> processedImage = processSingleImageGPUWithPadding(
                    srcImage, targetWidth, targetHeight, ratio, padTop, padLeft
                );
                
                if (!processedImage.empty()) {
                    m_outputResult.images.push_back(processedImage);
                    m_outputResult.imageSizes.push_back({targetWidth, targetHeight});
                    
                    // 添加预处理参数
                    MultiImagePreprocessResult::PreprocessParams params;
                    params.ratio = ratio;
                    params.padTop = padTop;
                    params.padLeft = padLeft;
                    params.originalWidth = srcImage.cols;
                    params.originalHeight = srcImage.rows;
                    params.targetWidth = targetWidth;
                    params.targetHeight = targetHeight;
                    m_outputResult.preprocessParams.push_back(params);
                    
                    LOG(INFO) << "Successfully processed sub-image " << i 
                              << " -> " << targetWidth << "x" << targetHeight
                              << " (ratio: " << ratio << ", pad: " << padTop << "," << padLeft << ")";
                } else {
                    LOG(ERROR) << "Failed to process sub-image " << i;
                }
                
            } catch (const std::exception& e) {
                LOG(ERROR) << "Exception while processing sub-image " << i << ": " << e.what();
                continue;
            } catch (...) {
                LOG(ERROR) << "Unknown exception while processing sub-image " << i;
                continue;
            }
        }
    }
    
    LOG(INFO) << "ImagePreProcessGPU::execute status: success, processed " 
              << m_outputResult.images.size() << " images";
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
    
    // 检查GPU缓冲区状态
    if (!m_gpuInputBuffer || !m_gpuOutputBuffer || !m_gpuTempBuffer) {
        LOG(ERROR) << "GPU buffers not initialized: input=" << m_gpuInputBuffer 
                   << ", output=" << m_gpuOutputBuffer << ", temp=" << m_gpuTempBuffer;
        return std::vector<float>();
    }
    
    // 计算缩放比例 - 与CPU版本保持一致
    ratio = std::min(static_cast<float>(targetHeight) / srcImage.rows, static_cast<float>(targetWidth) / srcImage.cols);
    
    // 计算缩放后的尺寸
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
                      padTop, padLeft, 0.0f, m_cudaStream);
    
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
    
    size_t size = width * height * 3 * sizeof(float);
    output.resize(width * height * 3);
    
    cudaError_t status = cudaMemcpyAsync(output.data(), gpu_buffer, size,
                                        cudaMemcpyDeviceToHost, m_cudaStream);
    if (status != cudaSuccess) {
        LOG(ERROR) << "Failed to download image from GPU: " << cudaGetErrorString(status);
        return false;
    }
    
    // 同步流
    cudaStreamSynchronize(m_cudaStream);
    
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