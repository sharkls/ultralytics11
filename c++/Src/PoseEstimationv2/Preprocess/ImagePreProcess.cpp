/*******************************************************
 文件名：ImagePreProcess.cpp
 作者：sharkls
 描述：图像预处理模块实现(处理多个子图，等比缩放至指定尺寸)
 版本：v1.0
 日期：2025-05-15
 *******************************************************/

#include "ImagePreProcess.h"

// 注册模块
REGISTER_MODULE("PoseEstimation", ImagePreProcess, ImagePreProcess)

ImagePreProcess::~ImagePreProcess()
{
}

bool ImagePreProcess::init(void* p_pAlgParam)
{
    LOG(INFO) << "ImagePreProcess::init status: start ";
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
    LOG(INFO) << "ImagePreProcess::init status: success, max_model_size_: " << max_model_size_ << ", stride_: " << stride_;
    return true;
}

void ImagePreProcess::setInput(void* input)
{
    if (!input) {
        LOG(ERROR) << "Input is null";
        return;
    }
    m_inputData = *static_cast<CAlgResult*>(input);
}

void* ImagePreProcess::getOutput()
{
    return &m_outputResult;
}

std::vector<float> ImagePreProcess::processSingleImage(const cv::Mat& srcImage, int& outWidth, int& outHeight)
{
    std::vector<float> outputImage;
    
    try {
        int src_w = srcImage.cols;
        int src_h = srcImage.rows;
        
        // 1. 计算等比缩放比例
        float r = std::min(static_cast<float>(max_model_size_) / src_h, static_cast<float>(max_model_size_) / src_w);
        
        // 2. 计算缩放后的尺寸
        int new_unpad_w = static_cast<int>(src_w * r);
        int new_unpad_h = static_cast<int>(src_h * r);
        
        // 3. 确保是stride的整数倍
        new_unpad_w = (new_unpad_w / stride_) * stride_;
        new_unpad_h = (new_unpad_h / stride_) * stride_;
        
        // 4. 设置输出尺寸
        outWidth = new_unpad_w;
        outHeight = new_unpad_h;
        
        // 5. resize图像
        cv::Mat resized;
        cv::resize(srcImage, resized, cv::Size(new_unpad_w, new_unpad_h), 0, 0, cv::INTER_LINEAR);
        
        // 6. 颜色空间转换：BGR转RGB（与Python脚本保持一致）
        cv::Mat rgbImage;
        cv::cvtColor(resized, rgbImage, cv::COLOR_BGR2RGB);
        
        // 7. 归一化并转换为float
        cv::Mat floatImage;
        rgbImage.convertTo(floatImage, CV_32FC3, 1.0/255.0);
        
        // 8. HWC to CHW - 优化内存分配
        outputImage.clear();
        outputImage.reserve(3 * new_unpad_h * new_unpad_w);
        
        // 直接访问数据，避免split操作
        for (int c = 0; c < 3; ++c) {
            for (int h = 0; h < new_unpad_h; ++h) {
                for (int w = 0; w < new_unpad_w; ++w) {
                    outputImage.push_back(floatImage.at<cv::Vec3f>(h, w)[c]);
                }
            }
        }
        
        LOG(INFO) << "Processed image: " << src_w << "x" << src_h << " -> " << new_unpad_w << "x" << new_unpad_h 
                  << " (ratio: " << r << ")";
    }
    catch (const std::exception& e) {
        LOG(ERROR) << "Single image preprocessing failed: " << e.what();
        outputImage.clear();
    }
    catch (...) {
        LOG(ERROR) << "Unknown exception in single image preprocessing";
        outputImage.clear();
    }
    
    return outputImage;
}

std::vector<float> ImagePreProcess::processSingleImageWithPadding(const cv::Mat& srcImage, int targetWidth, int targetHeight, 
                                                                  float& ratio, int& padTop, int& padLeft)
{
    std::vector<float> outputImage;
    
    try {
        int src_w = srcImage.cols;
        int src_h = srcImage.rows;
        
        // 1. 计算等比缩放比例
        ratio = std::min(static_cast<float>(targetHeight) / src_h, static_cast<float>(targetWidth) / src_w);
        
        // 2. 计算缩放后的尺寸
        int new_unpad_w = static_cast<int>(src_w * ratio);
        int new_unpad_h = static_cast<int>(src_h * ratio);
        
        // 3. 计算填充
        int dh = targetHeight - new_unpad_h;
        int dw = targetWidth - new_unpad_w;
        padTop = dh / 2;
        padLeft = dw / 2;
        int padBottom = dh - padTop;
        int padRight = dw - padLeft;
        
        // 4. resize图像
        cv::Mat resized;
        cv::resize(srcImage, resized, cv::Size(new_unpad_w, new_unpad_h), 0, 0, cv::INTER_LINEAR);
        
        // 5. 颜色空间转换：BGR转RGB（与Python脚本保持一致）
        cv::Mat rgbImage;
        cv::cvtColor(resized, rgbImage, cv::COLOR_BGR2RGB);
        
        // 6. 添加填充
        cv::Mat paddedImage;
        cv::copyMakeBorder(rgbImage, paddedImage, padTop, padBottom, padLeft, padRight, 
                          cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
        
        // 7. 归一化并转换为float
        cv::Mat floatImage;
        paddedImage.convertTo(floatImage, CV_32FC3, 1.0/255.0);
        
        // 8. HWC to CHW - 优化内存分配
        outputImage.clear();
        outputImage.reserve(3 * targetHeight * targetWidth);
        
        // 直接访问数据，避免split操作
        for (int c = 0; c < 3; ++c) {
            for (int h = 0; h < targetHeight; ++h) {
                for (int w = 0; w < targetWidth; ++w) {
                    outputImage.push_back(floatImage.at<cv::Vec3f>(h, w)[c]);
                }
            }
        }
        
        LOG(INFO) << "Processed image with padding: " << src_w << "x" << src_h 
                  << " -> " << new_unpad_w << "x" << new_unpad_h 
                  << " -> " << targetWidth << "x" << targetHeight
                  << " (ratio: " << ratio << ", pad: " << padTop << "," << padLeft << ")";
    }
    catch (const std::exception& e) {
        LOG(ERROR) << "Single image preprocessing with padding failed: " << e.what();
        outputImage.clear();
    }
    catch (...) {
        LOG(ERROR) << "Unknown exception in single image preprocessing with padding";
        outputImage.clear();
    }
    
    return outputImage;
}

void ImagePreProcess::execute()
{
    LOG(INFO) << "ImagePreProcess::execute status: start ";
    
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
    
    LOG(INFO) << "Processing " << videoSrcData.size() << " sub-images";
    
    // 第一步：计算所有图像的目标尺寸，取最大值确保一致性（与Python脚本保持一致）
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
    m_outputResult.preprocessParams.reserve(srcImages.size());  // 新增：预分配预处理参数容器
    
    // 第二步：使用统一的目标尺寸处理所有图像
    for (size_t i = 0; i < srcImages.size(); ++i) {
        try {
            const cv::Mat& srcImage = srcImages[i];
            
            // 使用新的批量预处理函数
            float ratio;
            int padTop, padLeft;
            std::vector<float> processedImage = processSingleImageWithPadding(srcImage, targetWidth, targetHeight, 
                                                                             ratio, padTop, padLeft);
            
            if (!processedImage.empty()) {
                m_outputResult.images.push_back(std::move(processedImage));
                m_outputResult.imageSizes.push_back(std::make_pair(targetWidth, targetHeight));
                
                // 新增：保存预处理参数
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
    
    LOG(INFO) << "ImagePreProcess::execute status: success! Processed " << m_outputResult.size() << " images";
    
    // 如果启用了离线保存，保存预处理结果
    if (status_ && !m_outputResult.empty()) {
        LOG(INFO) << "Offline mode enabled, preprocessing results ready for saving";
    }
}