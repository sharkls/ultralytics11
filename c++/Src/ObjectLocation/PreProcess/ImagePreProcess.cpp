/*******************************************************
 文件名：ImagePreProcess.cpp
 作者：
 描述：图像预处理模块实现
 版本：v1.0
 日期：2024-03-21
 *******************************************************/

#include "ImagePreProcess.h"
#include "../../Factory/ModuleFactory.h"
#include <iostream>

// 注册模块
REGISTER_MODULE(ImagePreProcess, ImagePreProcess)

ImagePreProcess::ImagePreProcess()
{
}

ImagePreProcess::~ImagePreProcess()
{
}

bool ImagePreProcess::init(CSelfAlgParam* p_pAlgParam)
{
    // 这里可以从配置参数中读取预处理参数
    if (p_pAlgParam) {
        // 读取配置参数
        // TODO: 实现参数读取逻辑
    }
    return true;
}

void ImagePreProcess::setInput(void* input)
{
    if (!input) {
        std::cerr << "Input is null" << std::endl;
        return;
    }
    m_inputImage = *static_cast<cv::Mat*>(input);
}

void* ImagePreProcess::getOutput()
{
    return &m_outputImage;
}

void* ImagePreProcess::execute()
{
    if (m_inputImage.empty()) {
        std::cerr << "Input image is empty" << std::endl;
        return nullptr;
    }

    try {
        // 1. 调整图像大小
        cv::Mat resized;
        cv::resize(m_inputImage, resized, cv::Size(m_params.targetWidth, m_params.targetHeight));

        // 2. 转换为浮点型
        cv::Mat float_img;
        resized.convertTo(float_img, CV_32FC3, 1.0/255.0);

        // 3. 标准化
        std::vector<cv::Mat> channels;
        cv::split(float_img, channels);
        
        for (int i = 0; i < 3; ++i) {
            channels[i] = (channels[i] - m_params.mean[i]) / m_params.std[i];
        }

        cv::merge(channels, m_outputImage);

        return &m_outputImage;
    }
    catch (const std::exception& e) {
        std::cerr << "Preprocessing failed: " << e.what() << std::endl;
        return nullptr;
    }
} 