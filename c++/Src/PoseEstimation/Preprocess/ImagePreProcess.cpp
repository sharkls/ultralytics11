/*******************************************************
 文件名：ImagePreProcess.cpp
 作者：
 描述：图像预处理模块实现
 版本：v1.0
 日期：2024-03-21
 *******************************************************/

#include "ImagePreProcess.h"


// 注册模块
REGISTER_MODULE("PoseEstimation", ImagePreProcess, ImagePreProcess)

ImagePreProcess::ImagePreProcess()
{
}

ImagePreProcess::~ImagePreProcess()
{
}

bool ImagePreProcess::init(void* p_pAlgParam)
{
    // 1. 从配置参数中读取预处理参数
    if (!p_pAlgParam) {
        return false;
    }
    // 参数格式转换
    PoseConfig* poseConfig = static_cast<PoseConfig*>(p_pAlgParam);

    int src_w = poseConfig->mutable_yolo_model_config()->src_width();
    int src_h = poseConfig->mutable_yolo_model_config()->src_height();
    int model_w = poseConfig->mutable_yolo_model_config()->width();
    int model_h = poseConfig->mutable_yolo_model_config()->height();
    int stride = poseConfig->mutable_yolo_model_config()->stride();

    // 计算resize_ratio
    float r = poseConfig->yolo_model_config().resize_ratio();
    if (r == 0.0f && src_w > 0 && src_h > 0 && model_w > 0 && model_h > 0) {
        r = std::min(static_cast<float>(model_h) / src_h, static_cast<float>(model_w) / src_w);
        poseConfig->mutable_yolo_model_config()->set_resize_ratio(r);
    }

    // 计算缩放后未填充的尺寸
    int new_unpad_w = static_cast<int>(src_w * r);
    int new_unpad_h = static_cast<int>(src_h * r);
    // 保证是stride的整数倍
    if (stride > 0) {
        new_unpad_w = (new_unpad_w / stride) * stride;
        new_unpad_h = (new_unpad_h / stride) * stride;
    }
    poseConfig->mutable_yolo_model_config()->set_new_unpad_w(new_unpad_w);
    poseConfig->mutable_yolo_model_config()->set_new_unpad_h(new_unpad_h);

    // 计算padding
    int unpad_w = model_w - new_unpad_w;
    int unpad_h = model_h - new_unpad_h;
    poseConfig->mutable_yolo_model_config()->set_dw(unpad_w / 2);
    poseConfig->mutable_yolo_model_config()->set_dh(unpad_h / 2);

    m_poseConfig = *poseConfig;
    return true;
}

void ImagePreProcess::setInput(void* input)
{
    if (!input) {
        LOG(ERROR) << "Input is null";
        return;
    }
    m_inputImage = *static_cast<CMultiModalSrcData*>(input);
}

void* ImagePreProcess::getOutput()
{
    return &m_outputImage;
}

void* ImagePreProcess::execute()
{
    if (m_inputImage.vecVideoSrcData().empty()) {
        LOG(ERROR) << "Input image is empty";
        return nullptr;
    }

    try {
        CVideoSrcData rgbData = m_inputImage.vecVideoSrcData()[0];
        cv::Mat rgbImg(rgbData.usBmpLength(), rgbData.usBmpWidth(), CV_8UC3, rgbData.vecImageBuf().data());

        int input_h = m_poseConfig.yolo_model_config().height();
        int input_w = m_poseConfig.yolo_model_config().width();
        int new_unpad_w = m_poseConfig.yolo_model_config().new_unpad_w();
        int new_unpad_h = m_poseConfig.yolo_model_config().new_unpad_h();
        int dw = m_poseConfig.yolo_model_config().dw();
        int dh = m_poseConfig.yolo_model_config().dh();

        // resize
        cv::Mat resized;
        cv::resize(rgbImg, resized, cv::Size(new_unpad_w, new_unpad_h), 0, 0, cv::INTER_LINEAR);

        // letterbox填充
        cv::Mat padded(input_h, input_w, CV_8UC3, cv::Scalar(114, 114, 114));
        resized.copyTo(padded(cv::Rect(dw, dh, resized.cols, resized.rows)));

        // 归一化
        padded.convertTo(padded, CV_32FC3, 1.0/255.0);

        // HWC to CHW
        std::vector<cv::Mat> channels(3);
        cv::split(padded, channels);
        std::vector<float> chw;
        chw.reserve(3 * input_h * input_w);
        for (int c = 0; c < 3; ++c) {
            chw.insert(chw.end(), (float*)channels[c].datastart, (float*)channels[c].dataend);
        }

        m_outputImage = cv::Mat(3, input_h * input_w, CV_32F, chw.data()).clone();
        return &m_outputImage;
    }
    catch (const std::exception& e) {
        LOG(ERROR) << "Preprocessing failed: " << e.what();
        return nullptr;
    }
} 