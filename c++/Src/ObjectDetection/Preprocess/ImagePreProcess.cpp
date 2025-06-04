/*******************************************************
 文件名：ImagePreProcess.cpp
 作者：sharkls
 描述：图像预处理模块实现(等比缩放+窄边填充至stride整数倍)
 版本：v1.0
 日期：2025-05-15
 *******************************************************/

#include "ImagePreProcess.h"

// 注册模块
REGISTER_MODULE("ObjectDetection", ImagePreProcess, ImagePreProcess)

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
    objectdetection::YOLOModelConfig* yoloConfig = static_cast<objectdetection::YOLOModelConfig*>(p_pAlgParam);
    src_w_ = yoloConfig->src_width();
    src_h_ = yoloConfig->src_height();
    max_model_size_ = yoloConfig->width();
    stride_ = yoloConfig->stride(2);
    status_ = yoloConfig->run_status();

    // 3. 计算resize_ratio
    float r = yoloConfig->resize_ratio();
    if (r == 0.0f && src_w_ > 0 && src_h_ > 0 && max_model_size_ > 0) {
        r = std::min(static_cast<float>(max_model_size_) / src_h_, static_cast<float>(max_model_size_) / src_w_);
    }

    // 4. 计算缩放后未填充的尺寸
    new_unpad_w_ = static_cast<int>(src_w_ * r);
    new_unpad_h_ = static_cast<int>(src_h_ * r);

    // 5. 保证是stride的整数倍
    if (stride_ > 0) {
        new_unpad_w_ = (new_unpad_w_ / stride_) * stride_;
        new_unpad_h_ = (new_unpad_h_ / stride_) * stride_;
    } else {
        LOG(ERROR) << "stride value error!";
        return false;
    }
 
    // 6. 计算padding
    dw_ = 0;
    dh_ = 0;

    // 7. 填充结果
    yoloConfig->set_dw(0);
    yoloConfig->set_dh(0);
    yoloConfig->set_new_unpad_w(new_unpad_w_);   // 未填充前的宽度
    yoloConfig->set_new_unpad_h(new_unpad_h_);   // 未填充前的高度
    yoloConfig->set_resize_ratio(r);

    // 打印所有信息
    // LOG(INFO) << "src_w_ = " << src_w_ << ", src_h_ = " << src_h_ << ", max_model_size_ = " << max_model_size_ << ", new_unpad_w_ = " << new_unpad_w_ << ", new_unpad_h_ = " << new_unpad_h_ << ", stride_ = " << stride_;
    // LOG(INFO) << "dw_ = " << dw_ << ", dh_ = " << dh_;

    m_poseConfig = *yoloConfig; 
    LOG(INFO) << "ImagePreProcess::init status: success ";
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

void ImagePreProcess::execute()
{
    // LOG(INFO) << "ImagePreProcess::execute status: start ";
    if (m_inputImage.vecVideoSrcData().empty()) {
        LOG(ERROR) << "Input image is empty";
        return;
    }

    try {
        // 1. 获取原始图像
        CVideoSrcData rgbData = m_inputImage.vecVideoSrcData()[0];
        cv::Mat rgbImg(rgbData.usBmpLength(), rgbData.usBmpWidth(), CV_8UC3, rgbData.vecImageBuf().data());

        // 2. resize
        cv::Mat resized;
        cv::resize(rgbImg, resized, cv::Size(new_unpad_w_, new_unpad_h_), 0, 0, cv::INTER_LINEAR);

        // 3. letterbox填充，目标尺寸为new_unpad_w_+2*dw_, new_unpad_h_+2*dh_
        int input_w = new_unpad_w_;
        int input_h = new_unpad_h_;
        cv::Mat padded(input_h, input_w, CV_8UC3, cv::Scalar(114, 114, 114));
        resized.copyTo(padded(cv::Rect(0, 0, resized.cols, resized.rows)));

        // // 3.2 填充至640*640
        // int input_w = new_unpad_w_ + 2 * dw_;
        // int input_h = new_unpad_h_ + 2 * dh_;
        // cv::Mat padded(input_h, input_w, CV_8UC3, cv::Scalar(114, 114, 114));
        // resized.copyTo(padded(cv::Rect(dw_, dh_, resized.cols, resized.rows)));

        // 4. 归一化
        padded.convertTo(padded, CV_32FC3, 1.0/255.0);

        // 5. HWC to CHW
        std::vector<cv::Mat> channels(3);
        cv::split(padded, channels);
        m_outputImage.clear();
        m_outputImage.reserve(3 * input_h * input_w);
        for (int c = 0; c < 3; ++c) {
            m_outputImage.insert(m_outputImage.end(), (float*)channels[c].datastart, (float*)channels[c].dataend);
        }

        // LOG(INFO) << "ImagePreProcess::execute status: success!";
        // if (status_) {
        //     save_bin(m_outputImage, "./Save_Data/pose/result/preprocess_yolov11pose.bin"); // Yolov11Pose/Preprocess
        // }
    }
    catch (const std::exception& e) {
        LOG(ERROR) << "Preprocessing failed: " << e.what();
        return;
    }
}