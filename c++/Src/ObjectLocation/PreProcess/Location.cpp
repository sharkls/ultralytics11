/*******************************************************
 文件名：Location.cpp
 作者：sharkls
 描述：目标定位预处理模块实现
 版本：v1.0
 日期：2025-05-15
 *******************************************************/

#include "Location.h"

// 注册模块
REGISTER_MODULE("ObjectLocation", Location, Location)

Location::~Location()
{
}

bool Location::init(void* p_pAlgParam)
{
    LOG(INFO) << "Location::init status: start ";
    // 1. 从配置参数中读取预处理参数
    if (!p_pAlgParam) {
        return false;
    }
    // 2. 参数格式转换
    objectlocation::ObjectLocationConfig* taskConfig = static_cast<objectlocation::ObjectLocationConfig*>(p_pAlgParam);

    m_config = *taskConfig; 
    LOG(INFO) << "Location::init status: success ";
    return true;
}

void Location::setInput(void* input)
{
    if (!input) {
        LOG(ERROR) << "Input is null";
        return;
    }
    m_inputdata = *static_cast<CAlgResult*>(input);
}

void* Location::getOutput()
{
    return &m_outputdata;
}

void Location::execute()
{
    LOG(INFO) << "Location::execute status: start ";
    if (m_inputdata.vecVideoSrcData().empty()) {
        LOG(ERROR) << "Input image is empty";
        return;
    }

    try {
        // 1. 获取原始图像
        CVideoSrcData rgbData = m_inputdata.vecVideoSrcData()[0];
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
        m_outputdata.clear();
        m_outputdata.reserve(3 * input_h * input_w);
        for (int c = 0; c < 3; ++c) {
            m_outputdata.insert(m_outputdata.end(), (float*)channels[c].datastart, (float*)channels[c].dataend);
        }

        LOG(INFO) << "Location::execute status: success!";
        if (status_) {
            save_bin(m_outputdata, "objectlocation_preprocess.bin"); // ObjectLocation/Preprocess
        }
    }
    catch (const std::exception& e) {
        LOG(ERROR) << "Preprocessing failed: " << e.what();
        return;
    }
}