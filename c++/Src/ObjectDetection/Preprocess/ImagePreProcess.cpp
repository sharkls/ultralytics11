/*******************************************************
 文件名：ImagePreProcess.cpp
 作者：sharkls
 描述：图像预处理模块实现(等比缩放+窄边填充至stride整数倍)
 版本：v1.0
 日期：2025-05-15
 *******************************************************/

#include "ImagePreProcess.h"
#include <filesystem>
#include <chrono>
#include <iomanip>
#include <sstream>

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
    
    // 初始化单应性矩阵和保存路径（如果status_为true）
    if (status_) {
        // 初始化单应性矩阵（红外图像到可见光图像的映射矩阵）
        homography_matrix_ = (cv::Mat_<double>(3, 3) << 
            1.79421472e+00, 1.09884934e-01, 6.43046067e+01,
            8.43536173e-02, 1.71536520e+00, -5.75071295e+01,
            1.49725508e-04, 1.79701920e-04, 1.00000000e+00);
        
        // 初始化红外相机内参矩阵
        ir_camera_matrix_ = (cv::Mat_<double>(3, 3) << 
            809.16015764, 0.0, 274.06714321,
            0.0, 805.23134742, 272.29427031,
            0.0, 0.0, 1.0);
        
        // 初始化红外相机畸变系数
        ir_dist_coeffs_ = (cv::Mat_<double>(5, 1) << 
            -2.89250776e-01, -1.22839875e+00, -1.79035429e-04, 
            1.23858116e-02, 5.11098655e+00);
        
        save_path_ = "./Output/vis/homography_corrected/";
        
        // 创建保存目录
        std::filesystem::create_directories(save_path_);
        
        LOG(INFO) << "Visible-IR homography correction enabled, save path: " << save_path_;
    }
    
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

        // 2.1 如果status_为true，进行图像矫正和融合
        if (status_ && m_inputImage.vecVideoSrcData().size() >= 2) {
            processImageWithHomography(rgbImg);
        }

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

void ImagePreProcess::processImageWithHomography(const cv::Mat& visible_img)
{
    try {
        // 1. 获取红外图像
        CVideoSrcData irData = m_inputImage.vecVideoSrcData()[1];
        cv::Mat irImg(irData.usBmpLength(), irData.usBmpWidth(), CV_8UC3, irData.vecImageBuf().data());
        
        // 2. 对红外图像进行畸变矫正
        cv::Mat undistorted_ir_img;
        cv::undistort(irImg, undistorted_ir_img, ir_camera_matrix_, ir_dist_coeffs_);
        
        LOG(INFO) << "IR image undistorted successfully";
        
        // 3. 使用单应性矩阵将畸变矫正后的红外图像投影到可见光图像坐标系
        cv::Mat warped_ir_img;
        cv::warpPerspective(undistorted_ir_img, warped_ir_img, homography_matrix_, 
                           cv::Size(visible_img.cols, visible_img.rows));
        
        LOG(INFO) << "IR image warped to visible image coordinate system";
        
        // 4. 将投影后的红外图像与原始可见光图像进行融合
        // 使用与Python脚本相同的融合方法：简单加权融合
        cv::Mat fused_img;
        double alpha = 0.5;  // 与Python脚本保持一致
        cv::addWeighted(visible_img, alpha, warped_ir_img, 1-alpha, 0, fused_img);
        
        // 5. 生成时间戳文件名
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()) % 1000;
        
        std::stringstream ss;
        ss << save_path_ << "visible_ir_fused_" 
           << std::put_time(std::localtime(&time_t), "%Y%m%d_%H%M%S")
           << "_" << std::setfill('0') << std::setw(3) << ms.count() << ".jpg";
        
        // 6. 保存融合后的图像
        bool save_success = cv::imwrite(ss.str(), fused_img);
        if (save_success) {
            LOG(INFO) << "Visible-IR fused image saved: " << ss.str();
        } else {
            LOG(ERROR) << "Failed to save visible-IR fused image: " << ss.str();
        }

        LOG(INFO) << "Fused image size: " << fused_img.cols << "x" << fused_img.rows;
        
    } catch (const std::exception& e) {
        LOG(ERROR) << "Visible-IR homography processing failed: " << e.what();
    }
}