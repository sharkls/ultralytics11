/*******************************************************
 文件名：ImagePreProcess.cpp
 作者：sharkls
 描述：多模态融合算法图像预处理模块实现
 版本：v1.0
 日期：2025-05-14
 *******************************************************/

#include "ImagePreProcess.h"

// 注册模块
REGISTER_MODULE("MultiModalFusion", ImagePreProcess, ImagePreProcess)

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
    multimodalfusion::MultiModalFusionModelConfig* multiModalFusionConfig = static_cast<multimodalfusion::MultiModalFusionModelConfig*>(p_pAlgParam);
    stride_ = multiModalFusionConfig->stride(2);
    status_ = multiModalFusionConfig->run_status();

    max_model_size_ = multiModalFusionConfig->width(); // 假设width=height

    // 获取可见光参数
    src_w_rgb_ = multiModalFusionConfig->src_width_rgb();
    src_h_rgb_ = multiModalFusionConfig->src_height_rgb();
    r_rgb_ = multiModalFusionConfig->resize_ratio_rgb();
    if (r_rgb_ == 0.0f && src_w_rgb_ > 0 && src_h_rgb_ > 0 && max_model_size_ > 0) {
        r_rgb_ = std::min(static_cast<float>(max_model_size_) / src_h_rgb_, static_cast<float>(max_model_size_) / src_w_rgb_);
    }
    new_unpad_w_rgb_ = static_cast<int>(src_w_rgb_ * r_rgb_);
    new_unpad_h_rgb_ = static_cast<int>(src_h_rgb_ * r_rgb_);
    // 计算padding，保证最终尺寸为max_model_size_*max_model_size_
    dw_rgb_ = (max_model_size_ - new_unpad_w_rgb_) / 2;
    dh_rgb_ = (max_model_size_ - new_unpad_h_rgb_) / 2;
    // 边界对齐
    if (dw_rgb_ < 0) dw_rgb_ = 0;
    if (dh_rgb_ < 0) dh_rgb_ = 0;

    // 获取红外参数
    src_w_ir_ = multiModalFusionConfig->src_width_ir();
    src_h_ir_ = multiModalFusionConfig->src_height_ir();
    r_ir_ = multiModalFusionConfig->resize_ratio_ir();
    if (r_ir_ == 0.0f && src_w_ir_ > 0 && src_h_ir_ > 0 && max_model_size_ > 0) {
        r_ir_ = std::min(static_cast<float>(max_model_size_) / src_h_ir_, static_cast<float>(max_model_size_) / src_w_ir_);
    }
    new_unpad_w_ir_ = static_cast<int>(src_w_ir_ * r_ir_);
    new_unpad_h_ir_ = static_cast<int>(src_h_ir_ * r_ir_);
    dw_ir_ = (max_model_size_ - new_unpad_w_ir_) / 2;
    dh_ir_ = (max_model_size_ - new_unpad_h_ir_) / 2;
    if (dw_ir_ < 0) dw_ir_ = 0;
    if (dh_ir_ < 0) dh_ir_ = 0;

    // 更新multiModalFusionConfig参数（写回p_pAlgParam）
    multiModalFusionConfig->set_dw_rgb(dw_rgb_);
    multiModalFusionConfig->set_dh_rgb(dh_rgb_);
    multiModalFusionConfig->set_new_unpad_w_rgb(new_unpad_w_rgb_);
    multiModalFusionConfig->set_new_unpad_h_rgb(new_unpad_h_rgb_);
    multiModalFusionConfig->set_resize_ratio_rgb(r_rgb_);
    multiModalFusionConfig->set_dw_ir(dw_ir_);
    multiModalFusionConfig->set_dh_ir(dh_ir_);
    multiModalFusionConfig->set_new_unpad_w_ir(new_unpad_w_ir_);
    multiModalFusionConfig->set_new_unpad_h_ir(new_unpad_h_ir_);
    multiModalFusionConfig->set_resize_ratio_ir(r_ir_);

    m_config = *multiModalFusionConfig;

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

cv::Mat ImagePreProcess::preprocess(cv::Mat img, int new_pad_w, int new_pad_h, int dw, int dh, float r, std::vector<float>& output)
{
    // 2. resize
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(new_pad_w, new_pad_h), 0, 0, cv::INTER_LINEAR);
    
    // 3.letterbox填充，目标尺寸为max_model_size*max_model_size
    cv::Mat padded(max_model_size_, max_model_size_, CV_8UC3, cv::Scalar(114, 114, 114));
    resized.copyTo(padded(cv::Rect(dw, dh, resized.cols, resized.rows)));
    
    // 4. 归一化
    padded.convertTo(padded, CV_32FC3, 1.0/255.0);

    // 5. HWC to CHW
    std::vector<cv::Mat> channels(3);
    cv::split(padded, channels);
    output.clear();
    output.reserve(3 * max_model_size_ * max_model_size_);
    for (int c = 0; c < 3; ++c) {
        output.insert(output.end(), (float*)channels[c].datastart, (float*)channels[c].dataend);
    }
    // 返回仿射变换矩阵
    cv::Mat M = (cv::Mat_<float>(3,3) << r, 0, dw, 0, r, dh, 0, 0, 1);
    return M;
}

void ImagePreProcess::execute()
{
    LOG(INFO) << "ImagePreProcess::execute status: start ";
    if (m_inputImage.vecVideoSrcData().size() < 2) {
        LOG(ERROR) << "Input image is empty or not enough channels";
        return;
    }
    try {
        m_outputImage.clear();
        // 1. 获取原始图像
        const CVideoSrcData& rgbData = m_inputImage.vecVideoSrcData()[0];
        const CVideoSrcData& irData  = m_inputImage.vecVideoSrcData()[1];
        cv::Mat rgbImg(rgbData.usBmpLength(), rgbData.usBmpWidth(), CV_8UC3, (void*)rgbData.vecImageBuf().data());
        cv::Mat irImg(irData.usBmpLength(), irData.usBmpWidth(), CV_8UC3, (void*)irData.vecImageBuf().data());

        // 2. 预处理并获取变换矩阵
        std::vector<float> rgb_output, ir_output;
        cv::Mat M_rgb = preprocess(rgbImg, new_unpad_w_rgb_, new_unpad_h_rgb_, dw_rgb_, dh_rgb_, r_rgb_, rgb_output);
        cv::Mat M_ir  = preprocess(irImg,  new_unpad_w_ir_,  new_unpad_h_ir_,  dw_ir_,  dh_ir_,  r_ir_,  ir_output);
        m_outputImage.push_back(rgb_output);
        m_outputImage.push_back(ir_output);

        // 2.1 如果status_为true，进行图像矫正和融合
        if (status_) {
            processImageWithHomography(rgbImg);
        }
        
        // 3. 同步更新单应性矩阵
        std::vector<float> H_vec = m_inputImage.vecfHomography();
        if (H_vec.size() != 9) {
            LOG(ERROR) << "Homography matrix size error!";
            m_outputImage.push_back(std::vector<float>(9, 0));
        } else {
            // 假设H_vec为原始单应性矩阵的std::vector<float>，长度为9
            cv::Mat H_ir2rgb = cv::Mat(3, 3, CV_32F, H_vec.data()).clone();

            // 构建缩放和平移矩阵
            // std::cout << "r_rgb_: " << r_rgb_ << "r_ir_: " << r_ir_ <<std::endl;
            cv::Mat S_rgb = (cv::Mat_<float>(3,3) << r_rgb_, 0, 0, 0, r_rgb_, 0, 0, 0, 1);
            cv::Mat S_ir  = (cv::Mat_<float>(3,3) << r_ir_, 0, 0, 0, r_ir_, 0, 0, 0, 1);
            cv::Mat T_rgb = (cv::Mat_<float>(3,3) << 1, 0, dw_rgb_, 0, 1, dh_rgb_, 0, 0, 1);
            cv::Mat T_ir  = (cv::Mat_<float>(3,3) << 1, 0, dw_ir_, 0, 1, dh_ir_, 0, 0, 1);

            // 计算逆矩阵
            cv::Mat S_ir_inv = S_ir.inv();
            cv::Mat T_ir_inv = T_ir.inv();

            // 单应性矩阵更新
            cv::Mat H_new = T_rgb * S_rgb * H_ir2rgb * S_ir_inv * T_ir_inv;

            // 转为std::vector<float>存储
            std::vector<float> H_new_vec(9);
            memcpy(H_new_vec.data(), H_new.ptr<float>(), 9 * sizeof(float));

            m_outputImage.push_back(H_new_vec);
        }
        LOG(INFO) << "ImagePreProcess::execute status: success!";
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