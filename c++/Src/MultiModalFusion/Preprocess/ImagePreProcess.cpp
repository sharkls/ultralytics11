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
        std::cout << "rgbData.usBmpLength(): " << rgbData.usBmpLength() << std::endl;
        std::cout << "rgbData.usBmpWidth(): " << rgbData.usBmpWidth() << std::endl;
        std::cout << "irData.usBmpLength(): " << irData.usBmpLength() << std::endl;
        std::cout << "irData.usBmpWidth(): " << irData.usBmpWidth() << std::endl;
        cv::Mat rgbImg(rgbData.usBmpLength(), rgbData.usBmpWidth(), CV_8UC3, (void*)rgbData.vecImageBuf().data());
        cv::Mat irImg(irData.usBmpLength(), irData.usBmpWidth(), CV_8UC3, (void*)irData.vecImageBuf().data());

        // 2. 预处理并获取变换矩阵
        std::vector<float> rgb_output, ir_output;
        cv::Mat M_rgb = preprocess(rgbImg, new_unpad_w_rgb_, new_unpad_h_rgb_, dw_rgb_, dh_rgb_, r_rgb_, rgb_output);
        cv::Mat M_ir  = preprocess(irImg,  new_unpad_w_ir_,  new_unpad_h_ir_,  dw_ir_,  dh_ir_,  r_ir_,  ir_output);
        m_outputImage.push_back(rgb_output);
        m_outputImage.push_back(ir_output);
        
        // 3. 同步更新单应性矩阵
        std::vector<float> H_vec = m_inputImage.vecfHomography();
        if (H_vec.size() != 9) {
            LOG(ERROR) << "Homography matrix size error!";
            m_outputImage.push_back(std::vector<float>(9, 0));
        } else {
            // 假设H_vec为原始单应性矩阵的std::vector<float>，长度为9
            cv::Mat H_ir2rgb = cv::Mat(3, 3, CV_32F, H_vec.data()).clone();

            // 构建缩放和平移矩阵
            std::cout << "r_rgb_: " << r_rgb_ << "r_ir_: " << r_ir_ <<std::endl;
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

        // 离线调试代码
        if (status_) {
            save_bin(m_outputImage, "preprocess_multimodalfusion_output.bin"); // MultiModalFusion/Preprocess
            if (m_outputImage.size() >= 3) {
                // 1. 还原可见光和红外预处理后的Mat
                int img_area = max_model_size_ * max_model_size_;
                // --- 可见光 ---
                const std::vector<float>& rgb_output = m_outputImage[0];
                std::cout << "rgb_output.size(): " << rgb_output.size() << std::endl;
                std::vector<cv::Mat> rgb_channels(3);
                for (int c = 0; c < 3; ++c) {
                    rgb_channels[c] = cv::Mat(max_model_size_, max_model_size_, CV_32F, (void*)(rgb_output.data() + c * img_area)).clone();
                }
                cv::Mat rgb_preprocessed;
                cv::merge(rgb_channels, rgb_preprocessed);
                // --- 红外 ---
                const std::vector<float>& ir_output = m_outputImage[1];
                std::cout << "ir_output.size(): " << ir_output.size() << std::endl;
                std::vector<cv::Mat> ir_channels(3);
                for (int c = 0; c < 3; ++c) {
                    ir_channels[c] = cv::Mat(max_model_size_, max_model_size_, CV_32F, (void*)(ir_output.data() + c * img_area)).clone();
                }
                cv::Mat ir_preprocessed;
                cv::merge(ir_channels, ir_preprocessed);

                // 2. 归一化到[0,255]并转uint8
                cv::Mat rgb_u8, ir_u8;
                rgb_preprocessed = cv::min(rgb_preprocessed, 1.0f);
                rgb_preprocessed = cv::max(rgb_preprocessed, 0.0f);
                rgb_preprocessed.convertTo(rgb_u8, CV_8UC3, 255.0);
                cv::imwrite("rgb_preprocessed.jpg", rgb_u8);

                ir_preprocessed = cv::min(ir_preprocessed, 1.0f);
                ir_preprocessed = cv::max(ir_preprocessed, 0.0f);
                ir_preprocessed.convertTo(ir_u8, CV_8UC3, 255.0);
                cv::imwrite("ir_preprocessed.jpg", ir_u8);

                // 3. 红外图像投影到可见光坐标系
                const std::vector<float>& H_new_vec = m_outputImage[2];
                cv::Mat H_new = cv::Mat(3, 3, CV_32F, (void*)H_new_vec.data()).clone();
                cv::Mat ir_warped_u8;
                cv::warpPerspective(ir_u8, ir_warped_u8, H_new, rgb_u8.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, 0);

                // 4. 加权融合
                cv::Mat blend;
                cv::addWeighted(rgb_u8, 0.5, ir_warped_u8, 0.5, 0, blend);
                cv::imwrite("blend_ir_rgb_projected.jpg", blend);

                // 新增：原始图像域的红外映射与融合
                const CVideoSrcData& irData  = m_inputImage.vecVideoSrcData()[1];
                const CVideoSrcData& rgbData = m_inputImage.vecVideoSrcData()[0];
                cv::Mat irImg(irData.usBmpLength(), irData.usBmpWidth(), CV_8UC3, (void*)irData.vecImageBuf().data());
                cv::Mat rgbImg(rgbData.usBmpLength(), rgbData.usBmpWidth(), CV_8UC3, (void*)rgbData.vecImageBuf().data());

                const std::vector<float>& H_vec = m_inputImage.vecfHomography();
                cv::Mat H_ir2rgb = cv::Mat(3, 3, CV_32F, (void*)H_vec.data()).clone();

                cv::Mat ir2rgb_raw;
                cv::warpPerspective(irImg, ir2rgb_raw, H_ir2rgb, rgbImg.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, 0);

                cv::Mat blend_raw;
                cv::addWeighted(rgbImg, 0.5, ir2rgb_raw, 0.5, 0, blend_raw);
                cv::imwrite("blend_ir_rgb_raw.jpg", blend_raw);
            }
        }
    }
    catch (const std::exception& e) {
        LOG(ERROR) << "Preprocessing failed: " << e.what();
        return;
    }
}