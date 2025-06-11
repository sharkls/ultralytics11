#include "SGBM.h"
#include "log.h"

// 注册模块
REGISTER_MODULE("BinocularPositioning", SGBM, SGBM)

SGBM::~SGBM() {
    // 析构函数
}

bool SGBM::init(void* p_pAlgParam) 
{
    LOG(INFO) << "SGBM::init status: start ";
    if (!p_pAlgParam) {
        LOG(ERROR) << "Input parameter is null";
        return false;
    }

    // 从配置参数中读取SGBM参数
    binocularpositioning::ModelConfig* config = 
        static_cast<binocularpositioning::ModelConfig*>(p_pAlgParam);
    
    // 设置SGBM参数
    const auto& sgbmConfig = config->sgbm_config();
    if (!setParameters(
        sgbmConfig.min_disparity(),
        sgbmConfig.num_disparities(),
        sgbmConfig.block_size(),
        sgbmConfig.p1(),
        sgbmConfig.p2(),
        sgbmConfig.disp12_max_diff(),
        sgbmConfig.pre_filter_cap(),
        sgbmConfig.uniqueness_ratio(),
        sgbmConfig.speckle_window_size(),
        sgbmConfig.speckle_range(),
        sgbmConfig.mode())) {
        LOG(ERROR) << "Failed to set SGBM parameters";
        return false;
    }

    // 设置相机参数
    baseline_ = config->baseline();
    focusPixel_ = config->focus_pixel();
    focalLength_ = config->focal_length();
    status_ = config->run_status();

    LOG(INFO) << "SGBM::init status: success ";
    return true;
}

bool SGBM::setParameters(int minDisparity, int numDisparities, int blockSize,
                        int P1, int P2, int disp12MaxDiff, int preFilterCap,
                        int uniquenessRatio, int speckleWindowSize,
                        int speckleRange, int mode) {
    try {
        minDisparity_ = minDisparity;
        numDisparities_ = numDisparities;
        blockSize_ = blockSize;
        P1_ = P1;
        P2_ = P2;
        disp12MaxDiff_ = disp12MaxDiff;
        preFilterCap_ = preFilterCap;
        uniquenessRatio_ = uniquenessRatio;
        speckleWindowSize_ = speckleWindowSize;
        speckleRange_ = speckleRange;
        mode_ = mode;

        // 创建SGBM对象
        sgbm = cv::StereoSGBM::create(minDisparity_, numDisparities_, blockSize_,
                                     P1_, P2_, disp12MaxDiff_, preFilterCap_,
                                     uniquenessRatio_, speckleWindowSize_,
                                     speckleRange_, mode_);
        return true;
    }
    catch (const std::exception& e) {
        LOG(ERROR) << "Failed to set SGBM parameters: " << e.what();
        return false;
    }
}

void SGBM::setInput(void* input) {
    if (!input) {
        LOG(ERROR) << "Input is null";
        return;
    }
    try {
        m_inputImage = *static_cast<CMultiModalSrcData*>(input);
    }
    catch (const std::exception& e) {
        LOG(ERROR) << "Failed to set input: " << e.what();
    }
}

void* SGBM::getOutput() {
    return &m_outputResult;
}

void SGBM::execute() {
    LOG(INFO) << "SGBM::execute status: start ";
    status_ = false;

    // 获取输入图像
    const auto& videoData = m_inputImage.vecVideoSrcData();
    if (videoData.size() < 2) {
        LOG(ERROR) << "Invalid input data size";
        return;
    }

    // 获取左右图像
    cv::Mat leftImg = cv::Mat(videoData[0].usBmpLength(), videoData[0].usBmpWidth(), CV_8UC3, 
                             const_cast<uint8_t*>(videoData[0].vecImageBuf().data()));
    cv::Mat rightImg = cv::Mat(videoData[1].usBmpLength(), videoData[1].usBmpWidth(), CV_8UC3, 
                              const_cast<uint8_t*>(videoData[1].vecImageBuf().data()));

    LOG(INFO) << "Input image size - Left: [" << leftImg.cols << " x " << leftImg.rows 
              << "], Right: [" << rightImg.cols << " x " << rightImg.rows << "]";

    // 获取相机参数
    float baseline = baseline_;
    float focusPixel = focusPixel_;
    LOG(INFO) << "Camera parameters - Baseline: " << baseline << "mm, FocusPixel: " << focusPixel;

    // 转换为灰度图
    cv::Mat leftGray, rightGray;
    cv::cvtColor(leftImg, leftGray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(rightImg, rightGray, cv::COLOR_BGR2GRAY);

    // 计算视差图
    cv::Mat disparity;
    sgbm->compute(leftGray, rightGray, disparity);

    // 可视化并保存视差图（由status_控制）
    if (!status_) {
        std::string disp_path = "./disparity_vis.jpg"; // 可根据需要自定义路径
        saveDisparityVis(disparity, disp_path, true); // true为伪彩色
    }

    // 转换视差值（SGBM输出需要除以16）
    cv::Mat disparityFloat;
    disparity.convertTo(disparityFloat, CV_32F, 1.0/16.0);

    LOG(INFO) << "Disparity map size: [" << disparity.cols << " x " << disparity.rows 
              << "], type: " << disparity.type();

    // 统计有效视差值
    int validCount = 0;
    float minDisp = FLT_MAX, maxDisp = -FLT_MAX;
    for(int i = 0; i < disparityFloat.rows; i++) {
        for(int j = 0; j < disparityFloat.cols; j++) {
            float disp = disparityFloat.at<float>(i, j);
            if(disp > 0) {
                validCount++;
                minDisp = std::min(minDisp, disp);
                maxDisp = std::max(maxDisp, disp);
            }
        }
    }
    float validRatio = (float)validCount / (disparityFloat.rows * disparityFloat.cols) * 100;
    LOG(INFO) << "Valid disparities: " << validCount << " (" << validRatio 
              << "%), Min disparity: " << minDisp << ", Max disparity: " << maxDisp;

    // 计算深度图
    cv::Mat depthMap = cv::Mat::zeros(disparityFloat.size(), CV_32F);
    for(int i = 0; i < disparityFloat.rows; i++) {
        for(int j = 0; j < disparityFloat.cols; j++) {
            float disp = disparityFloat.at<float>(i, j);
            if(disp > 0) {
                depthMap.at<float>(i, j) = (baseline * focusPixel) / disp;
            }
        }
    }

    LOG(INFO) << "Depth map size: [" << depthMap.cols << " x " << depthMap.rows 
              << "], type: " << depthMap.type();

    // 统计有效深度值
    validCount = 0;
    float minDepth = FLT_MAX, maxDepth = -FLT_MAX;
    for(int i = 0; i < depthMap.rows; i++) {
        for(int j = 0; j < depthMap.cols; j++) {
            float depth = depthMap.at<float>(i, j);
            if(depth > 0) {
                validCount++;
                minDepth = std::min(minDepth, depth);
                maxDepth = std::max(maxDepth, depth);
            }
        }
    }
    validRatio = (float)validCount / (depthMap.rows * depthMap.cols) * 100;
    LOG(INFO) << "Valid depth pixels: " << validCount << " (" << validRatio 
              << "%), Min depth: " << minDepth << ", Max depth: " << maxDepth;

    // 将深度图转换为整数类型
    cv::Mat depthMapInt;
    depthMap.convertTo(depthMapInt, CV_32S);

    // 更新输出结果
    m_outputResult = CAlgResult();
    auto& frameResult = m_outputResult.vecFrameResult().emplace_back();
    frameResult.tCameraSupplement().usWidth(depthMapInt.cols);
    frameResult.tCameraSupplement().usHeight(depthMapInt.rows);
    std::vector<int32_t> depthData(depthMapInt.data, depthMapInt.data + depthMapInt.total());
    frameResult.tCameraSupplement().vecDistanceInfo(depthData);

    LOG(INFO) << "Output depth data size: " << depthData.size();

    status_ = true;
    LOG(INFO) << "SGBM::execute status: success ";
}

cv::Mat SGBM::computeDisparity(const cv::Mat& leftImg, const cv::Mat& rightImg) {
    if (leftImg.empty() || rightImg.empty()) {
        return cv::Mat();
    }

    // 确保图像是灰度图
    cv::Mat leftGray, rightGray;
    if (leftImg.channels() > 1) {
        cv::cvtColor(leftImg, leftGray, cv::COLOR_BGR2GRAY);
        cv::cvtColor(rightImg, rightGray, cv::COLOR_BGR2GRAY);
    } else {
        leftGray = leftImg.clone();
        rightGray = rightImg.clone();
    }

    // 计算视差图
    sgbm->compute(leftGray, rightGray, disparity);
    
    // 将视差图转换为可视化格式
    cv::Mat disparityVis;
    disparity.convertTo(disparityVis, CV_8U, 255.0 / (16 * numDisparities_));
    
    return disparityVis;
}

cv::Mat SGBM::computeDepth(const cv::Mat& disparity) {
    if (disparity.empty()) {
        return cv::Mat();
    }

    // 创建深度图
    depth = cv::Mat::zeros(disparity.size(), CV_32F);
    
    // 计算深度图
    const float maxDepth = 10000.0f;  // 最大深度限制（毫米）
    const float minDepth = 100.0f;    // 最小深度限制（毫米）
    
    for (int i = 0; i < disparity.rows; i++) {
        for (int j = 0; j < disparity.cols; j++) {
            float disp = disparity.at<float>(i, j);
            if (disp > 0) {
                float d = (baseline_ * focusPixel_) / disp;
                // 限制深度值在有效范围内
                if (d >= minDepth && d <= maxDepth) {
                    depth.at<float>(i, j) = d;
                }
            }
        }
    }

    return depth;
}

void SGBM::saveDisparityVis(const cv::Mat& disparity, const std::string& path, bool useColor) {
    if (disparity.empty()) {
        LOG(ERROR) << "Disparity map is empty, cannot save visualization.";
        return;
    }
    cv::Mat dispVis;
    // 归一化到0~255
    disparity.convertTo(dispVis, CV_8U, 255.0 / (16.0 * numDisparities_));
    if (useColor) {
        cv::applyColorMap(dispVis, dispVis, cv::COLORMAP_JET);
    }
    cv::imwrite(path, dispVis);
    LOG(INFO) << "Disparity visualization saved to: " << path;
} 