#include "SGBM.h"

SGBM::SGBM() {
    // 默认构造函数
}

SGBM::~SGBM() {
    // 析构函数
}

void SGBM::init(int minDisparity, int numDisparities, int blockSize,
                int P1, int P2, int disp12MaxDiff, int preFilterCap,
                int uniquenessRatio, int speckleWindowSize, int speckleRange,
                int mode) {
    sgbm = cv::StereoSGBM::create(minDisparity, numDisparities, blockSize,
                                 P1, P2, disp12MaxDiff, preFilterCap,
                                 uniquenessRatio, speckleWindowSize,
                                 speckleRange, mode);
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
    disparity.convertTo(disparityVis, CV_8U, 255.0 / (16 * numDisparities));
    
    return disparityVis;
}

cv::Mat SGBM::computeDepth(const cv::Mat& disparity, float baseline, float focalLength) {
    if (disparity.empty()) {
        return cv::Mat();
    }

    // 创建深度图
    depth = cv::Mat::zeros(disparity.size(), CV_32F);
    
    // 计算深度图
    for (int i = 0; i < disparity.rows; i++) {
        for (int j = 0; j < disparity.cols; j++) {
            float disp = disparity.at<float>(i, j);
            if (disp > 0) {
                depth.at<float>(i, j) = (baseline * focalLength) / disp;
            }
        }
    }

    // 归一化深度图用于可视化
    cv::Mat depthVis;
    cv::normalize(depth, depthVis, 0, 255, cv::NORM_MINMAX);
    depthVis.convertTo(depthVis, CV_8U);

    return depthVis;
} 