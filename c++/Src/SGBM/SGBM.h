#ifndef SGBM_H
#define SGBM_H

#include <opencv2/opencv.hpp>
#include <string>

class SGBM {
public:
    SGBM();
    ~SGBM();

    // 初始化SGBM参数
    void init(int minDisparity = 0,
              int numDisparities = 128,
              int blockSize = 5,
              int P1 = 8 * 3 * 5 * 5,
              int P2 = 32 * 3 * 5 * 5,
              int disp12MaxDiff = 1,
              int preFilterCap = 63,
              int uniquenessRatio = 15,
              int speckleWindowSize = 100,
              int speckleRange = 2,
              int mode = cv::StereoSGBM::MODE_SGBM);

    // 计算视差图
    cv::Mat computeDisparity(const cv::Mat& leftImg, const cv::Mat& rightImg);

    // 计算深度图
    cv::Mat computeDepth(const cv::Mat& disparity, float baseline, float focalLength);

private:
    cv::Ptr<cv::StereoSGBM> sgbm;
    cv::Mat disparity;
    cv::Mat depth;
};

#endif // SGBM_H 