#ifndef CRL_H
#define CRL_H

#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <string>

class CRL {
public:
    CRL();
    ~CRL();

    // 初始化模型
    bool init(const std::string& modelPath);

    // 计算视差图
    cv::Mat computeDisparity(const cv::Mat& leftImg, const cv::Mat& rightImg);

    // 计算深度图
    cv::Mat computeDepth(const cv::Mat& disparity, float baseline, float focalLength);

private:
    // 网络结构定义
    struct CRLNet : torch::nn::Module {
        CRLNet();
        
        // 特征提取网络
        torch::nn::Sequential feature_extraction{nullptr};
        
        // 级联残差模块
        struct CascadeResidual : torch::nn::Module {
            CascadeResidual(int in_channels, int out_channels);
            torch::Tensor forward(torch::Tensor x);
            torch::nn::Conv2d conv1{nullptr}, conv2{nullptr};
            torch::nn::BatchNorm2d bn1{nullptr}, bn2{nullptr};
        };
        
        // 视差回归模块
        struct DisparityRegression : torch::nn::Module {
            DisparityRegression(int max_disp);
            torch::Tensor forward(torch::Tensor x);
            int max_disp;
        };

        std::vector<std::shared_ptr<CascadeResidual>> cascade_modules;
        std::shared_ptr<DisparityRegression> disparity_regression;
        
        torch::Tensor forward(torch::Tensor left, torch::Tensor right);
    };

    // 预处理图像
    torch::Tensor preprocessImage(const cv::Mat& img);
    
    // 后处理视差图
    cv::Mat postprocessDisparity(const torch::Tensor& disp);

    std::shared_ptr<CRLNet> model;
    bool isInitialized;
    int maxDisparity;
    float baseline;
    float focalLength;
};

#endif // CRL_H 