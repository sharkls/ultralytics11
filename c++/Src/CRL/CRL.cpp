#include "CRL.h"

// CRLNet构造函数
CRL::CRLNet::CRLNet() {
    // 特征提取网络
    feature_extraction = torch::nn::Sequential(
        torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 32, 3).padding(1)),
        torch::nn::BatchNorm2d(32),
        torch::nn::ReLU(),
        torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 32, 3).padding(1)),
        torch::nn::BatchNorm2d(32),
        torch::nn::ReLU(),
        torch::nn::MaxPool2d(2)
    );

    // 级联残差模块
    for (int i = 0; i < 3; i++) {
        cascade_modules.push_back(
            std::make_shared<CascadeResidual>(32, 32)
        );
    }

    // 视差回归模块
    disparity_regression = std::make_shared<DisparityRegression>(192);
}

// CascadeResidual构造函数
CRL::CRLNet::CascadeResidual::CascadeResidual(int in_channels, int out_channels) {
    conv1 = torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, 3).padding(1));
    bn1 = torch::nn::BatchNorm2d(out_channels);
    conv2 = torch::nn::Conv2d(torch::nn::Conv2dOptions(out_channels, out_channels, 3).padding(1));
    bn2 = torch::nn::BatchNorm2d(out_channels);
}

// CascadeResidual前向传播
torch::Tensor CRL::CRLNet::CascadeResidual::forward(torch::Tensor x) {
    auto residual = x;
    x = torch::relu(bn1(conv1(x)));
    x = bn2(conv2(x));
    x += residual;
    return torch::relu(x);
}

// DisparityRegression构造函数
CRL::CRLNet::DisparityRegression::DisparityRegression(int max_disp) : max_disp(max_disp) {}

// DisparityRegression前向传播
torch::Tensor CRL::CRLNet::DisparityRegression::forward(torch::Tensor x) {
    auto disp = torch::zeros({x.size(0), 1, x.size(2), x.size(3)});
    for (int i = 0; i < max_disp; i++) {
        disp += i * x.slice(1, i, i + 1);
    }
    return disp;
}

// CRLNet前向传播
torch::Tensor CRL::CRLNet::forward(torch::Tensor left, torch::Tensor right) {
    // 特征提取
    auto left_feat = feature_extraction->forward(left);
    auto right_feat = feature_extraction->forward(right);

    // 构建代价体
    auto cost_volume = torch::zeros({left.size(0), max_disp, left_feat.size(2), left_feat.size(3)});
    for (int i = 0; i < max_disp; i++) {
        auto left_slice = left_feat.slice(3, i, left_feat.size(3));
        auto right_slice = right_feat.slice(3, 0, right_feat.size(3) - i);
        cost_volume.slice(1, i, i + 1) = torch::sum(torch::abs(left_slice - right_slice), 1, true);
    }

    // 级联残差处理
    auto x = cost_volume;
    for (auto& module : cascade_modules) {
        x = module->forward(x);
    }

    // 视差回归
    return disparity_regression->forward(x);
}

// CRL构造函数
CRL::CRL() : isInitialized(false), maxDisparity(192), baseline(0.1f), focalLength(1000.0f) {
    model = std::make_shared<CRLNet>();
}

CRL::~CRL() {}

bool CRL::init(const std::string& modelPath) {
    try {
        torch::load(model, modelPath);
        model->eval();
        isInitialized = true;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Failed to load model: " << e.what() << std::endl;
        return false;
    }
}

torch::Tensor CRL::preprocessImage(const cv::Mat& img) {
    cv::Mat float_img;
    img.convertTo(float_img, CV_32F, 1.0/255.0);
    
    auto tensor = torch::from_blob(float_img.data, {1, img.rows, img.cols, 3});
    tensor = tensor.permute({0, 3, 1, 2});  // NCHW格式
    return tensor.clone();
}

cv::Mat CRL::postprocessDisparity(const torch::Tensor& disp) {
    auto disp_cpu = disp.squeeze().detach().cpu();
    cv::Mat disparity(disp_cpu.size(0), disp_cpu.size(1), CV_32F);
    std::memcpy(disparity.data, disp_cpu.data_ptr(), disp_cpu.numel() * sizeof(float));
    
    // 归一化到0-255
    cv::Mat disparity_vis;
    cv::normalize(disparity, disparity_vis, 0, 255, cv::NORM_MINMAX);
    disparity_vis.convertTo(disparity_vis, CV_8U);
    
    return disparity_vis;
}

cv::Mat CRL::computeDisparity(const cv::Mat& leftImg, const cv::Mat& rightImg) {
    if (!isInitialized || leftImg.empty() || rightImg.empty()) {
        return cv::Mat();
    }

    // 预处理
    auto left_tensor = preprocessImage(leftImg);
    auto right_tensor = preprocessImage(rightImg);

    // 计算视差
    torch::NoGradGuard no_grad;
    auto disparity = model->forward(left_tensor, right_tensor);

    // 后处理
    return postprocessDisparity(disparity);
}

cv::Mat CRL::computeDepth(const cv::Mat& disparity, float baseline, float focalLength) {
    if (disparity.empty()) {
        return cv::Mat();
    }

    cv::Mat depth = cv::Mat::zeros(disparity.size(), CV_32F);
    
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