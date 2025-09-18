#pragma once
#include <Eigen/Dense>
#include <utility>

class KalmanFilterXYAH {
public:
    KalmanFilterXYAH();
    // 初始化卡尔曼滤波器，返回初始均值和协方差
    std::pair<Eigen::VectorXf, Eigen::MatrixXf> initiate(const Eigen::VectorXf& measurement);
    // 预测下一状态
    std::pair<Eigen::VectorXf, Eigen::MatrixXf> predict(const Eigen::VectorXf& mean, const Eigen::MatrixXf& covariance);
    // 更新状态
    std::pair<Eigen::VectorXf, Eigen::MatrixXf> update(const Eigen::VectorXf& mean, const Eigen::MatrixXf& covariance, const Eigen::VectorXf& measurement);
}; 