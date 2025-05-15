#include "KalmanFilterXYAH.h"

constexpr int ndim = 4;
constexpr int dim_x = 8;
constexpr int dim_z = 4;

KalmanFilterXYAH::KalmanFilterXYAH() {}

std::pair<Eigen::VectorXf, Eigen::MatrixXf> KalmanFilterXYAH::initiate(const Eigen::VectorXf& measurement) {
    Eigen::VectorXf mean = Eigen::VectorXf::Zero(dim_x);
    mean.head(dim_z) = measurement.head(dim_z);
    Eigen::MatrixXf covariance = Eigen::MatrixXf::Identity(dim_x, dim_x) * 10.0f;
    return {mean, covariance};
}

std::pair<Eigen::VectorXf, Eigen::MatrixXf> KalmanFilterXYAH::predict(const Eigen::VectorXf& mean, const Eigen::MatrixXf& covariance) {
    Eigen::MatrixXf F = Eigen::MatrixXf::Identity(dim_x, dim_x);
    for (int i = 0; i < ndim; ++i) F(i, ndim + i) = 1.0f;
    Eigen::VectorXf mean_pred = F * mean;
    Eigen::MatrixXf Q = Eigen::MatrixXf::Identity(dim_x, dim_x) * 1.0f;
    Eigen::MatrixXf cov_pred = F * covariance * F.transpose() + Q;
    return {mean_pred, cov_pred};
}

std::pair<Eigen::VectorXf, Eigen::MatrixXf> KalmanFilterXYAH::update(const Eigen::VectorXf& mean, const Eigen::MatrixXf& covariance, const Eigen::VectorXf& measurement) {
    Eigen::MatrixXf H = Eigen::MatrixXf::Zero(dim_z, dim_x);
    H.block(0, 0, dim_z, dim_z) = Eigen::MatrixXf::Identity(dim_z, dim_z);
    Eigen::MatrixXf R = Eigen::MatrixXf::Identity(dim_z, dim_z) * 1.0f;
    Eigen::VectorXf y = measurement.head(dim_z) - H * mean;
    Eigen::MatrixXf S = H * covariance * H.transpose() + R;
    Eigen::MatrixXf K = covariance * H.transpose() * S.inverse();
    Eigen::VectorXf mean_upd = mean + K * y;
    Eigen::MatrixXf cov_upd = (Eigen::MatrixXf::Identity(dim_x, dim_x) - K * H) * covariance;
    return {mean_upd, cov_upd};
} 