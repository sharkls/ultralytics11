/*******************************************************
 文件名：ByteTrack.h
 作者：sharkls
 描述：ByteTrack跟踪模块
 版本：v1.0
 日期：2025-05-15
 *******************************************************/

#ifndef BYTE_TRACK_H
#define BYTE_TRACK_H

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include "log.h"
#include "IBaseModule.h"
#include "ModuleFactory.h"
#include "FunctionHub.h"
#include "CMultiModalSrcData.h"
#include "MultiModalFusion_conf.pb.h"
#include <Eigen/Dense>
#include <memory>
#include <list>
#include "BaseTrack.h"
#include "BYTETracker.h"

// 跟踪状态
enum class TrackState { New = 0, Tracked = 1, Lost = 2, Removed = 3 };

// STrack类，单目标跟踪对象，继承自BaseTrack
class STrack : public BaseTrack {
public:
    Eigen::VectorXf _tlwh; // 原始检测框，格式为(top-left-x, top-left-y, w, h)
    Eigen::VectorXf mean;  // 卡尔曼滤波器的均值向量
    Eigen::MatrixXf covariance; // 卡尔曼滤波器的协方差矩阵
    float score;           // 检测置信度分数
    int tracklet_len;      // 轨迹长度
    int cls;               // 目标类别
    int idx;               // 检测框索引
    float angle;           // 角度（可选）
    std::shared_ptr<class KalmanFilterXYAH> kalman_filter; // 卡尔曼滤波器指针

    // 构造函数，初始化STrack对象
    STrack(const Eigen::VectorXf& xywh, float score, int cls);
    // 预测下一状态
    void predict() override;
    // 激活新track
    void activate(std::shared_ptr<KalmanFilterXYAH> kf, int frame_id);
    // 重新激活丢失的track
    void re_activate(const STrack& new_track, int frame_id, bool new_id = false);
    // 用新检测结果更新track
    void update(const STrack& new_track, int frame_id);
    // 坐标格式转换（tlwh->xyah）
    Eigen::VectorXf convert_coords(const Eigen::VectorXf& tlwh) const;
    // 静态方法：tlwh转xyah
    static Eigen::VectorXf tlwh_to_xyah(const Eigen::VectorXf& tlwh);
    // 获取当前tlwh格式的检测框
    Eigen::VectorXf tlwh() const;
    // 获取当前xywh格式的检测框
    Eigen::VectorXf xywh() const;
    // 获取当前xywha格式的检测框
    Eigen::VectorXf xywha() const;
    // 获取当前track的输出结果
    std::vector<float> result() const;
    // 标记为移除
    void mark_removed();
    // 标记为丢失
    void mark_lost();
};

class ByteTrack : public IBaseModule {
public:
    ByteTrack(const std::string& exe_path) : IBaseModule(exe_path) {}
    ~ByteTrack() override;

    // 实现基类接口
    std::string getModuleName() const override { return "ByteTrack"; }
    ModuleType getModuleType() const override { return ModuleType::POST_PROCESS; }
    bool init(void* p_pAlgParam) override;
    void execute() override;
    void setInput(void* input) override;
    void* getOutput() override;


private:
   multimodalfusion::MultiModalFusionModelConfig m_config;  // 任务配置参数
   CAlgResult m_inputdata;             // 预处理输入数据
   CAlgResult m_outputdata;           // 模型输入数据缓存区

   // 运行状态
   bool status_ = false;
};

#endif // BYTE_TRACK_H 