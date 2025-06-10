#pragma once
#include <vector>
#include <memory>
#include <Eigen/Dense>
#include "BaseTrack.h"
#include "KalmanFilterXYAH.h"
#include "STrack.h"

// BYTETracker类，多目标跟踪主控器
class BYTETracker {
public:
    std::vector<std::shared_ptr<STrack>> tracked_stracks;   // 当前激活的track列表
    std::vector<std::shared_ptr<STrack>> lost_stracks;      // 丢失的track列表
    std::vector<std::shared_ptr<STrack>> removed_stracks;   // 被移除的track列表
    int frame_id;                                           // 当前帧号
    int max_time_lost;                                      // 最大丢失帧数
    std::shared_ptr<KalmanFilterXYAH> kalman_filter;        // 卡尔曼滤波器
    float track_high_thresh;                                // 高置信度阈值
    float track_low_thresh;                                 // 低置信度阈值
    float match_thresh;                                     // 关联距离阈值
    float new_track_thresh;                                 // 新track激活阈值
    int class_history_len;                                  // 类别平滑窗口长度

    // 构造函数，所有参数可调，默认值与原代码一致
    BYTETracker(
        int track_buffer = 30,
        int frame_rate = 30,
        float track_high_thresh = 0.6f,
        float track_low_thresh = 0.1f,
        float match_thresh = 0.8f,
        float new_track_thresh = 0.7f,
        int class_history_len = 5
    );
    // 更新tracker，输入检测框、分数、类别，返回当前激活track的结果
    std::vector<std::vector<float>> update(const std::vector<Eigen::VectorXf>& dets, const std::vector<float>& scores, const std::vector<int>& clss, const std::vector<float>& distances);
    // 批量预测track状态
    void multi_predict(std::vector<std::shared_ptr<STrack>>& tracks);
    // 重置全局ID
    static void reset_id();
    // 合并两个track列表，去重
    static std::vector<std::shared_ptr<STrack>> joint_stracks(const std::vector<std::shared_ptr<STrack>>& a, const std::vector<std::shared_ptr<STrack>>& b);
    // 从a中去除b中存在的track
    static std::vector<std::shared_ptr<STrack>> sub_stracks(const std::vector<std::shared_ptr<STrack>>& a, const std::vector<std::shared_ptr<STrack>>& b);
    // 利用IoU距离去除重复track
    static void remove_duplicate_stracks(std::vector<std::shared_ptr<STrack>>& a, std::vector<std::shared_ptr<STrack>>& b);
}; 