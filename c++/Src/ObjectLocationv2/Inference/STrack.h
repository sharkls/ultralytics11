/*******************************************************
 文件名：STrack.h
 作者：sharkls
 描述：单目标跟踪对象类定义
 版本：v1.0
 日期：2025-05-15
 *******************************************************/

#ifndef S_TRACK_H
#define S_TRACK_H

#include <Eigen/Dense>
#include <memory>
#include <vector>
#include <deque>
#include <algorithm>
#include "BaseTrack.h"

// 前向声明
class KalmanFilterXYAH;

// STrack类，单目标跟踪对象，继承自BaseTrack
class STrack : public BaseTrack {
public:
    Eigen::VectorXf _tlwh; // 原始检测框，格式为(top-left-x, top-left-y, w, h)
    Eigen::VectorXf mean;  // 卡尔曼滤波器的均值向量
    Eigen::MatrixXf covariance; // 卡尔曼滤波器的协方差矩阵
    float score;           // 检测置信度分数
    float distance;        // 目标距离值
    int tracklet_len;      // 轨迹长度
    int cls;               // 目标类别
    int idx;               // 检测框索引
    float angle;           // 角度（可选）
    std::shared_ptr<KalmanFilterXYAH> kalman_filter; // 卡尔曼滤波器指针
    std::deque<int> class_history; // 类别历史
    int class_history_len;     // 平滑窗口长度

    // 构造函数，初始化STrack对象
    STrack(const Eigen::VectorXf& tlwh, float score, int cls, int class_history_len = 5);
    
    // 获取跟踪框信息
    Eigen::VectorXf tlwh() const;
    Eigen::VectorXf to_xyah() const;
    Eigen::VectorXf to_tlbr() const;
    
    // 实现基类纯虚函数
    void activate() override;  // 无参数版本
    void update() override;    // 无参数版本
    void predict() override;
    
    // 扩展功能函数
    void activate_with_kf(std::shared_ptr<KalmanFilterXYAH> kf, int frame_id);
    void update_with_track(const STrack& new_track, int frame_id);
    void re_activate(const STrack& new_track, int frame_id, bool new_id = false);
    
    // 获取结果
    std::vector<float> result() const;
    
    // 坐标格式转换（tlwh->xyah）
    Eigen::VectorXf convert_coords(const Eigen::VectorXf& tlwh) const;
    
    // 静态方法：tlwh转xyah
    static Eigen::VectorXf tlwh_to_xyah(const Eigen::VectorXf& tlwh);
    
    // 获取当前xywh格式的检测框
    Eigen::VectorXf xywh() const;
    
    // 获取当前xywha格式的检测框
    Eigen::VectorXf xywha() const;
    
    // 标记为移除
    void mark_removed();
    
    // 标记为丢失
    void mark_lost();

    void update_class_history(int new_cls) {
        class_history.push_back(new_cls);
        if (class_history.size() > class_history_len) class_history.pop_front();
        // 取众数
        std::vector<int> cnt(100, 0); // 假设类别数<100
        for (int c : class_history) cnt[c]++;
        int max_cnt = 0, mode = new_cls;
        for (int i = 0; i < cnt.size(); ++i) if (cnt[i] > max_cnt) { max_cnt = cnt[i]; mode = i; }
        cls = mode;
    }
};

#endif // S_TRACK_H 