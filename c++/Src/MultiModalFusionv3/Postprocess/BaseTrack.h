#pragma once
#include <Eigen/Dense>

// 跟踪状态
// 表示目标跟踪的不同生命周期状态
enum class TrackState { New = 0, Tracked = 1, Lost = 2, Removed = 3 };

// 基础Track类，所有跟踪对象的基类
class BaseTrack {
public:
    static int count;                // 全局唯一ID计数器
    int track_id;                    // 当前track的唯一ID
    bool is_activated;               // 是否已激活
    TrackState state;                // 当前track的状态
    int start_frame;                 // track起始帧
    int frame_id;                    // 当前帧号

    // 构造函数，初始化track基本属性
    BaseTrack();
    // 获取下一个唯一ID
    static int next_id();
    // 激活track（纯虚函数，需子类实现）
    virtual void activate() = 0;
    // 预测track状态（纯虚函数，需子类实现）
    virtual void predict() = 0;
    // 更新track（纯虚函数，需子类实现）
    virtual void update() = 0;
    // 标记为丢失
    void mark_lost();
    // 标记为移除
    void mark_removed();
    // 重置全局ID计数器
    static void reset_id();
    // 获取track最后出现的帧号
    int end_frame() const { return frame_id; }
}; 