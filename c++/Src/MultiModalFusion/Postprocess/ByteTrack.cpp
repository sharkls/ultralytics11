/*******************************************************
 文件名：ByteTrack.cpp
 作者：sharkls
 描述：多模态融合算法ByteTrack跟踪模块实现
 版本：v1.0
 日期：2025-05-14
 *******************************************************/

#include <Eigen/Dense>
#include <algorithm>
#include <memory>
#include "ByteTrack.h"
#include "KalmanFilterXYAH.h"
#include <cassert>
#include <iostream>

// 注册模块
REGISTER_MODULE("MultiModalFusion", ByteTrack, ByteTrack)

ByteTrack::~ByteTrack()
{
}

bool ByteTrack::init(void* p_pAlgParam)
{
    LOG(INFO) << "ByteTrack::init status: start ";
    // 1. 从配置参数中读取预处理参数
    if (!p_pAlgParam) {
        return false;
    }
    // 2. 参数格式转换
    multimodalfusion::MultiModalFusionModelConfig* multiModalFusionConfig = static_cast<multimodalfusion::MultiModalFusionModelConfig*>(p_pAlgParam);
    status_ = multiModalFusionConfig->run_status();

    m_config = *multiModalFusionConfig;
    LOG(INFO) << "ByteTrack::init status: success ";
    return true;
}

void ByteTrack::setInput(void* input)
{
    if (!input) {
        LOG(ERROR) << "Input is null";
        return;
    }
    m_inputdata = *static_cast<CAlgResult*>(input);
}

void* ByteTrack::getOutput()
{
    return &m_outputdata;
}


void ByteTrack::execute()
{
    LOG(INFO) << "ByteTrack::execute status: start ";
    if (m_inputdata.vecFrameResult().size() < 0) {
        LOG(ERROR) << "Input data is empty";
        return;
    }
    try {
        // 清空输出数据
        m_outputdata = CAlgResult();

        
        
        LOG(INFO) << "ByteTrack::execute status: success!";

        // 离线调试代码
        if (status_) {
            save_bin(m_outputdata, "bytetrack_multimodalfusion_output.bin"); // MultiModalFusion/Postprocess
        }
    }
    catch (const std::exception& e) {
        LOG(ERROR) << "Preprocessing failed: " << e.what();
        return;
    }
}

// BaseTrack实现
int BaseTrack::count = 0;
BaseTrack::BaseTrack() : track_id(0), is_activated(false), state(TrackState::New), start_frame(0), frame_id(0) {}
int BaseTrack::next_id() { return ++count; }
void BaseTrack::mark_lost() { state = TrackState::Lost; }
void BaseTrack::mark_removed() { state = TrackState::Removed; }
void BaseTrack::reset_id() { count = 0; }

// STrack实现
STrack::STrack(const Eigen::VectorXf& xywh, float score, int cls)
    : score(score), tracklet_len(0), cls(cls), idx(xywh[4]), angle(xywh.size() == 6 ? xywh[5] : 0.0f) {
    // xywh: [x, y, w, h, idx] or [x, y, w, h, a, idx]
    assert(xywh.size() == 5 || xywh.size() == 6);
    _tlwh = xywh.head(4);
    mean = Eigen::VectorXf();
    covariance = Eigen::MatrixXf();
    kalman_filter = nullptr;
    is_activated = false;
}

void STrack::predict() {
    if (!kalman_filter) return;
    Eigen::VectorXf mean_state = mean;
    if (state != TrackState::Tracked && mean.size() > 7) mean_state[7] = 0;
    std::tie(mean, covariance) = kalman_filter->predict(mean_state, covariance);
}

void STrack::activate(std::shared_ptr<KalmanFilterXYAH> kf, int frame_id) {
    kalman_filter = kf;
    track_id = next_id();
    std::tie(mean, covariance) = kalman_filter->initiate(convert_coords(_tlwh));
    tracklet_len = 0;
    state = TrackState::Tracked;
    is_activated = (frame_id == 1);
    this->frame_id = frame_id;
    start_frame = frame_id;
}

void STrack::re_activate(const STrack& new_track, int frame_id, bool new_id) {
    std::tie(mean, covariance) = kalman_filter->update(mean, covariance, convert_coords(new_track.tlwh()));
    tracklet_len = 0;
    state = TrackState::Tracked;
    is_activated = true;
    this->frame_id = frame_id;
    if (new_id) track_id = next_id();
    score = new_track.score;
    cls = new_track.cls;
    angle = new_track.angle;
    idx = new_track.idx;
}

void STrack::update(const STrack& new_track, int frame_id) {
    this->frame_id = frame_id;
    tracklet_len++;
    std::tie(mean, covariance) = kalman_filter->update(mean, covariance, convert_coords(new_track.tlwh()));
    state = TrackState::Tracked;
    is_activated = true;
    score = new_track.score;
    cls = new_track.cls;
    angle = new_track.angle;
    idx = new_track.idx;
}

Eigen::VectorXf STrack::convert_coords(const Eigen::VectorXf& tlwh) const {
    return tlwh_to_xyah(tlwh);
}

Eigen::VectorXf STrack::tlwh_to_xyah(const Eigen::VectorXf& tlwh) {
    Eigen::VectorXf ret = tlwh;
    ret[0] += ret[2] / 2;
    ret[1] += ret[3] / 2;
    ret[2] /= ret[3];
    return ret;
}

Eigen::VectorXf STrack::tlwh() const {
    if (mean.size() == 0) return _tlwh;
    Eigen::VectorXf ret = mean.head(4);
    ret[2] *= ret[3];
    ret[0] -= ret[2] / 2;
    ret[1] -= ret[3] / 2;
    return ret;
}

Eigen::VectorXf STrack::xywh() const {
    Eigen::VectorXf ret = tlwh();
    ret[0] += ret[2] / 2;
    ret[1] += ret[3] / 2;
    return ret;
}

Eigen::VectorXf STrack::xywha() const {
    Eigen::VectorXf ret(5);
    ret.head(4) = xywh();
    ret[4] = angle;
    return ret;
}

std::vector<float> STrack::result() const {
    Eigen::VectorXf coords = (angle == 0.0f) ? xywh() : xywha();
    std::vector<float> res(coords.data(), coords.data() + coords.size());
    res.push_back(track_id);
    res.push_back(score);
    res.push_back(cls);
    res.push_back(idx);
    return res;
}

void STrack::mark_removed() { state = TrackState::Removed; }
void STrack::mark_lost() { state = TrackState::Lost; }

// BYTETracker实现
BYTETracker::BYTETracker(int track_buffer, int frame_rate) {
    frame_id = 0;
    max_time_lost = int(frame_rate / 30.0 * track_buffer);
    kalman_filter = std::make_shared<KalmanFilterXYAH>();
    BaseTrack::reset_id();
}
void BYTETracker::multi_predict(std::vector<std::shared_ptr<STrack>>& tracks) {
    for (auto& t : tracks) t->predict();
}
std::vector<std::shared_ptr<STrack>> BYTETracker::joint_stracks(const std::vector<std::shared_ptr<STrack>>& a, const std::vector<std::shared_ptr<STrack>>& b) {
    std::vector<std::shared_ptr<STrack>> res = a;
    for (const auto& t : b) {
        auto it = std::find_if(res.begin(), res.end(), [&](const std::shared_ptr<STrack>& x) { return x->track_id == t->track_id; });
        if (it == res.end()) res.push_back(t);
    }
    return res;
}
std::vector<std::shared_ptr<STrack>> BYTETracker::sub_stracks(const std::vector<std::shared_ptr<STrack>>& a, const std::vector<std::shared_ptr<STrack>>& b) {
    std::vector<std::shared_ptr<STrack>> res;
    for (const auto& t : a) {
        auto it = std::find_if(b.begin(), b.end(), [&](const std::shared_ptr<STrack>& x) { return x->track_id == t->track_id; });
        if (it == b.end()) res.push_back(t);
    }
    return res;
}
void BYTETracker::remove_duplicate_stracks(std::vector<std::shared_ptr<STrack>>& a, std::vector<std::shared_ptr<STrack>>& b) {
    // 这里应实现IoU距离和去重逻辑，留空待matching实现
}
void BYTETracker::reset_id() { BaseTrack::reset_id(); }

// BYTETracker::update主流程伪代码
std::vector<std::vector<float>> BYTETracker::update(const std::vector<Eigen::VectorXf>& dets, const std::vector<float>& scores, const std::vector<int>& clss) {
    frame_id++;
    std::vector<std::shared_ptr<STrack>> activated_stracks, refind_stracks, lost_stracks, removed_stracks;
    // 1. 处理检测结果，按置信度阈值分类
    // 2. 初始化STrack对象
    // 3. 预测所有track位置
    // 4. 关联（调用matching.linear_assignment）
    // 5. 更新track状态
    // 6. 处理丢失、移除
    // 7. 返回当前激活的track
    // 具体实现需结合matching和kalman模块
    return {};
}