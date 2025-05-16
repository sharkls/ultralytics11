#include "STrack.h"
#include "KalmanFilterXYAH.h"
#include <cassert>
#include <iostream>

STrack::STrack(const Eigen::VectorXf& tlwh, float score, int cls)
    : score(score), cls(cls), _tlwh(tlwh) {
    kalman_filter = std::make_shared<KalmanFilterXYAH>();
}

void STrack::predict() {
    if (this->state != TrackState::Tracked) {
        this->mean[7] = 0;
    }
    auto [mean, covariance] = kalman_filter->predict(this->mean, this->covariance);
    this->mean = mean;
    this->covariance = covariance;
}

void STrack::activate() {
    this->state = TrackState::Tracked;
    this->is_activated = true;
}

void STrack::update() {
    this->state = TrackState::Tracked;
    this->is_activated = true;
}

void STrack::activate_with_kf(std::shared_ptr<KalmanFilterXYAH> kf, int frame_id) {
    this->kalman_filter = kf;
    this->track_id = next_id();
    
    auto [mean, covariance] = kalman_filter->initiate(to_xyah());
    this->mean = mean;
    this->covariance = covariance;
    
    this->state = TrackState::Tracked;
    this->is_activated = true;
    this->frame_id = frame_id;
    this->start_frame = frame_id;
}

void STrack::update_with_track(const STrack& new_track, int frame_id) {
    this->frame_id = frame_id;
    this->score = new_track.score;
    
    auto [mean, covariance] = kalman_filter->update(this->mean, this->covariance, new_track.to_xyah());
    this->mean = mean;
    this->covariance = covariance;
    
    this->state = TrackState::Tracked;
    this->is_activated = true;
    this->_tlwh = new_track._tlwh;
}

void STrack::re_activate(const STrack& new_track, int frame_id, bool new_id) {
    auto [mean, covariance] = kalman_filter->update(this->mean, this->covariance, new_track.to_xyah());
    this->mean = mean;
    this->covariance = covariance;
    
    this->state = TrackState::Tracked;
    this->is_activated = true;
    this->frame_id = frame_id;
    this->score = new_track.score;
    
    if (new_id) {
        this->track_id = next_id();
    }
}

Eigen::VectorXf STrack::to_xyah() const {
    return tlwh_to_xyah(_tlwh);
}

Eigen::VectorXf STrack::to_tlbr() const {
    Eigen::VectorXf ret = _tlwh;
    ret[2] += ret[0];
    ret[3] += ret[1];
    return ret;
}

Eigen::VectorXf STrack::tlwh() const {
    return _tlwh;
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
    std::vector<float> ret(7);
    ret[0] = _tlwh[0] + _tlwh[2] / 2;  // x center
    ret[1] = _tlwh[1] + _tlwh[3] / 2;  // y center
    ret[2] = _tlwh[2];                  // width
    ret[3] = _tlwh[3];                  // height
    ret[4] = static_cast<float>(track_id);
    ret[5] = score;
    ret[6] = static_cast<float>(cls);
    return ret;
}

void STrack::mark_removed() {
    this->state = TrackState::Removed;
}

void STrack::mark_lost() {
    this->state = TrackState::Lost;
}

Eigen::VectorXf STrack::tlwh_to_xyah(const Eigen::VectorXf& tlwh) {
    Eigen::VectorXf ret = tlwh;
    ret[0] += ret[2] / 2;
    ret[1] += ret[3] / 2;
    ret[2] /= ret[3];
    return ret;
} 