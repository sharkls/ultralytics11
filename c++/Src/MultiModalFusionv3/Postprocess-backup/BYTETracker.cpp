#include "BYTETracker.h"
#include "Matching.h"
#include "KalmanFilterXYAH.h"
#include <algorithm>
#include <memory>
#include <iostream>

BYTETracker::BYTETracker(int track_buffer, int frame_rate)
 {
    frame_id = 0;
    max_time_lost = int(frame_rate / 30.0 * track_buffer);
    kalman_filter = std::make_shared<KalmanFilterXYAH>();
    BaseTrack::reset_id();
}

void BYTETracker::multi_predict(std::vector<std::shared_ptr<STrack>>& tracks) 
{
    for (auto& t : tracks) t->predict();
}

std::vector<std::shared_ptr<STrack>> BYTETracker::joint_stracks(const std::vector<std::shared_ptr<STrack>>& a, const std::vector<std::shared_ptr<STrack>>& b) 
{
    std::vector<std::shared_ptr<STrack>> res = a;
    for (const auto& t : b) {
        auto it = std::find_if(res.begin(), res.end(), [&](const std::shared_ptr<STrack>& x) { return x->track_id == t->track_id; });
        if (it == res.end()) res.push_back(t);
    }
    return res;
}

std::vector<std::shared_ptr<STrack>> BYTETracker::sub_stracks(const std::vector<std::shared_ptr<STrack>>& a, const std::vector<std::shared_ptr<STrack>>& b) 
{
    std::vector<std::shared_ptr<STrack>> res;
    for (const auto& t : a) {
        auto it = std::find_if(b.begin(), b.end(), [&](const std::shared_ptr<STrack>& x) { return x->track_id == t->track_id; });
        if (it == b.end()) res.push_back(t);
    }
    return res;
}

void BYTETracker::remove_duplicate_stracks(std::vector<std::shared_ptr<STrack>>& a, std::vector<std::shared_ptr<STrack>>& b)
 {
    // IoU距离去重，阈值0.15，保留track时间长的
    if (a.empty() || b.empty()) return;
    Eigen::MatrixXf pdist = Matching::iou_distance(a, b);
    std::vector<int> dupa, dupb;
    for (int i = 0; i < pdist.rows(); ++i) {
        for (int j = 0; j < pdist.cols(); ++j) {
            if (pdist(i, j) < 0.15f) {
                int timep = a[i]->frame_id - a[i]->start_frame;
                int timeq = b[j]->frame_id - b[j]->start_frame;
                if (timep > timeq) dupb.push_back(j);
                else dupa.push_back(i);
            }
        }
    }
    // 去重
    std::vector<std::shared_ptr<STrack>> a_new, b_new;
    for (int i = 0; i < a.size(); ++i) if (std::find(dupa.begin(), dupa.end(), i) == dupa.end()) a_new.push_back(a[i]);
    for (int i = 0; i < b.size(); ++i) if (std::find(dupb.begin(), dupb.end(), i) == dupb.end()) b_new.push_back(b[i]);
    a = a_new;
    b = b_new;
}
void BYTETracker::reset_id() { BaseTrack::reset_id(); }

std::vector<std::vector<float>> BYTETracker::update(const std::vector<Eigen::VectorXf>& dets, const std::vector<float>& scores, const std::vector<int>& clss) {
    frame_id++;
    std::vector<std::shared_ptr<STrack>> activated_stracks, refind_stracks, lost_stracks, removed_stracks;
    
    // 1. 置信度筛选
    std::vector<int> high_inds, low_inds, second_inds;
    for (size_t i = 0; i < scores.size(); ++i) {
        if (scores[i] >= track_high_thresh) high_inds.push_back(i);
        else if (scores[i] > track_low_thresh && scores[i] < track_high_thresh) second_inds.push_back(i);
    }
    
    // 2. 初始化STrack对象
    std::vector<std::shared_ptr<STrack>> detections, detections_second;
    for (int idx : high_inds) detections.push_back(std::make_shared<STrack>(dets[idx], scores[idx], clss[idx]));
    for (int idx : second_inds) detections_second.push_back(std::make_shared<STrack>(dets[idx], scores[idx], clss[idx]));
    
    // 3. 预测所有track位置
    std::vector<std::shared_ptr<STrack>> unconfirmed, tracked_stracks;
    for (auto& t : tracked_stracks) {
        if (!t->is_activated) unconfirmed.push_back(t);
        else tracked_stracks.push_back(t);
    }
    auto strack_pool = joint_stracks(tracked_stracks, lost_stracks);
    multi_predict(strack_pool);
    
    // 4. 第一阶段关联
    Eigen::MatrixXf dists = Matching::iou_distance(strack_pool, detections);
    std::vector<std::pair<int, int>> matches;
    std::vector<int> u_track, u_detection;
    Matching::linear_assignment(dists, match_thresh, matches, u_track, u_detection);
    for (auto& p : matches) {
        auto& track = strack_pool[p.first];
        auto& det = detections[p.second];
        if (track->state == TrackState::Tracked) {
            track->update_with_track(*det, frame_id);
            activated_stracks.push_back(track);
        } else {
            track->re_activate(*det, frame_id, false);
            refind_stracks.push_back(track);
        }
    }
    
    // 5. 第二阶段关联
    std::vector<std::shared_ptr<STrack>> r_tracked_stracks;
    for (int i : u_track) if (strack_pool[i]->state == TrackState::Tracked) r_tracked_stracks.push_back(strack_pool[i]);
    Eigen::MatrixXf dists2 = Matching::iou_distance(r_tracked_stracks, detections_second);
    std::vector<std::pair<int, int>> matches2;
    std::vector<int> u_track2, u_detection2;
    Matching::linear_assignment(dists2, 0.5f, matches2, u_track2, u_detection2);
    for (auto& p : matches2) {
        auto& track = r_tracked_stracks[p.first];
        auto& det = detections_second[p.second];
        if (track->state == TrackState::Tracked) {
            track->update_with_track(*det, frame_id);
            activated_stracks.push_back(track);
        } else {
            track->re_activate(*det, frame_id, false);
            refind_stracks.push_back(track);
        }
    }
    for (int it : u_track2) {
        auto& track = r_tracked_stracks[it];
        if (track->state != TrackState::Lost) {
            track->mark_lost();
            lost_stracks.push_back(track);
        }
    }
    
    // 6. 处理unconfirmed
    std::vector<std::shared_ptr<STrack>> detections_u;
    for (int i : u_detection) detections_u.push_back(detections[i]);
    Eigen::MatrixXf dists3 = Matching::iou_distance(unconfirmed, detections_u);
    std::vector<std::pair<int, int>> matches3;
    std::vector<int> u_unconfirmed, u_detection3;
    Matching::linear_assignment(dists3, 0.7f, matches3, u_unconfirmed, u_detection3);
    for (auto& p : matches3) {
        unconfirmed[p.first]->update_with_track(*detections_u[p.second], frame_id);
        activated_stracks.push_back(unconfirmed[p.first]);
    }
    for (int it : u_unconfirmed) {
        unconfirmed[it]->mark_removed();
        removed_stracks.push_back(unconfirmed[it]);
    }
    
    // 7. 新增track
    for (int inew : u_detection3) {
        auto& track = detections_u[inew];
        if (track->score < new_track_thresh) continue;
        track->activate_with_kf(kalman_filter, frame_id);
        activated_stracks.push_back(track);
    }
    
    // 8. 更新状态
    for (auto& track : lost_stracks) {
        if (frame_id - track->end_frame() > max_time_lost) {
            track->mark_removed();
            removed_stracks.push_back(track);
        }
    }
    tracked_stracks.clear();
    for (auto& t : tracked_stracks) if (t->state == TrackState::Tracked) tracked_stracks.push_back(t);
    tracked_stracks = joint_stracks(tracked_stracks, activated_stracks);
    tracked_stracks = joint_stracks(tracked_stracks, refind_stracks);
    lost_stracks = sub_stracks(lost_stracks, tracked_stracks);
    lost_stracks.insert(lost_stracks.end(), lost_stracks.begin(), lost_stracks.end());
    lost_stracks = sub_stracks(lost_stracks, removed_stracks);
    remove_duplicate_stracks(tracked_stracks, lost_stracks);
    removed_stracks.insert(removed_stracks.end(), removed_stracks.begin(), removed_stracks.end());
    if (removed_stracks.size() > 1000) removed_stracks.erase(removed_stracks.begin(), removed_stracks.end() - 999);
    std::vector<std::vector<float>> results;
    for (auto& x : tracked_stracks) if (x->is_activated) results.push_back(x->result());
    return results;
} 