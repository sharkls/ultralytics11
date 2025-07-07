#include "BYTETracker.h"
#include "Matching.h"
#include "KalmanFilterXYAH.h"
#include <algorithm>
#include <memory>
#include <iostream>

BYTETracker::BYTETracker(
    int track_buffer,
    int frame_rate,
    float track_high_thresh,
    float track_low_thresh,
    float match_thresh,
    float new_track_thresh,
    int class_history_len
) :
    track_high_thresh(track_high_thresh),
    track_low_thresh(track_low_thresh),
    match_thresh(match_thresh),
    new_track_thresh(new_track_thresh),
    class_history_len(class_history_len)
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

std::vector<std::vector<float>> BYTETracker::update(const std::vector<Eigen::VectorXf>& dets, const std::vector<float>& scores, const std::vector<int>& clss, const std::vector<float>& distances) {
    frame_id++;
    std::vector<std::shared_ptr<STrack>> activated_stracks, refind_stracks;
    
    // 1. 置信度筛选
    std::vector<int> high_inds, low_inds, second_inds;
    for (size_t i = 0; i < scores.size(); ++i) {
        if (scores[i] >= track_high_thresh) high_inds.push_back(i);
        else if (scores[i] > track_low_thresh && scores[i] < track_high_thresh) second_inds.push_back(i);
    }

    // 打印本帧检测结果
    // std::cout << "[BYTETracker] Frame " << frame_id << " 检测结果数量: " << dets.size() << std::endl;
    // for (size_t i = 0; i < dets.size(); ++i) {
    //     std::cout << "  det " << i << ": class=" << clss[i] << ", conf=" << scores[i]
    //               << ", box=[" << dets[i][0] << ", " << dets[i][1] << ", " << dets[i][2] << ", " << dets[i][3] << "]" << std::endl;
    // }
    
    // 2. 初始化STrack对象
    std::vector<std::shared_ptr<STrack>> detections, detections_second;
    for (int idx : high_inds) {
        auto track = std::make_shared<STrack>(dets[idx], scores[idx], clss[idx], class_history_len);
        if (idx < distances.size()) track->distance = distances[idx];
        detections.push_back(track);
    }
    for (int idx : second_inds) {
        auto track = std::make_shared<STrack>(dets[idx], scores[idx], clss[idx], class_history_len);
        if (idx < distances.size()) track->distance = distances[idx];
        detections_second.push_back(track);
    }
    
    // 3. 预测所有track位置
    std::vector<std::shared_ptr<STrack>> unconfirmed;
    std::vector<std::shared_ptr<STrack>> confirmed_tracks;
    for (auto& t : tracked_stracks) {
        if (!t->is_activated) unconfirmed.push_back(t);
        else confirmed_tracks.push_back(t);
    }
    auto strack_pool = joint_stracks(confirmed_tracks, lost_stracks);
    multi_predict(strack_pool);
    
    // 打印track信息
    // std::cout << "[BYTETracker] 当前strack_pool中的track信息: " << strack_pool.size() << " 个" << std::endl;
    // for (size_t i = 0; i < strack_pool.size(); ++i) {
    //     auto& t = strack_pool[i];
    //     std::cout << "  track[" << i << "]: id=" << t->track_id
    //               << ", state=" << static_cast<int>(t->state)
    //               << ", is_activated=" << t->is_activated
    //               << ", start_frame=" << t->start_frame
    //               << ", frame_id=" << t->frame_id
    //               << ", class=" << t->cls
    //               << std::endl;
    // }
    
    // 4. 第一阶段关联
    Eigen::MatrixXf dists = Matching::iou_distance(strack_pool, detections);
    // if (dists.size() > 0) {
    //     float max_iou = dists.maxCoeff();
    //     float min_iou = dists.minCoeff();
    //     std::cout << "[BYTETracker] IOU矩阵最大值: " << max_iou << " 最小值: " << min_iou << std::endl;
    // }
    std::vector<std::pair<int, int>> matches;
    std::vector<int> u_track, u_detection;
    Matching::linear_assignment(dists, match_thresh, matches, u_track, u_detection);
    // 打印track与检测的匹配关系
    std::cout << "[BYTETracker] 第一阶段关联: 匹配对数: " << matches.size() << std::endl;
    // for (const auto& p : matches) {
    //     std::cout << "  track_id=" << strack_pool[p.first]->track_id << " <---> det_idx=" << p.second << std::endl;
    // }
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
    
    // 8. 更新状态（成员变量）
    // 8.1 处理丢失track
    for (auto& track : lost_stracks) {
        if (frame_id - track->end_frame() > max_time_lost) {
            track->mark_removed();
            removed_stracks.push_back(track);
        }
    }
    // 8.2 更新tracked_stracks
    std::vector<std::shared_ptr<STrack>> new_tracked_stracks;
    for (auto& t : activated_stracks) if (t->state == TrackState::Tracked) new_tracked_stracks.push_back(t);
    new_tracked_stracks = joint_stracks(new_tracked_stracks, refind_stracks);
    tracked_stracks = new_tracked_stracks;
    // 8.3 更新lost_stracks
    lost_stracks = sub_stracks(lost_stracks, tracked_stracks);
    // 8.4 去重
    remove_duplicate_stracks(tracked_stracks, lost_stracks);
    // 8.5 控制removed_stracks长度
    if (removed_stracks.size() > 1000) removed_stracks.erase(removed_stracks.begin(), removed_stracks.end() - 999);
    // 9. 输出结果
    std::vector<std::vector<float>> results;
    for (auto& x : tracked_stracks) {
        if (x->is_activated) {
            auto result = x->result();
            results.push_back(result);
        }
    }
    return results;
} 