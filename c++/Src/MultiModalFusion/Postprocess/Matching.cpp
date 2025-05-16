/*******************************************************
 文件名：Matching.cpp
 作者：sharkls
 描述：目标跟踪匹配算法实现
 版本：v1.0
 日期：2025-05-15
 *******************************************************/

#include "Matching.h"
#include <limits>
#include <algorithm>

// 计算两个bbox的IoU
float iou_single(const Eigen::VectorXf& box1, const Eigen::VectorXf& box2) {
    float x1 = std::max(box1[0], box2[0]);
    float y1 = std::max(box1[1], box2[1]);
    float x2 = std::min(box1[0] + box1[2], box2[0] + box2[2]);
    float y2 = std::min(box1[1] + box1[3], box2[1] + box2[3]);
    float w = std::max(0.0f, x2 - x1);
    float h = std::max(0.0f, y2 - y1);
    float inter = w * h;
    float area1 = box1[2] * box1[3];
    float area2 = box2[2] * box2[3];
    float iou = inter / (area1 + area2 - inter + 1e-6f);
    return iou;
}

namespace Matching {
    Eigen::MatrixXf iou_distance(const std::vector<std::shared_ptr<STrack>>& tracks, 
                                const std::vector<std::shared_ptr<STrack>>& detections) {
        int n = tracks.size(), m = detections.size();
        Eigen::MatrixXf dist = Eigen::MatrixXf::Ones(n, m);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                float iou = iou_single(tracks[i]->tlwh(), detections[j]->tlwh());
                dist(i, j) = 1.0f - iou;
            }
        }
        return dist;
    }

    // 简化版匈牙利算法（贪心），适合小规模
    void linear_assignment(const Eigen::MatrixXf& cost_matrix, 
                          float thresh,
                          std::vector<std::pair<int, int>>& matches,
                          std::vector<int>& unmatched_tracks,
                          std::vector<int>& unmatched_detections) {
        int n = cost_matrix.rows(), m = cost_matrix.cols();
        std::vector<bool> track_used(n, false), det_used(m, false);
        
        for (int k = 0; k < std::min(n, m); ++k) {
            float min_cost = thresh;
            int min_i = -1, min_j = -1;
            
            for (int i = 0; i < n; ++i) {
                if (track_used[i]) continue;
                for (int j = 0; j < m; ++j) {
                    if (det_used[j]) continue;
                    if (cost_matrix(i, j) < min_cost) {
                        min_cost = cost_matrix(i, j);
                        min_i = i;
                        min_j = j;
                    }
                }
            }
            
            if (min_i == -1) break;
            
            matches.emplace_back(min_i, min_j);
            track_used[min_i] = true;
            det_used[min_j] = true;
        }
        
        // 收集未匹配的tracks和detections
        for (int i = 0; i < n; ++i) {
            if (!track_used[i]) unmatched_tracks.push_back(i);
        }
        for (int j = 0; j < m; ++j) {
            if (!det_used[j]) unmatched_detections.push_back(j);
        }
    }
} 