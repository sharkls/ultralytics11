#pragma once
#include <vector>
#include <Eigen/Dense>
#include <memory>
#include "ByteTrack.h"

namespace matching {
    // 计算IoU距离矩阵
    Eigen::MatrixXf iou_distance(const std::vector<std::shared_ptr<STrack>>& tracks, const std::vector<std::shared_ptr<STrack>>& detections);
    // 匈牙利算法线性分配，返回匹配对、未匹配track、未匹配det
    void linear_assignment(const Eigen::MatrixXf& cost_matrix, float thresh,
                          std::vector<std::pair<int, int>>& matches,
                          std::vector<int>& unmatched_tracks,
                          std::vector<int>& unmatched_detections);
} 