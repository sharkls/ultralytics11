/*******************************************************
 文件名：Matching.h
 作者：sharkls
 描述：目标跟踪匹配算法
 版本：v1.0
 日期：2025-05-15
 *******************************************************/

#ifndef MATCHING_H
#define MATCHING_H

#include <vector>
#include <Eigen/Dense>
#include <memory>
#include "STrack.h"

// 匹配算法命名空间
namespace Matching {
    // 计算IoU距离矩阵
    Eigen::MatrixXf iou_distance(const std::vector<std::shared_ptr<STrack>>& tracks, 
                                const std::vector<std::shared_ptr<STrack>>& detections);
    
    // 匈牙利算法线性分配，返回匹配对、未匹配track、未匹配det
    void linear_assignment(const Eigen::MatrixXf& cost_matrix, 
                          float thresh,
                          std::vector<std::pair<int, int>>& matches,
                          std::vector<int>& unmatched_tracks,
                          std::vector<int>& unmatched_detections);
}

#endif // MATCHING_H 