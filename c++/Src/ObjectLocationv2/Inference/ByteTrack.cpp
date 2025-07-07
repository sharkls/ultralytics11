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
#include <fstream>

// 注册模块
REGISTER_MODULE("ObjectLocation", ByteTrack, ByteTrack)

ByteTrack::~ByteTrack()
{
}

bool ByteTrack::init(void* p_pAlgParam)
{
    LOG(INFO) << "ByteTrack::init status: start ";
    // 1. 参数核验
    if (!p_pAlgParam) {
        return false;
    }
    // 2. 参数格式转换与成员变量初始化
    // objectdetection::YOLOModelConfig* cfg = static_cast<objectdetection::YOLOModelConfig*>(p_pAlgParam);
    objectlocation::TaskConfig* cfg = static_cast<objectlocation::TaskConfig*>(p_pAlgParam);
    m_config_ = *cfg;
    status_ = cfg->run_status();

    // 3. 读取所有参数（带默认值）
    if (!cfg) {
        LOG(ERROR) << "Invalid configuration pointer";
        return false;
    }

    // 设置默认值
    tracker_buffer_size_ = 30;
    track_high_thresh_   = 0.6f;
    track_low_thresh_    = 0.1f;
    match_thresh_        = 0.8f;
    new_track_thresh_    = 0.7f;
    class_history_len_   = 5;
    max_time_lost_       = 30;
    min_confidence_      = 0.5f;
    nms_threshold_       = 0.5f;
    max_tracks_          = 100;

    // 尝试从配置中读取值
    try {
        tracker_buffer_size_ = cfg->tracker_buffer_size();
        track_high_thresh_   = cfg->track_high_thresh();
        track_low_thresh_    = cfg->track_low_thresh();
        match_thresh_        = cfg->match_thresh();
        new_track_thresh_    = cfg->new_track_thresh();
        class_history_len_   = cfg->class_history_len();
        max_time_lost_       = cfg->max_time_lost();
        min_confidence_      = cfg->conf_thres();
        nms_threshold_       = cfg->iou_thres();
        max_tracks_          = cfg->max_dets();
    } catch (const std::exception& e) {
        LOG(WARNING) << "Failed to read some configuration values, using defaults: " << e.what();
    }

    save_result_         = false;
    result_path_         = "bytetrack_multimodalfusion_output.bin";

    // 4. 初始化跟踪器（所有参数可调）
    m_tracker_ = std::make_shared<BYTETracker>(
        tracker_buffer_size_,
        30, // frame_rate
        track_high_thresh_,
        track_low_thresh_,
        match_thresh_,
        new_track_thresh_,
        class_history_len_
    );
    m_inputData_ = CAlgResult();
    m_outputData_ = CAlgResult();
    LOG(INFO) << "ByteTrack::init status: success ";
    return true;
}

void ByteTrack::setInput(void* input)
{
    if (!input) {
        LOG(ERROR) << "Input is null";
        return;
    }
    m_inputData_ = *static_cast<CAlgResult*>(input);
}

void* ByteTrack::getOutput()
{
    return &m_outputData_;
}

void ByteTrack::convertDetections(const std::vector<CObjectResult>& detections,
                                std::vector<Eigen::VectorXf>& dets,
                                std::vector<float>& scores,
                                std::vector<int>& clss,
                                std::vector<float>& distances)
{
    if (detections.empty()) {
        LOG(WARNING) << "Empty detections input";
        return;
    }

    dets.clear();
    scores.clear();
    clss.clear();
    distances.clear();
    
    for (const auto& det : detections) {
        // 检查检测框的有效性
        if (det.fBottomRightX() <= det.fTopLeftX() || det.fBottomRightY() <= det.fTopLeftY()) {
            LOG(WARNING) << "Invalid detection box coordinates";
            continue;
        }

        // 转换检测框格式 [x, y, w, h]
        Eigen::VectorXf det_vec(4);
        det_vec[0] = det.fTopLeftX();
        det_vec[1] = det.fTopLeftY();
        det_vec[2] = det.fBottomRightX() - det.fTopLeftX(); // width
        det_vec[3] = det.fBottomRightY() - det.fTopLeftY(); // height
        
        dets.push_back(det_vec);
        scores.push_back(det.fVideoConfidence());
        clss.push_back(std::stoi(det.strClass()));
        distances.push_back(det.fDistance());
        
        // 创建STrack对象时传入距离值
        auto track = std::make_shared<STrack>(det_vec, det.fVideoConfidence(), std::stoi(det.strClass()), class_history_len_);
        track->distance = det.fDistance();  // 设置距离值

        // std::cout << "convertDetections:   track->distance: -------------" << track->distance << std::endl;
    }
}

void ByteTrack::convertTracks(const std::vector<std::vector<float>>& tracks,
                            std::vector<CObjectResult>& output)
{
    if (tracks.empty()) {
        LOG(WARNING) << "Empty tracks input";
        return;
    }

    output.clear();
    
    for (const auto& track : tracks) {
        // 检查track数据的完整性
        if (track.size() < 8) {  // 修改为8，因为现在包含距离值
            LOG(WARNING) << "Invalid track data size";
            continue;
        }

        CObjectResult obj;
        // track格式: [x, y, w, h, track_id, score, class, distance]
        float x = track[0];
        float y = track[1];
        float w = track[2];
        float h = track[3];

        // 检查坐标的有效性
        if (w <= 0 || h <= 0) {
            LOG(WARNING) << "Invalid track dimensions";
            continue;
        }

        obj.fTopLeftX(x - w/2);
        obj.fTopLeftY(y - h/2);
        obj.fBottomRightX(x + w/2);
        obj.fBottomRightY(y + h/2);
        obj.usTargetId(static_cast<uint16_t>(track[4]));
        obj.fVideoConfidence(track[5]);
        obj.strClass(std::to_string(static_cast<int>(track[6])));
        obj.fDistance(track[7]);  // 设置距离值 

        // std::cout << "convertTracks:   obj.fDistance: -------------" << obj.fDistance() << std::endl;
        
        output.push_back(obj);
    }
}

void ByteTrack::processFrame(const CFrameResult& frame)
{
    if (!m_tracker_) {
        LOG(ERROR) << "Tracker not initialized";
        return;
    }

    // 1. 转换检测结果
    std::vector<Eigen::VectorXf> dets;
    std::vector<float> scores;
    std::vector<int> clss;
    std::vector<float> distances;
    convertDetections(frame.vecObjectResult(), dets, scores, clss, distances);
    
    if (dets.empty()) {
        LOG(WARNING) << "No valid detections after conversion";
        return;
    }
    
    // 2. 执行跟踪
    auto tracks = m_tracker_->update(dets, scores, clss, distances);
    
    // 3. 转换跟踪结果
    std::vector<CObjectResult> tracked_objects;
    convertTracks(tracks, tracked_objects);
    
    // 4. 更新输出
    CFrameResult output_frame;
    output_frame.vecObjectResult(tracked_objects);
    m_outputData_.vecFrameResult().push_back(output_frame);

    // std::cout << "ByteTrack::processFrame status: success!  :   tracked_objects:" << tracked_objects.size() << std::endl;
}

void ByteTrack::execute()
{
    // LOG(INFO) << "ByteTrack::execute status: start ";
    if (m_inputData_.vecFrameResult().empty()) {
        LOG(ERROR) << "Input data is empty";
        return;
    }
    
    try {
        // 清空输出数据
        m_outputData_ = CAlgResult();
        
        // 处理每一帧
        for (const auto& frame : m_inputData_.vecFrameResult()) {
            processFrame(frame);
        }
        
        // LOG(INFO) << "ByteTrack::execute status: success!";

        // 离线调试代码
        // if (save_result_) {
        //     save_bin(m_outputData_, result_path_);
        // }
    }
    catch (const std::exception& e) {
        LOG(ERROR) << "Tracking failed: " << e.what();
        return;
    }
    // LOG(INFO) << "ByteTrack::execute status: end! ";
}

// 保存二进制数据的函数实现
void save_bin(const CAlgResult& data, const std::string& filename) {
    try {
        std::ofstream outfile(filename, std::ios::binary);
        if (!outfile.is_open()) {
            LOG(ERROR) << "Failed to open file for writing: " << filename;
            return;
        }

        // 保存帧数量
        size_t frame_count = data.vecFrameResult().size();
        outfile.write(reinterpret_cast<const char*>(&frame_count), sizeof(frame_count));

        // 保存每一帧的数据
        for (const auto& frame : data.vecFrameResult()) {
            // 保存该帧中的目标数量
            size_t obj_count = frame.vecObjectResult().size();
            outfile.write(reinterpret_cast<const char*>(&obj_count), sizeof(obj_count));

            // 保存每个目标的数据
            for (const auto& obj : frame.vecObjectResult()) {
                // 保存目标ID
                uint16_t target_id = obj.usTargetId();
                outfile.write(reinterpret_cast<const char*>(&target_id), sizeof(target_id));

                // 保存置信度
                float confidence = obj.fVideoConfidence();
                outfile.write(reinterpret_cast<const char*>(&confidence), sizeof(confidence));

                // 保存边界框坐标
                float top_left_x = obj.fTopLeftX();
                float top_left_y = obj.fTopLeftY();
                float bottom_right_x = obj.fBottomRightX();
                float bottom_right_y = obj.fBottomRightY();
                outfile.write(reinterpret_cast<const char*>(&top_left_x), sizeof(top_left_x));
                outfile.write(reinterpret_cast<const char*>(&top_left_y), sizeof(top_left_y));
                outfile.write(reinterpret_cast<const char*>(&bottom_right_x), sizeof(bottom_right_x));
                outfile.write(reinterpret_cast<const char*>(&bottom_right_y), sizeof(bottom_right_y));

                // 保存类别字符串
                std::string class_str = obj.strClass();
                size_t str_len = class_str.length();
                outfile.write(reinterpret_cast<const char*>(&str_len), sizeof(str_len));
                outfile.write(class_str.c_str(), str_len);
            }
        }

        outfile.close();
        LOG(INFO) << "Successfully saved binary data to: " << filename;
    }
    catch (const std::exception& e) {
        LOG(ERROR) << "Error saving binary data: " << e.what();
    }
}