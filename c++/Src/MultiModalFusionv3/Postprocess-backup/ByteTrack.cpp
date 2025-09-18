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
    
    // 初始化跟踪器
    m_tracker = std::make_shared<BYTETracker>(30); // 30帧的跟踪缓冲区
    
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

void ByteTrack::convertDetections(const std::vector<CObjectResult>& detections,
                                std::vector<Eigen::VectorXf>& dets,
                                std::vector<float>& scores,
                                std::vector<int>& clss)
{
    if (detections.empty()) {
        LOG(WARNING) << "Empty detections input";
        return;
    }

    dets.clear();
    scores.clear();
    clss.clear();
    
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
        clss.push_back(0); // 默认类别为0，可以根据需要修改
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
        if (track.size() < 7) {
            LOG(WARNING) << "Invalid track data size";
            continue;
        }

        CObjectResult obj;
        // track格式: [x, y, w, h, track_id, score, class]
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
        
        output.push_back(obj);
    }
}

void ByteTrack::processFrame(const CFrameResult& frame)
{
    if (!m_tracker) {
        LOG(ERROR) << "Tracker not initialized";
        return;
    }

    // 1. 转换检测结果
    std::vector<Eigen::VectorXf> dets;
    std::vector<float> scores;
    std::vector<int> clss;
    convertDetections(frame.vecObjectResult(), dets, scores, clss);
    
    if (dets.empty()) {
        LOG(WARNING) << "No valid detections after conversion";
        return;
    }
    
    // 2. 执行跟踪
    auto tracks = m_tracker->update(dets, scores, clss);
    
    // 3. 转换跟踪结果
    std::vector<CObjectResult> tracked_objects;
    convertTracks(tracks, tracked_objects);
    
    // 4. 更新输出
    CFrameResult output_frame;
    output_frame.vecObjectResult(tracked_objects);
    m_outputdata.vecFrameResult().push_back(output_frame);
}

void ByteTrack::execute()
{
    LOG(INFO) << "ByteTrack::execute status: start ";
    if (m_inputdata.vecFrameResult().empty()) {
        LOG(ERROR) << "Input data is empty";
        return;
    }
    
    try {
        // 清空输出数据
        m_outputdata = CAlgResult();
        
        // 处理每一帧
        for (const auto& frame : m_inputdata.vecFrameResult()) {
            processFrame(frame);
        }
        
        LOG(INFO) << "ByteTrack::execute status: success!";

        // 离线调试代码
        if (status_) {
            save_bin(m_outputdata, "bytetrack_multimodalfusion_output.bin");
        }
    }
    catch (const std::exception& e) {
        LOG(ERROR) << "Tracking failed: " << e.what();
        return;
    }
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