/*******************************************************
 文件名：ByteTrack.h
 作者：sharkls
 描述：ObjectLocation-ByteTrack跟踪模块
 版本：v1.0
 日期：2025-05-15
 *******************************************************/

#ifndef BYTE_TRACK_Location_H
#define BYTE_TRACK_Location_H

// 三方库
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include "log.h"
#include <Eigen/Dense>
#include <memory>
#include <list>

// 框架头文件
#include "IBaseModule.h"
#include "ModuleFactory.h"

// idl 和 proto 头文件
#include "CMultiModalSrcData.h"
// #include "PoseEstimation_conf.pb.h"
#include "ObjectLocation_conf.pb.h"

#include "FunctionHub.h"
#include "BaseTrack.h"
#include "BYTETracker.h"
#include "STrack.h"
#include "Matching.h"

// 保存二进制数据的函数声明
void save_bin(const CAlgResult& data, const std::string& filename);

class ByteTrack : public IBaseModule {
public:
    ByteTrack(const std::string& exe_path) : IBaseModule(exe_path) {}
    ~ByteTrack() override;

    // 实现基类接口
    std::string getModuleName() const override { return "ByteTrack"; }
    ModuleType getModuleType() const override { return ModuleType::POST_PROCESS; }
    bool init(void* p_pAlgParam) override;
    void execute() override;
    void setInput(void* input) override;
    void* getOutput() override;

private:
    // 以下参数均在init()中初始化
    objectlocation::TaskConfig m_config_;       // 配置参数
    CAlgResult m_inputData_;    // 输入数据
    CAlgResult m_outputData_;   // 输出结果
    std::shared_ptr<BYTETracker> m_tracker_; // ByteTracker实例
    bool status_ = false;       // 运行状态

    // 跟踪相关参数（全部通过proto配置）
    int tracker_buffer_size_;
    float track_high_thresh_;
    float track_low_thresh_;
    float match_thresh_;
    float new_track_thresh_;
    int class_history_len_;
    int max_time_lost_;
    float min_confidence_;
    float nms_threshold_;
    int max_tracks_;
    bool save_result_;
    std::string result_path_;

    // 处理单帧数据
    void processFrame(const CFrameResult& frame);
    
    // 将检测结果转换为跟踪器输入格式
    void convertDetections(const std::vector<CObjectResult>& detections,
                          std::vector<Eigen::VectorXf>& dets,
                          std::vector<float>& scores,
                          std::vector<int>& clss,
                          std::vector<float>& distances);
    
    // 将跟踪结果转换为输出格式
    void convertTracks(const std::vector<std::vector<float>>& tracks,
                      std::vector<CObjectResult>& output);
};

#endif // BYTE_TRACK_Location_H 