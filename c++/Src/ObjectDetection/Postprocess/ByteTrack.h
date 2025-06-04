/*******************************************************
 文件名：ByteTrack.h
 作者：sharkls
 描述：ObjectDetection-ByteTrack跟踪模块
 版本：v1.0
 日期：2025-05-15
 *******************************************************/

#ifndef BYTE_TRACK_H
#define BYTE_TRACK_H

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
#include "ObjectDetection_conf.pb.h"

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
    objectdetection::YOLOModelConfig m_config_;       // 配置参数
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

    // 常用可调参数（init中赋默认值，可后续支持配置化）
    float iou_threshold_;       // IOU匹配阈值
    float min_confidence_;      // 最小检测置信度
    int max_lost_frames_;       // 最大丢失帧数
    float nms_threshold_;       // NMS阈值
    int max_tracks_;            // 最大track数量
    bool save_result_;          // 是否保存结果
    std::string result_path_;   // 结果保存路径

    // 处理单帧数据
    void processFrame(const CFrameResult& frame);
    
    // 将检测结果转换为跟踪器输入格式
    void convertDetections(const std::vector<CObjectResult>& detections,
                          std::vector<Eigen::VectorXf>& dets,
                          std::vector<float>& scores,
                          std::vector<int>& clss);
    
    // 将跟踪结果转换为输出格式
    void convertTracks(const std::vector<std::vector<float>>& tracks,
                      std::vector<CObjectResult>& output);
};

#endif // BYTE_TRACK_H 