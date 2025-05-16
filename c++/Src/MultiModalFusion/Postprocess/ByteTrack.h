/*******************************************************
 文件名：ByteTrack.h
 作者：sharkls
 描述：ByteTrack跟踪模块
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
#include "MultiModalFusion_conf.pb.h"

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
    multimodalfusion::MultiModalFusionModelConfig m_config;  // 任务配置参数
    CAlgResult m_inputdata;             // 预处理输入数据
    CAlgResult m_outputdata;           // 模型输入数据缓存区
    std::shared_ptr<BYTETracker> m_tracker; // ByteTracker实例

    // 运行状态
    bool status_ = false;

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