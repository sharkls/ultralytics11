/*******************************************************
 文件名：PosePostProcess.h
 作者：
 描述：姿态估计后处理模块
 版本：v1.0
 日期：2024-03-21
 *******************************************************/

#ifndef POSE_POST_PROCESS_H
#define POSE_POST_PROCESS_H

#include "../../../Common/IBaseModule.h"
#include <vector>

// 关键点结构
struct KeyPoint {
    float x;
    float y;
    float confidence;
};

// 姿态结果结构
struct PoseResult {
    std::vector<KeyPoint> keypoints;
    float score;
};

class PosePostProcess : public IBaseModule {
public:
    PosePostProcess();
    ~PosePostProcess() override;

    // 实现基类接口
    std::string getModuleName() const override { return "PosePostProcess"; }
    ModuleType getModuleType() const override { return ModuleType::POST_PROCESS; }
    bool init(CSelfAlgParam* p_pAlgParam) override;
    void* execute() override;
    void setInput(void* input) override;
    void* getOutput() override;

private:
    // 后处理参数
    struct PostProcessParams {
        float confidenceThreshold = 0.5f;
        int numKeypoints = 17;  // COCO格式的关键点数量
    };

    void processKeypoints(const float* data, int dataSize);
    bool isValidPose(const PoseResult& pose) const;

    PostProcessParams m_params;
    std::vector<float> m_inputData;
    PoseResult m_outputResult;
};

#endif // POSE_POST_PROCESS_H 