/*******************************************************
 文件名：PosePostProcess.cpp
 作者：
 描述：姿态估计后处理模块实现
 版本：v1.0
 日期：2024-03-21
 *******************************************************/

#include "PosePostProcess.h"
#include "../../Factory/ModuleFactory.h"
#include <iostream>

// 注册模块
REGISTER_MODULE(PosePostProcess, PosePostProcess)

PosePostProcess::PosePostProcess()
{
    m_outputResult.keypoints.resize(m_params.numKeypoints);
}

PosePostProcess::~PosePostProcess()
{
}

bool PosePostProcess::init(CSelfAlgParam* p_pAlgParam)
{
    if (p_pAlgParam) {
        // 从配置参数中读取后处理参数
        // TODO: 实现参数读取逻辑
    }
    return true;
}

void PosePostProcess::setInput(void* input)
{
    if (!input) {
        std::cerr << "Input is null" << std::endl;
        return;
    }
    
    float* data = static_cast<float*>(input);
    size_t size = sizeof(data) / sizeof(float);
    m_inputData.assign(data, data + size);
}

void* PosePostProcess::getOutput()
{
    return &m_outputResult;
}

void* PosePostProcess::execute()
{
    if (m_inputData.empty()) {
        std::cerr << "Input data is empty" << std::endl;
        return nullptr;
    }

    try {
        // 处理关键点数据
        processKeypoints(m_inputData.data(), m_inputData.size());

        // 验证姿态结果
        if (!isValidPose(m_outputResult)) {
            std::cerr << "Invalid pose detected" << std::endl;
            return nullptr;
        }

        return &m_outputResult;
    }
    catch (const std::exception& e) {
        std::cerr << "Post-processing failed: " << e.what() << std::endl;
        return nullptr;
    }
}

void PosePostProcess::processKeypoints(const float* data, int dataSize)
{
    // 假设输入数据格式为 [x1, y1, conf1, x2, y2, conf2, ...]
    for (int i = 0; i < m_params.numKeypoints; ++i) {
        int base_idx = i * 3;
        if (base_idx + 2 < dataSize) {
            m_outputResult.keypoints[i].x = data[base_idx];
            m_outputResult.keypoints[i].y = data[base_idx + 1];
            m_outputResult.keypoints[i].confidence = data[base_idx + 2];
        }
    }

    // 计算整体姿态得分
    float total_confidence = 0.0f;
    int valid_points = 0;
    
    for (const auto& kp : m_outputResult.keypoints) {
        if (kp.confidence > m_params.confidenceThreshold) {
            total_confidence += kp.confidence;
            valid_points++;
        }
    }

    m_outputResult.score = valid_points > 0 ? total_confidence / valid_points : 0.0f;
}

bool PosePostProcess::isValidPose(const PoseResult& pose) const
{
    // 检查是否有足够的关键点被检测到
    int valid_points = 0;
    for (const auto& kp : pose.keypoints) {
        if (kp.confidence > m_params.confidenceThreshold) {
            valid_points++;
        }
    }

    // 如果有效关键点数量太少，认为姿态无效
    return valid_points >= m_params.numKeypoints / 2;
} 