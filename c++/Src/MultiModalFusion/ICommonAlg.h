#ifndef ICOMMONALG_MULTIMODALFUSION_H
#define ICOMMONALG_MULTIMODALFUSION_H

#include <opencv2/core/mat.hpp>
#include "log.h"
#include "yaml.h"
#include "CSelfAlgParam.h"
#include "CMultiModalSrcData.h"

class ICommonData {
public:
    ICommonData() = default;
    ~ICommonData() = default;

public:
    // 预处理后的数据结构
    struct PreprocessedData {
        std::vector<float> rgbInput;        // RGB输入数据
        std::vector<float> irInput;         // IR输入数据
        std::array<float, 9> homography;    // 单应性矩阵
        int inputHeight;                    // 输入高度
        int inputWidth;                     // 输入宽度
        float dw = 0.0f;                    // letterbox padding宽
        float dh = 0.0f;                    // letterbox padding高
        float ratio = 1.0f;                 // resize比例
    };

    // video
    int64_t m_nVideoStartTime;
    CMultiModalSrcData m_multiModalSrcData;     // 图像原始数据(软件传输到算法的原始数据)
    PreprocessedData preprocessedData;          // 预处理后的数据，Inference模块的输入
    // void* m_pCpuBuffer;                         // Inference模块的输出，图像后处理算法的输入
    CAlgResult m_multiModalFusionAlgResult;     // Inference模块的输出，后处理算法输入
    CAlgResult m_postProcessResult;             // 图像后处理算法输出

    // // Track
    // int64_t m_nTestStartTime;
    // CAlgResult m_trackSrcData;                  // 跟踪算法原始数据
    // CAlgResult m_trackAlgResult;                // 跟踪算法输出
    // CAlgResult m_pcTrackSrcData;                // 雷达原始数据
    // CAlgResult m_pcTrackAlgResult;              // 雷达算法输出
};

using CCommonDataPtr = std::shared_ptr<ICommonData>;

class ICommonAlg {
public:
    ICommonAlg() = default;
    virtual ~ICommonAlg() = default;

    virtual void init(CSelfAlgParam* p_pAlgParam) = 0;
    virtual void execute() = 0;

    virtual void setCommonData(CCommonDataPtr p_commonData) = 0;
    virtual CCommonDataPtr getCommonData() = 0;

public:
    CCommonDataPtr m_pCommonData;
};

using ICommonAlgPtr = std::shared_ptr<ICommonAlg>;

#endif //ICOMMONALG_MULTIMODALFUSION_H