/*******************************************************
 文件名：PreProcess.h
 作者：
 描述：触发算法接口实现，用于触发算法预处理的运行及结果数据处理
 版本：v1.0
 日期：2025-04-28
 *******************************************************/

#ifndef MULTIMODALFUSION_PREPROCESS_H
#define MULTIMODALFUSION_PREPROCESS_H

#include <iostream>
#include <vector>
#include <fstream>
#include "ICommonAlg.h"
#include "CAlgResult.h"
#include "CSelfAlgParam.h"
#include <opencv2/opencv.hpp>

class PreProcess : public ICommonAlg
{
public:
    PreProcess();

    PreProcess(const CSelfAlgParam* p_pAlgParam);

    virtual ~PreProcess();

    // 初始化多模态融合预处理部分中参数
    void init(CSelfAlgParam* p_pAlgParam);

    // 执行多模态融合预处理算法
    void execute();

    
    void setCommonData(CCommonDataPtr p_commonData) override
    {
        m_pCommonData = p_commonData;
    }

    CCommonDataPtr getCommonData() override
    {
        return m_pCommonData;
    }

private:
    int m_nInputHeight = 0;
    int m_nInputWidth = 0;

    std::tuple<std::vector<float>, std::vector<float>, std::array<float, 9>>
    preprocessMultiModalData(const cv::Mat& p_rgbImg, 
                            const cv::Mat& p_irImg,
                            const std::array<float, 9>& p_homographyMatrix);
};

#endif //MULTIMODALFUSION_PREPROCESS_H