/*******************************************************
 文件名：PostProcess.h
 作者：
 描述：触发算法接口实现，用于触发算法预处理的运行及结果数据处理
 版本：v1.0
 日期：2024-1-11
 *******************************************************/

#ifndef MULTIMODALFUSION_POSTPROCESS_H
#define MULTIMODALFUSION_POSTPROCESS_H

#include <iostream>
#include <vector>
#include <fstream>
#include "ICommonAlg.h"
#include "CAlgResult.h"
#include "CSelfAlgParam.h"

class PostProcess : public ICommonAlg
{
public:
    PostProcess();

    PostProcess(const CSelfAlgParam* p_pAlgParam);

    virtual ~PostProcess();

    // 初始化视频检测预处理部分中参数
    void init(CSelfAlgParam* p_pAlgParam);

    // 执行多模态融合后处理算法
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
    CAlgResult m_postProcessResult;
};

#endif //MULTIMODALFUSION_PREPROCESS_H
