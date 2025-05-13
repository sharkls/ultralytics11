/*******************************************************
 文件名：ExportMultiModalFusionAlgLib.h
 作者：
 描述：多模态融合算法库的算法接口类导出函数头文件
 版本：v1.0
 日期：2024-04-27
 *******************************************************/
#pragma once
#include <string>

#include "CSelfAlgParam.h"

{
struct IMultiModalFusionAlg
    IMultiModalFusionAlg(){};
    virtual ~IMultiModalFusionAlg(){};

    //初始化算法接口对象，内部主要处理只需初始化一次的操作，比如模型加载之类的，成功返回true，失败返回false
    virtual bool initAlgorithm(CSelfAlgParam* p_pAlgParam,  const AlgCallback& alg_cb, void* hd)   = 0;

    //执行算法函数，传入原始数据体，算法执行成功返回处理后的数据或者检测结果（由算法类型而定），失败返回nullptr
    virtual void runAlgorithm(void* p_pSrcData)  = 0;
};

extern "C" __attribute__ ((visibility("default"))) IMultiModalFusionAlg*   CreateMultiModalFusionAlgObj(const std::string& p_strExePath);