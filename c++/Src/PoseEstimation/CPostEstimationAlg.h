/*******************************************************
 文件名：CPoseEstimationAlg.h
 作者：
 描述：姿态估计算法接口实现，用于姿态估计算法的运行及结果数据处理
 版本：v1.0
 日期：2025-05-08
 *******************************************************/

#pragma once


#include "ExportPoseEstimationAlgLib.h"
#include "CAlgResult.h"
#include "CMultiModalSrcData.h"
#include "CSelfAlgParam.h"
#include <iostream>
#include <vector>
#include <queue>
#include <thread>

#include <fstream>
#include "IMakeAlg.h"
#include "ICommonAlg.h"
#include "CMakeAlgs.h"
#include "CCompositeAlgs.h"
#include "GlobalContext.h"
// #include "CSafeDataDeque.h"

class CPoseEstimationAlg :public IPoseEstimationAlg
{
public:
    CPoseEstimationAlg(const std::string& p_strExePath);
    virtual ~CPoseEstimationAlg();

    //初始化算法接口对象，内部主要处理只需初始化一次的操作，比如模型加载之类的，成功返回true，失败返回false
    bool initAlgorithm(CSelfAlgParam* p_pAlgParam, const AlgCallback& p_algCallback, void* p_handle);

    //执行算法函数，传入原始数据体，算法执行成功返回处理后的数据或者检测结果（由算法类型而定），失败返回nullptr
    //参数1：数据源，目前约定传入 CMultiModalSrcData* 类型数据 
    //返回值：算法处理后的数据空间地址，根据算法类型不同有些差异，具体如下：
    void runAlgorithm(void* p_pSrcData);

    std::string getVersion();
    
    int64_t getTimeStamp();

    //  回调函数
    AlgCallback m_callback;
    void* m_handle;
    
    IMakeAlgPtr m_pMakeAlgorithms;
    CCompositeAlgsPtr m_pCompositeAlgorithms;
    
    std::string m_strOutPath;
    CAlgResult m_stMultiModalFusionAlgResult;
    CSelfAlgParam m_stSelfMultiModalFusionAlgParam;

    CMultiModalSrcData m_multiModalSrcData;
    CCommonDataPtr m_pMultiModalSrcData;
};