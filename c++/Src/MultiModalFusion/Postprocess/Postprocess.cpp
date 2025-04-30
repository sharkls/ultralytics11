#include "Postprocess.h"
#include "log.h"

/**
 * PostProcess构造函数
 * 功能：
 *   创建PostProcess对象时记录启动状态
 */
PostProcess::PostProcess()
{   
    LOG(INFO) << "PostProcess :: PostProcess status: Started." << std::endl;
}

/**
 * PostProcess析构函数
 * 功能：
 *   销毁PostProcess对象时执行的清理操作
 */
PostProcess::~PostProcess()
{
}

/**
 * 初始化多模态融合后处理部分参数
 * 参数：
 *   p_pAlgParam - 算法参数
 * 功能：
 *   初始化PostProcess对象的算法参数
 */
void PostProcess::init(CSelfAlgParam* p_pAlgParam) 
{   
    LOG(INFO) << "PostProcess :: init status: start." << std::endl;
    if (!p_pAlgParam) {
        LOG(ERROR) << "The PostProcess InitAlgorithm incoming parameter is empty" << std::endl;
        return;
    }

    LOG(INFO) << "PostProcess :: init status: finish!" << std::endl;
}

/**
 * 执行多模态融合后处理
 * 功能：
 *   获取输入数据，执行后处理逻辑，并设置输出数据
 */
void PostProcess::execute()
{
    LOG(INFO) << "PostProcess::execute status: start." << std::endl;

    // 获取输入数据
    if (!m_pCommonData) {
        LOG(ERROR) << "Common data is not set" << std::endl;
        return;
    }

    // 获取多模态融合推理结果
    m_postProcessResult = m_pCommonData->m_multiModalFusionAlgResult;
    if (m_postProcessResult.vecFrameResult().empty()) {
        LOG(INFO) << "No detection results to process" << std::endl;
        return;
    }

    // 设置输出数据
    m_pCommonData->m_postProcessResult = m_postProcessResult;

    LOG(INFO) << "PostProcess::execute status: finish!" << std::endl;
}