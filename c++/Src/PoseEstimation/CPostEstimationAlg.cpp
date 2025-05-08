#include "CPoseEstimationAlg.h"
#include "log.h"
#include "CSelfAlgParam.h"
#include <cstdio>
using namespace std;
/**
 * 获取当前ms UTC时间
 * 参数：
 * 返回值：ms UTC时间
 */
int64_t CPoseEstimationAlg::getTimeStamp()
{
    std::chrono::time_point<std::chrono::system_clock, std::chrono::milliseconds> tp =
        std::chrono::time_point_cast<std::chrono::milliseconds>(std::chrono::system_clock::now());

    auto tmp = std::chrono::duration_cast<std::chrono::milliseconds>(tp.time_since_epoch());
    return tmp.count();
}

std::string CPoseEstimationAlg::getVersion()
{
    return "AlgLib PoseEstimationAlg V1.0";
}

/**
 * 构造函数
 * 参数：
 *   p_strExePath - 算法库执行路径
 * 功能：
 *   初始化输出路径和日志记录版本信息
 */
CPoseEstimationAlg::CPoseEstimationAlg(const std::string& p_strExePath):m_strOutPath(p_strExePath)
{   
    m_strOutPath = p_strExePath;
    LOG(INFO)<<"CPoseEstimationAlg Version: "<< getVersion() <<std::endl; 
}

/**
 * 析构函数
 * 功能：
 *   用于清理在CPoseEstimationAlg生命周期中分配的资源
 */
CPoseEstimationAlg::~CPoseEstimationAlg()
{

}

/**
 * 初始化算法
 * 参数：
 *   p_pAlgParam - 算法参数
 *   p_algCallback - 算法回调函数
 *   p_handle - 回调函数的用户数据
 * 返回值：
 *   bool - 初始化成功返回true，否则返回false
 * 功能：
 *   设置算法参数，初始化算法选择和数据类型
 */
bool CPoseEstimationAlg::initAlgorithm(CSelfAlgParam* p_pAlgParam, const AlgCallback& p_algCallback, void* p_handle)
{   
    if (!p_pAlgParam){
        LOG(ERROR)<< "CPoseEstimationAlg::initAlgorithm ---- End >>> Failed : No PoseEstimation Data."<<std::endl;
        return false;
    }

    m_stSelfPoseEstimationAlgParam = *p_pAlgParam;
    // 回调函数相关参数
    m_callback = p_algCallback;
    m_handle = p_handle;

    // 算法选择
    std::string s_configPath = m_stSelfPoseEstimationAlgParam.m_strRootPath+ "Configs/Alg/Alg.yaml";
    m_pMakeAlgorithms = std::make_shared<CMakePoseEstimationAlg_strategy>(s_configPath);
    m_pMakeAlgorithms->makeAlgs();

    // 初始化数据指针及初始化各子模块
    m_pCompositeAlgorithms = std::make_shared<CCompositeAlgs>(m_pMakeAlgorithms);
    m_pCompositeAlgorithms->init(&m_stSelfMultiModalFusionAlgParam);

    LOG(INFO)<< "CPoseEstimationAlg::initAlgorithm ---- finish!"<<std::endl;
    return true;    
}

/**
 * 运行算法
 * 参数：
 *   p_pSrcData - 源数据指针
 * 返回值：
 *   void* - 算法处理结果
 * 功能：
 *   执行算法处理流程，包括多模态图像数据处理、算法执行、结果处理和回调函数调用
 */
void CPoseEstimationAlg::runAlgorithm(void* p_pSrcData)
{   
    LOG(INFO)<< "CPoseEstimationAlg::runAlgorithm ---- start."<<std::endl;

    if (!p_pSrcData){
        LOG(ERROR)<< "CPoseEstimationAlg::runAlgorithm ---- The MultiModalFusion processing incoming data is empty"<<std::endl;
        return;                         // 如果多模态图像输入数据为空 ，直接跳过多模态图像算法
    }
    int64_t nStartTimeStamp = getTimeStamp();

    // 1.图像数据处理
    m_multiModalSrcData = *(static_cast<CMultiModalSrcData *>(p_pSrcData));   

    // 2. 数据传输至算法内部
    CCommonDataPtr pCommonData = std::make_shared<ICommonData>();
    pCommonData->m_nVideoStartTime = nStartTimeStamp;
    pCommonData->m_multiModalSrcData = m_multiModalSrcData;
    m_pCompositeAlgorithms->setCommonAllData(pCommonData);

    // 3.执行算法操作
    m_pCompositeAlgorithms->execute();

    // 4.对算法输出结果进行处理
    m_stMultiModalFusionAlgResult = static_cast<CAlgResult> (pCommonData->m_postProcessResult);
    
    // 5.图像感知结果信息设置
    int64_t nEndTimeStamp = getTimeStamp();       
    auto nLatency = nEndTimeStamp - nStartTimeStamp;     // 图像检测算法耗时

    
    LOG(INFO) <<"runAlgorithm ---- End >>> All Time : " << nLatency << " ms." << std::endl;
    
    // 5. 通过回调函数返回结果
    m_callback(m_stMultiModalFusionAlgResult, m_handle); 
    LOG(INFO)<< "CPoseEstimationAlg::runAlgorithm once status:  over."<<std::endl;
}
