/*******************************************************
 文件名：Location.cpp
 作者：sharkls
 描述：目标定位预处理模块实现
 版本：v1.0
 日期：2025-05-15
 *******************************************************/

#include "Location.h"
#include "GlobalContext.h"

// 注册模块
REGISTER_MODULE("ObjectLocation", Location, Location)

Location::~Location()
{
}

bool Location::init(void* p_pAlgParam)
{
    LOG(INFO) << "Location::init status: start ";
    // 1. 从配置参数中读取预处理参数
    if (!p_pAlgParam) {
        return false;
    }
    // 2. 参数格式转换
    objectlocation::TaskConfig* taskConfig = static_cast<objectlocation::TaskConfig*>(p_pAlgParam);
    iou_thres_ = taskConfig->iou_thres();
    num_keys_ = taskConfig->num_keys();
    bucket_size_ = taskConfig->bucket_size();
    max_distance_ = taskConfig->max_distance();
    m_config = *taskConfig; 
    LOG(INFO) << "Location::init status: success ";
    return true;
}

void Location::setInput(void* input)
{
    if (!input) {
        LOG(ERROR) << "Input is null";
        return;
    }
    m_inputdata = *static_cast<CAlgResult*>(input);
}

void* Location::getOutput()
{
    return &m_outputdata;
}

void Location::execute()
{   
    LOG(INFO) << "Location::execute status: start ";
    
    // 处理空输入的情况 - 输出空的帧结果以保持frame_id更新
    if (m_inputdata.vecFrameResult().empty()) {
        LOG(WARNING) << "Input data is empty, outputting empty frame to maintain frame_id continuity";
        CFrameResult emptyResult;
        m_outputdata.vecFrameResult().clear();
        m_outputdata.vecFrameResult().push_back(emptyResult);
        LOG(INFO) << "Location::execute: empty input, outputting empty frame for frame_id continuity";
        return;
    }

    try {
        CFrameResult outputResult;
        LOG(INFO) << "Location::execute status: start m_inputdata.vecFrameResult().size(): " << m_inputdata.vecFrameResult().size();

        if (m_inputdata.vecFrameResult().size() == 1) {
            const auto& result = m_inputdata.vecFrameResult()[0];
            // 判断类型，直接填充到输出
            if (result.eDataType() == DATA_TYPE_POSEALG_RESULT || result.eDataType() == DATA_TYPE_OBJECTCLASSIFYALG_RESULT) {
                m_outputdata.vecFrameResult().clear();
                m_outputdata.vecFrameResult().push_back(result);
                LOG(INFO) << "Location::execute: only one result, directly output.";
                return;
            } else {
                LOG(ERROR) << "Unknown data type in vecFrameResult[0]";
                // 即使类型未知，也输出空帧以保持frame_id连续性
                CFrameResult emptyResult;
                m_outputdata.vecFrameResult().clear();
                m_outputdata.vecFrameResult().push_back(emptyResult);
                return;
            }
        } else if (m_inputdata.vecFrameResult().size() >= 2) {
            // 1. 获取多模态感知结果和姿态估计结果
            const auto& objectClassifyResult = m_inputdata.vecFrameResult()[0];
            const auto& poseResult = m_inputdata.vecFrameResult()[1];

            const auto& classify_objs = objectClassifyResult.vecObjectResult();
            const auto& pose_objs = poseResult.vecObjectResult();


            if(classify_objs.size() != pose_objs.size() || classify_objs.size() == 0)
            {   
                LOG(ERROR) << "classify_objs.size() != pose_objs.size() .... cls.size() vs pos.size(): " << classify_objs.size() << " " << pose_objs.size();
                // 即使数据不匹配，也输出空帧以保持frame_id连续性
                CFrameResult emptyResult;
                m_outputdata.vecFrameResult().clear();
                m_outputdata.vecFrameResult().push_back(emptyResult);
                return;
            }
            else
            {
                for(size_t i = 0; i < classify_objs.size(); ++i)
                {
                    const auto& cls_obj = classify_objs[i];
                    const auto& pose_obj = pose_objs[i];
                    CObjectResult merged_obj = cls_obj; // 先拷贝一份

                    // 当姿态估计网络对该目标有输出时，进行深度修正和类别修正
                    if(pose_obj.vecKeypoints().size() > 0) 
                    {   
                        // if(pose_obj.strClass() != "pose_0" && cls_obj.strClass() == "class_2")
                        // {   
                        //     merged_obj.strClass("class_0");
                        // }
                        merged_obj.fDistance(pose_obj.fDistance());
                        LOG(INFO) << "pose_obj.strClass() : " << pose_obj.strClass() << " cls_obj.strClass() : " << cls_obj.strClass()  << " merged_obj.strClass() : " << merged_obj.strClass();
                    }

                    if(merged_obj.strClass() == "class_0")
                    {
                        merged_obj.strClass("0");
                    }
                    else if (merged_obj.strClass() == "class_1")
                    {
                        merged_obj.strClass("1");
                    }
                    else{
                        merged_obj.strClass("2");
                    }
                    LOG(INFO) << "merged_obj.strClass() : " << merged_obj.strClass();
                    outputResult.vecObjectResult().push_back(merged_obj);
                }
            }

            // 8. 填入输出
            m_outputdata.vecFrameResult().clear();
            m_outputdata.vecFrameResult().push_back(outputResult);

            // LOG(INFO) << "Location::execute status: success!";
            // if (status_) {
            //     save_bin(m_outputdata, "./Save_Data/objectlocation/result/processed_objectlocation_preprocess.bin"); // ObjectLocation/Preprocess
            // }
        }
        LOG(INFO) << "Location::execute status: success! m_outputdata.vecFrameResult().size(): " << m_outputdata.vecFrameResult().size();
    }
    catch (const std::exception& e) {
        LOG(ERROR) << "Preprocessing failed: " << e.what();
        return;
    }
}