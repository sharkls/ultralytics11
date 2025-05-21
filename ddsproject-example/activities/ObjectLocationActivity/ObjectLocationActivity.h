#include "include/activity/base/activitybase.hpp"

#include <fastdds/dds/topic/TypeSupport.hpp>

// #include "activities/alg_activity/proto/alg_activity.pb.h"
#include "activities/ObjectLocationActivity/proto/ObjectLocationActivity.pb.h"
#include "activities/idl/CMultiModalSrcData/CMultiModalSrcData.h"
#include "activities/idl/CMultiModalSrcData/CMultiModalSrcDataPubSubTypes.h"
#include "activities/idl/CAlgResult/CAlgResult.h"
#include "activities/idl/CAlgResult/CAlgResultPubSubTypes.h"
#include "include/queue/CSafeDataDeque.h"
#include "include/Interface/ExportObjectLocationAlgLib.h"
#include "include/Interface/CSelfAlgParam.h"
#include "include/Common/Functions.h"
#include "include/Common/GlobalContext.h"
#include <fstream>

#include <opencv2/opencv.hpp>
#include <glog/logging.h>

// CounterTopic是话题，test0_acticity向test1_activity发送数据
class ObjectLocationActivity : public ActivityBase
{
public:
    ObjectLocationActivity();
    ~ObjectLocationActivity();

    // 软件处理算法结果的回调函数
    void GetObjectLocationResultResponseMessageCallbackFunc(const CAlgResult& res_message, void* data_handle);

protected:
    // 初始化，读取配置文件
    virtual bool Init() override;
    // 当收到master节点的RUN指令，则执行Start，用于启动线程
    virtual void Start() override;
    // 当收到master节点的PAUSE指令，则执行一些清除工作，比如delete线程
    virtual void PauseClear();

private:
    // 向 Multi_Modal_Fusion_Result_Topic 中发送消息
    void MessageProducerThreadFunc();
    // 处理 Multi_Modal_Fusion_Result_Topic 中的数据
    void MessageConsumerThreadFunc();
    
    // 处理多模态感知Topic结果
    void ReadMultiModalFusionCallbackFunc(const CAlgResult &message,
        void *data_handle, std::string node_name, std::string topic_name);
    // 处理姿态估计Topic结果
    void ReadPoseEstimationCallbackFunc(const CAlgResult &message,
        void *data_handle, std::string node_name, std::string topic_name);


private:
    // DDS 读取和写出节点
    std::shared_ptr<Reader<CAlgResult>> reader_multi_modal_fusion_result_;
    std::shared_ptr<Reader<CAlgResult>> reader_pose_estimation_result_;
    std::shared_ptr<Writer> writer_object_location_result_;

    // 消息队列
    CSafeDataDeque<std::shared_ptr<CAlgResult>> multi_modal_fusion_result_deque_;   // 多模态融合感知结果
    CSafeDataDeque<std::shared_ptr<CAlgResult>> pose_estimation_result_deque_;  // 姿态估计结果
    CSafeDataDeque<std::shared_ptr<CAlgResult>> object_location_result_deque_;  // 多模态融合结果


    // 消息发送和消费线程
    std::thread *message_producer_thread_{nullptr};
    std::thread *message_consumer_thread_{nullptr};
    
    // 算法实例
    IObjectLocationAlg *object_location_alg_{nullptr};
    std::string root_path_;
    CSelfAlgParam alg_param_;
};