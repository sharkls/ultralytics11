#include "include/activity/base/activitybase.hpp"

#include <fastdds/dds/topic/TypeSupport.hpp>

// #include "activities/alg_activity/proto/alg_activity.pb.h"
#include "activities/MultiModalFusionActivity/proto/MultiModalFusionActivity.pb.h"
#include "activities/idl/CMultiModalSrcData/CMultiModalSrcData.h"
#include "activities/idl/CMultiModalSrcData/CMultiModalSrcDataPubSubTypes.h"
#include "activities/idl/CAlgResult/CAlgResult.h"
#include "activities/idl/CAlgResult/CAlgResultPubSubTypes.h"
#include "include/queue/CSafeDataDeque.h"
#include "include/Interface/ExportMultiModalFusionAlgLib.h"
#include "include/Interface/CSelfAlgParam.h"
#include "include/Common/Functions.h"
#include <fstream>

#include <opencv2/opencv.hpp>
#include <glog/logging.h>

// CounterTopic是话题，test0_acticity向test1_activity发送数据
class MultiModalFusionActivity : public ActivityBase
{
public:
    MultiModalFusionActivity();
    ~MultiModalFusionActivity();

    // 软件处理算法结果的回调函数
    void GetMultiModalFusionResultResponseMessageCallbackFunc(const CAlgResult& res_message, void* data_handle);

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
    // 处理camera_merged_topic中的数据
    void MessageConsumerThreadFunc();
    
    void ReadCallbackFunc(const CMultiModalSrcData &message,
        void *data_handle, std::string node_name, std::string topic_name);


private:
    // DDS 读取和写出节点
    std::shared_ptr<Reader<CMultiModalSrcData>> reader_;
    std::shared_ptr<Writer> writer_;

    // 消息队列
    CSafeDataDeque<std::shared_ptr<CMultiModalSrcData>> camera_merged_data_deque_;   // 时间同步后的数据
    CSafeDataDeque<std::shared_ptr<CAlgResult>> multi_modal_fusion_result_deque_;  // 多模态融合结果

    // 消息生产线程
    std::thread *message_producer_thread_{nullptr};
    std::thread *message_consumer_thread_{nullptr};

    // 算法实例
    IMultiModalFusionAlg *multi_modal_fusion_alg_{nullptr};
    std::string root_path_;
    CSelfAlgParam alg_param_;
};