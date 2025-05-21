#include "ObjectLocationActivity.h"

// 启动activity方法二：使用REGISTER_ACTIVITY进行注册，然后通过activity_exec命令将activity启动，可传入参数（-c activity配置文件路径）
// REGISTER_ACTIVITY(ObjectLocationActivity)

ObjectLocationActivity::ObjectLocationActivity()
{
}

ObjectLocationActivity::~ObjectLocationActivity()
{
    if ((message_producer_thread_ != nullptr) && (message_producer_thread_->joinable()))
    {
        message_producer_thread_->join();
        delete message_producer_thread_;
        message_producer_thread_ = nullptr;
    }
}

bool ObjectLocationActivity::Init()
{
    // 1. 读取私有配置文件内容
    TopicConfig topic_config;
    TINFO << config_file_path_;
    if (!GetProtoConfig(&topic_config))
    {
        TINFO << "configure file parse failed";
        return false;
    }

    // 2. 创建multi_modal_fusion_result_topic对应的reader，接收数据
    eprosima::fastdds::dds::TypeSupport deal_multi_modal_data_type(new CAlgResultPubSubType());
    if (!node_->AddTopic(topic_config.multi_modal_fusion_result_topic(), deal_multi_modal_data_type))
    {
        return false;
    }
    reader_multi_modal_fusion_result_ = node_->CreateReader<CAlgResult>(topic_config.multi_modal_fusion_result_topic(),
        std::bind(&ObjectLocationActivity::ReadMultiModalFusionCallbackFunc,
            this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4), this);
    reader_multi_modal_fusion_result_->Init();

    // 2. 创建pose_estimation_result_topic对应的reader，接收数据
    eprosima::fastdds::dds::TypeSupport deal_pose_estimation_data_type(new CAlgResultPubSubType());
    if (!node_->AddTopic(topic_config.pose_estimation_result_topic(), deal_pose_estimation_data_type))
    {
        return false;
    }
    reader_pose_estimation_result_ = node_->CreateReader<CAlgResult>(topic_config.pose_estimation_result_topic(),
        std::bind(&ObjectLocationActivity::ReadPoseEstimationCallbackFunc,
            this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4), this);
    reader_pose_estimation_result_->Init();

    // 3. 创建multi_modal_fusion_result_topic对应的writer，发送数据
    eprosima::fastdds::dds::TypeSupport writer_data_type(new CAlgResultPubSubType());
    if (!node_->AddTopic(topic_config.object_location_result_topic(), writer_data_type))
    {
        return false;
    }
    writer_object_location_result_ = node_->CreateWriter(topic_config.object_location_result_topic());
    writer_object_location_result_->Init();

    // 4. 初始化消息队列
    multi_modal_fusion_result_deque_.SetMaxSize(10);  // 通过SetMaxSize方法设置队列长度
    pose_estimation_result_deque_.SetMaxSize(10);
    object_location_result_deque_.SetMaxSize(10);

    // 5.实例化算法节点并初始化
    std::string root_path = GetRootPath();
    root_path_ = root_path + "/Output/";
    alg_param_.m_strRootPath = root_path_;
    LOG(INFO) << "root_path: " << root_path_;
    object_location_alg_ = CreateObjectLocationAlgObj(root_path_);
    object_location_alg_->initAlgorithm(&alg_param_, std::bind(&ObjectLocationActivity::GetObjectLocationResultResponseMessageCallbackFunc, this, std::placeholders::_1, std::placeholders::_2), nullptr);
    return true;
}

// 处理multi_modal_fusion_result_topic中的数据
void ObjectLocationActivity::ReadMultiModalFusionCallbackFunc(const CAlgResult &message,
        void *data_handle, std::string node_name, std::string topic_name)
{
    std::shared_ptr<CAlgResult> message_ptr = std::make_shared<CAlgResult>(message);
    multi_modal_fusion_result_deque_.PushBack(message_ptr);
}

// 处理multi_modal_fusion_result_topic中的数据
void ObjectLocationActivity::ReadPoseEstimationCallbackFunc(const CAlgResult &message,
        void *data_handle, std::string node_name, std::string topic_name)
{
    std::shared_ptr<CAlgResult> message_ptr = std::make_shared<CAlgResult>(message);
    object_location_result_deque_.PushBack(message_ptr);
}

// 处理算法返回的感知数据
void ObjectLocationActivity::GetObjectLocationResultResponseMessageCallbackFunc(const CAlgResult& res_message, void* data_handle)
{
    std::shared_ptr<CAlgResult> res_ptr = std::make_shared<CAlgResult>(res_message);
    pose_estimation_result_deque_.PushBack(res_ptr);
}

void ObjectLocationActivity::Start()
{
    TINFO << "ObjectLocationActivity running";
    message_producer_thread_ = new std::thread(&ObjectLocationActivity::MessageProducerThreadFunc, this);
    message_consumer_thread_ = new std::thread(&ObjectLocationActivity::MessageConsumerThreadFunc, this);

}

// 当收到master节点的PAUSE指令，则执行一些清除工作，比如delete线程
void ObjectLocationActivity::PauseClear()
{   
    // 停止消息生产者线程
    if ((message_producer_thread_ != nullptr) && (message_producer_thread_->joinable()))
    {
        message_producer_thread_->join();
        delete message_producer_thread_;
        message_producer_thread_ = nullptr;
    }

    // 停止消息消费者线程
    if ((message_consumer_thread_ != nullptr) && (message_consumer_thread_->joinable()))
    {
        message_consumer_thread_->join();
        delete message_consumer_thread_;
        message_consumer_thread_ = nullptr;
    }

}

void ObjectLocationActivity::MessageProducerThreadFunc()
{
    // 向CounterTopic中循环发送数据
    // is_running_是线程结束的标志位，通过master的指令进行控制
    while (is_running_.load())
    {
        std::shared_ptr<CAlgResult> message;
        if (!pose_estimation_result_deque_.PopFront(message, 1))
        {
            // std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            continue;
        }
        writer_object_location_result_->SendMessage((void*)message.get());
        // std::this_thread::sleep_for(std::chrono::milliseconds(2000));
    }
}

void ObjectLocationActivity::MessageConsumerThreadFunc()
{
    // 输入到融合算法的数据
    std::shared_ptr<CAlgResult> l_pMultiModalResult = std::make_shared<CAlgResult>(); 
    std::shared_ptr<CAlgResult> l_pPoseEstimationResult = std::make_shared<CAlgResult>(); 
    std::shared_ptr<CAlgResult> l_pObjectLocationInputData = std::make_shared<CAlgResult>(); 

    while (is_running_.load())
    {      
        // 清空之前的数据
        l_pMultiModalResult->vecFrameResult().clear();
        l_pPoseEstimationResult->vecFrameResult().clear();
        l_pObjectLocationInputData->vecFrameResult().clear();

        // 获取多模态融合结果
        if (!multi_modal_fusion_result_deque_.PopFront(l_pMultiModalResult, 1))
        {
            continue;
        }

        // 检查多模态融合结果是否有效
        if (l_pMultiModalResult->vecFrameResult().empty())
        {
            LOG(WARNING) << "Empty multi-modal fusion result";
            continue;
        }

        // 深拷贝多模态融合结果
        *l_pObjectLocationInputData = *l_pMultiModalResult;

        // 获取多模态数据的时间戳
        auto multiModalTime = l_pMultiModalResult->vecFrameResult()[0].mapTimeStamp()[TIMESTAMP_TIME_MATCH];
        bool foundMatch = false;

        // 尝试匹配姿态估计结果
        while(pose_estimation_result_deque_.PopFront(l_pPoseEstimationResult, 100))
        {   
            // 检查姿态估计结果是否有效
            if (l_pPoseEstimationResult->vecFrameResult().empty())
            {
                LOG(WARNING) << "Empty pose estimation result";
                continue;
            }

            auto poseTime = l_pPoseEstimationResult->vecFrameResult()[0].mapTimeStamp()[TIMESTAMP_TIME_MATCH];
            
            if (multiModalTime == poseTime)
            {
                // 时间戳匹配，添加姿态估计结果
                l_pObjectLocationInputData->vecFrameResult().push_back(l_pPoseEstimationResult->vecFrameResult()[0]);
                foundMatch = true;
                break;
            }
            else if (multiModalTime < poseTime)
            {
                // 当前姿态估计结果时间戳更大，放回队列
                pose_estimation_result_deque_.PushBack(l_pPoseEstimationResult);
                break;
            }
            // 如果时间戳更小，继续循环查找匹配项
        }

        // 记录是否找到匹配的姿态估计结果
        if (!foundMatch)
        {
            LOG(WARNING) << "No matching pose estimation result found for timestamp: " << multiModalTime;
        }

        // 执行目标定位算法
        try
        {
            // 直接调用runAlgorithm，不检查返回值
            object_location_alg_->runAlgorithm(l_pObjectLocationInputData.get());
            LOG(INFO) << "ObjectLocationActivity Algorithm InputData processed successfully. "
                    << "Frame count: " << l_pObjectLocationInputData->vecFrameResult().size();
        }
        catch (const std::exception& e)
        {
            LOG(ERROR) << "Exception in object location algorithm: " << e.what();
            continue;
        }
        // std::this_thread::sleep_for(std::chrono::milliseconds(2000));
    }
}

// 启动activity方法1：写main函数，可通过命令行传参，int main(int argc, char*argv[]),需自行解析
int main()
{
    std::string activity_info_path = "../../../ddsproject-example/activities/conf/ObjectLocationActivity.info";
    ActivityInfo activity_info;
    // 解析ObjectLocationActivity配置文件
    GetProtoFromFile(activity_info_path, &activity_info);

    // 设置当前进程名，用于在日志中打印
    SetName("ObjectLocationActivity");

    ObjectLocationActivity *activity = new ObjectLocationActivity();
    // 初始化操作：添加topic，以及其他成员变量的初始化操作
    if (activity->Initialize(activity_info))
    {
        activity->Run();
    }
    delete activity;
    return 0;
}
