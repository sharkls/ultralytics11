#include "ObjectClassifyActivity.h"

// 启动activity方法二：使用REGISTER_ACTIVITY进行注册，然后通过activity_exec命令将activity启动，可传入参数（-c activity配置文件路径）
// REGISTER_ACTIVITY(ObjectClassifyActivity)

ObjectClassifyActivity::ObjectClassifyActivity()
{
}

ObjectClassifyActivity::~ObjectClassifyActivity() 
{
    PauseClear();
}

bool ObjectClassifyActivity::Init()
{
    // 1. 读取私有配置文件内容
    TopicConfig topic_config;
    TINFO << config_file_path_;
    if (!GetProtoConfig(&topic_config))
    {
        TINFO << "configure file parse failed";
        return false;
    }

    // 2. 创建camera_merged_topic对应的reader，接收数据
    eprosima::fastdds::dds::TypeSupport deal_data_type(new CAlgResultPubSubType());
    if (!node_->AddTopic(topic_config.camera_merged_topic(), deal_data_type))
    {
        return false;
    }
    reader_ = node_->CreateReader<CAlgResult>(topic_config.camera_merged_topic(),
        std::bind(&ObjectClassifyActivity::ReadCallbackFunc,
            this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4), this);
    reader_->Init();

    // 3. 创建multi_modal_fusion_result_topic对应的writer，发送数据
    eprosima::fastdds::dds::TypeSupport writer_data_type(new CAlgResultPubSubType());
    if (!node_->AddTopic(topic_config.pose_estimation_v2_result_topic(), writer_data_type))
    {
        return false;
    }
    writer_ = node_->CreateWriter(topic_config.pose_estimation_v2_result_topic());
    writer_->Init();

    // 4. 初始化消息队列
    camera_merged_data_deque_.SetMaxSize(10);  // 通过SetMaxSize方法设置队列长度
    pose_estimation_result_deque_.SetMaxSize(10);

    // 5.实例化算法节点并初始化
    std::string root_path = GetRootPath();
    root_path_ = root_path + "/Output/";
    alg_param_.m_strRootPath = root_path_;
    LOG(INFO) << "root_path: " << root_path_;
    pose_estimation_alg_ = CreateObjectClassifyAlgObj(root_path_);
    pose_estimation_alg_->initAlgorithm(&alg_param_, std::bind(&ObjectClassifyActivity::GetObjectClassifyResultResponseMessageCallbackFunc, this, std::placeholders::_1, std::placeholders::_2), nullptr);
    return true;
}

// 处理camera_merged_topic中的数据
void ObjectClassifyActivity::ReadCallbackFunc(const CAlgResult &message,
        void *data_handle, std::string node_name, std::string topic_name)
{   
    startTimeStamp_ = GetTimeStamp();
    if(startTimeStamp_ - count_time_ > 1000)
    {
        count_time_ = startTimeStamp_;
        LOG(INFO) << "ObjectClassifyActivity FPS =================================: " << count_;
        count_ = 0;
    }
    count_++;

    std::shared_ptr<CAlgResult> message_ptr = std::make_shared<CAlgResult>(message);
    camera_merged_data_deque_.PushBack(message_ptr);
}

// 处理算法返回的感知数据
void ObjectClassifyActivity::GetObjectClassifyResultResponseMessageCallbackFunc(const CAlgResult& res_message, void* data_handle)
{
    std::shared_ptr<CAlgResult> res_ptr = std::make_shared<CAlgResult>(res_message);
    pose_estimation_result_deque_.PushBack(res_ptr);
    if(res_ptr->vecFrameResult().size() > 0)
    {   
        LOG(INFO) << "ObjectClassifyv2 Result  FrameResult size: " << res_ptr->vecFrameResult().size();
        LOG(INFO) << "ObjectClassifyv2 Result size: " << res_ptr->vecFrameResult()[0].vecObjectResult().size();
    }
    else{
        LOG(INFO) << "ObjectClassifyv2 Result size: 0";
    }
}


void ObjectClassifyActivity::Start() 
{
    TINFO << "ObjectClassifyActivity running";
    std::lock_guard<std::mutex> lock(thread_mutex_);
    if (is_running_) return;
    is_running_ = true;
    message_producer_thread_ = std::make_unique<std::thread>(&ObjectClassifyActivity::MessageProducerThreadFunc, this);
    message_consumer_thread_ = std::make_unique<std::thread>(&ObjectClassifyActivity::MessageConsumerThreadFunc, this);
}

// 当收到master节点的PAUSE指令，则执行一些清除工作，比如delete线程
void ObjectClassifyActivity::PauseClear() 
{
    std::lock_guard<std::mutex> lock(thread_mutex_);
    if (!is_running_) return;
    is_running_ = false;
    if (message_producer_thread_ && message_producer_thread_->joinable()) {
        message_producer_thread_->join();
        message_producer_thread_.reset();
    }
    if (message_consumer_thread_ && message_consumer_thread_->joinable()) {
        message_consumer_thread_->join();
        message_consumer_thread_.reset();
    }
}

void ObjectClassifyActivity::MessageProducerThreadFunc()
{
    // 向CounterTopic中循环发送数据
    // LOG(INFO) << "is_running : " << is_running_;
    // is_running_是线程结束的标志位，通过master的指令进行控制
    while (is_running_.load())
    {   
        // LOG(INFO) << "Waiting ...";
        std::shared_ptr<CAlgResult> message;
        if (!pose_estimation_result_deque_.PopFront(message, 1))
        {
            // std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            continue;
        }
        // LOG(INFO) << "Get CAlgResult!!";
        endTimeStamp_ = GetTimeStamp();

        if(message->vecFrameResult().size() > 0)
        {   
            LOG(INFO) << "ObjectClassifyv2Act Publish Result FrameResult size: " << message->vecFrameResult().size();
            LOG(INFO) << "ObjectClassifyv2Act Publish Result size: " << message->vecFrameResult()[0].vecObjectResult().size();
        }
        else{
            LOG(INFO) << "ObjectClassifyv2Act Publish Result size: 0";
        }

        LOG(INFO) << "ObjectClassifyActivity MessageProducerThreadFunc time:----------------------------------- " << endTimeStamp_ - startTimeStamp_;
        LOG(INFO) << "ObjectClassifyActivity TimeStamp : " << message->lTimeStamp();
        writer_->SendMessage((void*)message.get());
        // std::this_thread::sleep_for(std::chrono::milliseconds(2000));
    }
}

void ObjectClassifyActivity::MessageConsumerThreadFunc()
{
    while (is_running_.load())
    {      
        // 输入到融合算法的数据
        std::shared_ptr<CAlgResult> l_pMultiModalSrcData = std::make_shared<CAlgResult>();             
        if (!camera_merged_data_deque_.PopFront(l_pMultiModalSrcData, 1))
        {
            // std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            continue;
        }
        // 执行融合算法
        // LOG(INFO) << "ObjectClassifyActivity Algorithm InputData get !!! ---------- CMultiModalSrcData : " << l_pMultiModalSrcData->vecVideoSrcData().size();
        pose_estimation_alg_->runAlgorithm(l_pMultiModalSrcData.get());
        // std::this_thread::sleep_for(std::chrono::milliseconds(2000));
    }
}

// 启动activity方法1：写main函数，可通过命令行传参，int main(int argc, char*argv[]),需自行解析
int main()
{   
    FLAGS_alsologtostderr = true;
    google::InitGoogleLogging("ObjectClassifyActivity");

    std::string activity_info_path = "../../../ddsproject-example/activities/conf/ObjectClassifyActivity.info";
    ActivityInfo activity_info;
    // 解析ObjectClassifyActivity配置文件
    GetProtoFromFile(activity_info_path, &activity_info);

    // 设置当前进程名，用于在日志中打印
    SetName("ObjectClassifyActivity");

    ObjectClassifyActivity *activity = new ObjectClassifyActivity();
    // 初始化操作：添加topic，以及其他成员变量的初始化操作
    if (activity->Initialize(activity_info))
    {
        activity->Run();
    }
    delete activity;
    return 0;
}
