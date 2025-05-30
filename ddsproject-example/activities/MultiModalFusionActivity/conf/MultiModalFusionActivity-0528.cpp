#include "MultiModalFusionActivity.h"

// 启动activity方法二：使用REGISTER_ACTIVITY进行注册，然后通过activity_exec命令将activity启动，可传入参数（-c activity配置文件路径）
// REGISTER_ACTIVITY(MultiModalFusionActivity)

MultiModalFusionActivity::MultiModalFusionActivity()
{
    // 初始化单应性矩阵
    homography_ = {
        1.666317f, -0.050601f, 75.329809f,
        0.021197f, 1.554738f, -48.619249f,
        0.000110f, -0.000091f, 1.000000f
    };
}

MultiModalFusionActivity::~MultiModalFusionActivity()
{
    PauseClear();
}


bool MultiModalFusionActivity::Init()
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
    eprosima::fastdds::dds::TypeSupport deal_data_type(new CMultiModalSrcDataPubSubType());
    if (!node_->AddTopic(topic_config.camera_merged_topic(), deal_data_type))
    {
        return false;
    }
    reader_ = node_->CreateReader<CMultiModalSrcData>(topic_config.camera_merged_topic(),
        std::bind(&MultiModalFusionActivity::ReadCallbackFunc,
            this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4), this);
    reader_->Init();

    // 3. 创建multi_modal_fusion_result_topic对应的writer，发送数据
    eprosima::fastdds::dds::TypeSupport writer_data_type(new CAlgResultPubSubType());
    if (!node_->AddTopic(topic_config.multi_modal_fusion_result_topic(), writer_data_type))
    {
        return false;
    }
    writer_ = node_->CreateWriter(topic_config.multi_modal_fusion_result_topic());
    writer_->Init();

    // 4. 初始化消息队列
    camera_merged_data_deque_.SetMaxSize(10);  // 通过SetMaxSize方法设置队列长度
    multi_modal_fusion_result_deque_.SetMaxSize(10);

    // 5.实例化算法节点并初始化
    std::string root_path = GetRootPath();
    root_path_ = root_path + "/Output/";
    alg_param_.m_strRootPath = root_path_;
    LOG(INFO) << "root_path: " << root_path_;
    multi_modal_fusion_alg_ = CreateMultiModalFusionAlgObj(root_path_);
    multi_modal_fusion_alg_->initAlgorithm(&alg_param_, std::bind(&MultiModalFusionActivity::GetMultiModalFusionResultResponseMessageCallbackFunc, this, std::placeholders::_1, std::placeholders::_2), nullptr);
    return true;
}

// 处理camera_merged_topic中的数据
void MultiModalFusionActivity::ReadCallbackFunc(const CMultiModalSrcData &message,
        void *data_handle, std::string node_name, std::string topic_name)
{   
    startTimeStamp_ = GetTimeStamp();
    if(startTimeStamp_ - count_time_ > 1000)
    {
        count_time_ = startTimeStamp_;
        LOG(INFO) << "MultiModalFusionActivity FPS =================================: " << count_;
        count_ = 0;
    }
    count_++;
    std::shared_ptr<CMultiModalSrcData> message_ptr = std::make_shared<CMultiModalSrcData>(message);
    camera_merged_data_deque_.PushBack(message_ptr);
}

// 处理算法返回的感知数据
void MultiModalFusionActivity::GetMultiModalFusionResultResponseMessageCallbackFunc(const CAlgResult& res_message, void* data_handle)
{
    std::shared_ptr<CAlgResult> res_ptr = std::make_shared<CAlgResult>(res_message);
    multi_modal_fusion_result_deque_.PushBack(res_ptr);
}

void MultiModalFusionActivity::Start()
{
    std::lock_guard<std::mutex> lock(thread_mutex_);
    if (is_running_) return;
    TINFO << "MultiModalFusionActivity running";
    is_running_ = true;
    message_producer_thread_ = std::make_unique<std::thread>(&MultiModalFusionActivity::MessageProducerThreadFunc, this);
    message_consumer_thread_ = std::make_unique<std::thread>(&MultiModalFusionActivity::MessageConsumerThreadFunc, this);
}

// 当收到master节点的PAUSE指令，则执行一些清除工作，比如delete线程
void MultiModalFusionActivity::PauseClear()
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

void MultiModalFusionActivity::MessageProducerThreadFunc()
{
    // 向CounterTopic中循环发送数据
    // is_running_是线程结束的标志位，通过master的指令进行控制
    while (is_running_.load())
    {
        std::shared_ptr<CAlgResult> message;
        if (!multi_modal_fusion_result_deque_.PopFront(message, 1))
        {
            // std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            continue;
        }
        endTimeStamp_ = GetTimeStamp();

        LOG(INFO) << "MultiModalFusionActivity MessageProducerThreadFunc time:----------------------------------- " << endTimeStamp_ - startTimeStamp_;
        writer_->SendMessage((void*)message.get());
        // std::this_thread::sleep_for(std::chrono::milliseconds(2000));
    }
}

void MultiModalFusionActivity::MessageConsumerThreadFunc()
{
    while (is_running_.load())
    {     
        // 输入到融合算法的数据
        std::shared_ptr<CMultiModalSrcData> l_pMultiModalSrcData = std::make_shared<CMultiModalSrcData>();             
        if (!camera_merged_data_deque_.PopFront(l_pMultiModalSrcData, 1))
        {
            // std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            continue;
        }

        // 设置单应性矩阵
        l_pMultiModalSrcData->vecfHomography(homography_);

        // 执行融合算法
        LOG(INFO) << "MultiModalFusionActivity Algorithm InputData get !!! ---------- CMultiModalSrcData : " << l_pMultiModalSrcData->vecVideoSrcData().size();
        multi_modal_fusion_alg_->runAlgorithm(l_pMultiModalSrcData.get());
        // std::this_thread::sleep_for(std::chrono::milliseconds(2000));
    }
}

// 启动activity方法1：写main函数，可通过命令行传参，int main(int argc, char*argv[]),需自行解析
int main()
{
    std::string activity_info_path = "../../../ddsproject-example/activities/conf/MultiModalFusionActivity.info";
    ActivityInfo activity_info;
    // 解析MultiModalFusionActivity配置文件
    GetProtoFromFile(activity_info_path, &activity_info);

    // 设置当前进程名，用于在日志中打印
    SetName("MultiModalFusionActivity");

    MultiModalFusionActivity *activity = new MultiModalFusionActivity();
    // 初始化操作：添加topic，以及其他成员变量的初始化操作
    if (activity->Initialize(activity_info))
    {
        activity->Run();
    }
    delete activity;
    return 0;
}
