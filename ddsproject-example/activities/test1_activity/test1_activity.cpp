#include "include/activity/base/activitybase.hpp"

#include <fastdds/dds/topic/TypeSupport.hpp>

#include "include/queue/data_queue.hpp"
#include "activities/test1_activity/proto/test1_activity.pb.h"
#include "activities/idl/counter_message/counter_message.h"
#include "activities/idl/counter_message/counter_messagePubSubTypes.h"
#include "activities/idl/counter_response_message/counter_response_message.h"
#include "activities/idl/counter_response_message/counter_response_messagePubSubTypes.h"

#include "activities/test1_activity/test_obj.hpp"

class ActivityTest1 : public ActivityBase
{
public:
    ActivityTest1();
    ~ActivityTest1();

protected:
    // 初始化，读取配置文件、其他成员的初始化操作
    virtual bool Init() override;
    // 当收到master节点的RUN指令，则执行Start，用于启动线程
    virtual void Start() override;
    // 当收到master节点的PAUSE指令，则执行一些清除工作，比如delete线程
    virtual void PauseClear();

private:
    // 从CounterTopic中获取CounterMessage的回调函数，将CounterMessage存储到消息队列中
    void ReadCounterMessageCallbackFunc(const CounterMessage &message, void *data_handle, std::string node_name, std::string topic_name);
    // 从CounterMessage消息队列中获取数据，交由TestObj进行处理
    void DealCounterMessageThreadFunc();
    // 从CounterResponseMessage消息队列中获取属具，并通过CounterResponseTopic发送给test2_activity
    void SendCounterResponseMessageThreadFunc();
    // TestObj所需的回调函数，获取TestObj处理CounterMessage之后的CounterResponseMesssage数据
    void GetCounterResponseMessageCallbackFunc(const CounterResponseMessage& res_message, void* data_handle);

private:
    // CounterMessage消息队列
    DataQueue<CounterMessage>* counter_message_queue_;
    // 处理CounterMessage消息队列中数据的线程
    std::thread* counter_message_customer_thread_{nullptr};

    // CounterResponseMessage消息队列
    DataQueue<CounterResponseMessage>* counter_response_message_queue_;
    // 将CounterResponseMessage消息队列中的数据发送给CounterResponseTopic的线程
    std::thread* counter_response_message_producer_thread_{nullptr};

    // 从CounterTopic中读取CounterMessage数据
    std::shared_ptr<Reader<CounterMessage>> reader_;
    // 向CounterResponseTopic发送CounterResponseMessage数据
    std::shared_ptr<Writer> writer_;

    // 处理CounterMessage数据的对象，生成CounterResponseMessage数据
    TestObj* test_obj_;
};

// 启动activity方法二：使用REGISTER_ACTIVITY进行注册，然后通过activity_exec命令将activity启动，可传入参数（-c activity配置文件路径）
// REGISTER_ACTIVITY(ActivityTest1)

ActivityTest1::ActivityTest1()
{
}

ActivityTest1::~ActivityTest1()
{
    if ((counter_message_customer_thread_!= nullptr) && (counter_message_customer_thread_->joinable()))
    {
        counter_message_customer_thread_->join();
        delete counter_message_customer_thread_;
        counter_message_customer_thread_ = nullptr;
    }
    if ((counter_response_message_producer_thread_ != nullptr) && (counter_response_message_producer_thread_->joinable()))
    {
        counter_response_message_producer_thread_->join();
        delete counter_response_message_producer_thread_;
        counter_response_message_producer_thread_ = nullptr;
    }
    delete counter_response_message_queue_;
    delete counter_message_queue_;
    delete test_obj_;
}

bool ActivityTest1::Init()
{
    // 读取私有配置文件内容
    TopicConfig topic_config;
    TINFO << config_file_path_;
    if (!GetProtoConfig(&topic_config))
    {
        TINFO << "configure file parse failed";
        return false;
    }

    // 添加CounterTopic
    eprosima::fastdds::dds::TypeSupport deal_data_type(new CounterMessagePubSubType());
    if (!node_->AddTopic(topic_config.counter_topic(), deal_data_type))
    {
        return false;
    }
    // 创建CounterTopic对应的reader，接收数据
    reader_ = node_->CreateReader<CounterMessage>(topic_config.counter_topic(), std::bind(&ActivityTest1::ReadCounterMessageCallbackFunc, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4), this);
    reader_->Init();

    // 添加CounterResponseTopic
    eprosima::fastdds::dds::TypeSupport res_data_type(new CounterResponseMessagePubSubType());
    if (!node_->AddTopic(topic_config.counter_response_topic(), res_data_type))
    {
        return false;
    }
    // 创建CounterResponseTopic对应的writer，发送数据
    writer_ = node_->CreateWriter(topic_config.counter_response_topic());
    // 初始化
    writer_->Init();

    counter_message_queue_ = new DataQueue<CounterMessage>(10);
    counter_response_message_queue_ = new DataQueue<CounterResponseMessage>(10);

    test_obj_ = new TestObj();
    test_obj_->Init(std::bind(&ActivityTest1::GetCounterResponseMessageCallbackFunc, this, std::placeholders::_1, std::placeholders::_2), this);

    return true;
}

void ActivityTest1::Start()
{
    // TINFO << "ActivityTest1 is start";

    counter_message_customer_thread_ = new std::thread(&ActivityTest1::DealCounterMessageThreadFunc, this);
    counter_response_message_producer_thread_ = new std::thread(&ActivityTest1::SendCounterResponseMessageThreadFunc, this);
}

void ActivityTest1::PauseClear()
{
    if ((counter_message_customer_thread_!= nullptr) && (counter_message_customer_thread_->joinable()))
    {
        counter_message_customer_thread_->join();
        delete counter_message_customer_thread_;
        counter_message_customer_thread_ = nullptr;
    }
    if ((counter_response_message_producer_thread_ != nullptr) && (counter_response_message_producer_thread_->joinable()))
    {
        counter_response_message_producer_thread_->join();
        delete counter_response_message_producer_thread_;
        counter_response_message_producer_thread_ = nullptr;
    }
}

void ActivityTest1::ReadCounterMessageCallbackFunc(const CounterMessage &message, void *data_handle, std::string node_name, std::string topic_name) 
{
    std::shared_ptr<CounterMessage> message_ptr = std::make_shared<CounterMessage>(message);
    counter_message_queue_->Fill(message_ptr);
}

void ActivityTest1::DealCounterMessageThreadFunc()
{
    while (is_running_.load())
    {
        std::shared_ptr<CounterMessage> message;
        if (!counter_message_queue_->Fetch(message))
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            continue;
        }
        TINFO << "************************************";
        std::shared_ptr<CounterResponseMessage> res_message;
        test_obj_->Run(message);
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }
}

void ActivityTest1::SendCounterResponseMessageThreadFunc()
{
    while (is_running_.load())
    {
        std::shared_ptr<CounterResponseMessage> message;
        if (!counter_response_message_queue_->Fetch(message))
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            continue;
        }
        writer_->SendMessage((void*)message.get());
        std::this_thread::sleep_for(std::chrono::milliseconds(2000));
    }
}

void ActivityTest1::GetCounterResponseMessageCallbackFunc(const CounterResponseMessage &res_message, void *data_handle)
{
    std::shared_ptr<CounterResponseMessage> res_ptr = std::make_shared<CounterResponseMessage>(res_message);
    counter_response_message_queue_->Fill(res_ptr);
}

// 启动activity方法1：写main函数，可通过命令行传参，int main(int argc, char*argv[]),需自行解析
int main()
{
    std::string activity_info_path = "/workspace/ddsproject-example/activities/conf/test1_activity.info";
    ActivityInfo activity_info;
    // 解析test0_activity配置文件
    GetProtoFromFile(activity_info_path, &activity_info);

    // 设置当前进程名，用于在日志中打印
    SetName("ActivityTest1");

    ActivityTest1* activity = new ActivityTest1();
    // 初始化操作：添加topic，以及其他成员变量的初始化操作
    if (activity->Initialize(activity_info))
    {
        activity->Run();
    }
    delete activity;
    return 0;
}

