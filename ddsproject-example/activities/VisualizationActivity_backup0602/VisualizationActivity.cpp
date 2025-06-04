#include "VisualizationActivity.h"

#include <unistd.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <netinet/in.h>


// 启动activity方法二：使用REGISTER_ACTIVITY进行注册，然后通过activity_exec命令将activity启动，可传入参数（-c activity配置文件路径）
// REGISTER_ACTIVITY(VisualizationActivity)

VisualizationActivity::VisualizationActivity()
{
}

VisualizationActivity::~VisualizationActivity() 
{
    PauseClear();
    
    closeSocket();
}

void VisualizationActivity::openSocket()
{
    serv_sock = socket(AF_INET, SOCK_STREAM, 0);
    
    int opt = 1;
    if (setsockopt(serv_sock, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) == -1)
    {
        close(serv_sock);
        std::cout << "設置Sock選項失敗" << std::endl;
    }

    sockaddr_in serv_addr;
    memset(&serv_addr, 0, sizeof(serv_addr));
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_addr.s_addr = inet_addr("0.0.0.0"); // inet_addr("localhost");
    serv_addr.sin_port = htons(1234);

    if (bind(serv_sock, (sockaddr*)&serv_addr, sizeof(serv_addr)) == -1)
    {
        close(serv_sock);
        std::cout << "=====bind failed, " << "errno: " << errno << std::endl;
    } else {
        std::cout << "=====bind success" << std::endl;
    }

    listen(serv_sock, 1);

    sockaddr_in clnt_addr;
    socklen_t clnt_addr_size = sizeof(clnt_addr);

    clnt_sock = accept(serv_sock, (sockaddr*)&clnt_addr, &clnt_addr_size);

    if (-1 == clnt_sock)
    {
        std::cout << "accept failed, " << "errno: " << errno <<std::endl;
    }
}

void VisualizationActivity::closeSocket()
{
    if (clnt_sock != -1)
        close(clnt_sock);
    
    if (serv_sock != -1)
        close(serv_sock);
}

bool VisualizationActivity::Init()
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
    reader_multi_modal_src_data_ = node_->CreateReader<CMultiModalSrcData>(topic_config.camera_merged_topic(),
        std::bind(&VisualizationActivity::ReadMultiModalSrcDataCallbackFunc,
            this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4), this);
    reader_multi_modal_src_data_->Init();

    // 2. 创建object_location_result_topic对应的reader，接收数据
    eprosima::fastdds::dds::TypeSupport deal_data_type_object_location(new CAlgResultPubSubType());
    if (!node_->AddTopic(topic_config.object_location_result_topic(), deal_data_type_object_location))
    {
        return false;
    }
    reader_object_location_result_ = node_->CreateReader<CAlgResult>(topic_config.object_location_result_topic(),
        std::bind(&VisualizationActivity::ReadObjectLocationResultCallbackFunc,
            this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4), this);
    reader_object_location_result_->Init();

    // // 3. 创建multi_modal_fusion_result_topic对应的writer，发送数据
    // eprosima::fastdds::dds::TypeSupport writer_data_type(new CAlgResultPubSubType());
    // if (!node_->AddTopic(topic_config.pose_estimation_result_topic(), writer_data_type))
    // {
    //     return false;
    // }
    // writer_ = node_->CreateWriter(topic_config.pose_estimation_result_topic());
    // writer_->Init();

    // 4. 初始化消息队列
    multi_modal_src_data_deque_.SetMaxSize(10);  // 通过SetMaxSize方法设置队列长度
    object_location_result_deque_.SetMaxSize(10);

    // 5.实例化算法节点并初始化
    // std::string root_path = GetRootPath();
    // root_path_ = root_path + "/Output/";
    // alg_param_.m_strRootPath = root_path_;
    // LOG(INFO) << "root_path: " << root_path_;
    // pose_estimation_alg_ = CreateVisualizationAlgObj(root_path_);
    // pose_estimation_alg_->initAlgorithm(&alg_param_, std::bind(&VisualizationActivity::GetVisualizationResultResponseMessageCallbackFunc, this, std::placeholders::_1, std::placeholders::_2), nullptr);
    
    openSocket();

    return true;
}

// 处理camera_merged_topic中的数据
void VisualizationActivity::ReadMultiModalSrcDataCallbackFunc(const CMultiModalSrcData &message,
        void *data_handle, std::string node_name, std::string topic_name)
{
    std::shared_ptr<CMultiModalSrcData> message_ptr = std::make_shared<CMultiModalSrcData>(message);
    multi_modal_src_data_deque_.PushBack(message_ptr);
}

// 处理object_location_result_topic中的数据
void VisualizationActivity::ReadObjectLocationResultCallbackFunc(const CAlgResult &message,
        void *data_handle, std::string node_name, std::string topic_name)
{
    std::shared_ptr<CAlgResult> message_ptr = std::make_shared<CAlgResult>(message);
    object_location_result_deque_.PushBack(message_ptr);
}



void VisualizationActivity::Start() 
{
    TINFO << "VisualizationActivity running";
    std::lock_guard<std::mutex> lock(thread_mutex_);
    if (is_running_) return;
    is_running_ = true;
    // message_producer_thread_ = std::make_unique<std::thread>(&VisualizationActivity::MessageProducerThreadFunc, this);
    message_producer_thread_ = nullptr; 
    message_consumer_thread_ = std::make_unique<std::thread>(&VisualizationActivity::MessageConsumerThreadFunc, this);
}

// 当收到master节点的PAUSE指令，则执行一些清除工作，比如delete线程
void VisualizationActivity::PauseClear() 
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

// void VisualizationActivity::MessageProducerThreadFunc()
// {
//     // 向CounterTopic中循环发送数据
//     // LOG(INFO) << "is_running : " << is_running_;
//     // is_running_是线程结束的标志位，通过master的指令进行控制
//     while (is_running_.load())
//     {   
//         // LOG(INFO) << "Waiting ...";
//         std::shared_ptr<CAlgResult> message;
//         if (!object_location_result_deque_.PopFront(message, 1))
//         {
//             // std::this_thread::sleep_for(std::chrono::milliseconds(1000));
//             continue;
//         }
//         LOG(INFO) << "Get CAlgResult!!";
//         writer_->SendMessage((void*)message.get());
//         // std::this_thread::sleep_for(std::chrono::milliseconds(2000));
//     }
// }

long long getTimeStamp()
{
    auto now = std::chrono::system_clock::now();
    auto duration = now.time_since_epoch();
    return std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
}

void VisualizationActivity::MessageConsumerThreadFunc()
{
    // 输入到融合算法的数据
    std::shared_ptr<CMultiModalSrcData> l_pMultiModalSrcData = std::make_shared<CMultiModalSrcData>(); 
    std::shared_ptr<CAlgResult> l_pObjectLocationResult = std::make_shared<CAlgResult>(); 
    LOG(INFO) << "MessageConsumerThreadFunc start : " << is_running_.load();


    int count = 0;
    long long tmpTimeStamp = getTimeStamp();
    while (is_running_.load())
    {     
        // 清空之前的数据
        l_pMultiModalSrcData->vecVideoSrcData().clear();
        l_pObjectLocationResult->vecFrameResult().clear();

        // 获取多模态融合结果
        if (!multi_modal_src_data_deque_.PopFront(l_pMultiModalSrcData, 1))
        {
            // LOG(INFO) << "MessageConsumerThreadFunc multi_modal_src_data_deque_ PopFront failed";
            continue;
        }

        // 检查多模态融合结果是否有效
        if (l_pMultiModalSrcData->vecVideoSrcData().empty())
        {
            LOG(WARNING) << "Empty multi-modal fusion srcdata";
            continue;
        }

        // 深拷贝多模态融合结果
        // *l_pObjectLocationInputData = *l_pMultiModalSrcData;
        // LOG(INFO) << "MessageConsumerThreadFunc multi_modal_fusion_result_deque_ PopFront success";

        // 获取多模态数据的时间戳
        auto multiModalTime = l_pMultiModalSrcData->vecVideoSrcData()[0].lTimeStamp(); // ->lTimeStamp();
        bool foundMatch = false;

        // std::cout << "==========2" << std::endl;

        int i = 0;
        // 尝试匹配姿态估计结果
        while(object_location_result_deque_.PopFront(l_pObjectLocationResult, 100))
        {   
            // 检查姿态估计结果是否有效
            if (l_pObjectLocationResult->vecFrameResult().empty())
            {
                // LOG(WARNING) << "Empty pose estimation result";
                continue;
            }

            // auto poseTime = l_pPoseEstimationResult->vecFrameResult()[0].mapTimeStamp()[TIMESTAMP_TIME_MATCH];
            auto poseTime = l_pObjectLocationResult->lTimeStamp();
            
            if (multiModalTime == poseTime)
            {
                // 时间戳匹配，添加姿态估计结果
                l_pObjectLocationResult->vecFrameResult().push_back(l_pObjectLocationResult->vecFrameResult()[0]);
                foundMatch = true;

                std::vector<uint8_t> vec = l_pMultiModalSrcData->vecVideoSrcData()[0].vecImageBuf();
                auto width = l_pMultiModalSrcData->vecVideoSrcData()[0].usBmpWidth();
                auto length = l_pMultiModalSrcData->vecVideoSrcData()[0].usBmpLength();
                cv::Mat mat(length, width, CV_8UC3, vec.data());
                
                for (const auto& item: l_pObjectLocationResult->vecFrameResult())
                {
                    for (const auto& obRes: item.vecObjectResult())
                    {
                        std::cout << " " << obRes.fTopLeftX() << " " << obRes.fTopLeftY() << " " <<obRes.fBottomRightX() << " " << obRes.fBottomRightY() << std::endl;
                        cv::rectangle(mat, cv::Point(obRes.fTopLeftX(), obRes.fTopLeftY()), cv::Point(obRes.fBottomRightX(), obRes.fBottomRightY()), cv::Scalar(0, 0, 255), 3);
                        cv::putText(mat, obRes.strClass() + ", " + std::to_string(obRes.fDistance()), cv::Point(obRes.fTopLeftX(), obRes.fTopLeftY()),
                            cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255));
                    }
                }

                count++;
                if (getTimeStamp() - tmpTimeStamp > 1000)
                {
                    std::cout << "==========fps: " << count << std::endl;

                    tmpTimeStamp = getTimeStamp();
                    count = 0;
                }

                // cv::imwrite(std::to_string(i) +  ".png", mat);
                if (-1 != clnt_sock)
                {
                    int nSize = write(clnt_sock, mat.data, length * width * 3);
                    std::cout << "=-=-=-=-=-= " << clnt_sock << " " << width << " " << length << " " << nSize << std::endl;
                } else {
                    std::cout << "clnt_sock is -1" << std::endl;
                }
                // std::cout << "=-=-=-=-=-= " << clnt_sock << " " << width << " " << length << " " << nSize << std::endl;
                break;
            }
            else if (multiModalTime < poseTime)
            {
                // 当前姿态估计结果时间戳更大，放回队列
                object_location_result_deque_.PushBack(l_pObjectLocationResult);
                break;
            }
            // 如果时间戳更小，继续循环查找匹配项
        }

        // 记录是否找到匹配的姿态估计结果
        // if (!foundMatch)
        // {
        //     LOG(WARNING) << "No matching pose estimation result found for timestamp: " << multiModalTime;
        //     continue;
        // }

        // // 执行目标定位算法
        // try
        // {
        //     // 直接调用runAlgorithm，不检查返回值
        //     // LOG(INFO) << "runAlgorithm InputData TimeStamp : " << l_pObjectLocationInputData->lTimeStamp();
        //     object_location_alg_->runAlgorithm(l_pObjectLocationInputData.get());
        //     // LOG(INFO) << "ObjectLocationActivity Algorithm InputData processed successfully. "
        //     //         << "Frame count: " << l_pObjectLocationInputData->vecFrameResult().size();
        // }
        // catch (const std::exception& e)
        // {
        //     LOG(ERROR) << "Exception in object location algorithm: " << e.what();
        //     continue;
        // }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}


// 启动activity方法1：写main函数，可通过命令行传参，int main(int argc, char*argv[]),需自行解析
int main()
{
    std::string activity_info_path = "../../../ddsproject-example/activities/conf/VisualizationActivity.info";
    ActivityInfo activity_info;
    // 解析VisualizationActivity配置文件
    GetProtoFromFile(activity_info_path, &activity_info);

    // 设置当前进程名，用于在日志中打印
    SetName("VisualizationActivity");

    VisualizationActivity *activity = new VisualizationActivity();
    // 初始化操作：添加topic，以及其他成员变量的初始化操作
    if (activity->Initialize(activity_info))
    {
        activity->Run();
    }
    delete activity;
    return 0;
}
