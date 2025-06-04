#include "include/activity/base/activitybase.hpp"

#include <fastdds/dds/topic/TypeSupport.hpp>

#include "activities/multiModal_activity/proto/multiModal_activity.pb.h"
#include "activities/idl/CMultiModalSrcData/CMultiModalSrcData.h"
#include "activities/idl/CMultiModalSrcData/CMultiModalSrcDataPubSubTypes.h"

#include <opencv2/opencv.hpp>

#include "include/queue/data_queue.hpp"

#include "include/Common/GlobalContext.h"
#include "include/queue/CSafeDataDeque.h"


long long getTimestamp()
{
    auto now = std::chrono::system_clock::now();
    auto duration = now.time_since_epoch();
    return std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
}

class ActivityMultiModal : public ActivityBase
{
public:
    ActivityMultiModal();
    ~ActivityMultiModal();

protected:
    // 初始化，读取配置文件
    virtual bool Init() override;
    // 当收到master节点的RUN指令，则执行Start，用于启动线程
    virtual void Start() override;
    // 当收到master节点的PAUSE指令，则执行一些清除工作，比如delete线程
    virtual void PauseClear();

private:
    // 向test1_activity发送数据的线程，向CounterTopic中发送消息
    void ImgProducerThreadFunc();

    void stopAndDeleteThreads();

    void ReadXYZCallbackFunc(const CMultiModalSrcData &message,
        void *data_handle, std::string node_name, std::string topic_name);
    
    void ReadM3JUVCCallbackFunc(const CVideoSrcData &message,
        void *data_handle, std::string node_name, std::string topic_name);


private:
    std::shared_ptr<Reader<CMultiModalSrcData>> reader_XYZ;
    std::shared_ptr<Reader<CVideoSrcData>> reader_M3JUVC;
    std::shared_ptr<Writer> writer_;

    std::thread *counter_message_producer_thread_{nullptr};
    std::shared_ptr<std::thread> message_producer_thread_{nullptr};




    DataQueue<CMultiModalSrcData>* XYZ_queue_;
    DataQueue<CVideoSrcData>* M3JUVC_queue_;

    CSafeDataDeque<std::shared_ptr<CMultiModalSrcData>> XYZ_safeDeque;
    CSafeDataDeque<std::shared_ptr<CVideoSrcData>> M3JUVC_safeDeque;




    CSafeDataDeque<std::shared_ptr<CMultiModalSrcData>> safeDeque;
    void MessageProducerThreadFunc();




    int count_readXYZ = 0;
    long long tmpTimeStamp_readXYZ = getTimestamp();

    int count_readM3J = 0;
    long long tmpTimeStamp_readM3J = getTimestamp();

};

// 启动activity方法二：使用REGISTER_ACTIVITY进行注册，然后通过activity_exec命令将activity启动，可传入参数（-c activity配置文件路径）
// REGISTER_ACTIVITY(ActivityTest0)

ActivityMultiModal::ActivityMultiModal()
{
}

ActivityMultiModal::~ActivityMultiModal()
{
    if ((counter_message_producer_thread_ != nullptr) && (counter_message_producer_thread_->joinable()))
    {
        counter_message_producer_thread_->join();
        delete counter_message_producer_thread_;
        counter_message_producer_thread_ = nullptr;
    }

    if (message_producer_thread_ && message_producer_thread_->joinable()) {
        message_producer_thread_->join();
        message_producer_thread_.reset();
    }

    stopAndDeleteThreads();

    delete XYZ_queue_;
    delete M3JUVC_queue_;
}

bool ActivityMultiModal::Init()
{
    TopicConfig topic_config;
    TINFO << config_file_path_;
    if (!GetProtoConfig(&topic_config))
    {
        TINFO << "configure file parse failed";
        return false;
    }

    // 添加CounterTopic
    eprosima::fastdds::dds::TypeSupport deal_data_type(new CMultiModalSrcDataPubSubType());
    if (!node_->AddTopic(topic_config.camera_xyz_topic(), deal_data_type))
    {
        return false;
    }
    // 创建CounterTopic对应的reader，接收数据
    reader_XYZ = node_->CreateReader<CMultiModalSrcData>(topic_config.camera_xyz_topic(),
        std::bind(&ActivityMultiModal::ReadXYZCallbackFunc,
            this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4), this);
    reader_XYZ->Init();




    // 添加CounterTopic
    eprosima::fastdds::dds::TypeSupport deal_data_type1(new CVideoSrcDataPubSubType());
    if (!node_->AddTopic(topic_config.camera_m3juvc_topic(), deal_data_type1))
    {
        return false;
    }
    // 创建CounterTopic对应的reader，接收数据
    reader_M3JUVC = node_->CreateReader<CVideoSrcData>(topic_config.camera_m3juvc_topic(),
        std::bind(&ActivityMultiModal::ReadM3JUVCCallbackFunc,
            this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4), this);
    reader_M3JUVC->Init();




    // 添加CounterResponseTopic
    eprosima::fastdds::dds::TypeSupport res_data_type(new CMultiModalSrcDataPubSubType());
    if (!node_->AddTopic(topic_config.camera_merged_topic(), res_data_type))
    {
        return false;
    }
    writer_ = node_->CreateWriter(topic_config.camera_merged_topic());
    // 初始化
    writer_->Init();



    XYZ_queue_ = new DataQueue<CMultiModalSrcData>(10);
    M3JUVC_queue_ = new DataQueue<CVideoSrcData>(10);

    XYZ_safeDeque.SetMaxSize(5);
    M3JUVC_safeDeque.SetMaxSize(5);
    // counter_response_message_queue_ = new DataQueue<CounterResponseMessage>(10);

    safeDeque.SetMaxSize(10);

    return true;
}

void ActivityMultiModal::ReadXYZCallbackFunc(const CMultiModalSrcData &message,
    void *data_handle, std::string node_name, std::string topic_name) 
{
    std::shared_ptr<CMultiModalSrcData> message_ptr = std::make_shared<CMultiModalSrcData>(message);
    // XYZ_queue_->Fill(message_ptr);

    XYZ_safeDeque.PushBack(std::make_shared<CMultiModalSrcData>(*message_ptr));



    count_readXYZ++;
    if (getTimestamp() - tmpTimeStamp_readXYZ > 1000)
    {
        std::cout << "==========fps_readXYZ: " << count_readXYZ << std::endl;

        tmpTimeStamp_readXYZ = getTimestamp();
        count_readXYZ = 0;
    }
}

void ActivityMultiModal::ReadM3JUVCCallbackFunc(const CVideoSrcData &message,
    void *data_handle, std::string node_name, std::string topic_name)
{
    std::shared_ptr<CVideoSrcData> message_ptr = std::make_shared<CVideoSrcData>(message);
    // M3JUVC_queue_->Fill(message_ptr);

    M3JUVC_safeDeque.PushBack(std::make_shared<CVideoSrcData>(*message_ptr));



    count_readM3J++;
    if (getTimestamp() - tmpTimeStamp_readM3J > 1000)
    {
        std::cout << "==========fps_readM3J: " << count_readM3J << std::endl;

        tmpTimeStamp_readM3J = getTimestamp();
        count_readM3J = 0;
    }
}

void ActivityMultiModal::Start()
{
    TINFO << "ActivityMultiModal running";



    counter_message_producer_thread_ = new std::thread(&ActivityMultiModal::ImgProducerThreadFunc, this);

    message_producer_thread_ = std::make_unique<std::thread>(&ActivityMultiModal::MessageProducerThreadFunc, this);
}

void saveToDisk_M3JUVC(const int serial, const std::vector<uint8_t>& img)
{
    // 640 * 512 == 327680
    void *p = malloc(640 * 512 * 3);
    memcpy(p, img.data(), 640 * 512 * 3);
    cv::Mat matImg(cv::Size(640, 512), CV_8UC3, p);

    std::filesystem::path dir = "/workspace/ddsproject-example/tmp/M3JUVC_multiModal/";
    std::filesystem::create_directories(dir);

    std::string filePath = dir.string() + std::to_string(serial) + ".jpg";
    if (cv::imwrite(filePath, matImg)) {
        ;
    } else {

    }

    free(p);
}

void ActivityMultiModal::ImgProducerThreadFunc()
{
    std::shared_ptr<CVideoSrcData> img_M3JUVC;
    std::shared_ptr<CMultiModalSrcData> img_XYZ;

    TINFO << "is_running_: " << is_running_;
    
    int count = 0;
    long long tmpTimeStamp = getTimestamp();
    
    while (is_running_.load())
    {
        

        if (nullptr == img_XYZ)
        {
            int retry_XYZ = 0;
            while(!XYZ_safeDeque.PopFront(img_XYZ, 1))
            {
                // std::this_thread::sleep_for(std::chrono::milliseconds(10));
                if (++retry_XYZ > 1000)
                {
                    std::cout << "There has been no data from XYZ for " << retry_XYZ / 1000 << " second." << std::endl;
                    // break;
                }
            }
        }


        if (nullptr == img_M3JUVC)
        {
            int retry_M3JUVC = 0;
            while(!M3JUVC_safeDeque.PopFront(img_M3JUVC, 1))
            {
                // std::this_thread::sleep_for(std::chrono::milliseconds(10));
                if (++retry_M3JUVC > 1000)
                {
                    std::cout << "There has been no data from M3JUVC for " << retry_M3JUVC / 1000 << " second." << std::endl;
                    // break;
                };
            }
        }

        long long timestamp_M3JUVC = img_M3JUVC->lTimeStamp();//[TIMESTAMP_IR_ARRIVE];
        long long timestamp_XYZ = img_XYZ->lTimeStamp();//[TIMESTAMP_RGB_ARRIVE];

        if (abs(timestamp_M3JUVC - timestamp_XYZ) < 20)
        {
            CMultiModalSrcData multiModalSrcData;

            std::cout << "=====matched: "
                << "M3JUVC: " << timestamp_M3JUVC << " " << img_M3JUVC->unFrameId() << " " << img_M3JUVC->lTimeStamp() << " ========== "
                << "XYZ: " << timestamp_XYZ << " " << img_XYZ->unFrameId() << " " << img_XYZ->lTimeStamp() << std::endl;


            img_M3JUVC->unFrameId(img_XYZ->unFrameId());
            img_XYZ->mapTimeStamp()[TIMESTAMP_TIME_MATCH] = timestamp_XYZ;
            img_M3JUVC->mapTimeStamp()[TIMESTAMP_TIME_MATCH] = timestamp_XYZ;

            multiModalSrcData.vecVideoSrcData().push_back(CVideoSrcData( img_XYZ->vecVideoSrcData()[0]));
            multiModalSrcData.vecVideoSrcData().push_back(*img_M3JUVC);
            multiModalSrcData.tDisparityResult(img_XYZ->tDisparityResult());





            // saveToDisk_M3JUVC(img_M3JUVC->unFrameId(), img_M3JUVC->vecImageBuf());



            
            count++;
            if (getTimestamp() - tmpTimeStamp > 1000)
            {
                std::cout << "==========fps: " << count << std::endl;

                tmpTimeStamp = getTimestamp();
                count = 0;
            }

            // std::cout << "aaaaaaaaaaaaa: " << multiModalSrcData.tDisparityResult().vecDistanceInfo()[100] << std::endl;



            /*
            std::vector<uint8_t> tmp;
            CDisparityResult dr = multiModalSrcData.tDisparityResult();
            tmp.resize(dr.usWidth() * dr.usHeight());
            for (int i = 0; i < tmp.size(); ++i)
            {
                tmp[i] = (uint8_t)dr.vecDistanceInfo()[i];
                // if (0!=tmp[i])
                // {
                //     std::cout << "99999999999999999999999999999999999999999" << std::endl;
                // }
            }

            // cv::Mat px_depth(Size(dr.usWidth(), dr.usHeight()), CV_16UC1, gDepthImgBuf);//
            cv::Mat px_depth_temp(cv::Size(dr.usWidth(), dr.usHeight()), CV_8UC1, tmp.data());//
            cv::imwrite("aaaaaaa.png", px_depth_temp);

            // for (int y = 0; y < gColorHeight; y++) {
            //     for (int x = 0; x < gColorWidth; x++) {
            //         px_depth_temp.at<uchar>(y, x) = px_depth.at<ushort>(y, x)>>3;
            //     }
            // }
*/











            safeDeque.PushBack(std::make_shared<CMultiModalSrcData>(multiModalSrcData));
            // writer_->SendMessage((void*)&multiModalSrcData);

            
            img_M3JUVC = nullptr;
            img_XYZ = nullptr;
        } else {
            std::cout << "=====mismatched: "
                << "M3JUVC: " << timestamp_M3JUVC << " " << img_M3JUVC->unFrameId() << " " << img_M3JUVC->lTimeStamp() << " ========== "
                << "XYZ: " << timestamp_XYZ << " " << img_XYZ->unFrameId() << " " << img_XYZ->lTimeStamp() << std::endl;
            


            if (timestamp_M3JUVC < timestamp_XYZ)
            {
                img_M3JUVC = nullptr;
            } else {
                img_XYZ = nullptr;
            }
        }
        

        // std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

void ActivityMultiModal::MessageProducerThreadFunc()
{
    // 向CounterTopic中循环发送数据
    // is_running_是线程结束的标志位，通过master的指令进行控制
    while (is_running_.load())
    {
        std::shared_ptr<CMultiModalSrcData> message;
        if (!safeDeque.PopFront(message, 1))
        {
            // std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            continue;
        }

        writer_->SendMessage((void*)message.get());
        // std::this_thread::sleep_for(std::chrono::milliseconds(2000));
    }
}

void ActivityMultiModal::stopAndDeleteThreads()
{
    // if (color_thread_ != nullptr)
    // {
    //     if (color_thread_->joinable())
    //     {
    //         color_thread_->join();
    //     }
        
    //     delete color_thread_;
    //     color_thread_ = nullptr;
    // }

    // if (depth_thread_ != nullptr)
    // {
    //     if (depth_thread_->joinable())
    //     {
    //         depth_thread_->join();
    //     }
        
    //     delete depth_thread_;
    //     depth_thread_ = nullptr;
    // }
}

void ActivityMultiModal::PauseClear()
{
    if ((counter_message_producer_thread_ != nullptr) && (counter_message_producer_thread_->joinable()))
    {
        counter_message_producer_thread_->join();
        delete counter_message_producer_thread_;
        counter_message_producer_thread_ = nullptr;
    }

    stopAndDeleteThreads();
}

// 启动activity方法1：写main函数，可通过命令行传参，int main(int argc, char*argv[]),需自行解析
int main()
{
    std::string activity_info_path = "../../../ddsproject-example/activities/conf/multiModal_activity.info";
    ActivityInfo activity_info;
    // 解析test0_activity配置文件
    GetProtoFromFile(activity_info_path, &activity_info);

    // 设置当前进程名，用于在日志中打印
    SetName("multiModal");

    ActivityMultiModal *activity = new ActivityMultiModal();
    // 初始化操作：添加topic，以及其他成员变量的初始化操作
    if (activity->Initialize(activity_info))
    {
        activity->Run();
    }
    delete activity;
    return 0;                // if (0!=tmp[i])
                // {
                //     std::cout << "99999999999999999999999999999999999999999" << std::endl;
                // }
}
