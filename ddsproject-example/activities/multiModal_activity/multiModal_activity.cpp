#include "include/activity/base/activitybase.hpp"

#include <fastdds/dds/topic/TypeSupport.hpp>

#include "activities/multiModal_activity/proto/multiModal_activity.pb.h"
#include "activities/idl/CMultiModalSrcData/CMultiModalSrcData.h"
#include "activities/idl/CMultiModalSrcData/CMultiModalSrcDataPubSubTypes.h"

#include <opencv2/opencv.hpp>

#include "include/queue/data_queue.hpp"

#include "include/Common/GlobalContext.h"
#include "include/queue/CSafeDataDeque.h"

#include "pullFromStream.h"
#include "DepthConverter.h"

long long getTimestamp()
{
    auto now = std::chrono::system_clock::now();
    auto duration = now.time_since_epoch();
    return std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
}

class FrameFromStream
{
public:
    int width;
    int height;
    long long timestamp;
    std::vector<uint8_t> vecBuf;
};

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
    void PullStreamThreadFunc_M3J();
    void PullStreamThreadFunc_XYZ_Color();
    void PullStreamThreadFunc_XYZ_Depth();

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

    std::thread *pull_stream_thread_M3J_{nullptr};
    std::thread *pull_stream_thread_XYZ_Color{nullptr};
    std::thread *pull_stream_thread_XYZ_Depth{nullptr};

    std::thread *counter_message_producer_thread_{nullptr};
    std::shared_ptr<std::thread> message_producer_thread_{nullptr};

    CSafeDataDeque<std::shared_ptr<FrameFromStream>> frame_safeDeque_M3J;
    CSafeDataDeque<std::shared_ptr<FrameFromStream>> frame_safeDeque_XYZ_Color;
    CSafeDataDeque<std::shared_ptr<FrameFromStream>> frame_safeDeque_XYZ_Depth;


    int index{0};

    int count_M3J = 0;
    long long tmpTimeStamp_M3J = getTimestamp();

    int count_XYZ_Color = 0;
    long long tmpTimeStamp_XYZ_Color = getTimestamp();

    int count_XYZ_Depth = 0;
    long long tmpTimeStamp_XYZ_Depth = getTimestamp();

    long long frameId{0};

    DataQueue<CMultiModalSrcData>* XYZ_queue_;
    DataQueue<CVideoSrcData>* M3JUVC_queue_;

    CSafeDataDeque<std::shared_ptr<CMultiModalSrcData>> XYZ_safeDeque;
    CSafeDataDeque<std::shared_ptr<CVideoSrcData>> M3JUVC_safeDeque;




    CSafeDataDeque<std::shared_ptr<CMultiModalSrcData>> safeDeque;
    void MessageProducerThreadFunc();


    PullFromStream pullM3J;
    PullFromStream pullXYZ_Color;
    PullFromStream pullXYZ_Depth;


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
    // eprosima::fastdds::dds::TypeSupport deal_data_type(new CMultiModalSrcDataPubSubType());
    // if (!node_->AddTopic(topic_config.camera_xyz_topic(), deal_data_type))
    // {
    //     return false;
    // }
    // // 创建CounterTopic对应的reader，接收数据
    // reader_XYZ = node_->CreateReader<CMultiModalSrcData>(topic_config.camera_xyz_topic(),
    //     std::bind(&ActivityMultiModal::ReadXYZCallbackFunc,
    //         this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4), this);
    // reader_XYZ->Init();




    // 添加CounterTopic
    // eprosima::fastdds::dds::TypeSupport deal_data_type1(new CVideoSrcDataPubSubType());
    // if (!node_->AddTopic(topic_config.camera_m3juvc_topic(), deal_data_type1))
    // {
    //     return false;
    // }
    // // 创建CounterTopic对应的reader，接收数据
    // reader_M3JUVC = node_->CreateReader<CVideoSrcData>(topic_config.camera_m3juvc_topic(),
    //     std::bind(&ActivityMultiModal::ReadM3JUVCCallbackFunc,
    //         this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4), this);
    // reader_M3JUVC->Init();




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



    frame_safeDeque_M3J.SetMaxSize(5);
    frame_safeDeque_XYZ_Color.SetMaxSize(5);
    frame_safeDeque_XYZ_Depth.SetMaxSize(5);

    
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

    bool bInitM3J = pullM3J.init("rtsp://192.168.3.56:8554/camera/M3J");
    if (!bInitM3J)
    {
        TINFO << "can not open stream M3J";

        return;
    }

    bool bInitXYZ_Color = pullXYZ_Color.init("rtsp://192.168.3.56:8554/camera/XYZ_Color"); // 6
    if (!bInitXYZ_Color)
    {
        TINFO << "can not open stream XYZ_Color";

        return;
    }

    bool bInitXYZ_Depth = pullXYZ_Depth.init("rtsp://192.168.3.56:8554/camera/XYZ_Depth"); // 7
    if (!bInitXYZ_Depth)
    {
        TINFO << "can not open stream XYZ_Depth";

        return;
    }


    pull_stream_thread_M3J_ = new std::thread(&ActivityMultiModal::PullStreamThreadFunc_M3J, this);
    pull_stream_thread_XYZ_Color = new std::thread(&ActivityMultiModal::PullStreamThreadFunc_XYZ_Color, this);
    pull_stream_thread_XYZ_Depth = new std::thread(&ActivityMultiModal::PullStreamThreadFunc_XYZ_Depth, this);

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


void ActivityMultiModal::PullStreamThreadFunc_M3J()
{
    pullM3J.pull(
        [this](char* data, const int width, const int height){
            FrameFromStream frame;
            frame.width = width;
            frame.height = height;
            frame.timestamp = getTimestamp();
            frame.vecBuf.resize(width * height * 3);
            memcpy(&frame.vecBuf[0], data, width * height * 3);

            frame_safeDeque_M3J.PushBack(std::make_shared<FrameFromStream>(frame));


            // cv::Mat mat(512, 640, CV_8UC3, data);
            // cv::imwrite(std::string("/share/tmpimage/M3J/a") + std::to_string(index++) + ".jpg", mat);


            count_M3J++;
            if (getTimestamp() - tmpTimeStamp_M3J > 1000)
            {
                std::cout << "==========fps_M3J: " << count_M3J << std::endl;

                tmpTimeStamp_M3J = getTimestamp();
                count_M3J = 0;
            }

        }
    );
}

void ActivityMultiModal::PullStreamThreadFunc_XYZ_Color()
{
    pullXYZ_Color.pull(
        [this](char* data, const int width, const int height){
            FrameFromStream frame;
            frame.width = width;
            frame.height = height;
            frame.timestamp = getTimestamp();
            frame.vecBuf.resize(width * height * 3);
            memcpy(&frame.vecBuf[0], data, width * height * 3);

            frame_safeDeque_XYZ_Color.PushBack(std::make_shared<FrameFromStream>(frame));

            

            count_XYZ_Color++;
            if (getTimestamp() - tmpTimeStamp_XYZ_Color > 1000)
            {
                std::cout << "==========fps_Color: " << count_XYZ_Color << std::endl;

                tmpTimeStamp_XYZ_Color = getTimestamp();
                count_XYZ_Color = 0;
            }
        }
    );
}

void ActivityMultiModal::PullStreamThreadFunc_XYZ_Depth()
{
    pullXYZ_Depth.pull(
        [this](char* data, const int width, const int height){
            FrameFromStream frame;
            frame.width = width;
            frame.height = height;
            frame.timestamp = getTimestamp();
            frame.vecBuf.resize(width * height * 3);
            memcpy(&frame.vecBuf[0], data, width * height * 3);

            frame_safeDeque_XYZ_Depth.PushBack(std::make_shared<FrameFromStream>(frame));

            count_XYZ_Depth++;
            if (getTimestamp() - tmpTimeStamp_XYZ_Depth > 1000)
            {
                std::cout << "==========fps_Depth: " << count_XYZ_Depth << std::endl;

                tmpTimeStamp_XYZ_Depth = getTimestamp();
                count_XYZ_Depth = 0;
            }
        }
    );
}

void ActivityMultiModal::ImgProducerThreadFunc()
{

    DepthConverter depthConverter(1280, 720);

    std::shared_ptr<FrameFromStream> ptr_M3J;
    std::shared_ptr<FrameFromStream> ptr_XYZ_Color;
    std::shared_ptr<FrameFromStream> ptr_XYZ_Depth;




    TINFO << "is_running_: " << is_running_;
    
    int count = 0;
    long long tmpTimeStamp = getTimestamp();
    
    while (is_running_.load())
    {
        if (nullptr == ptr_M3J)
        {
            while(!frame_safeDeque_M3J.PopFront(ptr_M3J, 1))
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        }

        if (nullptr == ptr_XYZ_Color)
        {
            while(!frame_safeDeque_XYZ_Color.PopFront(ptr_XYZ_Color, 1))
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        }

        if (nullptr == ptr_XYZ_Depth)
        {
            while(!frame_safeDeque_XYZ_Depth.PopFront(ptr_XYZ_Depth, 1))
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        }

        long long timestamp_M3J = ptr_M3J->timestamp;
        long long timestamp_XYZ_Color = ptr_XYZ_Color->timestamp;
        long long timestamp_XYZ_Depth = ptr_XYZ_Depth->timestamp;

        if (abs(timestamp_M3J - timestamp_XYZ_Color) < 50 && abs(timestamp_XYZ_Color - timestamp_XYZ_Depth) < 50)
        {
            std::cout << "=====matched: "
                << "M3J: " << timestamp_M3J << " ========== "
                << "XYZ_Color: " << timestamp_XYZ_Color << " ========== "
                << "XYZ_Depth: " << timestamp_XYZ_Depth << " ========== "
                << std::endl;

            count++;
            if (getTimestamp() - tmpTimeStamp > 1000)
            {
                std::cout << "==========fps: " << count << std::endl;

                tmpTimeStamp = getTimestamp();
                count = 0;
            }

            

            CMultiModalSrcData multiModalSrcData;
            // will set timestamp_XYZ_Color to all timestamps

            // XYZ_Color
            CVideoSrcData videoSrcData_XYZ;
            videoSrcData_XYZ.eDataSourceType(0);
            videoSrcData_XYZ.unFrameId(frameId);
            videoSrcData_XYZ.mapTimeStamp()[TIMESTAMP_RGB_ARRIVE] = timestamp_XYZ_Color;
            videoSrcData_XYZ.lTimeStamp(timestamp_XYZ_Color);
            videoSrcData_XYZ.ucCameraId(0);
            videoSrcData_XYZ.usBmpLength(ptr_XYZ_Color->height);
            videoSrcData_XYZ.usBmpWidth(ptr_XYZ_Color->width);
            videoSrcData_XYZ.unBmpBytes(3 * ptr_XYZ_Color->height * ptr_XYZ_Color->width);
            videoSrcData_XYZ.vecImageBuf(std::move(ptr_XYZ_Color->vecBuf));

            multiModalSrcData.vecVideoSrcData().push_back(videoSrcData_XYZ);


            cv::Mat mat_C(ptr_XYZ_Color->height, ptr_XYZ_Color->width, CV_8UC3, videoSrcData_XYZ.vecImageBuf().data());
            // cv::imwrite(std::string("/share/tmpimage/Depth/a_C") + std::to_string(index) + ".jpg", mat_C);
        








            // M3J
            CVideoSrcData videoSrcData_M3J;
            videoSrcData_M3J.eDataSourceType(1);
            videoSrcData_M3J.unFrameId(frameId);
            videoSrcData_M3J.mapTimeStamp()[TIMESTAMP_RGB_ARRIVE] = timestamp_XYZ_Color;
            videoSrcData_M3J.lTimeStamp(timestamp_XYZ_Color);
            videoSrcData_M3J.ucCameraId(1);
            videoSrcData_M3J.usBmpLength(ptr_M3J->height);
            videoSrcData_M3J.usBmpWidth(ptr_M3J->width);
            videoSrcData_M3J.unBmpBytes(3 * ptr_M3J->height * ptr_M3J->width);
            videoSrcData_M3J.vecImageBuf(std::move(ptr_M3J->vecBuf));

            multiModalSrcData.vecVideoSrcData().push_back(videoSrcData_M3J);

            // Disparity
            CDisparityResult disparityResult;
            disparityResult.usWidth(ptr_XYZ_Depth->width);
            disparityResult.usHeight(ptr_XYZ_Depth->height);

            std::vector<uint8_t> tmpSrcVec;
            tmpSrcVec.resize(1280 * 720 * 2);
            for (int i = 0; i < 1280 * 720; ++i)
            {
                tmpSrcVec[i*2 + 0] = ptr_XYZ_Depth->vecBuf[i*3 + 0];
                tmpSrcVec[i*2 + 1] = ptr_XYZ_Depth->vecBuf[i*3 + 1];
            }

            std::cout << (int)ptr_XYZ_Depth->vecBuf[60*3 + 1] << std::endl;
            std::cout << (int)ptr_XYZ_Depth->vecBuf[61*3 + 1] << std::endl;
            std::cout << (int)ptr_XYZ_Depth->vecBuf[62*3 + 1] << std::endl;

            std::cout << (int)ptr_XYZ_Depth->vecBuf[60*3 + 2] << std::endl;
            std::cout << (int)ptr_XYZ_Depth->vecBuf[61*3 + 2] << std::endl;
            std::cout << (int)ptr_XYZ_Depth->vecBuf[62*3 + 2] << std::endl;

            std::vector<int32_t> tmpVec;
            tmpVec.resize(1280 * 720);
            depthConverter.process(tmpSrcVec, tmpVec);
            // depthConverter.process(ptr_XYZ_Depth->vecBuf, tmpVec);

            disparityResult.vecDistanceInfo(tmpVec);

            int center = 1280 * 720 / 2;
            int center_a1 = center - 1280;
            int center_a2 = center - 2 * 1280;
            int center_b1 = center + 1280;
            int center_b2 = center + 2 *1280;
            std::cout
                << tmpVec[center_a2 - 2] << " " << tmpVec[center_a2 - 1] << " " << tmpVec[center_a2] << " " << tmpVec[center_a2 + 1] << " " << tmpVec[center_a2 + 2] << "\n"
                << tmpVec[center_a1 - 2] << " " << tmpVec[center_a1 - 1] << " " << tmpVec[center_a1] << " " << tmpVec[center_a1 + 1] << " " << tmpVec[center_a1 + 2] << "\n"
                << tmpVec[center - 2] << " " << tmpVec[center - 1] << " " << tmpVec[center] << " " << tmpVec[center + 1] << " " << tmpVec[center + 2] << "\n"
                << tmpVec[center_b1 - 2] << " " << tmpVec[center_b1 - 1] << " " << tmpVec[center_b1] << " " << tmpVec[center_b1 + 1] << " " << tmpVec[center_b1 + 2] << "\n"
                << tmpVec[center_b2 - 2] << " " << tmpVec[center_b2 - 1] << " " << tmpVec[center_b2] << " " << tmpVec[center_b2 + 1] << " " << tmpVec[center_b2 + 2] << "\n"
                << std::endl;



            cv::Mat mat_0(ptr_XYZ_Depth->height, ptr_XYZ_Depth->width, CV_8UC3, ptr_XYZ_Depth->vecBuf.data());
            // cv::imwrite(std::string("/share/tmpimage/Depth/a0") + std::to_string(index) + ".jpg", mat_0);
        


            std::vector<uint8_t> tmptmptmp;
            tmptmptmp.resize(1280 * 720);
            for (int i = 0; i < 1280 * 720; ++i)
            {
                tmptmptmp[i] = tmpVec[i] / 256;
            }
            cv::Mat mat(ptr_XYZ_Depth->height, ptr_XYZ_Depth->width, CV_8UC1, tmptmptmp.data());
            // cv::imwrite(std::string("/share/tmpimage/Depth/a") + std::to_string(index) + ".jpg", mat);


            index++;

            multiModalSrcData.tDisparityResult(disparityResult);




            safeDeque.PushBack(std::make_shared<CMultiModalSrcData>(multiModalSrcData));
            // writer_->SendMessage((void*)&multiModalSrcData);

            
            ptr_M3J = nullptr;
            ptr_XYZ_Color = nullptr;
            ptr_XYZ_Depth = nullptr;

            frameId++;
        } else {
            std::cout << "=====mismatched: "
                << "M3J: " << timestamp_M3J << " ========== "
                << "XYZ_Color: " << timestamp_XYZ_Color << " ========== "
                << "XYZ_Depth: " << timestamp_XYZ_Depth << " ========== "
                << std::endl;
            

            // only keep the max
            if (timestamp_M3J < timestamp_XYZ_Color)
            {
                ptr_M3J = nullptr;

                if (timestamp_XYZ_Color < timestamp_XYZ_Depth)
                {
                    ptr_XYZ_Color = nullptr;
                } else {
                    ptr_XYZ_Depth = nullptr;
                }
            } else {
                ptr_XYZ_Color = nullptr;
                
                if (timestamp_M3J < timestamp_XYZ_Depth)
                {
                    ptr_M3J = nullptr;
                } else {
                    ptr_XYZ_Depth = nullptr;
                }
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

        //-----test-----
        ;;;;;

        writer_->SendMessage((void*)message.get());
        // std::this_thread::sleep_for(std::chrono::milliseconds(2000));
    }
}

void ActivityMultiModal::stopAndDeleteThreads()
{
    if ((counter_message_producer_thread_ != nullptr) && (counter_message_producer_thread_->joinable()))
    {
        counter_message_producer_thread_->join();
        delete counter_message_producer_thread_;
        counter_message_producer_thread_ = nullptr;
    }

    if ((pull_stream_thread_M3J_ != nullptr) && (pull_stream_thread_M3J_->joinable()))
    {
        pull_stream_thread_M3J_->join();
        delete pull_stream_thread_M3J_;
        pull_stream_thread_M3J_ = nullptr;
    }

    if ((pull_stream_thread_XYZ_Color != nullptr) && (pull_stream_thread_XYZ_Color->joinable()))
    {
        pull_stream_thread_XYZ_Color->join();
        delete pull_stream_thread_XYZ_Color;
        pull_stream_thread_XYZ_Color = nullptr;
    }

    if ((pull_stream_thread_XYZ_Depth != nullptr) && (pull_stream_thread_XYZ_Depth->joinable()))
    {
        pull_stream_thread_XYZ_Depth->join();
        delete pull_stream_thread_XYZ_Depth;
        pull_stream_thread_XYZ_Depth = nullptr;
    }

    if (message_producer_thread_ && message_producer_thread_->joinable()) {
        message_producer_thread_->join();
        message_producer_thread_.reset();
    }
}

void ActivityMultiModal::PauseClear()
{
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
    return 0;
}
