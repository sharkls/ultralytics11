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

#include <stdio.h>

#include <cstring>

#include <unistd.h>

#include <arpa/inet.h>
#include <sys/socket.h>
#include <netinet/in.h>

#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/iostreams/device/back_inserter.hpp>

// 解压缩函数，将输入的压缩数据in解压缩后存入out中
void decompress(const char *p, const int size, std::vector<char>& out)
{
    using namespace boost::iostreams;
    filtering_ostream fos;  // 创建一个具有filter功能的输出流
    fos.push(gzip_decompressor());  // 添加gzip解压缩器
    fos.push(boost::iostreams::back_inserter(out));  // 将输出流指定为out，即解压缩后的数据存放位置
    fos.write(p, size);  // 将压缩数据写入解压缩流
    boost::iostreams::close(fos);  // 关闭流，确保数据被完整解压并存入out
}

ssize_t myRead(const int sock, void *p, const size_t length)
{
    ssize_t result = 0;

    int left = length;
    while (left > 0)
    {
        int n = read(sock, p, left);
        if (n < 0)
        {
            result = n;
            break;
        } else if (n == 0)
        {
            break;
        } else {
            result += n;
            p += n;
            left -= n;
        }
    }

    return result;
}

enum class ErrorState
{
    Empty,
    A,
    B,
    C,
    D,
    E,
    F,
    G
};

void findHeader(const int sock)
{
    ErrorState errorState {ErrorState::Empty};
    char c;
    int n;
    while (true)
    {
        switch (errorState)
        {
        case ErrorState::Empty:
            n = myRead(sock, &c, 1);
            if (1 == n && 'A' == c)
            {
                errorState = ErrorState::A;
            }
            break;
        case ErrorState::A:
            n = myRead(sock, &c, 1);
            if (1 == n && 'B' == c)
            {
                errorState = ErrorState::B;
            } else {
                errorState = ErrorState::Empty;
            }
            break;
        case ErrorState::B:
            n = myRead(sock, &c, 1);
            if (1 == n && 'C' == c)
            {
                errorState = ErrorState::C;
            } else {
                errorState = ErrorState::Empty;
            }
            break;
        case ErrorState::C:
            n = myRead(sock, &c, 1);
            if (1 == n && 'D' == c)
            {
                errorState = ErrorState::D;
            } else {
                errorState = ErrorState::Empty;
            }
            break;
        case ErrorState::D:
            n = myRead(sock, &c, 1);
            if (1 == n && 'E' == c)
            {
                errorState = ErrorState::E;
            } else {
                errorState = ErrorState::Empty;
            }
            break;
        case ErrorState::E:
            n = myRead(sock, &c, 1);
            if (1 == n && 'F' == c)
            {
                errorState = ErrorState::F;
            } else {
                errorState = ErrorState::Empty;
            }
            break;
        case ErrorState::F:
            n = myRead(sock, &c, 1);
            if (1 == n && 'G' == c)
            {
                errorState = ErrorState::G;
            } else {
                errorState = ErrorState::Empty;
            }
            break;
        case ErrorState::G:
            n = myRead(sock, &c, 1);
            if (1 == n && 'H' == c)
            {
                TINFO << "recover";
                return;
            } else {
                errorState = ErrorState::Empty;
            }
            break;
        }
    }
}

enum class DataState
{
    Header,
    Size,
    Data,
    Error,
};

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

    int sock{-1};
    char tmpMem[1024 * 1024 * 10];


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
        TINFO << "fps_readXYZ: " << count_readXYZ;

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
        TINFO << "fps_readM3J: " << count_readM3J;

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

    pull_stream_thread_M3J_ = new std::thread(&ActivityMultiModal::PullStreamThreadFunc_M3J, this);
    pull_stream_thread_XYZ_Color = new std::thread(&ActivityMultiModal::PullStreamThreadFunc_XYZ_Color, this);
    pull_stream_thread_XYZ_Depth = new std::thread(&ActivityMultiModal::PullStreamThreadFunc_XYZ_Depth, this);

    counter_message_producer_thread_ = new std::thread(&ActivityMultiModal::ImgProducerThreadFunc, this);

    message_producer_thread_ = std::make_unique<std::thread>(&ActivityMultiModal::MessageProducerThreadFunc, this);
}

// void saveToDisk_M3JUVC(const int serial, const std::vector<uint8_t>& img)
// {
//     // 640 * 512 == 327680
//     void *p = malloc(640 * 512 * 3);
//     memcpy(p, img.data(), 640 * 512 * 3);
//     cv::Mat matImg(cv::Size(640, 512), CV_8UC3, p);

//     std::filesystem::path dir = "/workspace/ddsproject-example/tmp/M3JUVC_multiModal/";
//     std::filesystem::create_directories(dir);

//     std::string filePath = dir.string() + std::to_string(serial) + ".jpg";
//     if (cv::imwrite(filePath, matImg)) {
//         ;
//     } else {

//     }

//     free(p);
// }


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
                TINFO << "==========fps_M3J: " << count_M3J;

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
                TINFO << "fps_Color: " << count_XYZ_Color;

                tmpTimeStamp_XYZ_Color = getTimestamp();
                count_XYZ_Color = 0;
            }
        }
    );
}

void ActivityMultiModal::PullStreamThreadFunc_XYZ_Depth()
{
    sock = socket(AF_INET, SOCK_STREAM, 0);

    sockaddr_in serv_addr;
    memset(&serv_addr, 0, sizeof(serv_addr));
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_addr.s_addr = inet_addr("192.168.3.56");
    serv_addr.sin_port = htons(1234);

    int ret = connect(sock, (sockaddr*)&serv_addr, sizeof(serv_addr));
    if (ret != 0)
    {
        TINFO << "connect failed. errno: " << errno;

        return;
    }

    {
        
        DataState state {DataState::Header};
        char Header[] = "ABCDEFGH";
        int size = 0;
        int n;

        while(is_running_.load())
        {
            switch (state)
            {
            case DataState::Header:
                n = myRead(sock, tmpMem, 8);
                if (8 != n)
                {
                    state = DataState::Error;
                } else {
                    tmpMem[8] = '\0';
                    if (0 != strcmp(tmpMem, Header))
                    {
                        state = DataState::Error;
                    } else {
                        state = DataState::Size;
                    }
                }
                break;
            case DataState::Size:
                n = myRead(sock, &size, 4);
                if (4 != n)
                {
                    state = DataState::Error;
                } else {
                    state = DataState::Data;
                }
                break;
            case DataState::Data:
                // no leak here

                n = myRead(sock, tmpMem, size);
                if (size != n)
                {
                    state = DataState::Error;
                } else {
                    ///////// already leak

                    bool bSuccess {true};
                    std::vector<char> out;
                    try {
                        decompress(tmpMem, size, out);
                    } catch (boost::wrapexcept<boost::iostreams::gzip_error> e)
                    {
                        TINFO << e.what() << std::endl;
                        bSuccess = false;
                    }

                    if (bSuccess)
                    {
                        FrameFromStream frame;
                        frame.width = 1280;
                        frame.height = 720;
                        frame.timestamp = getTimestamp();
                        frame.vecBuf.resize(1280 * 720 * 2);
                        memcpy(&frame.vecBuf[0], out.data(), 1280 * 720 * 2);

                        frame_safeDeque_XYZ_Depth.PushBack(std::make_shared<FrameFromStream>(frame));

                        count_XYZ_Depth++;
                        if (getTimestamp() - tmpTimeStamp_XYZ_Depth > 1000)
                        {
                            TINFO << "fps_Depth: " << count_XYZ_Depth;

                            tmpTimeStamp_XYZ_Depth = getTimestamp();
                            count_XYZ_Depth = 0;
                        }

                        state = DataState::Header;
                    } else {
                        state = DataState::Error;
                    }
                }
                break;
            case DataState::Error:
                findHeader(sock);
                state = DataState::Size;
                break;
            }
        }
            
    }

    close(sock);
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
            TINFO << "=====matched: "
                << "M3J: " << timestamp_M3J << " ========== "
                << "XYZ_Color: " << timestamp_XYZ_Color << " ========== "
                << "XYZ_Depth: " << timestamp_XYZ_Depth << " ========== ";

            count++;
            if (getTimestamp() - tmpTimeStamp > 1000)
            {
                TINFO << "\fps: " << count;

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


            // cv::Mat mat_C(ptr_XYZ_Color->height, ptr_XYZ_Color->width, CV_8UC3, videoSrcData_XYZ.vecImageBuf().data());
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

            // std::vector<uint8_t> tmpSrcVec;
            // tmpSrcVec.resize(1280 * 720 * 2);
            // for (int i = 0; i < 1280 * 720; ++i)
            // {
            //     tmpSrcVec[i*2 + 0] = ptr_XYZ_Depth->vecBuf[i*3 + 0];
            //     tmpSrcVec[i*2 + 1] = ptr_XYZ_Depth->vecBuf[i*3 + 1];
            // }

            // std::cout << (int)ptr_XYZ_Depth->vecBuf[60*3 + 1] << std::endl;
            // std::cout << (int)ptr_XYZ_Depth->vecBuf[61*3 + 1] << std::endl;
            // std::cout << (int)ptr_XYZ_Depth->vecBuf[62*3 + 1] << std::endl;

            // std::cout << (int)ptr_XYZ_Depth->vecBuf[60*3 + 2] << std::endl;
            // std::cout << (int)ptr_XYZ_Depth->vecBuf[61*3 + 2] << std::endl;
            // std::cout << (int)ptr_XYZ_Depth->vecBuf[62*3 + 2] << std::endl;

            std::vector<int32_t> tmpVec;
            tmpVec.resize(1280 * 720);
            auto t1 = std::chrono::high_resolution_clock::now();
            depthConverter.process_gpu(ptr_XYZ_Depth->vecBuf, tmpVec);
            auto t2 = std::chrono::high_resolution_clock::now();
            double gpu_cost = std::chrono::duration<double, std::milli>(t2 - t1).count();
            TINFO << "[GPU] depthConverter.process_gpu 耗时: " << gpu_cost << " ms";
            // depthConverter.process(ptr_XYZ_Depth->vecBuf, tmpVec);

            disparityResult.vecDistanceInfo(tmpVec);

            // int center = 1280 * 720 / 2;
            // int center_a1 = center - 1280;
            // int center_a2 = center - 2 * 1280;
            // int center_b1 = center + 1280;
            // int center_b2 = center + 2 *1280;
            // std::cout
            //     << tmpVec[center_a2 - 2] << " " << tmpVec[center_a2 - 1] << " " << tmpVec[center_a2] << " " << tmpVec[center_a2 + 1] << " " << tmpVec[center_a2 + 2] << "\n"
            //     << tmpVec[center_a1 - 2] << " " << tmpVec[center_a1 - 1] << " " << tmpVec[center_a1] << " " << tmpVec[center_a1 + 1] << " " << tmpVec[center_a1 + 2] << "\n"
            //     << tmpVec[center - 2] << " " << tmpVec[center - 1] << " " << tmpVec[center] << " " << tmpVec[center + 1] << " " << tmpVec[center + 2] << "\n"
            //     << tmpVec[center_b1 - 2] << " " << tmpVec[center_b1 - 1] << " " << tmpVec[center_b1] << " " << tmpVec[center_b1 + 1] << " " << tmpVec[center_b1 + 2] << "\n"
            //     << tmpVec[center_b2 - 2] << " " << tmpVec[center_b2 - 1] << " " << tmpVec[center_b2] << " " << tmpVec[center_b2 + 1] << " " << tmpVec[center_b2 + 2] << "\n"
            //     << std::endl;



            // cv::Mat mat_0(ptr_XYZ_Depth->height, ptr_XYZ_Depth->width, CV_8UC3, ptr_XYZ_Depth->vecBuf.data());
            // cv::imwrite(std::string("/share/tmpimage/Depth/a0") + std::to_string(index) + ".jpg", mat_0);
        


            // std::vector<uint8_t> tmptmptmp;
            // tmptmptmp.resize(1280 * 720);
            // for (int i = 0; i < 1280 * 720; ++i)
            // {
            //     tmptmptmp[i] = tmpVec[i] / 256;
            // }
            // cv::Mat mat(ptr_XYZ_Depth->height, ptr_XYZ_Depth->width, CV_8UC1, tmptmptmp.data());
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
            TINFO << "=====mismatched: "
                << "M3J: " << timestamp_M3J << " ========== "
                << "XYZ_Color: " << timestamp_XYZ_Color << " ========== "
                << "XYZ_Depth: " << timestamp_XYZ_Depth << " ========== ";
            

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
