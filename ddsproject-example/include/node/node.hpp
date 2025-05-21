#ifndef NODEBASE_HPP
#define NODEBASE_HPP

#include <fastdds/dds/domain/DomainParticipant.hpp>
#include <string>

#include "include/common/log.hpp"
#include "include/node/reader.hpp"
#include "include/node/writer.hpp"
#include "proto/node_conf.pb.h"

using eprosima::fastdds::dds::TypeSupport;

// DDS通信类型：UDP、SHM、DEFAULT
enum class COMMUNICATION_TYPE
{
    UDP,
    SHM,
    DEFAULT
};

/** 每个活动节点的配置信息
 *      域名
 *      通信方式
 *      segment size:针对SHM通信方式
 *      whitelist:针对udp通信方式
 */
struct NodeConf
{
    uint32_t domain_ID;
    COMMUNICATION_TYPE comm_type;
    uint32_t segment_size;
    std::string white_list; // 白名单，用于指定使用的网卡

    NodeConf()
    {
        domain_ID = 0;
        comm_type = COMMUNICATION_TYPE::DEFAULT;
        segment_size = 100 * 1024 * 1024;
        white_list = "";
    }

    NodeConf(const NodeConf &config)
    {
        if (this == &config)
        {
            return;
        }

        domain_ID = config.domain_ID;
        comm_type = config.comm_type;
        segment_size = config.segment_size;
        white_list = config.white_list;
    }

    NodeConf &operator=(const NodeConf &config)
    {
        if (this == &config)
        {
            return *this;
        }

        domain_ID = config.domain_ID;
        comm_type = config.comm_type;
        segment_size = config.segment_size;
        white_list = config.white_list;
    }

    void print()
    {
        TINFO << "NodeDDSConfig: ";
        TINFO << "\t"
              << "domianID: " << domain_ID;
        switch (comm_type)
        {
        case COMMUNICATION_TYPE::UDP:
            TINFO << "\t"
                  << "communication type: UDP";
            TINFO << "\t"
                  << "white list: " << white_list;
            break;
        case COMMUNICATION_TYPE::SHM:
            TINFO << "\t"
                  << "communication type: SHM";
            TINFO << "\t"
                  << "segment size: " << segment_size;
            break;
        case COMMUNICATION_TYPE::DEFAULT:
            TINFO << "\t"
                  << "communication type: DEFAULT(SHM+UDP)";
            TINFO << "\t"
                  << "segment size: " << segment_size;
            TINFO << "\t"
                  << "white list: " << white_list;
            break;
        default:
            TINFO << "\t"
                  << "comm type: unsurported.";
            break;
        }
    }
};

template <typename MessageT>
using CallbackFunc = std::function<void(const MessageT &, // 消息
                                        void *,           // data_handler
                                        std::string,      // node_name
                                        std::string)>;    // topic_name

class Node
{
public:
    // template <typename MessageT>
    // using CallbackFunc = std::function<void(const MessageT &, // 消息
    //                                         void *,           // data_handler
    //                                         std::string,      // node_name
    //                                         std::string)>;    // topic_name

    Node() = default;
    virtual ~Node();

    int8_t Init(const NodeConfig& config);
    // int8_t Init(const std::string &name, const NodeConf &config);
    bool AddTopic(const std::string &topic_name, eprosima::fastdds::dds::TypeSupport type);
    bool RemoveTopic(const std::string &topic_name);

    template <typename MessageT>
    auto CreateReader(const std::string &topic_name,
                      const CallbackFunc<MessageT> &read_func,
                      void *data_handler = nullptr) -> std::shared_ptr<Reader<MessageT>>;

    auto CreateWriter(const std::string &topic_name) -> std::shared_ptr<Writer>;

private:
    eprosima::fastdds::dds::DataReaderQos GetDefaultDataReaderQos();
    eprosima::fastdds::dds::DataWriterQos GetDefaultDataWriterQos();

private:
    std::string name_;

    std::map<std::string, eprosima::fastdds::dds::TypeSupport> message_types_; // key为topic data type name，value为数据类型对象
    std::map<std::string, eprosima::fastdds::dds::Topic *> topics_;            // key为topic name，value为话题对象

    eprosima::fastdds::dds::DomainParticipant *participant_;
    eprosima::fastdds::dds::Publisher *publisher_;
    eprosima::fastdds::dds::Subscriber *subscriber_;
};

template <typename MessageT>
auto Node::CreateReader(const std::string &topic_name,
                        const CallbackFunc<MessageT> &read_func,
                        void *data_handler) -> std::shared_ptr<Reader<MessageT>>
{
    auto it = topics_.find(topic_name);
    if (it == topics_.end())
    {
        TERROR << "not add " << topic_name << " , please add this topic by call Node::AddTopic forstly ...";
        return nullptr;
    }
    return std::make_shared<Reader<MessageT>>(subscriber_, it->second, GetDefaultDataReaderQos(), read_func, data_handler);
}

#endif