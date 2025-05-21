#ifndef READER_HPP
#define READER_HPP

#include <fastdds/dds/topic/Topic.hpp>
#include <fastdds/dds/subscriber/Subscriber.hpp>
#include <fastdds/dds/subscriber/DataReader.hpp>
#include <fastdds/dds/subscriber/DataReaderListener.hpp>
#include <fastdds/dds/subscriber/SampleInfo.hpp>
#include <fastdds/dds/subscriber/qos/DataReaderQos.hpp>
#include <fastdds/dds/domain/DomainParticipant.hpp>

#include "include/common/log.hpp"

template <typename MessageT>
class Reader
{
public:
    using CallbackFunc = std::function<void(const MessageT &, // 消息
                                            void *,           // data_handler
                                            std::string,      // node_name
                                            std::string)>;    // topic_name

    Reader(eprosima::fastdds::dds::Subscriber *sub,
           eprosima::fastdds::dds::Topic *topic,
           const eprosima::fastdds::dds::DataReaderQos &qos,
           const CallbackFunc &read_func,
           void *data_handle);
    ~Reader();

    bool Init(void);

private:
    class ReaderListener : public eprosima::fastdds::dds::DataReaderListener
    {
    public:
        explicit ReaderListener(const CallbackFunc &callback, void *data_handle)
            : callback_(callback), data_handle_{data_handle} {}
        virtual ~ReaderListener() {}

        void on_subscription_matched(
            eprosima::fastdds::dds::DataReader *reader,
            const eprosima::fastdds::dds::SubscriptionMatchedStatus &info);

        void on_data_available(eprosima::fastdds::dds::DataReader *reader);

        int matched_{0};

    private:
        CallbackFunc callback_{NULL};
        void *data_handle_{NULL};
    } listener_;

    eprosima::fastdds::dds::Subscriber *subscriber_;
    eprosima::fastdds::dds::Topic *topic_;
    eprosima::fastdds::dds::DataReader *data_reader_;
    const eprosima::fastdds::dds::DataReaderQos data_reader_qos_;
};

template <typename MessageT>
Reader<MessageT>::Reader(eprosima::fastdds::dds::Subscriber *sub,
                         eprosima::fastdds::dds::Topic *topic,
                         const eprosima::fastdds::dds::DataReaderQos &qos,
                         const CallbackFunc &read_func,
                         void *data_handle)
    : subscriber_(sub),
      topic_(topic),
      data_reader_qos_(qos),
      listener_(read_func, data_handle),
      data_reader_(nullptr)
{
}

template <typename MessageT>
Reader<MessageT>::~Reader()
{
    if (data_reader_)
    {
        subscriber_->delete_datareader(data_reader_);
        data_reader_ = nullptr;
    }
}

template <typename MessageT>
bool Reader<MessageT>::Init(void)
{
    if (!data_reader_)
    {
        data_reader_ = subscriber_->create_datareader(topic_, data_reader_qos_, &listener_);
    }
    if (!data_reader_)
    {
        return false;
    }
    return true;
}

template <typename MessageT>
void Reader<MessageT>::ReaderListener::on_subscription_matched(eprosima::fastdds::dds::DataReader *reader,
                                                               const eprosima::fastdds::dds::SubscriptionMatchedStatus &info)
{
    matched_ = info.current_count;
    std::string node_name = reader->get_subscriber()->get_participant()->get_qos().name().to_string();
    std::string topic_name = reader->get_topicdescription()->get_name();

    if (info.current_count_change == 1)
    {
        TINFO << node_name << " : " << topic_name << " reader matched. currently matched " << matched_;
    }
    else if (info.current_count_change == -1)
    {
        TINFO << node_name << " : " << topic_name << " reader unmatched. currently matched " << matched_;
    }
}

template <typename MessageT>
void Reader<MessageT>::ReaderListener::on_data_available(eprosima::fastdds::dds::DataReader *reader)
{
    if (!callback_)
    {
        TERROR << "please initialize a data processing callback func.";
        return;
    }

    std::string node_name = reader->get_subscriber()->get_participant()->get_qos().name().to_string();
    std::string topic_name = reader->get_topicdescription()->get_name();

    FASTDDS_CONST_SEQUENCE(DataSeq, MessageT);

    DataSeq data;
    eprosima::fastdds::dds::SampleInfoSeq info_seq;

    while (ReturnCode_t::RETCODE_OK == reader->take(data, info_seq))
    {
        for (int i = 0; i < info_seq.length(); i++)
        {
            if (info_seq[i].valid_data)
            {
                callback_(data[i], data_handle_, node_name, topic_name);
            }
        }
        reader->return_loan(data, info_seq);
    }
}

#endif