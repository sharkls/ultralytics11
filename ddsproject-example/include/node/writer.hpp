#ifndef WRITER_HPP
#define WRITER_HPP

#include <fastdds/dds/topic/Topic.hpp>
#include <fastdds/dds/publisher/Publisher.hpp>
#include <fastdds/dds/publisher/DataWriter.hpp>
#include <fastdds/dds/publisher/DataWriterListener.hpp>
#include <fastdds/dds/domain/DomainParticipant.hpp>
#include <fastdds/dds/domain/DomainParticipantFactory.hpp>

#include "include/common/log.hpp"

class Writer
{
public:
    Writer(eprosima::fastdds::dds::Publisher *pub, eprosima::fastdds::dds::Topic *topic, const eprosima::fastdds::dds::DataWriterQos &qos);
    ~Writer();

    bool Init(void);

    void* LoanSample();
    int SendMessage(void* msg, bool zero = false);

private:
    class WriterListener : public eprosima::fastdds::dds::DataWriterListener
    {
    public:
        WriterListener() = default;
        ~WriterListener() override = default;

        void on_publication_matched(eprosima::fastdds::dds::DataWriter *writer,
                                    const eprosima::fastdds::dds::PublicationMatchedStatus &info) override;

        int matched_{0};
    } listener_;

    eprosima::fastdds::dds::Publisher *publisher_{nullptr};
    eprosima::fastdds::dds::DataWriter *data_writer_{nullptr};
    eprosima::fastdds::dds::Topic *topic_{nullptr};
    eprosima::fastdds::dds::DataWriterQos qos_;
};

#endif