#ifndef LOGGER_LOGGER_HPP
#define LOGGER_LOGGER_HPP

#define GLOG_USE_GLOG_EXPORT
#include <glog/logging.h>

#include <mutex>

class Logger : public google::base::Logger
{
public:
    Logger(google::base::Logger* wrapped);
    ~Logger();
    void Write(bool force_flush,
               const std::chrono::system_clock::time_point &timestamp,
               const char *message, size_t message_len) override;
    void Flush() override;
    uint32_t LogSize() override;

private:
    google::base::Logger* wrapped_;
    std::mutex mtx_;
};

#endif