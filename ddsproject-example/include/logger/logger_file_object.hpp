#ifndef LOGGER_LOGGER_FILE_OBJECT_HPP
#define LOGGER_LOGGER_FILE_OBJECT_HPP

#define GLOG_USE_GLOG_EXPORT 
#include <glog/logging.h>
#include <mutex>
#include <iomanip>
#include <cstdint>
#include <fcntl.h>
#include <unistd.h>
#include <ctime>
#include <time.h>
#include <assert.h>

#include "include/common/log.hpp"
#include "include/logger/logger_util.hpp"

class LoggerFileObject : public google::base::Logger
{
public:
    LoggerFileObject(google::LogSeverity severity, const char* base_filename);
    ~LoggerFileObject();
    void Write(bool force_flush,
               const std::chrono::system_clock::time_point &timestamp,
               const char *message, size_t message_len) override;
    void Flush() override;
    uint32_t LogSize() override;

    void SetBasename(const char* basename);
    void SetExtension(const char* ext);
    void SetSymlinkBasename(const char* symlink_basename);

    void FlushUnlocked();

private:
    // Acturally create a logfile using the value of base_filename_ and the optional argument time_pid_string
    bool CreateLogfile(const std::string& time_pid_string);
    const std::string& Hostname();

    static const uint32_t kRolloverAttemptFrequency = 0x20;

    std::mutex mtx_;
    bool base_filename_selected_;
    std::string base_filename_;
    std::string symlink_basename_;
    std::string filename_extension_;
    std::unique_ptr<FILE> file_;
    google::LogSeverity severity_;
    uint32_t bytes_since_flush_{0};
    uint32_t dropped_mem_length_{0};
    uint32_t file_length_{0};
    unsigned int rollover_attempt_;
    int64_t next_flush_time_;

    std::string hostname_;
};


#endif