#ifndef LOGGER_LOGGER_UTIL_HPP
#define LOGGER_LOGGER_UTIL_HPP

#include <cstdint>
#include <unistd.h>
#include <sys/utsname.h>
#define GLOG_USE_GLOG_EXPORT 
#include <glog/logging.h>

#include <sys/time.h>

inline uint32_t MaxLogSize()
{
    return (FLAGS_max_log_size > 0 && FLAGS_max_log_size < 4096) ? FLAGS_max_log_size : 1;
}

bool PidHasChanged();

int32_t GetMainThreadPid();

static inline void GetHostName(std::string* hostname)
{
    struct utsname buf;
    if (uname(&buf) < 0) {
        *buf.nodename = '\0';
    }
    *hostname = buf.nodename;
}

int64_t GetNowTimestamp();

void FindModuleName(std::string* message, std::string *module_name);

#endif