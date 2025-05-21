#ifndef LOG_HPP
#define LOG_HPP

#include <cstdarg>
#include <string>

#define GLOG_USE_GLOG_EXPORT
#include <glog/log_severity.h>
#include <glog/logging.h>

#include "include/common/binary.hpp"

#ifndef MOUDLE_NAME
#define MODULE_NAME GetName().c_str()
#endif

#define LEFT_BRACKET "["
#define RIGHT_BRACKET "]"

#define DEBUG_MODULE(module) VLOG(4) << LEFT_BRACKET << module << RIGHT_BRACKET << "[DEBUG] "

#ifndef LOG_MODULE_STREAM
#define LOG_MODULE_STREAM(log_severity) \
        LOG_MODULE_STREAM_##log_severity
#endif

#ifndef LOG_MODULE
#define LOG_MODULE(module, log_severity) \
        LOG_MODULE_STREAM(log_severity)(module)
#endif

#define LOG_MODULE_STREAM_INFO(module)                                  \
        google::LogMessage(__FILE__, __LINE__, google::INFO).stream()   \
        << LEFT_BRACKET << module << RIGHT_BRACKET

#define LOG_MODULE_STREAM_WARN(module)                                      \
        google::LogMessage(__FILE__, __LINE__, google::WARNING).stream()    \
        << LEFT_BRACKET << module << RIGHT_BRACKET

#define LOG_MODULE_STREAM_ERROR(module)                                 \
        google::LogMessage(__FILE__, __LINE__, google::ERROR).stream()  \
        << LEFT_BRACKET << module << RIGHT_BRACKET

#define LOG_MODULE_STREAM_FATAL(module)                                 \
        google::LogMessage(__FILE__, __LINE__, google::FATAL).stream()  \
        << LEFT_BRACKET << module << RIGHT_BRACKET

#define TDEBUG  DEBUG_MODULE(MODULE_NAME)
#define TINFO   LOG_MODULE(MODULE_NAME, INFO)
#define TWARN   LOG_MODULE(MODULE_NAME, WARN)
#define TERROR  LOG_MODULE(MODULE_NAME, ERROR)
#define TFATAL  LOG_MODULE(MODULE_NAME, FATAL)

#endif