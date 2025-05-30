#ifndef LOGGER_ASYNC_LOGGER_HPP
#define LOGGER_ASYNC_LOGGER_HPP

#define GLOG_USE_GLOG_EXPORT 
#include <glog/logging.h>

#include <atomic>
#include <deque>
#include <memory>
#include <unordered_map>
#include <thread>

#include "include/logger/logger_file_object.hpp"
#include "include/common/macros.hpp"

class AsyncLogger : public google::base::Logger 
{
public:
    AsyncLogger();
    ~AsyncLogger();

    void Start();
    void Stop();

    void Write(bool force_flush, const std::chrono::system_clock::time_point& timestamp, const char* message, size_t message_len) override;
    void Flush() override;
    uint32_t LogSize() override;

    std::thread* LogThread();

private:
    struct Msg 
    {
        time_t ts;
        std::string message;
        int32_t level;
        Msg() : ts(0), message(), level(google::INFO) {}
        Msg(time_t ts, std::string&& message, int32_t level)
        : ts(ts), message(std::move(message)), level(level){}
        Msg(const Msg& rsh) {
            ts = rsh.ts;
            message = rsh.message;
            level = rsh.level;
        }
        Msg(Msg&& rsh) {
            ts = rsh.ts;
            message = rsh.message;
            level = rsh.level;
        }
        Msg& operator=(Msg&& rsh) {
            ts = rsh.ts;
            message = std::move(rsh.message);
            level = rsh.level;
            return *this;
        }
        Msg& operator=(const Msg& rsh) {
            ts = rsh.ts;
            message = rsh.message;
            level = rsh.level;
            return *this;
        }
    };

    void RunThread();
    void FlushBuffer(const std::unique_ptr<std::deque<Msg>>& msg);

    std::thread log_thread_;

    std::atomic<uint64_t> flush_count_{0};
    uint64_t drop_count_{0};

    std::unique_ptr<std::deque<Msg>> active_buf_;
    std::unique_ptr<std::deque<Msg>> flushing_buf_;
    
    enum State {INTITED, RUNNING, STOPPED};
    std::atomic<State> state_ = {INTITED};
    std::atomic_flag flag_ = ATOMIC_FLAG_INIT;
    std::unordered_map<std::string, std::unique_ptr<LoggerFileObject>> module_logger_map_;

    DISALLOW_COPY_AND_ASSIGN(AsyncLogger);
};

#endif