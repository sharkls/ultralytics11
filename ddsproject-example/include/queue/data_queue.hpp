#ifndef CACHE_DATA_QUEUE
#define CACHE_DATA_QUEUE

#include <memory>
#include <condition_variable>

#include "include/queue/data_cache.hpp"
#include "include/common/log.hpp"

template <typename T>
using BufferType = CacheBuffer<std::shared_ptr<T>>;

template <typename T>
class DataQueue
{
public:
    explicit DataQueue(const uint32_t& size);
    ~DataQueue();

    bool Latest(std::shared_ptr<T>& m);
    bool Fetch(std::shared_ptr<T>& m);
    bool Fill(const std::shared_ptr<T>& value);

private:
    std::shared_ptr<BufferType<T>> buffer_;
    uint64_t next_msg_index_{0};
    std::condition_variable cv_;
};


template <typename T>
DataQueue<T>::DataQueue(const uint32_t& size)
{
    buffer_ = std::shared_ptr<BufferType<T>>(new BufferType<T>(size));
}

template <typename T>
DataQueue<T>::~DataQueue()
{
}

template <typename T>
bool DataQueue<T>::Latest(std::shared_ptr<T> &m)
{
    std::lock_guard<std::mutex> lock(buffer_->Mutex());
    if (buffer_->Empty())
    {
        return false;
    }
    m = buffer_->Back();
    return true;
}

template <typename T>
bool DataQueue<T>::Fetch(std::shared_ptr<T> &m)
{
    std::lock_guard<std::mutex> lock(buffer_->Mutex());
    if (buffer_->Empty())
    {
        return false;
    }
    
    if (next_msg_index_ == 0)
    {
        next_msg_index_ = buffer_->Tail();
    }
    else if (next_msg_index_ == buffer_->Tail() + 1)
    {
        return false;
    }
    else
    {
        auto interval = buffer_->Tail() - next_msg_index_;
        next_msg_index_ = buffer_->Tail();
    }
    m = buffer_->at(next_msg_index_);
    next_msg_index_++;
    return true;
}

template <typename T>
bool DataQueue<T>::Fill(const std::shared_ptr<T> &value)
{
    {
        std::lock_guard<std::mutex> lock(buffer_->Mutex());
        buffer_->Fill(value);
    }
    return true;
}

#endif