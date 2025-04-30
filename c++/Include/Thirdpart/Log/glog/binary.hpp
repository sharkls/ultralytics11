#ifndef COMMON_BINARY_HPP
#define COMMON_BINARY_HPP

#include <string>
#include <mutex>

std::string GetName();
void SetName(const std::string& name); // 函数声明

#endif

namespace
{
    std::mutex m;
    std::string binary_name = "";
}

inline std::string GetName()
{
    std::lock_guard<std::mutex> lock(m);
    return binary_name;
}

// 添加 SetName 函数的实现
inline void SetName(const std::string& name)
{
    std::lock_guard<std::mutex> lock(m);
    binary_name = name;
}