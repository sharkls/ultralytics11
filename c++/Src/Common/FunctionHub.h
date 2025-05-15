/*******************************************************
 文件名：FunctionHub.h
 作者：sharkls
 描述：函数集，用于基础模块的运行及结果数据处理
 版本：v1.0
 日期：2025-05-13
 *******************************************************/

#pragma once
#include <vector>
#include <string>
#include <fstream>
#include <filesystem>
#include <chrono>

// 保存 float 数组为二进制文件
inline void save_bin(const std::vector<float>& input_data, const std::string& filename) 
{
    // 自动创建父目录
    std::filesystem::path file_path(filename);
    std::filesystem::create_directories(file_path.parent_path());

    std::ofstream ofs(filename, std::ios::binary);
    ofs.write(reinterpret_cast<const char*>(input_data.data()), input_data.size() * sizeof(float));
    ofs.close();
}

inline void save_bin(const std::vector<std::vector<float>>& input_data, const std::string& filename) 
{   
    // 自动创建父目录
    std::filesystem::path file_path(filename);
    std::filesystem::create_directories(file_path.parent_path());

    std::ofstream ofs(filename, std::ios::binary);
    for (const auto& v : input_data) {
        ofs.write(reinterpret_cast<const char*>(v.data()), v.size() * sizeof(float));
    }
    ofs.close();
}

/**
 * 获取当前ms UTC时间
 * 参数：
 * 返回值：ms UTC时间
 */
inline int64_t GetTimeStamp()
{
    std::chrono::time_point<std::chrono::system_clock, std::chrono::milliseconds> tp =
        std::chrono::time_point_cast<std::chrono::milliseconds>(std::chrono::system_clock::now());

    auto tmp = std::chrono::duration_cast<std::chrono::milliseconds>(tp.time_since_epoch());
    return tmp.count();
}
