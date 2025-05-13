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

// 保存 float 数组为二进制文件
inline void save_bin(const std::vector<float>& input_data, const std::string& filename) {
    std::ofstream ofs(filename, std::ios::binary);
    ofs.write(reinterpret_cast<const char*>(input_data.data()), input_data.size() * sizeof(float));
    ofs.close();
}