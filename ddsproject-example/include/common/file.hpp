#ifndef COMMON_FILE_HPP
#define COMMON_FILE_HPP

#include <string>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>

#include "include/common/log.hpp"

/**
 * @brief check if the path exists.
 * @param path a file name
 * @return if the path exists, return true
 */
bool PathExists(const std::string &path);

/**
 * @brief 通过拼接prefix和relative_path获取绝对路径
 * @param prefix            整个工程的绝对路径
 * @param relative_path     在当前项目中的相对路径
 * @return
 */
std::string GetAbsolutePath(const std::string &prefix,
                            const std::string &relative_path);

/**
 * @brief 将proto格式的内容保存为二进制文件
 */
bool SetProtoToBinaryFile(const google::protobuf::Message &message, const std::string &file_name);

/**
 * @brief 以二进制的方式去解析文件内容
 * @param file_name : 待解析文件
 * @param message : 以proto格式去解析文件内容
 */
bool GetProtoFromBinaryFile(const std::string &file_name, google::protobuf::Message *message);

/**
 * @brief 将proto格式的内容保存为ASCII文件
 */
bool SetProtoToASCIIFile(const google::protobuf::Message &message, const std::string &file_name);

/**
 * @brief 以ascii的方式解些文件内容
 * @param file_name : 待解析文件
 * @param message : 以proto格式去解析文件内容
 */
bool GetProtoFromASCIIFile(const std::string &file_name, google::protobuf::Message *message);

/**
 * @brief 使用proto格式去解析文件中的内容
 * @param file_name : 待解析文件
 * @param message : 以proto格式去解析文件内容
 */
bool GetProtoFromFile(const std::string &file_name, google::protobuf::Message *message);


std::string GetCurrentPath();

std::string GetFileName(const std::string& path, const bool remove_extension = true);

#endif