#pragma once
#include <string>
#include <google/protobuf/message.h>

class AlgorithmConfig {
public:
    virtual ~AlgorithmConfig() = default;
    // 加载配置文件（可重载支持不同格式）
    virtual bool loadFromFile(const std::string& path) = 0;
    // 获取protobuf配置对象
    virtual const google::protobuf::Message* getConfigMessage() const = 0;
}; 