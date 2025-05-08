/*******************************************************
 文件名：AlgorithmConfig.cpp
 作者：
 描述：算法配置类实现
 版本：v1.0
 日期：2024-03-21
 *******************************************************/

#include "AlgorithmConfig.h"
#include <iostream>
#include <fstream>

AlgorithmConfig::AlgorithmConfig()
{
}

AlgorithmConfig::~AlgorithmConfig()
{
}

bool AlgorithmConfig::loadFromFile(const std::string& configPath)
{
    try {
        // 加载YAML文件
        YAML::Node config = YAML::LoadFile(configPath);
        
        // 解析全局参数
        if (config["global_params"]) {
            if (!parseParams(config["global_params"], m_globalParams)) {
                std::cerr << "Failed to parse global parameters" << std::endl;
                return false;
            }
        }

        // 解析模块配置
        if (!config["modules"] || !config["modules"].IsSequence()) {
            std::cerr << "No modules configuration found" << std::endl;
            return false;
        }

        m_moduleConfigs.clear();
        for (const auto& moduleNode : config["modules"]) {
            ModuleConfig moduleConfig;
            if (!parseModuleConfig(moduleNode, moduleConfig)) {
                std::cerr << "Failed to parse module configuration" << std::endl;
                return false;
            }
            m_moduleConfigs.push_back(moduleConfig);
        }

        return true;
    }
    catch (const YAML::Exception& e) {
        std::cerr << "YAML parsing error: " << e.what() << std::endl;
        return false;
    }
    catch (const std::exception& e) {
        std::cerr << "Error loading config file: " << e.what() << std::endl;
        return false;
    }
}

bool AlgorithmConfig::parseModuleConfig(const YAML::Node& node, ModuleConfig& config)
{
    try {
        // 解析模块名称
        if (!node["name"] || !node["name"].IsScalar()) {
            std::cerr << "Module name is required" << std::endl;
            return false;
        }
        config.moduleName = node["name"].as<std::string>();

        // 解析模块类型
        if (!node["type"] || !node["type"].IsScalar()) {
            std::cerr << "Module type is required" << std::endl;
            return false;
        }
        config.moduleType = node["type"].as<std::string>();

        // 解析模块参数
        if (node["params"]) {
            if (!parseParams(node["params"], config.params)) {
                std::cerr << "Failed to parse module parameters" << std::endl;
                return false;
            }
        }

        return true;
    }
    catch (const YAML::Exception& e) {
        std::cerr << "Error parsing module config: " << e.what() << std::endl;
        return false;
    }
}

bool AlgorithmConfig::parseParams(const YAML::Node& node, std::map<std::string, std::string>& params)
{
    try {
        if (!node.IsMap()) {
            std::cerr << "Parameters must be a map" << std::endl;
            return false;
        }

        params.clear();
        for (const auto& param : node) {
            std::string key = param.first.as<std::string>();
            std::string value;

            // 根据参数类型转换为字符串
            if (param.second.IsScalar()) {
                value = param.second.as<std::string>();
            }
            else if (param.second.IsSequence()) {
                // 对于数组类型的参数，将其转换为逗号分隔的字符串
                std::vector<std::string> values;
                for (const auto& v : param.second) {
                    values.push_back(v.as<std::string>());
                }
                value = values[0];
                for (size_t i = 1; i < values.size(); ++i) {
                    value += "," + values[i];
                }
            }
            else {
                std::cerr << "Unsupported parameter type for: " << key << std::endl;
                continue;
            }

            params[key] = value;
        }

        return true;
    }
    catch (const YAML::Exception& e) {
        std::cerr << "Error parsing parameters: " << e.what() << std::endl;
        return false;
    }
} 