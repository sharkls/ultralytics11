/*******************************************************
 文件名：AlgorithmConfig.h
 作者：
 描述：算法配置类，用于解析和管理算法配置
 版本：v1.0
 日期：2024-03-21
 *******************************************************/

#ifndef ALGORITHM_CONFIG_H
#define ALGORITHM_CONFIG_H

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <yaml-cpp/yaml.h>

// 模块配置结构
struct ModuleConfig {
    std::string moduleName;           // 模块名称
    std::string moduleType;           // 模块类型
    std::map<std::string, std::string> params;  // 模块参数
};

class AlgorithmConfig {
public:
    AlgorithmConfig();
    ~AlgorithmConfig();

    // 从YAML文件加载配置
    bool loadFromFile(const std::string& configPath);
    
    // 获取模块配置列表
    const std::vector<ModuleConfig>& getModuleConfigs() const { return m_moduleConfigs; }
    
    // 获取全局参数
    const std::map<std::string, std::string>& getGlobalParams() const { return m_globalParams; }

private:
    // 解析YAML节点到模块配置
    bool parseModuleConfig(const YAML::Node& node, ModuleConfig& config);
    
    // 解析YAML节点到参数映射
    bool parseParams(const YAML::Node& node, std::map<std::string, std::string>& params);

private:
    std::vector<ModuleConfig> m_moduleConfigs;  // 模块配置列表
    std::map<std::string, std::string> m_globalParams;  // 全局参数
};

#endif // ALGORITHM_CONFIG_H 