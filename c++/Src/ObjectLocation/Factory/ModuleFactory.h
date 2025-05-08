/*******************************************************
 文件名：ModuleFactory.h
 作者：
 描述：模块工厂类，负责创建各种类型的模块实例
 版本：v1.0
 日期：2024-03-21
 *******************************************************/

#ifndef MODULE_FACTORY_H
#define MODULE_FACTORY_H

#include <memory>
#include <string>
#include <map>
#include "../Common/IBaseModule.h"

// 模块创建函数类型
using ModuleCreator = std::function<std::shared_ptr<IBaseModule>()>;

class ModuleFactory {
public:
    static ModuleFactory& getInstance();

    // 注册模块创建器
    void registerModule(const std::string& moduleName, ModuleCreator creator);
    
    // 创建模块实例
    std::shared_ptr<IBaseModule> createModule(const std::string& moduleName);

private:
    ModuleFactory() = default;
    ~ModuleFactory() = default;
    
    // 禁用拷贝和赋值
    ModuleFactory(const ModuleFactory&) = delete;
    ModuleFactory& operator=(const ModuleFactory&) = delete;

private:
    std::map<std::string, ModuleCreator> m_moduleCreators;  // 模块创建器映射表
};

// 模块注册宏
#define REGISTER_MODULE(moduleName, className) \
    static bool moduleName##_registered = []() { \
        ModuleFactory::getInstance().registerModule(#moduleName, \
            []() { return std::make_shared<className>(); }); \
        return true; \
    }();

#endif // MODULE_FACTORY_H 