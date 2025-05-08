/*******************************************************
 文件名：IBaseModule.h
 作者：
 描述：基础模块接口实现，用于基础模块的运行及结果数据处理
 版本：v1.0
 日期：2025-05-08
 *******************************************************/

#ifndef IBASE_MODULE_H
#define IBASE_MODULE_H

#include <string>
#include <memory>
#include <vector>
#include "CSelfAlgParam.h"

// 模块类型枚举
enum class ModuleType {
    PRE_PROCESS,    // 前处理模块
    INFERENCE,      // 推理模块
    POST_PROCESS    // 后处理模块
};

class IBaseModule {
public:
    IBaseModule() = default;
    virtual ~IBaseModule() = default;

    // 获取模块名称
    virtual std::string getModuleName() const = 0;
    
    // 获取模块类型
    virtual ModuleType getModuleType() const = 0;

    // 初始化模块，返回是否成功
    virtual bool init(CSelfAlgParam* p_pAlgParam) = 0;

    // 执行模块功能，返回执行结果
    virtual void* execute() = 0;

    // 设置输入数据
    virtual void setInput(void* input) = 0;

    // 获取输出数据
    virtual void* getOutput() = 0;

    // 禁用拷贝构造和赋值操作
    IBaseModule(const IBaseModule&) = delete;
    IBaseModule& operator=(const IBaseModule&) = delete;
};

// 模块工厂基类
class IModuleFactory {
public:
    virtual ~IModuleFactory() = default;
    virtual std::shared_ptr<IBaseModule> createModule(const std::string& moduleName) = 0;
};

#endif // IBASE_MODULE_H
