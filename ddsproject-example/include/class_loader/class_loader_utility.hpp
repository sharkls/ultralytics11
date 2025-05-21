#ifndef CLASS_LOADER_CLASS_LOADER_UTILITY_HPP
#define CLASS_LOADER_CLASS_LOADER_UTILITY_HPP

#include <string>
#include <map>
#include <mutex>

#include "include/class_loader/class_factory.hpp"
#include "include/common/log.hpp"
#include "include/class_loader/shared_library/shared_library.hpp"

namespace Utility
{
using ClassClassFactoryMap = std::map<std::string, AbstractClassFactoryBase *>;      // class_name : 工厂类对象 
using BaseToClassClassFactoryMapMap = std::map<std::string, ClassClassFactoryMap>;   // base(基类) : <class_name : 工厂类对象>
using ClassFactoryVector = std::vector<AbstractClassFactoryBase*>;                     // 工厂类对象数组

using SharedLibraryPtr = std::shared_ptr<SharedLibrary>;                            // 动态库
using LibPathSharedLibraryVector = std::vector<std::pair<std::string, SharedLibraryPtr>>;   // 动态库地址 ： 动态库对象

std::recursive_mutex& GetBaseToClassClassFactoryMapMapMutex();
BaseToClassClassFactoryMapMap& GetBaseToClassClassFactoryMapMap();

ClassClassFactoryMap& GetClassClassFactoryMapByBase(const std::string& base_name);
ClassFactoryVector GetAllClassFactoryObjects(const ClassClassFactoryMap& class_class_factory_map);
ClassFactoryVector GetAllClassFactoryObjects();
ClassFactoryVector GetAllClassFactoryObjectsByLibrary(const std::string& library_path);

std::recursive_mutex& GetLibPathSharedLibraryVectorMutex();
LibPathSharedLibraryVector& GetLibPathSharedLibraryVector();

/**
 * @brief 设置/获取当前动态的上下文环境：路径，用于之后在进行RegisterClass时为对应的类工厂对象设置上下文
*/
std::string GetCurrLoadingLibraryPath();
void SetCurrLoadingLibraryPath(const std::string& library_path);

/**
 * @brief 设置/获取当前动态的上下文环境：ClassLoader，用于之后在进行RegisterClass时为对应的类工厂对象设置上下文
*/
ClassLoader* GetCurrActiveClassLoader();
void SetCurrActiveClassLoader(ClassLoader* class_loader);

/**
 * @brief 判断当前动态库是否被加载
*/
bool IsLibraryLoaded(const std::string& library_path, ClassLoader* class_loader);
/**
 * @brief 判断当前的动态库是否被anybody加载
*/
bool IsLibraryLoadedByAnyBody(const std::string& library_path);

/**
 * @brief 加载动态库，设置动态库加载的上下文：librarypath和ClassLoader，用于设置各个组建对象初始化的类工厂对象的上下文
*/
bool LoadLibrary(const std::string& library_path, ClassLoader* class_loader);

/**
 * @brief unload动态库，class_loader对应的工厂类对象delete，若当前动态库的所有的工厂类都已delete，那么Unload 动态库
*/
void UnloadLibrary(const std::string& library_path, ClassLoader* class_loader);

/**
 * @brief 在进行动态库加载的时候，生成工厂类对象，并设置对应的上下文
*/
template <typename ClassName, typename Base>
void RegisterClass(const std::string &class_name, const std::string &base_name);

/**
 * @brief 获取class_loader对应的所有的valid的class name
*/
template <typename Base>
std::vector<std::string> GetValidClassNames(ClassLoader* class_loader);

/**
 * @brief 使用工厂类对象创建类对象
*/
template <typename Base>
Base* CreateClassObj(const std::string& class_name, ClassLoader* class_loader);

template <typename ClassName, typename Base>
inline void RegisterClass(const std::string &class_name, const std::string &base_name)
{
    TINFO << "register class [ " << class_name << " ], [ " << base_name << " ], [" << GetCurrLoadingLibraryPath() << " ]";
    AbstractClassFactory<Base>* class_factory_obj =
        new ClassFactory<ClassName, Base>(class_name, base_name);
    // 设置类工厂对象的上下文
    class_factory_obj->SetRelativeLibraryPath(GetCurrLoadingLibraryPath());
    class_factory_obj->AddOwnedClassLoader(GetCurrActiveClassLoader());
        
    GetBaseToClassClassFactoryMapMapMutex().lock();
    ClassClassFactoryMap& class_class_factory_map = GetClassClassFactoryMapByBase(typeid(Base).name());
    class_class_factory_map[class_name] = class_factory_obj;
    GetBaseToClassClassFactoryMapMapMutex().unlock();
}

template <typename Base>
std::vector<std::string> GetValidClassNames(ClassLoader* class_loader)
{
    std::lock_guard<std::recursive_mutex> lock(GetBaseToClassClassFactoryMapMapMutex());

    ClassClassFactoryMap& class_class_factory_map = GetClassClassFactoryMapByBase(typeid(Base).name());
    std::vector<std::string> valid_class_names;
    for (auto& class_class_factory : class_class_factory_map)
    {
        AbstractClassFactoryBase* class_factory = class_class_factory.second;
        if (class_factory && class_factory->IsOwnedBy(class_loader))
        {
            valid_class_names.emplace_back(class_class_factory.first);
        }
    }

    return valid_class_names;
}
template <typename Base>
Base *CreateClassObj(const std::string &class_name, ClassLoader *class_loader)
{
    GetBaseToClassClassFactoryMapMapMutex().lock();
    ClassClassFactoryMap& class_class_factory_map = GetClassClassFactoryMapByBase(typeid(Base).name());
    AbstractClassFactory<Base>* class_factory = nullptr;
    if (class_class_factory_map.find(class_name) != class_class_factory_map.end())
    {
        class_factory = dynamic_cast<AbstractClassFactory<Base>*>(class_class_factory_map[class_name]);
    }
    GetBaseToClassClassFactoryMapMapMutex().unlock();

    Base* classobj = nullptr;
    if (class_factory && class_factory->IsOwnedBy(class_loader))
    {
        classobj = class_factory->CreateObj();
    }
    return classobj;
}
}

#endif
