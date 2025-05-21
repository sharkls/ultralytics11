#ifndef CLASS_LOADER_CLASS_FACTORY_HPP
#define CLASS_LOADER_CLASS_FACTORY_HPP

#include <string>
#include <typeinfo>
#include <vector>
#include <algorithm>

#include "include/common/macros.hpp"

class ClassLoader;

class AbstractClassFactoryBase
{
public:
    AbstractClassFactoryBase(const std::string &class_name, const std::string &base_name)
        : class_name_(class_name),
          base_name_(base_name)
    {
    }
    virtual ~AbstractClassFactoryBase() {}

    void SetRelativeLibraryPath(const std::string &library_path);
    void AddOwnedClassLoader(ClassLoader *class_loader);
    void RemoveOwnedClassLoader(const ClassLoader *class_loader);
    bool IsOwnedBy(const ClassLoader *class_loader);
    bool IsOwnedByAnybody();

    std::vector<ClassLoader *> GetRelativeClassLoaders();
    const std::string GetRelativeLibraryPath() const;
    const std::string GetBaseClassName() const;
    const std::string GetClassName() const;

private:
    std::string class_name_;
    std::string base_name_;
    std::string relative_library_path_;
    std::vector<ClassLoader *> relative_class_loaders_;
};

template <typename Base>
class AbstractClassFactory : public AbstractClassFactoryBase
{
public:
    AbstractClassFactory(const std::string &class_name, const std::string &base_name)
        : AbstractClassFactoryBase(class_name, base_name) {}
    ~AbstractClassFactory() {}

    virtual Base *CreateObj() const = 0;

private:
    DISALLOW_COPY_AND_ASSIGN(AbstractClassFactory)
};

template <typename ClassObject, typename Base>
class ClassFactory : public AbstractClassFactory<Base>
{
public:
    ClassFactory(const std::string &class_name, const std::string &base_name)
        : AbstractClassFactory<Base>(class_name, base_name) {}

    Base *CreateObj() const
    {
        return new ClassObject();
    }

private:
};

#endif
