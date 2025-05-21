#ifndef CLASS_LOADER_CLASS_LOADER_HPP
#define CLASS_LOADER_CLASS_LOADER_HPP

#include <functional>

#include "include/common/log.hpp"
#include "include/class_loader/class_loader_utility.hpp"

class ClassLoader
{
public:
    ClassLoader(const std::string &library_path);
    ~ClassLoader();

    bool LoadLibrary();
    bool IsLibraryLoaded();
    /**
     * @brief unload shared library ： 当当前应用的动态库的数量为0时，那么unload shared library
     * @return 返回当前动态被引用（加载）的次数
     */
    int UnloadLibrary();

    const std::string GetLibraryPath() const;

    /**
     * @brief 获取当前类加载器对应的所有valid class name
    */
    template <typename Base>
    std::vector<std::string> GetValidClassNames();

    template <typename Base>
    bool IsValidClass(const std::string &class_name);

    template <typename Base>
    std::shared_ptr<Base> CreateClassObj(const std::string &class_name);

    template <typename Base>
    bool IsClassValid(const std::string &class_name);

private:
    template <typename Base>
    void OnClassObjDeleter(Base *obj);

private:
    std::string library_path_;
    int loadlib_ref_count_;
    std::mutex loadlib_ref_count_mutex_;
    int classobj_ref_count_;
    std::mutex classobj_ref_count_mutex_;
};

template <typename Base>
std::vector<std::string> ClassLoader::GetValidClassNames()
{
    return Utility::GetValidClassNames<Base>(this);
}

template <typename Base>
bool ClassLoader::IsValidClass(const std::string &class_name)
{
    std::vector<std::string> valid_classes = GetValidClassNames<Base>();
    return (std::find(valid_classes.begin(), valid_classes.end(), class_name) != valid_classes.end());
}

template <typename Base>
std::shared_ptr<Base> ClassLoader::CreateClassObj(const std::string &class_name)
{
    if (!IsLibraryLoaded())
    {
        LoadLibrary();
    }

    Base *class_obj = Utility::CreateClassObj<Base>(class_name, this);
    if (class_obj == nullptr)
    {
        TWARN << "CreateClassObj failed, ensure class has been registered. classname [ " << class_name << " ], lib [ " << library_path_ << "]";
        return std::shared_ptr<Base>();
    }
    TINFO << "CreateClassObj [ " << class_name << " ]" << " addr: [ " << class_obj << "]";
    std::lock_guard<std::mutex> lock(classobj_ref_count_mutex_);
    classobj_ref_count_++;
    std::shared_ptr<Base> class_obj_shared_ptr(class_obj, std::bind(&ClassLoader::OnClassObjDeleter<Base>, this, std::placeholders::_1));
    return class_obj_shared_ptr;
}

template <typename Base>
bool ClassLoader::IsClassValid(const std::string &class_name)
{
    std::vector<std::string> class_names = GetValidClassNames<Base>();
    return (std::find(class_names.begin(), class_names.end(), class_name) != class_names.end());
}

template <typename Base>
void ClassLoader::OnClassObjDeleter(Base *obj)
{
    if (obj == nullptr)
    {
        return;
    }
    std::lock_guard<std::mutex> lock(classobj_ref_count_mutex_);
    delete obj;
    --classobj_ref_count_;
}

#endif
