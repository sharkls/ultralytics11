#ifndef CLASS_LOADER_CLASS_LOADER_MANAGER_HPP
#define CLASS_LOADER_CLASS_LOADER_MANAGER_HPP

#include <map>
#include <mutex>

#include "include/class_loader/class_loader.hpp"

class ClassLoaderManager
{
public:
    ClassLoaderManager();
    ~ClassLoaderManager();

    bool LoadLibrary(const std::string& library_path);
    bool IsLibraryLoaded(const std::string& library_path);
    void UnloadAllLibrary();

    template <typename Base>
    std::shared_ptr<Base> CreateClassObj(const std::string& class_name);
    
    template <typename Base>
    std::shared_ptr<Base> CreateClassObj(const std::string& class_name, const std::string& library_path);

    template <typename Base>
    bool IsClassVaild(const std::string& class_name);

    template <typename Base>
    std::vector<std::string> GetAllValidClassNames();

private:
    std::vector<ClassLoader*> GetAllValidClassLoader();
    std::vector<std::string> GetAllLoadedLibraryPath();
    ClassLoader* GetClassLoaderByLibraryPath(const std::string& library_path);
    int UnloadLibrary(const std::string& library_path);

private:
    std::map<std::string, ClassLoader*> libpath_class_loader_map_;
    std::mutex libpath_class_loader_map_mutex_;
};

template <typename Base>
std::shared_ptr<Base> ClassLoaderManager::CreateClassObj(const std::string &class_name)
{
    std::vector<ClassLoader*> class_loaders = GetAllValidClassLoader();
    for (auto class_loader : class_loaders)
    {
        if (class_loader && class_loader->IsValidClass<Base>(class_name))
        {
            return (class_loader->CreateClassObj<Base>(class_name));
        }
    }
    TERROR << "Invalid class name [ " << class_name << " ]";
    return std::shared_ptr<Base>();
}

template <typename Base>
std::shared_ptr<Base> ClassLoaderManager::CreateClassObj(const std::string &class_name, const std::string &library_path)
{
    ClassLoader* class_loader = GetClassLoaderByLibraryPath(library_path);
    if (class_loader)
    {
        return (class_loader->CreateClassObj<Base>(class_name));
    }
    TERROR << "Could not create obj, there is no ClassLoader for [ " << library_path << " ]";
    return std::shared_ptr<Base>();
}

template <typename Base>
bool ClassLoaderManager::IsClassVaild(const std::string &class_name)
{
    std::vector<std::string> all_class_names = GetAllValidClassNames<Base>();
    return (std::find(all_class_names.begin(), all_class_names.end(), class_name) != all_class_names.end());
}

template <typename Base>
std::vector<std::string> ClassLoaderManager::GetAllValidClassNames()
{
    std::vector<ClassLoader*> class_loaders = GetAllValidClassLoader();

    std::vector<std::string> class_names;
    for (auto class_loader : class_loaders)
    {
        std::vector<std::string> class_names_for_class_loader = class_loader->GetValidClassNames<Base>();
        class_names.insert(class_names.end(), class_names_for_class_loader.begin(), class_names_for_class_loader.end());
    }
    return class_names;
}

#endif
