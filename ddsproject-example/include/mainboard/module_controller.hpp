#ifndef MAINBOARD_MODULE_CONTROLLER_HPP
#define MAINBOARD_MOUDLE_CONTROLLER_HPP

#include <thread>

#include "include/common/environment.hpp"
#include "include/common/file.hpp"
#include "include/mainboard/module_argument.hpp"
#include "include/class_loader/class_loader_manager.hpp"
#include "include/activity/base/activitybase.hpp"
#include "proto/activity_cfg/dag_conf.pb.h"

class ModuleController
{
public:
    explicit ModuleController(const ModuleArgument& args);
    ~ModuleController() = default;

    bool Init();
    bool LoadAll();
    void Clear();
    void Run();

private:
    int GetActivityNum(const std::string& path);

private:
    bool LoadModule(const std::string& path);
    bool LoadModule(const DagConfig& dag_config);

    ModuleArgument args_;
    ClassLoaderManager class_loader_manager_;

    int total_activity_nums_ = 0;

    std::vector<std::shared_ptr<ActivityBase>> activity_list_;

    std::vector<std::thread*> processor_;
};

#endif