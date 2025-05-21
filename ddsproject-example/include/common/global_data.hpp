#ifndef COMMON_GLOBAL_DATA_HPP
#define COMMON_GLOBAL_DATA_HPP

#include <string>

#include "include/common/macros.hpp"
#include "include/common/file.hpp"
#include "include/common/environment.hpp"

class GlobalData
{
public:
    ~GlobalData();
    
    /**
     * @brief 设置master节点 node 的相关配置信息
    */
    void SetMasterConfigFilePath(std::string);
    const std::string& GetMasterConfigFilePath();

    /**
     * @brief 设置master节点和其他activity节点之间的topic：master获取activity对于指令的反馈状态
    */
    void SetMasterActivityStatusFeedbackTopic(const std::string &master_activity_status_feedback_topic);
    /**
     * @brief 获取master节点和其他activity节点之间的topic
    */
    std::string GetMasterActivityStatusFeedbackTopic();

    /**
     * @brief 设置master节点和其他activity节点之间的topic：master向activity发送控制指令
    */
    void SetMasterActivityCmdCtrlTopic(const std::string& master_activity_cmd_ctrl_topic);
    /**
     * @brief 获取master节点和其他activity节点之间的topic
    */
    std::string GetMasterActivityCmdCtrlTopic();

    /**
     * @brief 设置master读取各个activity运行设置的配置文件路径
    */
    void SetActivityLaunchConfigFilePath(const std::string& file_path);
    /**
     * @brief 获取master读取各个activity运行设置的配置文件路径
    */
    std::string GetActivityLaunchConfigFilePath();

    /**
     * @brief 设置master读取activity运行配置文件的时间间隔
    */
    void SetActivityLaunchConfigUpdateTimeGap(const int16_t time_gap);
    /**
     * @brief 获取master读取activity运行配置文件的时间间隔
    */
    int16_t GetActivityLaunchConfigUpdateTimeGap();

private:
    std::string master_config_file_name_{"include/conf/master_activity_communicate.conf"};
    std::string master_activity_status_feedback_topic_{"ActivityStatusFeedback"};
    std::string master_activity_cmd_ctrl_topic_{"ActivityCmdControl"};
    std::string activity_launch_config_file_path_{"include/conf/activity_launch.conf"};

    int16_t activity_launch_config_update_time_gap_{5000};

    DECLARE_SINGLETON(GlobalData)
};

#endif