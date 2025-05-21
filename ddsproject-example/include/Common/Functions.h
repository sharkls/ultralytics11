#ifndef FUNCTIONS_H  // 包含保护
#define FUNCTIONS_H

#include <string>
#include <unistd.h>
#include <cstring>

// 函数实现
inline std::string GetExecutionPath()
{
    static std::string m_strExecutionPath;  // 使用 static 以保持路径的状态
    if (!m_strExecutionPath.empty())
    {
        return m_strExecutionPath;
    }

    char *p = NULL;
    const int len = 1024;
    char l_pExecutionPath[len];
    memset(l_pExecutionPath, 0, len);

    // 读取 /proc/self/exe 链接，获取当前可执行程序的路径
    int n = readlink("/proc/self/exe", l_pExecutionPath, len);
    if (n != -1)  // 确保读取成功
    {
        if (NULL != (p = strrchr(l_pExecutionPath, '/')))
        {
            *p = '\0';  // 去掉路径中的可执行文件名，只保留目录部分
        }

        m_strExecutionPath = l_pExecutionPath;  // 保存路径
        m_strExecutionPath += "/..";  // 返回上一级目录
    }

    return m_strExecutionPath;
}

inline std::string GetRootPath()
{
    std::string execution_path = GetExecutionPath();
    
    // 提取根路径
    return execution_path.substr(0, execution_path.find("/output"));
}

#endif // FUNCTIONS_H