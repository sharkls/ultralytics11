#include "ExportObjectLocationAlgLib.h"
#include "CObjectLocationAlg.h"

extern "C" __attribute__ ((visibility("default"))) IObjectLocationAlg* CreateObjectLocationAlgObj(const std::string& p_strExePath)
{
    return new CObjectLocationAlg(p_strExePath);
}