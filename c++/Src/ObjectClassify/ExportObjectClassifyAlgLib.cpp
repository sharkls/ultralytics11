#include "ExportObjectClassifyAlgLib.h"
#include "CObjectClassifyAlg.h"

extern "C" __attribute__ ((visibility("default"))) IObjectClassifyAlg* CreateObjectClassifyAlgObj(const std::string& p_strExePath)
{
    return new CObjectClassifyAlg(p_strExePath);
}