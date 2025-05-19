#include "ExportObjectDetectionAlgLib.h"
#include "CObjectDetectionAlg.h"

extern "C" __attribute__ ((visibility("default"))) IObjectDetectionAlg* CreateObjectDetectionAlgObj(const std::string& p_strExePath)
{
    return new CObjectDetectionAlg(p_strExePath);
}