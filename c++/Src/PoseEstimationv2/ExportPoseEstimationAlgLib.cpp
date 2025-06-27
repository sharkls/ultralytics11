#include "ExportPoseEstimationAlgLib.h"
#include "CPoseEstimationAlg.h"

extern "C" __attribute__ ((visibility("default"))) IPoseEstimationAlg* CreatePoseEstimationAlgObj(const std::string& p_strExePath)
{
    return new CPoseEstimationAlg(p_strExePath);
}