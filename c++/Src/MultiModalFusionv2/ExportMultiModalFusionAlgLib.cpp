#include "ExportMultiModalFusionAlgLib.h"
#include "CMultiModalFusionAlg.h"

extern "C" __attribute__ ((visibility("default"))) IMultiModalFusionAlg* CreateMultiModalFusionAlgObj(const std::string& p_strExePath)
{
    return new CMultiModalFusionAlg(p_strExePath);
}