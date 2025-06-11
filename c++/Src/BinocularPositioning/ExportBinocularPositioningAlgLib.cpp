#include "ExportBinocularPositioningAlgLib.h"
#include "CBinocularPositioningAlg.h"

extern "C" __attribute__ ((visibility("default"))) IBinocularPositioningAlg* CreateBinocularPositioningAlgObj(const std::string& p_strExePath)
{
    return new CBinocularPositioningAlg(p_strExePath);
}