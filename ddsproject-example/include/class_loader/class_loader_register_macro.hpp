#ifndef CLASS_LOADER_CLASS_LOADER_REGISTER_MACRO_HPP
#define CLASS_LOADER_CLASS_LOADER_REGISTER_MACRO_HPP

#include "include/class_loader/class_loader_utility.hpp"

#define CLASS_LOADER_REGISTER_CLASS_INTERNAL(Device, Base, UniqueID)  \
    namespace                                                         \
    {                                                                 \
        struct ProxyType##UniqueID                                    \
        {                                                             \
            ProxyType##UniqueID()                                     \
            {                                                         \
                Utility::RegisterClass<Device, Base>(#Device, #Base); \
            }                                                         \
        };                                                            \
        static ProxyType##UniqueID g_register_class_##UniqueID;       \
    }

#define CLASS_LOADER_REGISTER_CLASS_INTERNAL_1(Device, Base, UniqueID) \
    CLASS_LOADER_REGISTER_CLASS_INTERNAL(Device, Base, UniqueID)

#define CLASS_LOADER_REGISTER_CLASS(Device, Base) \
    CLASS_LOADER_REGISTER_CLASS_INTERNAL_1(Device, Base, __COUNTER__)

#endif