#ifndef CLASS_LOADER_SHARED_LIBRARY_EXCEPTION_HPP
#define CLASS_LOADER_SHARED_LIBRARY_EXCEPTION_HPP

#include <stdexcept>
#include <string>

#define DECLARE_SHARED_LIBRARY_EXCEPTION(CLS, BASE)                 \
    class CLS : public BASE                                          \
    {                                                               \
    public:                                                         \
        explicit CLS(const std::string &err_msg) : BASE(err_msg) {} \
        ~CLS() throw() {}                                           \
    };

DECLARE_SHARED_LIBRARY_EXCEPTION(LibraryAlreadyLoadedException, std::runtime_error);
DECLARE_SHARED_LIBRARY_EXCEPTION(LibraryLoadException, std::runtime_error);
DECLARE_SHARED_LIBRARY_EXCEPTION(SymbolNotFoundException, std::runtime_error);

#endif