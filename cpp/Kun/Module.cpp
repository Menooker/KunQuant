#include "Module.hpp"
#include <memory>
#include <cstdio>

#ifdef _WIN32
#include <Windows.h>
#include <system_error>
#else
#include <dlfcn.h>
#endif

namespace kun {

#ifdef _WIN32
const Module *Library::getModule(const char *name) {
    return (const Module *)GetProcAddress((HMODULE)handle, name);
}
std::shared_ptr<Library> Library::load(const char *filename) {
    auto handle = (void*)LoadLibrary(filename);
    if (!handle) {
        DWORD error = ::GetLastError();
        std::string message = std::system_category().message(error);
        fprintf(stderr, "LoadLibrary failed: %s %s\n",filename, message.c_str());
        return nullptr;
    }
    return std::make_shared<Library>(handle);
}
Library::~Library() {
    if (handle) {
        FreeLibrary((HMODULE)handle);
    }
}
#else
const Module *Library::getModule(const char *name) {
    return (const Module *)dlsym(handle, name);
}
std::shared_ptr<Library> Library::load(const char *filename) {
    auto handle = dlopen(filename, RTLD_LOCAL | RTLD_NOW);
    if (!handle) {
        fprintf(stderr, "dlopen failed: %s %s\n",filename, dlerror());
        return nullptr;
    }
    return std::make_shared<Library>(handle);
}
Library::~Library() {
    if (handle) {
        dlclose(handle);
    }
}
#endif

} // namespace kun