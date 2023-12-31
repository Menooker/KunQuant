#include "Module.hpp"
#include <dlfcn.h>
#include <memory>

namespace kun {

const Module *Library::getModule(const char *name) {
    return (const Module *)dlsym(handle, name);
}
std::shared_ptr<Library> Library::load(const char *filename) {
    auto handle = dlopen(filename, RTLD_LOCAL);
    if (!handle) {
        return nullptr;
    }
    return std::make_shared<Library>(handle);
}
Library::~Library() {
    if (handle) {
        dlclose(handle);
    }
}

} // namespace kun