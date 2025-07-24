#include "CApi.h"
#include "Module.hpp"
#include "RunGraph.hpp"
#include <string>
#include <unordered_map>

using namespace kun;
extern "C" {

KUN_API KunExecutorHandle kunCreateSingleThreadExecutor() {
    return new std::shared_ptr<Executor>{kun::createSingleThreadExecutor()};
}

KUN_API KunExecutorHandle kunCreateMultiThreadExecutor(int numthreads) {
    return new std::shared_ptr<Executor>{
        kun::createMultiThreadExecutor(numthreads)};
}

static std::shared_ptr<Executor> *unwrapExecutor(KunExecutorHandle ptr) {
    return reinterpret_cast<std::shared_ptr<Executor> *>(ptr);
}

KUN_API void kunDestoryExecutor(KunExecutorHandle ptr) {
    delete unwrapExecutor(ptr);
}

KUN_API KunLibraryHandle kunLoadLibrary(const char *path_or_name) {
    return new std::shared_ptr<Library>{Library::load(path_or_name)};
}

static std::shared_ptr<Library> *unwrapLibrary(KunLibraryHandle ptr) {
    return reinterpret_cast<std::shared_ptr<Library> *>(ptr);
}

KUN_API KunModuleHandle kunGetModuleFromLibrary(KunLibraryHandle lib,
                                                const char *name) {
    auto &plib = *unwrapLibrary(lib);
    return (KunModuleHandle)plib->getModule(name);
}

KUN_API void kunUnloadLibrary(KunLibraryHandle ptr) {
    delete unwrapLibrary(ptr);
}

KUN_API KunBufferNameMapHandle kunCreateBufferNameMap() {
    return new std::unordered_map<std::string, float *>{};
}

static std::unordered_map<std::string, float *> *
unwrapMap(KunBufferNameMapHandle ptr) {
    return reinterpret_cast<std::unordered_map<std::string, float *> *>(ptr);
}

KUN_API void kunDestoryBufferNameMap(KunBufferNameMapHandle ptr) {
    delete unwrapMap(ptr);
}

KUN_API void kunSetBufferNameMap(KunBufferNameMapHandle ptr, const char *name,
                                 float *buffer) {
    auto &map = *unwrapMap(ptr);
    map[name] = buffer;
}

KUN_API void kunEraseBufferNameMap(KunBufferNameMapHandle ptr,
                                   const char *name) {
    auto &map = *unwrapMap(ptr);
    map.erase(name);
}

KUN_API void kunRunGraph(KunExecutorHandle exec, KunModuleHandle m,
                         KunBufferNameMapHandle buffers, size_t num_stocks,
                         size_t total_time, size_t cur_time, size_t length) {
    auto &pexec = *unwrapExecutor(exec);
    auto modu = reinterpret_cast<Module *>(m);
    auto map = unwrapMap(buffers);
    runGraph(pexec, modu, *map, num_stocks, total_time, cur_time, length);
}

KUN_API KunStreamContextHandle kunCreateStream(KunExecutorHandle exec,
                                               KunModuleHandle m,
                                               size_t num_stocks) {
    auto &pexec = *unwrapExecutor(exec);
    auto modu = reinterpret_cast<Module *>(m);
    return new kun::StreamContext{pexec, modu, num_stocks};
}

KUN_API size_t kunQueryBufferHandle(KunStreamContextHandle context,
                                    const char *name) {
    return reinterpret_cast<kun::StreamContext *>(context)->queryBufferHandle(
        name);
}

KUN_API const float *kunStreamGetCurrentBuffer(KunStreamContextHandle context,
                                               size_t handle) {
    return reinterpret_cast<kun::StreamContext *>(context)
        ->getCurrentBufferPtrFloat(handle);
}

KUN_API void kunStreamPushData(KunStreamContextHandle context, size_t handle,
                               const float *buffer) {
    reinterpret_cast<kun::StreamContext *>(context)->pushData(handle, buffer);
}

KUN_API void kunStreamRun(KunStreamContextHandle context) {
    reinterpret_cast<kun::StreamContext *>(context)->run();
}

KUN_API void kunDestoryStream(KunStreamContextHandle context) {
    delete reinterpret_cast<kun::StreamContext *>(context);
}
}