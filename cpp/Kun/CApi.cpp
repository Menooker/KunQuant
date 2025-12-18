#include "CApi.h"
#include "IO.hpp"
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
    try {
        // Create a StreamContext with no initial states
        return new kun::StreamContext{pexec, modu, num_stocks};
    } catch (...) {
        // If there is an error, return nullptr
        return nullptr;
    }
}


KUN_API KunStatus kunCreateStreamEx(KunExecutorHandle exec,
                                               KunModuleHandle m,
                                               size_t num_stocks,
                                               const KunStreamExtraArgs *extra_args,
                                               KunStreamContextHandle *out_handle) {
    auto &pexec = *unwrapExecutor(exec);
    auto modu = reinterpret_cast<Module *>(m);
    FileInputStream file_stream;
    MemoryInputStream memory_stream {nullptr, 0};
    InputStreamBase *states = nullptr;
    if (extra_args) {
        if (extra_args->version != KUN_API_VERSION) {
            return KUN_INVALID_ARGUMENT;
        }
        if (extra_args->init_kind == KUN_INIT_FILE) {
            if (!extra_args->init.path) {
                return KUN_INVALID_ARGUMENT;
            }
            file_stream.file.open(extra_args->init.path, std::ios::binary);
            states = &file_stream;
        } else if (extra_args->init_kind == KUN_INIT_MEMORY) {
            if (!extra_args->init.memory.buffer) {
                return KUN_INVALID_ARGUMENT;
            }
            memory_stream.data = extra_args->init.memory.buffer;
            memory_stream.size = extra_args->init.memory.size;
            states = &memory_stream;
        } else if (extra_args->init_kind == KUN_INIT_NONE) {
            // do nothing
        } else {
            return KUN_INVALID_ARGUMENT;
        }
    }
    try {
        auto ctx = new kun::StreamContext{pexec, modu, num_stocks, states};
        *out_handle = reinterpret_cast<KunStreamContextHandle>(ctx);
    } catch (const std::exception &e) {
        *out_handle = nullptr;
        // If there is an error, return KUN_INIT_ERROR
        return KUN_INIT_ERROR;
    }
    return KUN_SUCCESS;
}

KUN_API KunStatus kunStreamSerializeStates(KunStreamContextHandle context,
                                           size_t dump_kind,
                                           char *path_or_buffer,
                                           size_t *size) {
    auto ctx = reinterpret_cast<kun::StreamContext *>(context);
    if (dump_kind == KUN_INIT_FILE) {
        kun::FileOutputStream stream(path_or_buffer);
        if (!ctx->serializeStates(&stream)) {
            return KUN_INVALID_ARGUMENT;
        }
        return KUN_SUCCESS;
    } else if (dump_kind == KUN_INIT_MEMORY) {
        if (!path_or_buffer || !size) {
            return KUN_INVALID_ARGUMENT;
        }
        size_t in_size = *size;
        kun::MemoryRefOutputStream stream {path_or_buffer, in_size};
        if (!ctx->serializeStates(&stream)) {
            return KUN_INVALID_ARGUMENT;
        }
        *size = stream.pos; // Update size to the actual written size
        return stream.pos > in_size ? KUN_INIT_ERROR : KUN_SUCCESS;
    } else {
        return KUN_INVALID_ARGUMENT;
    }
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