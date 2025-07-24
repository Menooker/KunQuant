#pragma once

#include "Context.hpp"
#include "Module.hpp"
#include <unordered_map>
#include <string>
#include <vector>

namespace kun {
KUN_API void runGraph(std::shared_ptr<Executor> exec, const Module *m,
                      std::unordered_map<std::string, float *> &buffers,
                      size_t num_stocks, size_t total_time, size_t cur_time,
                      size_t length);
KUN_API void corrWith(std::shared_ptr<Executor> exec, MemoryLayout layout, bool rank_inputs,
              std::vector<float*>& buffers,
              float* corr_with_buffer,
              std::vector<float*>& outbuffers,
              size_t num_stocks, size_t total_time, size_t cur_time,
              size_t length);
struct AlignedPtr {
    void* ptr;
#if CHECKED_PTR
    size_t size;
#endif
    char* get() const noexcept {
        return (char*)ptr;
    }
    AlignedPtr(void* ptr, size_t size) noexcept;
    AlignedPtr(AlignedPtr&& other) noexcept;
    AlignedPtr& operator=(AlignedPtr&& other) noexcept;
    void release() noexcept;
    ~AlignedPtr();
};

struct KUN_API StreamContext {
    std::vector<AlignedPtr> buffers;
    Context ctx;
    const Module *m;
    StreamContext(std::shared_ptr<Executor> exec, const Module *m,
                  size_t num_stocks);
    // query the buffer handle of a named buffer
    size_t queryBufferHandle(const char *name) const;
    // get the current readable position of the named buffer. The returned
    // buffer length should be num_stocks.
    const float *getCurrentBufferPtrFloat(size_t handle) const;
    const double *getCurrentBufferPtrDouble(size_t handle) const;
    // push new data on the named buffer and move forward the internal data
    // position register.
    void pushData(size_t handle, const float *data);
    void pushData(size_t handle, const double *data);
    void run();
    StreamContext(const StreamContext&) = delete;
    StreamContext& operator=(const StreamContext&) = delete;
    ~StreamContext();
};

} // namespace kun