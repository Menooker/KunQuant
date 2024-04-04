#pragma once

#include "Stage.hpp"
#include <memory>

namespace kun {

enum class MemoryLayout {
    STs,
    TS,
    STREAM,
};

enum class Datatype {
    Float,
    Double,
};

struct Module {
    size_t required_version;
    size_t num_stages;
    Stage *stages;
    size_t num_buffers;
    BufferInfo *buffers;
    MemoryLayout input_layout;
    MemoryLayout output_layout;
    size_t blocking_len;
    Datatype dtype;
};

struct Library {
    void *handle;
    KUN_API const Module *getModule(const char *name);
    KUN_API static std::shared_ptr<Library> load(const char *filename);
    Library(const Library &) = delete;
    Library(void *handle) : handle{handle} {}
    KUN_API ~Library();
};

} // namespace kun