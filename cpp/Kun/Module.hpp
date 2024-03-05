#pragma once

#include "Stage.hpp"
#include <memory>

namespace kun {

enum class OutputLayout {
    ST8s,
    TS,
};

struct Module {
    size_t num_stages;
    Stage *stages;
    size_t num_buffers;
    BufferInfo *buffers;
    OutputLayout layout;
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