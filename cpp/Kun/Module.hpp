#pragma once

#include "Stage.hpp"
#include <memory>

namespace kun {

enum class OutputLayout {
    ST8s,
    FTS,
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
    const Module *getModule(const char *name);
    static std::shared_ptr<Library> load(const char *filename);
    Library(const Library &) = delete;
    Library(void *handle) : handle{handle} {}
    ~Library();
};

} // namespace kun