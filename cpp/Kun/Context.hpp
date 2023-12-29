#pragma once

#include "Stage.hpp"
#include <memory>
#include <stdlib.h>

namespace kun {

struct RuntimeStage {
    const Stage *stage;
    Context *ctx;

    std::atomic<size_t> pending;
    std::atomic<size_t> doing_index;
    std::atomic<size_t> done_count;

    RuntimeStage(Stage *stage, Context *ctx) : stage{stage}, ctx{ctx} {
        reset();
    }

    void reset() {
        pending = stage->orig_pending;
        doing_index = 0;
        done_count = 0;
    }

    void beforeEnqueue(Context *ctx);
    void onDone(Context *ctx);
};

struct Buffer {
    float *ptr;
    std::atomic<int> refcount;

    void alloc(size_t count) {
        ptr = (float *)aligned_alloc(32, count * sizeof(float));
        refcount = 0;
    }

    Buffer() {
        ptr = nullptr;
        refcount = 0;
    }

    Buffer(const Buffer &) = delete;

    Buffer(float *inptr) {
        ptr = inptr;
        refcount = -1000;
    }

    void deref() {
        auto new_cnt = --refcount;
        if (new_cnt == 0) {
            free(ptr);
            ptr = nullptr;
        }
    }

    ~Buffer() {
        if(ptr && refcount.load() >= 0) {
            free(ptr);
        }
    }
};

struct Executor;

struct Context {
    std::vector<Buffer> buffers;
    std::vector<RuntimeStage> stages;
    std::shared_ptr<Executor> executor;
    size_t buffer_len;
};

} // namespace kun