#pragma once

#include "Stage.hpp"
#include <atomic>
#include <memory>
#include <stdlib.h>

namespace kun {

struct KUN_API RuntimeStage {
    const Stage *stage;
    Context *ctx;

    std::atomic<size_t> pending;
    std::atomic<size_t> doing_index;
    std::atomic<size_t> done_count;

    RuntimeStage(Stage *stage, Context *ctx) : stage{stage}, ctx{ctx} {
        reset();
    }

    bool doJob();

    void reset() {
        pending = stage->orig_pending;
        doing_index = 0;
        done_count = stage->num_tasks;
    }

    void enqueue();
    void onDone(size_t cnt);
};

struct KUN_API Executor {
    virtual void enqueue(RuntimeStage *stage) = 0;
    virtual void dequeue(RuntimeStage *stage) = 0;
    virtual bool takeSingleJob() = 0;
    virtual void runUntilDone() = 0;
    virtual ~Executor() = 0;
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

    void ref() { ++refcount; }

    void deref() {
        auto new_cnt = --refcount;
        if (new_cnt == 0) {
            free(ptr);
            ptr = nullptr;
        }
    }

    ~Buffer() {
        if (ptr && refcount.load() >= 0) {
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

    size_t stock_count;
    size_t total_time;
    size_t start;
    size_t length;
};

} // namespace kun