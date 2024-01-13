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
        reset(ctx);
    }

    RuntimeStage(RuntimeStage &&other) noexcept {
        stage = other.stage;
        ctx = other.ctx;
        pending = other.pending.load(std::memory_order_relaxed);
        doing_index = other.doing_index.load(std::memory_order_relaxed);
        done_count = other.done_count.load(std::memory_order_relaxed);
    }

    bool doJob();

    bool hasJobToDo() const {
        auto cur_idx = doing_index.load();
        auto num_tasks = getNumTasks();
        return (cur_idx < num_tasks);
    }
    size_t getNumTasks() const;

    void reset(Context *ctx) {
        pending = stage->orig_pending;
        doing_index = 0;
        done_count = getNumTasks();
    }

    void enqueue();
    void onDone(size_t cnt);
};

struct KUN_API Executor {
    virtual void enqueue(RuntimeStage *stage) = 0;
    virtual void dequeue(RuntimeStage *stage) = 0;
    // virtual bool takeSingleJob() = 0;
    virtual void runUntilDone() = 0;
    virtual ~Executor() = default;
};

struct Buffer {
    float *ptr;
    std::atomic<int> refcount;

    KUN_API void alloc(size_t count, size_t use_count);

    Buffer() {
        ptr = nullptr;
        refcount = 0;
    }

    Buffer(const Buffer &) = delete;
    Buffer(Buffer &&other) noexcept {
        ptr = other.ptr;
        refcount = other.refcount.load();
        other.ptr = nullptr;
    }

    Buffer(float *inptr) {
        ptr = inptr;
        refcount = -1000;
    }

    void ref() { ++refcount; }

    KUN_API void deref() ;

    KUN_API ~Buffer() ;
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

KUN_API std::shared_ptr<Executor> createSingleThreadExecutor();
KUN_API std::shared_ptr<Executor> createMultiThreadExecutor(int num_threads);
namespace ops {
   KUN_API void RankStocksST8s_ST8s(RuntimeStage *stage, size_t __stock_idx,
                         size_t __total_time, size_t __start, size_t __length);
   KUN_API void RankStocksST8s_TS(RuntimeStage *stage, size_t __stock_idx,
                         size_t __total_time, size_t __start, size_t __length);
   KUN_API void RankStocksTS_ST8s(RuntimeStage *stage, size_t __stock_idx,
                         size_t __total_time, size_t __start, size_t __length);
   KUN_API void RankStocksTS_TS(RuntimeStage *stage, size_t __stock_idx,
                         size_t __total_time, size_t __start, size_t __length);
}
} // namespace kun