#include "Context.hpp"
#include "Module.hpp"
#include <algorithm>
#include <cstdio>
#include <list>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include "RunGraph.hpp"
#ifdef _WIN32
#include <malloc.h>
#define kunAlignedAlloc(x, y) _aligned_malloc(y, x)
#define kunAlignedFree(x) _aligned_free(x)
#else
#define kunAlignedAlloc(x, y) aligned_alloc(x, y)
#define kunAlignedFree(x) free(x)
#endif

namespace kun {

void Buffer::alloc(size_t count, size_t use_count) {
    if (!ptr) {
        ptr = (float *)kunAlignedAlloc(32, count * sizeof(float));
        refcount = use_count;
    }
}

void Buffer::deref() {
    auto new_cnt = --refcount;
    if (new_cnt == 0) {
        kunAlignedFree(ptr);
        ptr = nullptr;
    }
}

Buffer::~Buffer() {
    if (ptr && refcount.load() >= 0) {
        free(ptr);
    }
}

static size_t divideAndCeil(size_t x, size_t y) { return (x + y - 1) / y; }

size_t RuntimeStage::getNumTasks() const {
    return stage->kind == TaskExecKind::SLICE_BY_STOCK
               ? divideAndCeil(ctx->stock_count, simd_len)
               : divideAndCeil(ctx->length, time_stride);
}

bool RuntimeStage::doJob() {
    auto cur_idx = doing_index.load();
    auto num_tasks = getNumTasks();
    while (cur_idx < num_tasks) {
        if (doing_index.compare_exchange_strong(cur_idx, cur_idx + 1)) {
            if (stage->kind == TaskExecKind::SLICE_BY_STOCK) {
                stage->f.f(ctx, cur_idx, ctx->total_time, ctx->start,
                           ctx->length);
            } else {
                stage->f.rankf(this, cur_idx, ctx->total_time, ctx->start,
                               ctx->length);
            }
            onDone(1);
            return true;
        }
    }
    return false;
}

void RuntimeStage::enqueue() {
    for (size_t i = 0; i < stage->num_out_buffers; i++) {
        auto buf_id = stage->out_buffers[i];
        ctx->buffers[buf_id->id].alloc(ctx->buffer_len,
                                       stage->out_buffers[i]->num_users);
    }
    ctx->executor->enqueue(this);
}

void RuntimeStage::onDone(size_t cnt) {
    auto newdone = --done_count;
    if (newdone == 0) {
        // current stage is done
        for (size_t i = 0; i < stage->num_dependers; i++) {
            auto id = stage->dependers[i]->id;
            auto &rtl_stage = ctx->stages[id];
            auto newpending = --rtl_stage.pending;
            if (newpending == 0) {
                rtl_stage.enqueue();
            }
        }
        for (size_t i = 0; i < stage->num_in_buffers; i++) {
            auto buf_id = stage->in_buffers[i];
            ctx->buffers[buf_id->id].deref();
        }
        ctx->executor->dequeue(this);
    }
}

void runGraph(std::shared_ptr<Executor> exec, const Module *m,
              std::unordered_map<std::string, float *> &buffers,
              size_t num_stocks, size_t total_time, size_t cur_time,
              size_t length) {
    std::vector<Buffer> rtlbuffers;
    rtlbuffers.reserve(m->num_buffers);
    for (size_t i = 0; i < m->num_buffers; i++) {
        auto &buf = m->buffers[i];
        if (buf.kind != BufferKind::TEMP) {
            auto itr = buffers.find(buf.name);
            if (itr == buffers.end()) {
                throw std::runtime_error("Buffer name not found: " +
                                         std::string(buf.name));
            }
            rtlbuffers.emplace_back(itr->second, total_time);
        } else {
            rtlbuffers.emplace_back(length);
        }
    }
    Context ctx{std::move(rtlbuffers),
                {},
                exec,
                num_stocks * length,
                num_stocks,
                total_time,
                cur_time,
                length};
    std::vector<RuntimeStage> &stages = ctx.stages;
    stages.reserve(m->num_stages);
    for (size_t i = 0; i < m->num_stages; i++) {
        auto &stage = m->stages[i];
        stages.emplace_back(&stage, &ctx);
    }
    for (size_t i = 0; i < m->num_stages; i++) {
        auto &stage = m->stages[i];
        if (stage.orig_pending == 0) {
            stages[i].enqueue();
        }
    }
    exec->runUntilDone();
}
} // namespace kun
