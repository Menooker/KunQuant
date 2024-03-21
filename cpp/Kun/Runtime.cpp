#include "Context.hpp"
#include "Module.hpp"
#include "RunGraph.hpp"
#include <algorithm>
#include <cstdio>
#include <list>
#include <stdexcept>
#include <string.h>
#include <string>
#include <unordered_map>
#ifdef _WIN32
#include <malloc.h>
#define kunAlignedAlloc(x, y) _aligned_malloc(y, x)
#define kunAlignedFree(x) _aligned_free(x)
#else
#define kunAlignedAlloc(x, y) aligned_alloc(x, y)
#define kunAlignedFree(x) free(x)
#endif

namespace kun {
static const uint64_t VERSION = 0x00000001;

void Buffer::alloc(size_t count, size_t use_count) {
    if (!ptr) {
        ptr = (float *)kunAlignedAlloc(32, count * sizeof(float));
        refcount = use_count;
    }
}

void Buffer::deref() {
    if (refcount < 0) {
        return;
    }
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
    if (m->required_version != VERSION) {
        throw std::runtime_error("The required version in the module does not "
                                 "match the runtime version");
    }
    if (m->layout == OutputLayout::STREAM) {
        throw std::runtime_error(
            "Cannot run stream mode module via runGraph()");
    }
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
                length,
                false};
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

void StreamContext::Deleter::operator()(char *b) { kunAlignedFree(b); }

char *StreamBuffer::make(size_t stock_count, size_t window_size) {
    auto ret = kunAlignedAlloc(
        32, StreamBuffer::getBufferSize(stock_count, window_size));
    auto buf = (StreamBuffer *)ret;
    for (size_t i = 0; i < stock_count * window_size; i++) {
        buf->getBuffer()[i] = NAN;
    }
    for (size_t i = 0; i < stock_count / simd_len; i++) {
        *buf->getPos(i, stock_count, window_size) = 0;
    }
    return (char *)ret;
}

StreamContext::StreamContext(std::shared_ptr<Executor> exec, const Module *m,
                             size_t num_stocks)
    : m{m} {
    if (m->required_version != VERSION) {
        throw std::runtime_error("The required version in the module does not "
                                 "match the runtime version");
    }
    if (m->layout != OutputLayout::STREAM) {
        throw std::runtime_error(
            "Cannot run batch mode module via StreamContext");
    }
    std::vector<Buffer> rtlbuffers;
    rtlbuffers.reserve(m->num_buffers);
    buffers.reserve(m->num_buffers);
    for (size_t i = 0; i < m->num_buffers; i++) {
        auto &buf = m->buffers[i];
        buffers.emplace_back(StreamBuffer::make(num_stocks, buf.window),
                             StreamContext::Deleter{});
        rtlbuffers.emplace_back((float *)buffers.back().get(), 1);
    }
    ctx.buffers = std::move(rtlbuffers);
    ctx.executor = exec;
    ctx.buffer_len = num_stocks * 1;
    ctx.stock_count = num_stocks;
    ctx.total_time = 1;
    ctx.start = 0;
    ctx.length = 1;
    ctx.is_stream = true;
}

size_t StreamContext::queryBufferHandle(const char *name) const {
    for (size_t i = 0; i < m->num_buffers; i++) {
        auto &buf = m->buffers[i];
        if (!strcmp(buf.name, name)) {
            return i;
        }
    }
    throw std::runtime_error("Cannot find the buffer name");
}

const float *StreamContext::getCurrentBufferPtr(size_t handle) const {
    auto buf = (StreamBuffer *)buffers.at(handle).get();
    return buf->getCurrentBufferPtr(ctx.stock_count, m->buffers[handle].window);
}

void StreamContext::pushData(size_t handle, const float *data) {
    auto buf = (StreamBuffer *)buffers.at(handle).get();
    float* ptr = buf->pushData(ctx.stock_count, m->buffers[handle].window);
    memcpy(ptr, data, ctx.stock_count);
    printf("INPUT %f\n", *data);
}

void StreamContext::run() {
    std::vector<RuntimeStage> &stages = ctx.stages;
    stages.clear();
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
    ctx.executor->runUntilDone();
}

StreamContext::~StreamContext() = default;
} // namespace kun
