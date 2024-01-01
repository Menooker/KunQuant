#include "Context.hpp"
#include "Module.hpp"
#include <algorithm>
#include <list>
#include <unordered_map>

namespace kun {

struct SingleThreadExecutor : Executor {
    std::list<RuntimeStage *> q;
    virtual void enqueue(RuntimeStage *stage) override { q.push_front(stage); }

    virtual void dequeue(RuntimeStage *stage) override {
        q.erase(std::find(q.begin(), q.end(), stage));
    }

    bool takeSingleJob() override {
        if (q.empty()) {
            return false;
        }
        q.front()->doJob();
    }

    void runUntilDone() override {
        while (takeSingleJob()) {
            /* code */
        }
    }

    ~SingleThreadExecutor() = default;
};

std::shared_ptr<Executor> createSingleThreadExecutor() {
    return std::make_shared<SingleThreadExecutor>();
}

bool RuntimeStage::doJob() {
    auto cur_idx = doing_index.load();
    while (cur_idx < stage->num_tasks) {
        if (doing_index.compare_exchange_strong(cur_idx, cur_idx + 1)) {
            stage->f(ctx, cur_idx, ctx->total_time, ctx->start, ctx->length);
            onDone(1);
            return true;
        }
    }
    return false;
}

void RuntimeStage::enqueue() {
    for (size_t i = 0; i < stage->num_in_buffers; i++) {
        auto buf_id = stage->in_buffers[i];
        ctx->buffers[buf_id->id].ref();
    }
    for (size_t i = 0; i < stage->num_out_buffers; i++) {
        auto buf_id = stage->out_buffers[i];
        ctx->buffers[buf_id->id].alloc(ctx->buffer_len);
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

void runGraph(std::shared_ptr<Executor> exec, Module *m,
              std::unordered_map<std::string, float *> &buffers,
              size_t num_stocks, size_t total_time, size_t cur_time) {
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
            rtlbuffers.emplace_back(itr->second);
        } else {
            rtlbuffers.emplace_back();
        }
    }
    Context ctx{std::move(rtlbuffers),
                {},
                exec,
                num_stocks * (total_time - cur_time),
                num_stocks,
                cur_time,
                total_time};
    std::vector<RuntimeStage> &stages = ctx.stages;
    stages.reserve(m->num_stages);
    for (size_t i = 0; i < m->num_stages; i++) {
        auto &stage = m->stages[i];
        stages.emplace_back(&stage, &ctx);
    }
    for (size_t i = 0; i < m->num_stages; i++) {
        auto &stage = m->stages[i];
        if (stage.orig_pending == 0) {
            exec->enqueue(&stages[i]);
        }
    }
    exec->runUntilDone();
}
} // namespace kun
