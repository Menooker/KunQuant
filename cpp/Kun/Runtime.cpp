#include "Context.hpp"
#include <list>
#include <algorithm>

namespace kun {

struct SingleThreadExecutor: Executor
{
    std::list<RuntimeStage*> q;
    virtual void enqueue(RuntimeStage* stage) override {
        q.push_front(stage);
    }
    
    virtual void dequeue(RuntimeStage* stage) override {
        q.erase(std::find(q.begin(), q.end(), stage));
    }

    bool takeSingleJob() override {
        if (q.empty()) {
            return false;
        }
        q.front()->doJob();
    }
    

    void runUntilDone() override {
        while (takeSingleJob())
        {
            /* code */
        }
        
    }

    ~SingleThreadExecutor() = default;
};


bool RuntimeStage::doJob() {
    auto cur_idx = doing_index.load();
    while (cur_idx < stage->num_tasks) {
        if(doing_index.compare_exchange_strong(cur_idx, cur_idx + 1)) {
            stage->f(ctx, cur_idx * 8, ctx->total_time, ctx->start, ctx->length);
            onDone(1);
            return true;
        }
    }
    return false;
}

void RuntimeStage::enqueue() {
    for (size_t i = 0; i < stage->num_in_buffers; i++) {
        auto buf_id = stage->in_buffers[i];
        ctx->buffers[buf_id].ref();
    }
    for (size_t i = 0; i < stage->num_out_buffers; i++) {
        auto buf_id = stage->out_buffers[i];
        ctx->buffers[buf_id].alloc(ctx->buffer_len);
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
            ctx->buffers[buf_id].deref();
        }
        ctx->executor->dequeue(this);
    }
}
} // namespace kun
