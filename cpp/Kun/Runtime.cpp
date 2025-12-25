#include "Context.hpp"
#include "CorrWith.hpp"
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

#if CHECKED_PTR
#include <assert.h>
#include <sys/mman.h>
static void fill_memory(char *start, char *end) {
    memset(start, 0xcc, end - start);
}

void *checkedAlloc(size_t sz, size_t alignment) {
    size_t page_sz = 4096; // should get OS page size
    size_t data_size = divideAndCeil(sz, page_sz) * page_sz;
    size_t real_sz = data_size + page_sz * 2;
    assert(real_sz > 2 * page_sz && "At least 3 pages should be allocated");
    auto ret = mmap(nullptr, real_sz, PROT_READ | PROT_WRITE,
                    MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    assert(ret && "mmap failed");
    mprotect(ret, page_sz, PROT_NONE);
    auto protect_page_rhs = (char *)ret + real_sz - page_sz;
    mprotect(protect_page_rhs, page_sz, PROT_NONE);
    auto result = protect_page_rhs - divideAndCeil(sz, alignment) * alignment;
    fill_memory(result + sz, protect_page_rhs);
    fill_memory((char *)ret + page_sz, result);
    return result;
}

static void check(uint8_t *start, uint8_t *end, void *base_buffer,
                  size_t real_sz) {
    for (uint8_t *p = start; p < end; p++) {
        if (*p != 0xcc) {
            fputs("Buffer overflow detected\n", stderr);
            munmap(base_buffer, real_sz);
            std::abort();
        }
    }
}
void checkedDealloc(void *ptr, size_t sz) {
    size_t page_sz = 4096; // should get OS page size
    size_t data_size = divideAndCeil(sz, page_sz) * page_sz;
    size_t real_sz = data_size + page_sz * 2;
    auto buffer = (uint8_t *)((size_t)ptr / page_sz * page_sz - page_sz);
    auto protect_page_rhs = (uint8_t *)buffer + real_sz - page_sz;
    check((uint8_t *)ptr + sz, protect_page_rhs, buffer, real_sz);
    check(buffer + page_sz, (uint8_t *)ptr, buffer, real_sz);
    munmap(buffer, real_sz);
}

#undef kunAlignedAlloc
#undef kunAlignedFree
#define kunAlignedAlloc(x, y) checkedAlloc(y, x)
#define kunAlignedFree(x) checkedDealloc(x, size)
#endif

namespace kun {

void Buffer::alloc(size_t count, size_t use_count, size_t elem_size) {
    if (!ptr && count > 0) {
        ptr = (float *)kunAlignedAlloc(KUN_MALLOC_ALIGNMENT, count * elem_size);
        refcount = (int)use_count;
#if CHECKED_PTR
        size = count * elem_size;
#endif
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
        kunAlignedFree(ptr);
    }
}

size_t RuntimeStage::getNumTasks() const {
    return stage->kind == TaskExecKind::SLICE_BY_STOCK
               ? divideAndCeil(ctx->stock_count, ctx->simd_len)
               : divideAndCeil(ctx->length, time_stride);
}

bool RuntimeStage::doJob() {
    auto cur_idx = doing_index.load();
    auto num_tasks = getNumTasks();
    auto total_time = ctx->total_time;
    auto start = ctx->start;
    auto length = ctx->length;
    while (cur_idx < num_tasks) {
        if (doing_index.compare_exchange_strong(cur_idx, cur_idx + 1)) {
            stage->f.f(this, cur_idx, total_time, start, length);
            if (!onDone(1)) {
                return false;
            }
        }
    }
    return false;
}

static size_t getSizeofDtype(Datatype dtype) {
    if (dtype == Datatype::Double) {
        return sizeof(double);
    }
    return sizeof(float);
}

void RuntimeStage::enqueue() {
    size_t sz = getSizeofDtype(ctx->dtype);
    for (size_t i = 0; i < stage->num_out_buffers; i++) {
        auto buf_id = stage->out_buffers[i];
        ctx->buffers[buf_id->id].alloc(ctx->buffer_len,
                                       stage->out_buffers[i]->num_users, sz);
    }
    ctx->executor->enqueue(this);
}

bool RuntimeStage::onDone(size_t cnt) {
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
        return false;
    }
    return true;
}

static void setContextStagesAndRun(Context &ctx, Stage *mstages,
                                   size_t num_stages, Executor *exec) {
    std::vector<RuntimeStage> &stages = ctx.stages;
    stages.reserve(num_stages);
    for (size_t i = 0; i < num_stages; i++) {
        auto &stage = mstages[i];
        stages.emplace_back(&stage, &ctx);
    }
    for (size_t i = 0; i < num_stages; i++) {
        auto &stage = mstages[i];
        if (stage.orig_pending == 0) {
            stages[i].enqueue();
        }
    }
    exec->runUntilDone();
}

void corrWith(std::shared_ptr<Executor> exec, MemoryLayout layout,
              bool rank_inputs, std::vector<float *> &buffers,
              float *corr_with_buffer, std::vector<float *> &outbuffers,
              size_t num_stocks, size_t total_time, size_t cur_time,
              size_t length) {
    decltype(&ops::CorrWith<ops::MapperTS<float, 8>>) thefunc = nullptr;
    if (layout == MemoryLayout::TS) {
        if (rank_inputs) {
            thefunc = &ops::RankCorrWith<ops::MapperTSFloat>;
        } else {
            thefunc = &ops::CorrWith<ops::MapperTSFloat>;
        }
    } else {
        if (rank_inputs) {
            thefunc = &ops::RankCorrWith<ops::MapperSTsFloat>;
        } else {
            thefunc = &ops::CorrWith<ops::MapperSTsFloat>;
        }
    }
    std::vector<BufferInfo> buffer_info;
    std::vector<Buffer> rtlbuffers;
    std::vector<Stage> mstages;
    std::vector<BufferInfo *> temp;
    rtlbuffers.reserve(buffers.size() * 2 + 1);
    buffer_info.reserve(buffers.size() * 2 + 1);
    mstages.reserve(buffers.size() * 2 + 1);
    temp.reserve(buffers.size() * 3);
    for (float *b : buffers) {
        rtlbuffers.emplace_back(b, total_time);
        buffer_info.emplace_back(
            BufferInfo{buffer_info.size(), "", 1, BufferKind::INPUT, 0, 0});
    }
    for (float *b : outbuffers) {
        rtlbuffers.emplace_back(b, total_time);
        buffer_info.emplace_back(
            BufferInfo{buffer_info.size(), "", 0, BufferKind::OUTPUT, 0, 0});
    }
    rtlbuffers.emplace_back(corr_with_buffer, total_time);
    buffer_info.emplace_back(
        BufferInfo{buffer_info.size(), "", 0, BufferKind::INPUT, 0, 0});
    for (size_t i = 0; i < buffers.size(); i++) {
        temp.push_back(&buffer_info[i]);
        auto *inbuf = &temp.back();
        temp.push_back(&buffer_info.back());
        temp.push_back(&buffer_info[i + buffers.size()]);
        auto *outbuf = &temp.back();
        mstages.emplace_back(Stage{thefunc, nullptr, 0, inbuf, 2, outbuf, 1, 0,
                                   TaskExecKind::SLICE_BY_TIME,
                                   mstages.size()});
    }
    Context ctx{std::move(rtlbuffers),
                {},
                exec,
                0,
                num_stocks,
                total_time,
                cur_time,
                length,
                8,
                Datatype::Float,
                false,
                nullptr};
    setContextStagesAndRun(ctx, mstages.data(), mstages.size(), exec.get());
}

namespace ops {
void aggregrationFloat(RuntimeStage *stage, size_t stock_idx,
                       size_t __total_time, size_t __start, size_t __length);
void aggregrationDouble(RuntimeStage *stage, size_t stock_idx,
                        size_t __total_time, size_t __start, size_t __length);
} // namespace ops

void aggregrate(std::shared_ptr<Executor> exec, size_t num_aggregrations,
                  float **buffers, float **labels, Datatype dtype,
                  const AggregrationOutput *outbuffers, size_t num_stocks,
                  size_t total_time, size_t cur_time, size_t length) {
    decltype(&ops::aggregrationFloat) thefunc = nullptr;
    size_t simd_len = 0;
    if (dtype == Datatype::Float) {
        simd_len = 8;
        thefunc = &ops::aggregrationFloat;
    } else {
        simd_len = 4;
        thefunc = &ops::aggregrationDouble;
    }
    std::vector<BufferInfo> buffer_info;
    std::vector<Buffer> rtlbuffers;
    std::vector<Stage> mstages;
    std::vector<BufferInfo *> temp;
    // for each aggregration, we need num_aggregration_buffers + 2 buffers
    // 1 for the labels, 1 for the input and num_aggregration_buffers for the
    // output
    rtlbuffers.reserve((AGGREGRATION_NUM_KINDS + 2) * num_aggregrations);
    buffer_info.reserve((AGGREGRATION_NUM_KINDS + 2) * num_aggregrations);
    mstages.reserve(num_aggregrations);
    temp.reserve((AGGREGRATION_NUM_KINDS + 2) * num_aggregrations);
    auto pushBuffer = [&](float *buffer, BufferKind kind, size_t num_users) {
        rtlbuffers.emplace_back(buffer, total_time);
        buffer_info.emplace_back(
            BufferInfo{buffer_info.size(), "", num_users, kind, 0, 0});
        temp.push_back(&buffer_info.back());
    };
    for (size_t i = 0; i < num_aggregrations; i++) {
        float *input = buffers[i];
        float *label = labels[i];
        pushBuffer(label, BufferKind::INPUT, 1);
        auto *inbuf = &temp.back();
        pushBuffer(input, BufferKind::INPUT, 1);
        auto *outbuf = &temp.back() + 1;
        for (size_t j = 0; j < AGGREGRATION_NUM_KINDS; j++) {
            pushBuffer(outbuffers[i].buffers[j], BufferKind::OUTPUT, 0);
        }
        mstages.emplace_back(
            Stage{thefunc, nullptr, 0, inbuf, 2, outbuf, AGGREGRATION_NUM_KINDS,
                  0, TaskExecKind::SLICE_BY_STOCK, mstages.size()});
    }
    Context ctx{std::move(rtlbuffers),
                {},
                exec,
                0,
                num_stocks,
                total_time,
                cur_time,
                length,
                simd_len,
                dtype,
                false,
                nullptr};
    setContextStagesAndRun(ctx, mstages.data(), mstages.size(), exec.get());
}

void runGraph(std::shared_ptr<Executor> exec, const Module *m,
              std::unordered_map<std::string, float *> &buffers,
              size_t num_stocks, size_t total_time, size_t cur_time,
              size_t length) {
    if (m->required_version != VERSION) {
        throw std::runtime_error("The required version in the module does not "
                                 "match the runtime version");
    }
    if (m->output_layout == MemoryLayout::STREAM) {
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
                divideAndCeil(num_stocks, m->blocking_len) * m->blocking_len *
                    length,
                num_stocks,
                total_time,
                cur_time,
                length,
                m->blocking_len,
                m->dtype,
                false,
                nullptr};
    setContextStagesAndRun(ctx, m->stages, m->num_stages, exec.get());
}

AlignedPtr::AlignedPtr(void *ptr, size_t size) noexcept {
    this->ptr = ptr;
    this->size = size;
}
AlignedPtr::AlignedPtr(AlignedPtr &&other) noexcept {
    ptr = other.ptr;
    other.ptr = nullptr;
    size = other.size;
}

void AlignedPtr::release() noexcept {
    if (ptr) {
        kunAlignedFree(ptr);
        ptr = nullptr;
    }
}

AlignedPtr &AlignedPtr::operator=(AlignedPtr &&other) noexcept {
    if (&other == this) {
        return *this;
    }
    release();
    ptr = other.ptr;
    other.ptr = nullptr;
    size = other.size;
    return *this;
}

AlignedPtr::~AlignedPtr() { release(); }

template <typename T>
char *StreamBuffer<T>::make(size_t stock_count, size_t window_size,
                            size_t simd_len) {
    auto ret = kunAlignedAlloc(
        sizeof(T) * simd_len,
        StreamBuffer::getBufferSize(stock_count, window_size, simd_len));
    auto buf = (StreamBuffer *)ret;
    auto data = buf->getBuffer();
    auto rounded_stock_count = roundUp(stock_count, simd_len);
    // fill the buffer with NANs
    for (size_t i = 0; i < rounded_stock_count * window_size; i++) {
        data[i] = NAN;
    }
    for (size_t i = 0; i < divideAndCeil(stock_count, simd_len); i++) {
        *buf->getPos(i, rounded_stock_count, window_size) = 0;
    }
    return (char *)ret;
}

template struct StreamBuffer<float>;
template struct StreamBuffer<double>;

template <typename T>
static void pushBuffer(std::vector<Buffer> &rtlbuffers,
                       std::vector<AlignedPtr> &buffers, size_t num_stocks,
                       size_t blocking_len, const BufferInfo &buf,
                       InputStreamBase *states) {
    auto ptr = StreamBuffer<T>::make(num_stocks, buf.window, blocking_len);
    size_t buf_size =
        StreamBuffer<T>::getBufferSize(num_stocks, buf.window, blocking_len);
    buffers.emplace_back(ptr, buf_size);
    if (states) {
        if (!states->read(ptr, buf_size)) {
            throw std::runtime_error("Failed to read initial stream buffer");
        }
    }
    rtlbuffers.emplace_back((float *)ptr, 1);
}

StreamContext::StreamContext(std::shared_ptr<Executor> exec, const Module *m,
                             size_t num_stocks, InputStreamBase *states)
    : m{m} {
    if (m->required_version != VERSION) {
        throw std::runtime_error("The required version in the module does not "
                                 "match the runtime version");
    }
    if (m->output_layout != MemoryLayout::STREAM) {
        throw std::runtime_error(
            "Cannot run batch mode module via StreamContext");
    }
    std::vector<Buffer> rtlbuffers;
    rtlbuffers.reserve(m->num_buffers);
    buffers.reserve(m->num_buffers);
    if (m->dtype == Datatype::Float) {
        for (size_t i = 0; i < m->num_buffers; i++) {
            pushBuffer<float>(rtlbuffers, buffers, num_stocks, m->blocking_len,
                              m->buffers[i], states);
        }
    } else if (m->dtype == Datatype::Double) {
        for (size_t i = 0; i < m->num_buffers; i++) {
            pushBuffer<double>(rtlbuffers, buffers, num_stocks, m->blocking_len,
                               m->buffers[i], states);
        }
    } else {
        throw std::runtime_error("Unknown type");
    }
    if (m->init_state_buffers) {
        state_buffers = m->init_state_buffers(num_stocks);
        if (states) {
            for (auto &buf : state_buffers) {
                if (!buf->deserialize(states)) {
                    throw std::runtime_error(
                        "Failed to deserialize state buffer");
                }
            }
        } else {
            for (auto &buf : state_buffers) {
                buf->initialize();
            }
        }
    }
    ctx.buffers = std::move(rtlbuffers);
    ctx.executor = exec;
    ctx.buffer_len = num_stocks * 1;
    ctx.stock_count = num_stocks;
    ctx.total_time = 1;
    ctx.start = 0;
    ctx.length = 1;
    ctx.dtype = m->dtype;
    ctx.is_stream = true;
    ctx.simd_len = m->blocking_len;
    ctx.state_buffers = state_buffers.data();
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

const float *StreamContext::getCurrentBufferPtrFloat(size_t handle) const {
    auto buf = (StreamBuffer<float> *)buffers.at(handle).get();
    return buf->getCurrentBufferPtr(ctx.stock_count, m->buffers[handle].window,
                                    m->blocking_len);
}

const double *StreamContext::getCurrentBufferPtrDouble(size_t handle) const {
    auto buf = (StreamBuffer<double> *)buffers.at(handle).get();
    return buf->getCurrentBufferPtr(ctx.stock_count, m->buffers[handle].window,
                                    m->blocking_len);
}

void StreamContext::pushData(size_t handle, const float *data) {
    auto buf = (StreamBuffer<float> *)buffers.at(handle).get();
    float *ptr = buf->pushData(ctx.stock_count, m->buffers[handle].window,
                               m->blocking_len);
    memcpy(ptr, data, ctx.stock_count * sizeof(float));
}

void StreamContext::pushData(size_t handle, const double *data) {
    auto buf = (StreamBuffer<double> *)buffers.at(handle).get();
    double *ptr = buf->pushData(ctx.stock_count, m->buffers[handle].window,
                                m->blocking_len);
    memcpy(ptr, data, ctx.stock_count * sizeof(double));
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

bool StreamContext::serializeStates(OutputStreamBase *stream) {
    for (auto &buf : buffers) {
        if (!stream->write(buf.get(), buf.size)) {
            return false;
        }
    }
    for (auto &ptr : state_buffers) {
        if (!ptr->serialize(stream)) {
            return false;
        }
    }
    return true;
}

StateBuffer *StateBuffer::make(size_t num_objs, size_t elem_size,
                               CtorFn_t ctor_fn, DtorFn_t dtor_fn,
                               SerializeFn_t serialize_fn,
                               DeserializeFn_t deserialize_fn) {
    auto ret = kunAlignedAlloc(KUN_MALLOC_ALIGNMENT,
                               sizeof(StateBuffer) + num_objs * elem_size);
    auto buf = (StateBuffer *)ret;
    buf->num_objs = num_objs;
    buf->elem_size = elem_size;
    buf->initialized = 0;
    buf->ctor_fn = ctor_fn;
    buf->dtor_fn = dtor_fn;
    buf->serialize_fn = serialize_fn;
    buf->deserialize_fn = deserialize_fn;
    return buf;
}

void StateBuffer::Deleter::operator()(StateBuffer *buf) {
    if (buf->initialized) {
        buf->dtor_fn(buf);
    }
    kunAlignedFree(buf);
}
} // namespace kun
