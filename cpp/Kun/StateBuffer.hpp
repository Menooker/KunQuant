#pragma once

#include "Base.hpp"
#include "MathUtil.hpp"
#include <cstdint>
#include <memory>

namespace kun {

#ifdef __AVX__
#define KUN_MALLOC_ALIGNMENT 64 // AVX-512 alignment
#else
#define KUN_MALLOC_ALIGNMENT 16 // NEON alignment
#endif

struct InputStreamBase {
    virtual bool read(void* buf, size_t len) = 0;
    virtual ~InputStreamBase() = default;
};

struct OutputStreamBase {
    virtual bool write(const void* buf, size_t len) = 0;
    virtual ~OutputStreamBase() = default;
};

struct StateBuffer {
    using DtorFn_t = void (*)(StateBuffer *obj);
    using CtorFn_t = void (*)(StateBuffer *obj);
    using SerializeFn_t = bool (*)(StateBuffer *obj, OutputStreamBase *stream);
    using DeserializeFn_t = bool (*)(StateBuffer *obj, InputStreamBase *stream);

    alignas(KUN_MALLOC_ALIGNMENT) size_t num_objs;
    uint32_t elem_size;
    uint32_t initialized;
    CtorFn_t ctor_fn;
    DtorFn_t dtor_fn;
    SerializeFn_t serialize_fn;
    DeserializeFn_t deserialize_fn;
    alignas(KUN_MALLOC_ALIGNMENT) char buf[0];

    KUN_API static StateBuffer *make(size_t num_objs, size_t elem_size,
                             CtorFn_t ctor_fn, DtorFn_t dtor_fn, SerializeFn_t serialize_fn,
                             DeserializeFn_t deserialize_fn);

    // for std::unique_ptr
    struct Deleter {
        KUN_API void operator()(StateBuffer *buf);
    };

    template <typename T>
    T &get(size_t idx) {
        return *reinterpret_cast<T *>(buf + idx * sizeof(T));
    }

    void initialize() {
        initialized = 1;
        ctor_fn(this);
    }
    void destroy() {
        if (initialized) {
            dtor_fn(this);
        }
        initialized = 0;
    }
    bool serialize(OutputStreamBase *stream) {
        return serialize_fn(this, stream);
    }
    bool deserialize(InputStreamBase *stream) {
        if (deserialize_fn(this, stream)) {
            initialized = 1;
            return true;
        }
        return false;
    }
  private:
    StateBuffer() = default;
};

using StateBufferPtr = std::unique_ptr<StateBuffer, StateBuffer::Deleter>;

} // namespace kun